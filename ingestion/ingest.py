import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from warnings import filterwarnings

from agent.db_utils import close_database, db_pool, execute_init_sql, initialize_database
from agent.models import IngestionConfig, IngestionResult
from agent.providers import get_embedding_model
from .chunker import ChunkingConfig, DocumentChunk, create_chunker
from .extract_files import PDFExtractionConfig, create_pdf_extractor

filterwarnings("ignore", category=UserWarning)

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """将文档导入 PostgreSQL + pgvector 的流水线。"""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = False,
        sql_schema_path: str = "sql/schema.sql",
        include_images: bool = True,
        include_tables: bool = True,
    ):
        """初始化导入流水线。"""
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        self.sql_schema_path = sql_schema_path

        # 配置 PDF 提取
        self.extractor_config = PDFExtractionConfig(
            enable_ocr=False,
            images_scale=1.0,
            include_images=include_images,
            include_tables=include_tables,
        )

        # 配置分块
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking,
        )

        # 创建提取器和分块器
        self.extractor = create_pdf_extractor(self.extractor_config)
        self.chunker = create_chunker(self.chunker_config)
        self._initialized = False

    async def initialize(self):
        """初始化数据库连接。"""
        if self._initialized:
            return

        logger.info("Initializing ingestion pipeline...")
        await initialize_database()
        await execute_init_sql(self.sql_schema_path)

        self._initialized = True
        logger.info("Ingestion pipeline initialized")

    async def close(self):
        """关闭数据库连接。"""
        if self._initialized:
            await close_database()
            self._initialized = False

    async def _clean_databases(self):
        """清理数据库表中的已有数据。"""
        logger.warning("Cleaning existing data from databases...")

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")

        logger.info("Cleaned PostgreSQL database")

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """导入单个文档。"""
        start_time = datetime.now()

        document_content, document_metadata = self.extractor.extract_pdf_content(file_path)
        document_source = os.path.relpath(file_path, self.documents_folder)
        document_title = document_metadata.get("title", document_source)

        logger.info(f"Processing document: {document_title}")
        logger.info(
            f"Found {document_metadata.get('pictures', 0)} images and {document_metadata.get('tables', 0)} tables"
        )

        main_chunks = self.chunker.chunk_content(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata,
        )

        if not main_chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        logger.info(f"Total chunks created: {len(main_chunks)}")

        embedded_chunks = await self.aembed_chunks(
            chunks=main_chunks,
            model=get_embedding_model(),
        )
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

        document_id = await self._save_to_postgres(
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata,
        )

        logger.info(f"Saved document to PostgreSQL with ID: {document_id}")

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(main_chunks),
            processing_time_ms=processing_time,
        )

    async def ingest_documents(self, progress_callback: Optional[callable] = None) -> List[IngestionResult]:
        """导入文档目录中的所有文档。"""
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest:
            await self._clean_databases()

        pdf_files = self._find_pdfs_in_directory(self.documents_folder)

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_folder}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        results = []
        for i, file_path in enumerate(pdf_files):
            try:
                logger.info(f"Processing file {i + 1}/{len(pdf_files)}: {file_path}")
                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(pdf_files))
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(
                    IngestionResult(
                        document_id="",
                        title=os.path.basename(file_path),
                        chunks_created=0,
                        processing_time_ms=0,
                    )
                )

        total_chunks = sum(r.chunks_created for r in results)
        logger.info(f"Ingestion complete: {len(results)} documents, {total_chunks} chunks")
        return results

    async def aembed_chunks(
        self,
        chunks: List[DocumentChunk],
        model: str = "text-embedding-v4",
    ) -> List[DocumentChunk]:
        """使用 OpenAI 兼容客户端批量生成分块嵌入向量。"""
        if not chunks:
            return []

        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        texts = [chunk.content for chunk in chunks]
        embedded_chunks = []
        batch_size = 10  # 百炼 text-embedding-v4 单次最多 10 条

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            resp = await client.embeddings.create(
                model=model,
                input=batch_texts,
                dimensions=1024,
                encoding_format="float",
            )
            vectors = [item.embedding for item in resp.data]

            logger.info(
                "Embedding batch %s/%s processed",
                start // batch_size + 1,
                (len(texts) + batch_size - 1) // batch_size,
            )

            for chunk, vector in zip(chunks[start:start + batch_size], vectors):
                embedded_chunk = DocumentChunk(
                    content=chunk.content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={
                        **chunk.metadata,
                        "embedding_model": model,
                        "embedding_generated_at": datetime.now().isoformat(),
                    },
                )
                embedded_chunk.embedding = vector
                embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    def _find_pdfs_in_directory(self, directory: str, recursive: bool = True) -> List[str]:
        """查找目录中的所有 PDF 文件。"""
        directory_path = Path(directory)

        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(f"Directory not found or not a directory: {directory_path}")

        if recursive:
            pdf_files = list(directory_path.rglob("*.pdf"))
        else:
            pdf_files = list(directory_path.glob("*.pdf"))

        pdf_paths = [str(pdf.resolve()) for pdf in pdf_files if pdf.is_file()]
        logger.info(f"Found {len(pdf_paths)} PDF files in {directory_path}")
        return pdf_paths

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any],
    ) -> str:
        """将文档及其分块保存到 PostgreSQL。"""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata),
                )

                document_id = document_result["id"]

                for chunk in chunks:
                    embedding_data = None
                    if hasattr(chunk, "embedding") and chunk.embedding:
                        embedding_data = "[" + ",".join(map(str, chunk.embedding)) + "]"

                    chunk_metadata = {
                        **chunk.metadata,
                        "chunk_type": chunk.metadata.get("content_type", "text"),
                    }

                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(chunk_metadata),
                        chunk.token_count if hasattr(chunk, "token_count") else len(chunk.content.split()),
                    )

                return document_id


async def main():
    """运行导入流程的主函数。"""
    parser = argparse.ArgumentParser(description="Document ingestion with table/image processing")
    parser.add_argument("--documents", "-d", default="documents", help="Documents folder path")
    parser.add_argument("--clean", "-c", action="store_true", help="Clean existing data before ingestion")
    parser.add_argument("--chunk-size", type=int, default=850, help="Chunk size for splitting documents")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic chunking")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--sql-schema-path", "-sql", default="sql/schema.sql", help="Path to SQL schema file")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")
    parser.add_argument("--no-tables", action="store_true", help="Skip table extraction")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast text-only ingestion: disables semantic chunking, image extraction, and table extraction",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    no_semantic = bool(args.no_semantic or args.fast)
    no_images = bool(args.no_images or args.fast)
    no_tables = bool(args.no_tables or args.fast)

    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not no_semantic,
    )
    include_images = not no_images
    include_tables = not no_tables

    if include_images or include_tables:
        logger.warning(
            "Image/table extraction can be slow. Use --fast or --no-images --no-tables for faster text-only ingestion."
        )

    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        sql_schema_path=args.sql_schema_path,
        include_images=include_images,
        include_tables=include_tables,
    )

    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")

    try:
        start_time = datetime.now()
        results = await pipeline.ingest_documents(progress_callback)
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Images extracted: {pipeline.extractor_config.include_images}")
        print(f"Tables extracted: {pipeline.extractor_config.include_tables}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print("=" * 60)

        for result in results:
            if result.chunks_created > 0:
                logger.info(f"{result.title}: {result.chunks_created} chunks ({result.processing_time_ms/1000:.1f}s)")
            else:
                logger.warning(f"{result.title}: Failed to process")

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
