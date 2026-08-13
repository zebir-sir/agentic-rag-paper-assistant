import argparse
import asyncio
from collections import Counter
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from warnings import filterwarnings

from agent.db_utils import close_database, db_pool, execute_init_sql, get_document, get_document_pdf_bytes, initialize_database
from agent.models import IngestionConfig, IngestionResult
from agent.providers import build_embedding_request_kwargs, get_embedding_model
from agent.embedding_runtime import detect_embedding_language, get_embedding_client_for_route, get_embedding_route
from agent.graph_runtime import refresh_paper_graph
from .chunker import ChunkingConfig, DocumentChunk, create_chunker
from .extract_files import PDFExtractionConfig, create_pdf_extractor
from .vision_client import ArkVisionClient, figure_context

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
        reset_kb_before_ingest: bool = False,
        sql_schema_path: str = "sql/schema.sql",
        include_images: bool = True,
        include_tables: bool = True,
    ):
        """初始化导入流水线。"""
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        self.reset_kb_before_ingest = reset_kb_before_ingest
        self.sql_schema_path = sql_schema_path

        # 配置 PDF 提取
        self.extractor_config = PDFExtractionConfig(
            enable_ocr=False,
            images_scale=1.0,
            include_images=include_images,
            include_tables=include_tables,
            image_output_dir=os.getenv("VISION_IMAGE_ASSET_DIR", os.path.join(documents_folder, ".vision_assets")),
            max_images=None,
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
        if os.getenv("ALLOW_DESTRUCTIVE_INGEST_CLEAN", "").strip().lower() != "true":
            raise RuntimeError(
                "--clean 会删除 sessions/messages/documents/chunks。"
                "如确认要执行，请设置环境变量 ALLOW_DESTRUCTIVE_INGEST_CLEAN=true 后重试。"
            )
        logger.warning("Cleaning existing data from databases...")

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")

        logger.info("Cleaned PostgreSQL database")

    async def _reset_knowledge_base(self):
        if os.getenv("ALLOW_KB_RESET", "").strip().lower() != "true":
            raise RuntimeError(
                "--reset-kb 会删除 documents/chunks 并要求重新入库。"
                "如确认要执行，请设置环境变量 ALLOW_KB_RESET=true 后重试。"
            )

        logger.warning("Resetting knowledge base: deleting chunks and documents only...")
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        logger.info("Knowledge base reset complete; sessions/messages were preserved.")

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """导入单个文档。"""
        start_time = datetime.now()

        print("INGEST_STAGE=extract", flush=True)
        document_content, document_metadata = self.extractor.extract_pdf_content(file_path)
        document_source = os.path.relpath(file_path, self.documents_folder)
        document_title = str(document_metadata.get("title") or document_source).strip()
        document_metadata["document_language"] = detect_embedding_language(document_content)
        document_metadata["include_artifacts"] = bool(self.extractor_config.include_images or self.extractor_config.include_tables)

        print("INGEST_STAGE=vision", flush=True)
        vision_chunks = await self._build_vision_chunks(document_content, document_metadata, document_title, document_source)
        if vision_chunks:
            document_metadata["vision_analyses"] = [chunk.metadata.get("vision_analysis") for chunk in vision_chunks]

        logger.info(f"Processing document: {document_title}")
        logger.info(
            f"Found {document_metadata.get('pictures', 0)} images and {document_metadata.get('tables', 0)} tables"
        )

        print("INGEST_STAGE=structure", flush=True)
        main_chunks = self.chunker.chunk_content(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata,
        )

        main_chunks.extend(vision_chunks)

        if not main_chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        # 视觉证据块在普通文本分块之后追加，重新编号保证数据库排序稳定。
        total_chunks = len(main_chunks)
        for index, chunk in enumerate(main_chunks):
            chunk.index = index
            chunk.metadata.update({"chunk_index": index, "total_chunks": total_chunks})
        logger.info(f"Total chunks created: {len(main_chunks)}")
        method_counts = Counter(
            str((chunk.metadata or {}).get("chunk_method") or "unknown")
            for chunk in main_chunks
        )
        section_titles = {
            str((chunk.metadata or {}).get("section_title") or "").strip()
            for chunk in main_chunks
            if str((chunk.metadata or {}).get("section_title") or "").strip()
        }
        logger.info("Chunk method distribution for %s: %s", document_title, dict(method_counts))
        logger.info("Detected %s unique sections for %s", len(section_titles), document_title)

        print("INGEST_STAGE=embed", flush=True)
        embedded_chunks = await self.aembed_chunks(
            chunks=main_chunks,
            model=get_embedding_model(),
        )
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

        print("INGEST_STAGE=persist", flush=True)
        document_id = await self._save_to_postgres(
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata,
            pdf_bytes=Path(file_path).read_bytes(),
        )

        try:
            print("INGEST_STAGE=graph", flush=True)
            await refresh_paper_graph(document_id)
        except Exception as exc:
            logger.exception("Paper graph refresh failed for %s: %s", document_id, exc)

        logger.info(f"Saved document to PostgreSQL with ID: {document_id}")

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(main_chunks),
            processing_time_ms=processing_time,
        )

    async def upgrade_document_to_full(self, document_id: str) -> IngestionResult:
        """Rebuild one fast-ingested paper into full text/algorithm/table/figure evidence."""
        document = await get_document(document_id)
        pdf_bytes = await get_document_pdf_bytes(document_id)
        if not document or not pdf_bytes:
            raise ValueError("该论文没有保存可用于补充入库的原始 PDF")

        import tempfile
        with tempfile.TemporaryDirectory(prefix="upgrade_ingest_") as temp_dir:
            path = Path(temp_dir) / "paper.pdf"
            path.write_bytes(pdf_bytes)
            content, metadata = self.extractor.extract_pdf_content(str(path))
            metadata["document_language"] = detect_embedding_language(content)
            metadata["include_artifacts"] = True
            title = str(document["title"])
            source = str(document["source"])
            vision_chunks = await self._build_vision_chunks(content, metadata, title, source)
            chunks = self.chunker.chunk_content(content, title=title, source=source, metadata=metadata)
            chunks.extend(vision_chunks)
            for index, chunk in enumerate(chunks):
                chunk.index = index
                chunk.metadata.update({"chunk_index": index, "total_chunks": len(chunks)})
            embedded = await self.aembed_chunks(chunks, model=get_embedding_model())

            async with db_pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("DELETE FROM chunks WHERE document_id=$1::uuid", document_id)
                    await conn.execute("DELETE FROM artifacts WHERE document_id=$1::uuid", document_id)
                    await self._save_visual_artifacts(conn, document_id, embedded)
                    for chunk in embedded:
                        embedding_data = "[" + ",".join(map(str, chunk.embedding)) + "]"
                        metadata_value = {**chunk.metadata, "chunk_type": chunk.metadata.get("content_type", "text")}
                        await conn.execute(
                            """INSERT INTO chunks(document_id,content,embedding,chunk_index,metadata,token_count)
                               VALUES($1::uuid,$2,$3::vector,$4,$5::jsonb,$6)""",
                            document_id, chunk.content, embedding_data, chunk.index,
                            json.dumps(metadata_value, ensure_ascii=False), chunk.token_count,
                        )
                    metadata["include_artifacts"] = True
                    metadata["upgraded_to_full_at"] = datetime.now().isoformat()
                    await conn.execute(
                        "UPDATE documents SET content=$2, metadata=$3::jsonb, updated_at=CURRENT_TIMESTAMP WHERE id=$1::uuid",
                        document_id, content, json.dumps(metadata, ensure_ascii=False),
                    )
            try:
                await refresh_paper_graph(document_id)
            except Exception as exc:
                logger.exception("Paper graph refresh failed for upgraded document %s: %s", document_id, exc)
            return IngestionResult(document_id=document_id, title=title, chunks_created=len(embedded), processing_time_ms=0)

    async def _build_vision_chunks(
        self,
        markdown: str,
        metadata: Dict[str, Any],
        title: str,
        source: str,
    ) -> List[DocumentChunk]:
        """将已导出的论文图逐张转为带图注与邻近语境的检索证据块。"""
        assets = metadata.get("vision_assets") or []
        if not assets:
            return []
        chunks: List[DocumentChunk] = []
        failures: List[str] = []
        for asset in assets:
            caption = str(asset.get("caption") or "").strip()
            context_before, context_after = figure_context(markdown, caption)
            try:
                analysis = await ArkVisionClient().analyze_figure(
                    asset["path"], caption=caption, context_before=context_before, context_after=context_after
                )
            except Exception as exc:
                failures.append(str(exc)[:300])
                logger.warning("Vision analysis skipped for %s: %s", source, exc)
                continue
            analysis_dict = analysis.to_dict()
            figure_number = int(asset.get("index", len(chunks))) + 1
            body = (
            "[Artifact: Figure]\n"
            f"[Document: {title}]\n"
            f"[Caption: {caption or '[无图注]'}]\n"
            f"[Context before: {context_before or '[无]'}]\n"
            f"[Context after: {context_after or '[无]'}]\n\n"
            f"Visual summary: {analysis.summary or '[无]'}\n"
            f"Figure type: {analysis.figure_type}\n"
            f"Research purpose: {analysis.research_purpose or '[无]'}\n"
            f"Experimental task: {analysis.experimental_task or '[无]'}\n"
            f"Axes and units: {'; '.join(analysis.axes_and_units) or '[无]'}\n"
            f"Series or methods: {'; '.join(analysis.series_or_methods) or '[无]'}\n"
            f"Visible text: {'; '.join(analysis.visible_text) or '[无]'}\n"
            f"Observations: {'; '.join(analysis.observations) or '[无]'}\n"
            f"Quantitative findings: {'; '.join(analysis.quantitative_findings) or '[无]'}\n"
            f"Comparative claims: {'; '.join(analysis.comparative_claims) or '[无]'}\n"
            f"Visual tags: {'; '.join(analysis.visual_tags) or '[无]'}\n"
            f"Evidence confidence: {analysis.evidence_confidence}\n"
                f"Limitations: {'; '.join(analysis.limitations) or '[无]'}"
            )
            chunks.append(DocumentChunk(
            content=body,
            index=0,
            start_char=0,
            end_char=len(body),
            metadata={
                "title": title,
                "source": source,
                "content_type": "artifact",
                "artifact_type": "figure",
                "chunk_method": "vision_figure",
                "retrieval_title": caption or f"{title} figure {figure_number}",
                "caption": caption,
                "image_asset_path": asset["path"],
                "vision_analysis": analysis_dict,
                "vision_context_before": context_before,
                "vision_context_after": context_after,
                "figure_number": figure_number,
                "page_number": asset.get("page"),
                # 图卡由视觉模型生成时可能使用与原论文不同的表述语言；
                # 检索向量仍必须与整篇论文共享同一语言模型。
                "document_language": str(metadata.get("document_language") or "").strip().lower(),
            },
            ))
        metadata["vision_analysis_status"] = "success" if chunks else "failed"
        metadata["vision_model"] = chunks[0].metadata["vision_analysis"]["model"] if chunks else None
        if failures:
            metadata["vision_analysis_errors"] = failures
        return chunks

    @staticmethod
    def _media_type_for_path(path: str) -> str:
        suffix = Path(path).suffix.lower()
        return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
            suffix, "image/png"
        )

    async def _save_visual_artifacts(self, conn: Any, document_id: str, chunks: List[DocumentChunk]) -> None:
        """Persist figures and tables as complete research evidence records.

        Only figure/table chunks are promoted. Algorithms deliberately keep the
        existing text-chunk representation because their original text is already
        directly searchable and citable.
        """
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            artifact_type = str(metadata.get("artifact_type") or "").lower()
            if metadata.get("content_type") != "artifact" or artifact_type not in {"figure", "table"}:
                continue

            image_path = str(metadata.get("image_asset_path") or "")
            image_blob = None
            image_media_type = None
            if artifact_type == "figure" and image_path:
                path = Path(image_path)
                if path.is_file():
                    image_blob = path.read_bytes()
                    image_media_type = self._media_type_for_path(image_path)

            structured_data = dict(metadata.get("vision_analysis") or {})
            structured_data.update(
                {
                    "artifact_index": metadata.get("artifact_index"),
                    "figure_number": metadata.get("figure_number"),
                    "artifact_start_line": metadata.get("artifact_start_line"),
                    "artifact_end_line": metadata.get("artifact_end_line"),
                    "document_language": metadata.get("document_language"),
                }
            )
            artifact_row = await conn.fetchrow(
                """
                INSERT INTO artifacts (
                    document_id, artifact_type, caption, page_number, section_path,
                    context_before, context_after, raw_content, structured_data,
                    retrieval_text, image_blob, image_media_type
                )
                VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11, $12)
                RETURNING id::text
                """,
                document_id,
                artifact_type,
                str(metadata.get("caption") or ""),
                metadata.get("page") or metadata.get("page_number"),
                str(metadata.get("section_path_text") or ""),
                str(metadata.get("vision_context_before") or metadata.get("context_before") or ""),
                str(metadata.get("vision_context_after") or metadata.get("context_after") or ""),
                str(metadata.get("raw_artifact_content") or chunk.content),
                json.dumps(structured_data, ensure_ascii=False),
                chunk.content,
                image_blob,
                image_media_type,
            )
            metadata["artifact_id"] = artifact_row["id"]
            chunk.metadata = metadata

    async def ingest_documents(self, progress_callback: Optional[callable] = None) -> List[IngestionResult]:
        """导入文档目录中的所有文档。"""
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest and self.reset_kb_before_ingest:
            raise ValueError("Use either --clean or --reset-kb, not both.")

        if self.clean_before_ingest:
            await self._clean_databases()

        if self.reset_kb_before_ingest:
            await self._reset_knowledge_base()

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

        failed_results = [result for result in results if not str(result.document_id or "").strip()]
        if failed_results:
            failed_titles = ", ".join(str(result.title or "unknown") for result in failed_results[:3])
            raise RuntimeError(f"{len(failed_results)} document(s) were not persisted: {failed_titles}")

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

        routed_chunks: Dict[tuple[str, str], List[DocumentChunk]] = {}
        for chunk in chunks:
            route = get_embedding_route(language=str(chunk.metadata.get("document_language") or "") or None)
            routed_chunks.setdefault((route.language, route.model), []).append(chunk)

        embedded_by_index: Dict[int, DocumentChunk] = {}
        batch_size = 10  # 百炼 text-embedding-v4 单次最多 10 条

        for (language, route_model), route_chunks in routed_chunks.items():
            route = get_embedding_route(language=language)
            client = get_embedding_client_for_route(route)
            for start in range(0, len(route_chunks), batch_size):
                batch_chunks = route_chunks[start:start + batch_size]
                try:
                    resp = await client.embeddings.create(
                        **build_embedding_request_kwargs(
                            model=route_model,
                            input_value=[chunk.content for chunk in batch_chunks],
                            encoding_format="float",
                        )
                    )
                    vectors = [item.embedding for item in resp.data]
                except Exception as exc:
                    logger.warning(
                        "Embedding batch %s-%s was rejected; isolating individual chunks: %s",
                        batch_chunks[0].index,
                        batch_chunks[-1].index,
                        exc,
                    )
                    vectors = [
                        await self._embed_chunk_with_formula_fallback(client, route_model, chunk)
                        for chunk in batch_chunks
                    ]
                for chunk, vector in zip(batch_chunks, vectors):
                    embedded_chunk = DocumentChunk(
                        content=chunk.content,
                        index=chunk.index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={
                            **chunk.metadata,
                            "embedding_model": route_model,
                            "embedding_language": language,
                            "embedding_generated_at": datetime.now().isoformat(),
                        },
                    )
                    embedded_chunk.embedding = vector
                    embedded_by_index[chunk.index] = embedded_chunk
        return [embedded_by_index[chunk.index] for chunk in chunks]

    async def _embed_chunk_with_formula_fallback(
        self,
        client: AsyncOpenAI,
        model: str,
        chunk: DocumentChunk,
    ) -> List[float]:
        """Keep the original retrieval chunk while splitting only a rejected model input."""
        try:
            response = await client.embeddings.create(
                **build_embedding_request_kwargs(
                    model=model,
                    input_value=[chunk.content],
                    encoding_format="float",
                )
            )
            return response.data[0].embedding
        except Exception as exc:
            fragments = [chunk.content[offset:offset + 500] for offset in range(0, len(chunk.content), 500)]
            if len(fragments) < 2:
                raise RuntimeError(
                    f"Embedding rejected chunk {chunk.index} ({len(chunk.content)} chars): {exc}"
                ) from exc
            logger.warning(
                "Embedding formula-dense chunk %s (%s chars) as %s weighted fragments",
                chunk.index,
                len(chunk.content),
                len(fragments),
            )
            vectors_and_weights: List[tuple[List[float], int]] = []
            for fragment in fragments:
                response = await client.embeddings.create(
                    **build_embedding_request_kwargs(
                        model=model,
                        input_value=[fragment],
                        encoding_format="float",
                    )
                )
                vectors_and_weights.append((response.data[0].embedding, len(fragment)))
            total_weight = sum(weight for _, weight in vectors_and_weights)
            return [
                sum(vector[dimension] * weight for vector, weight in vectors_and_weights) / total_weight
                for dimension in range(len(vectors_and_weights[0][0]))
            ]

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
        pdf_bytes: Optional[bytes] = None,
    ) -> str:
        """将文档及其分块保存到 PostgreSQL。"""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata, pdf_blob, pdf_media_type)
                    VALUES ($1, $2, $3, $4, $5, 'application/pdf')
                    RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata, ensure_ascii=False),
                    pdf_bytes,
                )

                document_id = document_result["id"]

                await self._save_visual_artifacts(conn, document_id, chunks)

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
                        json.dumps(chunk_metadata, ensure_ascii=False),
                        chunk.token_count if hasattr(chunk, "token_count") else len(chunk.content.split()),
                    )

                return document_id


async def main():
    """运行导入流程的主函数。"""
    parser = argparse.ArgumentParser(description="Document ingestion with table/image processing")
    parser.add_argument("--documents", "-d", default="documents", help="Documents folder path")
    parser.add_argument("--clean", "-c", action="store_true", help="Clean existing data before ingestion")
    parser.add_argument(
        "--reset-kb",
        action="store_true",
        help="Delete existing documents and chunks before ingestion, while preserving sessions and messages.",
    )
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
        reset_kb_before_ingest=args.reset_kb,
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
