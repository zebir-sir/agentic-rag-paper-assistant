import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class ChunkingConfig:
    """PDF 分块的基础配置。"""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    use_semantic_splitting: bool = True

    def __post_init__(self):
        """校验配置。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


@dataclass
class DocumentChunk:
    """表示一个文档分块。"""

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None

    def __post_init__(self):
        """如果未提供 token 数，则自动计算。"""
        if self.token_count is None:
            self.token_count = len(self.content) // 4


class PDFSemanticChunker:
    """PDF 文档的语义分块器。"""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.embeddings = None
        self.semantic_splitter = None

        # 语义切分器
        if config.use_semantic_splitting:
            self.embeddings = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_BASE_URL"),
                tiktoken_enabled=False,
            )
            self.semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",  # 按百分位阈值切分差异点
            )

        # 递归切分器
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )

    def chunk_content(
        self,
        content: str,
        title: str = "PDF Document",
        source: str = "pdf",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """将 PDF 内容切分为语义片段，并转换为 DocumentChunk 对象。"""
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "content_type": "pdf",
            **(metadata or {}),
        }

        doc = Document(page_content=content, metadata=base_metadata)

        try:
            if self.config.use_semantic_splitting and len(content) > self.config.chunk_size:
                chunks = self.semantic_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["chunk_method"] = "semantic"
            else:
                chunks = self.fallback_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["chunk_method"] = "recursive"

        except Exception as e:
            logger.warning(f"Semantic chunking failed, using fallback: {e}")
            chunks = self.fallback_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata["chunk_method"] = "fallback"

        # 过滤过小的分块，并转换为 DocumentChunk
        final_chunks = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content.strip()
            if len(text) >= self.config.min_chunk_size:
                chunk.metadata.update(
                    {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(text),
                    }
                )
                final_chunks.append(
                    DocumentChunk(
                        content=text,
                        index=i,
                        start_char=0,
                        end_char=len(text),
                        metadata=chunk.metadata,
                    )
                )

        return final_chunks

    def chunk_pdf_documents(self, documents: List[Document]) -> List[Document]:
        """对 PDF 文档进行分块。"""
        all_chunks = []

        for doc in documents:
            title = doc.metadata.get("title", "PDF Document")
            source = doc.metadata.get("source", "pdf")

            chunks = self.chunk_content(
                content=doc.page_content,
                title=title,
                source=source,
                metadata=doc.metadata,
            )
            all_chunks.extend(chunks)

        return all_chunks


def create_chunker(config: ChunkingConfig) -> PDFSemanticChunker:
    """使用基础配置创建 PDF 分块器。"""
    return PDFSemanticChunker(config)
