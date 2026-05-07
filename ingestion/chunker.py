import logging
import os
import re
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


@dataclass
class MarkdownSection:
    title: str
    level: int
    path: List[str]
    content: str
    start_line: int
    end_line: int


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

    def _is_standard_paper_section_title(self, title: str) -> bool:
        normalized = str(title or "").strip().lower()
        if not normalized:
            return False
        keywords = [
            "abstract",
            "introduction",
            "related work",
            "background",
            "method",
            "methodology",
            "approach",
            "algorithm",
            "experiment",
            "experiments",
            "evaluation",
            "results",
            "discussion",
            "conclusion",
            "references",
            "appendix",
        ]
        return any(k in normalized for k in keywords)

    def _looks_like_document_title(self, title: str) -> bool:
        value = str(title or "").strip()
        if not value:
            return False
        if self._is_standard_paper_section_title(value):
            return False
        return len(value) >= 40 or ":" in value or " - " in value

    def _build_chunk_title(
        self,
        section: MarkdownSection,
        section_chunk_index: int = 0,
        section_chunk_count: int = 1,
    ) -> str:
        base = " > ".join(section.path).strip() or str(section.title or "").strip() or "Section"
        if section_chunk_count > 1:
            base = f"{base} · Part {section_chunk_index + 1}/{section_chunk_count}"
        if len(base) > 180:
            base = base[:177].rstrip() + "..."
        return base

    def _split_markdown_sections(self, content: str) -> List[MarkdownSection]:
        lines = content.splitlines()
        heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
        matches = []
        for idx, line in enumerate(lines, start=1):
            match = heading_re.match(line)
            if not match:
                continue
            title = str(match.group(2) or "").strip()
            if not title:
                continue
            matches.append((idx, len(match.group(1)), title))

        if not matches:
            return []

        sections: List[MarkdownSection] = []
        first_heading_line = matches[0][0]
        if first_heading_line > 1:
            preamble_lines = lines[: first_heading_line - 1]
            preamble_text = "\n".join(preamble_lines).strip()
            if preamble_text:
                sections.append(
                    MarkdownSection(
                        title="Document Preamble",
                        level=0,
                        path=["Document Preamble"],
                        content=preamble_text,
                        start_line=1,
                        end_line=first_heading_line - 1,
                    )
                )

        path_stack: List[str] = []
        title_root: Optional[str] = None
        for idx, (start_line, level, title) in enumerate(matches):
            is_standard = self._is_standard_paper_section_title(title)
            if idx == 0 and level == 1 and self._looks_like_document_title(title):
                title_root = title

            force_top_level = False
            if level == 1 and is_standard:
                force_top_level = True
            elif level == 2 and is_standard:
                current_parent = path_stack[0] if path_stack else ""
                if current_parent and self._is_standard_paper_section_title(current_parent):
                    force_top_level = True
                elif current_parent and title_root and current_parent == title_root:
                    force_top_level = True

            if force_top_level:
                path_stack = [title]
            else:
                if len(path_stack) >= level:
                    path_stack = path_stack[: level - 1]
                path_stack.append(title)

            end_line = matches[idx + 1][0] - 1 if idx + 1 < len(matches) else len(lines)
            section_lines = lines[start_line - 1 : end_line]
            section_text = "\n".join(section_lines).strip()
            if not section_text:
                continue
            sections.append(
                MarkdownSection(
                    title=title,
                    level=level,
                    path=list(path_stack),
                    content=section_text,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        return sections

    def _chunk_whole_document(
        self,
        content: str,
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
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
        return chunks

    def _chunk_section(
        self,
        section: MarkdownSection,
        base_metadata: Dict[str, Any],
        title: str,
        source: str,
    ) -> List[DocumentChunk]:
        section_text = section.content.strip()
        if len(section_text) < self.config.min_chunk_size:
            return []

        section_metadata = {
            **base_metadata,
            "title": title,
            "source": source,
            "section_title": section.title,
            "section_level": section.level,
            "section_path": list(section.path),
            "section_path_text": " > ".join(section.path),
            "section_start_line": section.start_line,
            "section_end_line": section.end_line,
        }

        section_chunks: List[DocumentChunk] = []
        if len(section_text) <= self.config.max_chunk_size:
            section_chunk_count = 1
            chunk_title = self._build_chunk_title(section, 0, section_chunk_count)
            section_path_text = " > ".join(section.path).strip()
            prefix_parts: List[str] = []
            if section_path_text:
                prefix_parts.append(f"[Section: {section_path_text}]")
            if section_chunk_count > 1:
                prefix_parts.append(f"[Chunk: {chunk_title}]")
            prefix = "\n".join(prefix_parts).strip()
            enhanced_content = f"{prefix}\n\n{section_text}".strip() if prefix else section_text
            section_chunks.append(
                DocumentChunk(
                    content=enhanced_content,
                    index=0,
                    start_char=0,
                    end_char=len(enhanced_content),
                    metadata={
                        **section_metadata,
                        "chunk_method": "section",
                        "raw_chunk_size": len(section_text),
                        "chunk_title": chunk_title,
                        "retrieval_title": chunk_title,
                    },
                )
            )
        else:
            docs = self.fallback_splitter.split_documents(
                [Document(page_content=section_text, metadata=section_metadata)]
            )
            filtered_docs = []
            for doc_chunk in docs:
                text = doc_chunk.page_content.strip()
                if len(text) >= self.config.min_chunk_size:
                    filtered_docs.append(text)
            section_chunk_count = len(filtered_docs)
            for idx, text in enumerate(filtered_docs):
                chunk_title = self._build_chunk_title(section, idx, section_chunk_count)
                section_path_text = " > ".join(section.path).strip()
                prefix_parts: List[str] = []
                if section_path_text:
                    prefix_parts.append(f"[Section: {section_path_text}]")
                if section_chunk_count > 1:
                    prefix_parts.append(f"[Chunk: {chunk_title}]")
                prefix = "\n".join(prefix_parts).strip()
                enhanced_content = f"{prefix}\n\n{text}".strip() if prefix else text
                section_chunks.append(
                    DocumentChunk(
                        content=enhanced_content,
                        index=idx,
                        start_char=0,
                        end_char=len(enhanced_content),
                        metadata={
                            **section_metadata,
                            "chunk_method": "section_recursive",
                            "section_chunk_index": idx,
                            "section_chunk_count": section_chunk_count,
                            "raw_chunk_size": len(text),
                            "chunk_title": chunk_title,
                            "retrieval_title": chunk_title,
                        },
                    )
                )
            return section_chunks

        section_chunk_count = len(section_chunks)
        for idx, chunk in enumerate(section_chunks):
            chunk.index = idx
            chunk.metadata["section_chunk_index"] = idx
            chunk.metadata["section_chunk_count"] = section_chunk_count
        return section_chunks

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
        try:
            sections = self._split_markdown_sections(content)
            if sections:
                final_chunks: List[DocumentChunk] = []
                for section in sections:
                    final_chunks.extend(
                        self._chunk_section(
                            section=section,
                            base_metadata=base_metadata,
                            title=title,
                            source=source,
                        )
                    )
                if final_chunks:
                    total_chunks = len(final_chunks)
                    for idx, chunk in enumerate(final_chunks):
                        text = chunk.content.strip()
                        chunk.index = idx
                        chunk.metadata.update(
                            {
                                "chunk_index": idx,
                                "total_chunks": total_chunks,
                                "chunk_size": len(text),
                            }
                        )
                    return final_chunks
        except Exception as e:
            logger.warning(f"Section-aware chunking failed, falling back to whole-document splitting: {e}")

        chunks = self._chunk_whole_document(content, base_metadata)

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
