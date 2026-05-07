from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .agent_runtime import AgentDependencies, _resolve_default_search_type
from .tool_payloads import (
    collect_hits,
    run_get_document_payload,
    run_hybrid_search_payload,
    run_list_documents_payload,
    run_openalex_payload,
    run_section_search_payload,
    run_vector_search_payload,
)
from .tools import (
    web_search_tool,
    WebSearchInput,
)


class KnowledgeSearchArgs(BaseModel):
    query: str
    limit: Optional[int] = None


class VectorSearchArgs(BaseModel):
    query: str
    limit: int = 10


class HybridSearchArgs(BaseModel):
    query: str
    limit: int = 10
    text_weight: float = 0.3


class SectionSearchArgs(BaseModel):
    query: str
    section_query: str
    document_id: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)


class DocumentArgs(BaseModel):
    document_id: str


class DocumentListArgs(BaseModel):
    limit: int = 20
    offset: int = 0


class OpenAlexSearchArgs(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=10)


class WebSearchArgs(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=10)


def build_langchain_tools(deps: AgentDependencies) -> list:
    async def vector_search_async(query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return await run_vector_search_payload(deps=deps, query=query, limit=limit)

    async def hybrid_search_async(
        query: str,
        limit: int = 10,
        text_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        return await run_hybrid_search_payload(
            deps=deps,
            query=query,
            limit=limit,
            text_weight=text_weight,
        )

    async def search_knowledge_base_async(
        query: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        search_type = _resolve_default_search_type(deps)
        default_limit = int((deps.search_preferences or {}).get("default_limit", 10) or 10)
        effective_limit = max(1, min(int(limit or default_limit), 50))
        if search_type == "vector":
            return await vector_search_async(query=query, limit=effective_limit)
        return await hybrid_search_async(query=query, limit=effective_limit)

    async def section_search_async(
        query: str,
        section_query: str,
        document_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return await run_section_search_payload(
            deps=deps,
            query=query,
            section_query=section_query,
            document_id=document_id,
            limit=limit,
        )

    async def get_document_async(document_id: str) -> Optional[Dict[str, Any]]:
        return await run_get_document_payload(deps=deps, document_id=document_id)

    async def list_documents_async(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        return await run_list_documents_payload(limit=limit, offset=offset)

    async def search_openalex_papers_async(query: str, limit: int = 5) -> List[Dict[str, Any]]:
        return await run_openalex_payload(deps=deps, query=query, limit=limit)

    async def search_web_async(query: str, limit: int = 5) -> List[Dict[str, Any]]:
        results = await web_search_tool(WebSearchInput(query=query, limit=limit))
        payload: List[Dict[str, Any]] = []
        for item in results:
            title = item.get("title") or "Web Result"
            url = str(item.get("url") or "").strip()
            snippet = item.get("snippet") or ""
            source = item.get("source") or url
            metadata = {
                "source_type": "web",
                "source_kind": "general_web",
                "url": url,
                "provider": item.get("provider"),
                "published_date": item.get("published_date"),
            }
            hit = {
                "content": snippet,
                "score": None,
                "document_title": title,
                "document_source": source,
                "chunk_id": url or None,
                "document_id": url or None,
                "metadata": metadata,
            }
            collect_hits(deps, [hit])
            payload.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": source,
                    "published_date": item.get("published_date"),
                }
            )
        return payload

    return [
        StructuredTool.from_function(
            coroutine=search_knowledge_base_async,
            name="search_knowledge_base",
            description=(
                "Search uploaded/local papers in the knowledge base using configured default search type. "
                "Use when the user asks about uploaded documents, paper summaries, methods, experiments, "
                "limitations, or evidence from local papers."
            ),
            args_schema=KnowledgeSearchArgs,
        ),
        StructuredTool.from_function(
            coroutine=section_search_async,
            name="section_search",
            description=(
                "Search chunks within a specific paper section using section metadata such as "
                "section_title or section_path_text. Use when the user asks for Method/Experiments/"
                "References/Abstract/Conclusion or a specific section."
            ),
            args_schema=SectionSearchArgs,
        ),
        StructuredTool.from_function(
            coroutine=vector_search_async,
            name="vector_search",
            description="Run vector search against local knowledge base.",
            args_schema=VectorSearchArgs,
        ),
        StructuredTool.from_function(
            coroutine=hybrid_search_async,
            name="hybrid_search",
            description="Run hybrid search against local knowledge base.",
            args_schema=HybridSearchArgs,
        ),
        StructuredTool.from_function(
            coroutine=get_document_async,
            name="get_document",
            description="Get full document by document_id.",
            args_schema=DocumentArgs,
        ),
        StructuredTool.from_function(
            coroutine=list_documents_async,
            name="list_documents",
            description="List indexed documents.",
            args_schema=DocumentListArgs,
        ),
        StructuredTool.from_function(
            coroutine=search_openalex_papers_async,
            name="search_openalex_papers",
            description=(
                "Search academic papers and metadata via OpenAlex. Use for related work, literature discovery, "
                "authors, publication year, DOI, venue, open-access links, and papers outside the local knowledge base."
            ),
            args_schema=OpenAlexSearchArgs,
        ),
        StructuredTool.from_function(
            coroutine=search_web_async,
            name="search_web",
            description=(
                "Search the open web. Use for current information, general technical explanations, web sources, "
                "non-paper sources, and open-domain questions."
            ),
            args_schema=WebSearchArgs,
        ),
    ]
