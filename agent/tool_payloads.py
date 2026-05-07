from typing import Any, Dict, List, Optional

from .agent_runtime import AgentDependencies, _collect_evidence_hit
from .tools import (
    DocumentInput,
    DocumentListInput,
    HybridSearchInput,
    OpenAlexSearchInput,
    SectionSearchInput,
    VectorSearchInput,
    get_document_tool,
    hybrid_search_tool,
    list_documents_tool,
    openalex_search_tool,
    section_search_tool,
    vector_search_tool,
)

def _mark_retrieval_error(deps: AgentDependencies, message: str) -> None:
    prefs = dict(deps.search_preferences or {})
    prefs["retrieval_error"] = message
    deps.search_preferences = prefs


def chunk_result_to_payload(result: Any) -> Dict[str, Any]:
    return {
        "content": result.content,
        "score": result.score,
        "document_title": result.document_title,
        "document_source": result.document_source,
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "metadata": result.metadata,
    }


def collect_hits(deps: AgentDependencies, payloads: List[Dict[str, Any]]) -> None:
    for hit in payloads:
        _collect_evidence_hit(deps, hit)


async def run_vector_search_payload(deps: AgentDependencies, query: str, limit: int) -> List[Dict[str, Any]]:
    try:
        results = await vector_search_tool(VectorSearchInput(query=query, limit=limit))
        payload = [chunk_result_to_payload(r) for r in results]
        collect_hits(deps, payload)
        return payload
    except Exception as exc:
        _mark_retrieval_error(deps, f"vector_search_failed:{type(exc).__name__}")
        return []


async def run_hybrid_search_payload(
    deps: AgentDependencies,
    query: str,
    limit: int,
    text_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    try:
        results = await hybrid_search_tool(HybridSearchInput(query=query, limit=limit, text_weight=text_weight))
        payload = [chunk_result_to_payload(r) for r in results]
        collect_hits(deps, payload)
        return payload
    except Exception as exc:
        _mark_retrieval_error(deps, f"hybrid_search_failed:{type(exc).__name__}")
        return []


async def run_get_document_payload(deps: AgentDependencies, document_id: str) -> Optional[Dict[str, Any]]:
    try:
        document = await get_document_tool(DocumentInput(document_id=document_id))
        if not document:
            return None
        collect_hits(
            deps,
            [
                {
                    "document_id": document.get("id"),
                    "document_title": document.get("title"),
                    "document_source": document.get("source"),
                    "chunk_id": None,
                    "content": document.get("content", ""),
                    "score": None,
                    "metadata": {},
                }
            ],
        )
        return {
            "id": document["id"],
            "title": document["title"],
            "source": document["source"],
            "content": document["content"],
            "chunk_count": len(document.get("chunks", [])),
            "created_at": document["created_at"],
        }
    except Exception as exc:
        _mark_retrieval_error(deps, f"get_document_failed:{type(exc).__name__}")
        return None


async def run_list_documents_payload(limit: int, offset: int) -> List[Dict[str, Any]]:
    documents = await list_documents_tool(DocumentListInput(limit=limit, offset=offset))
    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat(),
        }
        for d in documents
    ]


async def run_openalex_payload(deps: AgentDependencies, query: str, limit: int) -> List[Dict[str, Any]]:
    results = await openalex_search_tool(OpenAlexSearchInput(query=query, limit=limit))
    payload: List[Dict[str, Any]] = []
    for item in results:
        abstract = str(item.get("abstract") or "").strip()
        snippet = abstract or f"{item.get('title', '')} ({item.get('year') or 'N/A'})"
        metadata = {
            "source_type": "web",
            "source_kind": "openalex",
            "authors": item.get("authors") or [],
            "year": item.get("year"),
            "venue": item.get("source"),
            "doi": item.get("doi"),
            "landing_page_url": item.get("landing_page_url"),
            "pdf_url": item.get("pdf_url"),
            "openalex_id": item.get("openalex_id"),
            "is_oa": item.get("is_oa"),
            "has_pdf": item.get("has_pdf"),
            "has_fulltext": item.get("has_fulltext"),
            "cited_by_count": item.get("cited_by_count"),
        }
        collect_hits(
            deps,
            [
                {
                    "content": snippet,
                    "score": None,
                    "document_title": item.get("title") or "OpenAlex Paper",
                    "document_source": item.get("source") or "OpenAlex",
                    "chunk_id": item.get("openalex_id"),
                    "document_id": item.get("openalex_id"),
                    "metadata": metadata,
                }
            ],
        )
        payload.append(item)
    return payload


async def run_section_search_payload(
    deps: AgentDependencies,
    query: str,
    section_query: str,
    limit: int,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        results = await section_search_tool(
            SectionSearchInput(
                query=query,
                section_query=section_query,
                limit=limit,
                document_id=document_id,
            )
        )
        payload = [chunk_result_to_payload(r) for r in results]
        collect_hits(deps, payload)
        return payload
    except Exception as exc:
        _mark_retrieval_error(deps, f"section_search_failed:{type(exc).__name__}")
        return []
