import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .db_utils import (
    vector_search,
    hybrid_search,
    section_search,
    get_document,
    list_documents,
    get_document_chunks,
)
from .models import ChunkResult, DocumentMetadata
from .providers import get_embedding_client, get_embedding_model

load_dotenv()
logger = logging.getLogger(__name__)

embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()
EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "20"))
LOCAL_SEARCH_TIMEOUT_SECONDS = float(os.getenv("LOCAL_SEARCH_TIMEOUT_SECONDS", "30"))
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_GENERAL_WEB_ENDPOINTS = {
    "tavily": "https://api.tavily.com/search",
    "serpapi": "https://serpapi.com/search.json",
    "brave": "https://api.search.brave.com/res/v1/web/search",
    "bocha": "https://api.bochaai.com/v1/web-search",
}


async def generate_embedding(text: str) -> List[float]:
    try:
        response = await asyncio.wait_for(
            embedding_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                dimensions=1024,
                encoding_format="float",
            ),
            timeout=EMBEDDING_TIMEOUT_SECONDS,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


class VectorSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")


class HybridSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    text_weight: float = Field(default=0.3, description="Weight for text similarity (0-1)")


class SectionSearchInput(BaseModel):
    query: str = Field(..., description="Content query within the section")
    section_query: str = Field(..., description="Section title/path keyword, e.g. Method, Experiments, References")
    document_id: Optional[str] = Field(default=None, description="Optional document UUID to restrict search")
    limit: int = Field(default=10, ge=1, le=50)


class DocumentInput(BaseModel):
    document_id: str = Field(..., description="Document ID to retrieve")


class DocumentListInput(BaseModel):
    limit: int = Field(default=20, description="Maximum number of documents")
    offset: int = Field(default=0, description="Number of documents to skip")


class OpenAlexSearchInput(BaseModel):
    query: str = Field(..., description="Search keywords")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of OpenAlex works")


class WebSearchInput(BaseModel):
    query: str = Field(..., description="General web search query")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of web results")


def _as_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def is_general_web_search_enabled() -> bool:
    if not _as_bool_env("GENERAL_WEB_SEARCH_ENABLED", False):
        return False
    provider = str(os.getenv("GENERAL_WEB_SEARCH_PROVIDER", "") or "").strip().lower()
    api_key = str(os.getenv("GENERAL_WEB_SEARCH_API_KEY", "") or "").strip()
    endpoint = str(os.getenv("GENERAL_WEB_SEARCH_ENDPOINT", "") or "").strip()
    if provider in {"tavily", "serpapi", "brave", "bing", "bocha"}:
        return bool(api_key)
    if provider == "custom":
        return bool(api_key and endpoint)
    return False


def get_general_web_search_provider() -> str:
    return str(os.getenv("GENERAL_WEB_SEARCH_PROVIDER", "custom") or "custom").strip().lower()


def _decode_openalex_abstract(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not abstract_inverted_index:
        return ""
    max_pos = -1
    for positions in abstract_inverted_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    tokens = [""] * (max_pos + 1)
    for token, positions in abstract_inverted_index.items():
        for pos in positions:
            if 0 <= pos < len(tokens):
                tokens[pos] = token
    text = " ".join([tok for tok in tokens if tok]).strip()
    return text


def _extract_pdf_url(work: Dict[str, Any]) -> Optional[str]:
    open_access = work.get("open_access") or {}
    best_oa = work.get("best_oa_location") or {}
    primary_location = work.get("primary_location") or {}

    for candidate in [
        open_access.get("oa_url"),
        best_oa.get("pdf_url"),
        best_oa.get("landing_page_url"),
        primary_location.get("pdf_url"),
        primary_location.get("landing_page_url"),
    ]:
        if isinstance(candidate, str) and candidate.startswith("http"):
            return candidate
    return None


def _extract_venue(work: Dict[str, Any]) -> str:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    return str(source.get("display_name") or "")


def _extract_authors(work: Dict[str, Any]) -> List[str]:
    authors: List[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if isinstance(name, str) and name.strip():
            authors.append(name.strip())
    return authors


def _extract_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").strip().lower()
    except Exception:
        return ""


def _safe_request_json(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    payload = None
    request_headers = dict(headers or {})
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = Request(url, data=payload, headers=request_headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_general_web_result(item: Dict[str, Any], fallback_source: str) -> Optional[Dict[str, Any]]:
    title = str(item.get("title") or "").strip()
    url = str(item.get("url") or item.get("link") or "").strip()
    snippet = str(
        item.get("snippet")
        or item.get("content")
        or item.get("description")
        or item.get("body")
        or ""
    ).strip()
    published_date = item.get("published_date") or item.get("date") or item.get("published") or item.get("age")
    source = str(item.get("source") or item.get("domain") or _extract_domain(url) or fallback_source).strip()
    if not title or not url:
        return None
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "source": source,
        "published_date": published_date,
        "provider": fallback_source,
    }


def _normalize_bocha_result(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = str(item.get("name") or item.get("title") or "").strip()
    url = str(item.get("url") or item.get("link") or "").strip()
    snippet = str(
        item.get("summary")
        or item.get("snippet")
        or item.get("content")
        or item.get("description")
        or ""
    ).strip()
    published_date = (
        item.get("datePublished")
        or item.get("dateLastCrawled")
        or item.get("date")
        or item.get("published_date")
    )
    source = str(
        item.get("siteName")
        or item.get("site")
        or item.get("source")
        or item.get("domain")
        or _extract_domain(url)
        or "bocha"
    ).strip()
    if not title or not url:
        return None
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "source": source,
        "published_date": published_date,
        "provider": "bocha",
    }


def _sync_general_web_search(query: str, limit: int) -> List[Dict[str, Any]]:
    if not is_general_web_search_enabled():
        return []

    provider = get_general_web_search_provider()
    api_key = str(os.getenv("GENERAL_WEB_SEARCH_API_KEY", "") or "").strip()
    endpoint = str(os.getenv("GENERAL_WEB_SEARCH_ENDPOINT", "") or "").strip()

    try:
        safe_limit = max(1, min(limit, 10))
        if provider == "tavily":
            url = endpoint or DEFAULT_GENERAL_WEB_ENDPOINTS["tavily"]
            if not api_key:
                logger.warning("General web search skipped: missing API key for provider '%s'", provider)
                return []
            payload = _safe_request_json(
                url,
                method="POST",
                body={
                    "api_key": api_key,
                    "query": query,
                    "max_results": safe_limit,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            items = payload.get("results") or []
        elif provider == "serpapi":
            base_url = endpoint or DEFAULT_GENERAL_WEB_ENDPOINTS["serpapi"]
            if not api_key:
                logger.warning("General web search skipped: missing API key for provider '%s'", provider)
                return []
            url = f"{base_url}?{urlencode({'engine': 'google', 'q': query, 'num': safe_limit, 'api_key': api_key})}"
            payload = _safe_request_json(url)
            items = payload.get("organic_results") or []
        elif provider == "brave":
            base_url = endpoint or DEFAULT_GENERAL_WEB_ENDPOINTS["brave"]
            if not api_key:
                logger.warning("General web search skipped: missing API key for provider '%s'", provider)
                return []
            url = f"{base_url}?{urlencode({'q': query, 'count': safe_limit})}"
            payload = _safe_request_json(url, headers={"X-Subscription-Token": api_key, "Accept": "application/json"})
            items = ((payload.get("web") or {}).get("results") or [])
        elif provider == "bing":
            if not endpoint:
                logger.warning("General web search skipped: missing endpoint for provider '%s'", provider)
                return []
            if not api_key:
                logger.warning("General web search skipped: missing API key for provider '%s'", provider)
                return []
            url = f"{endpoint}?{urlencode({'q': query, 'count': safe_limit})}"
            payload = _safe_request_json(url, headers={"Ocp-Apim-Subscription-Key": api_key})
            items = ((payload.get("webPages") or {}).get("value") or [])
        elif provider == "bocha":
            url = endpoint or DEFAULT_GENERAL_WEB_ENDPOINTS["bocha"]
            if not api_key:
                logger.warning("General web search skipped: missing API key for provider '%s'", provider)
                return []
            payload = _safe_request_json(
                url,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                body={
                    "query": query,
                    "summary": True,
                    "count": safe_limit,
                    "freshness": "noLimit",
                },
            )
            items = (
                ((payload.get("data") or {}).get("webPages") or {}).get("value")
                or ((payload.get("webPages") or {}).get("value"))
                or payload.get("results")
                or ((payload.get("data") or {}).get("results"))
                or (payload.get("data") if isinstance(payload.get("data"), list) else [])
                or []
            )
            normalized: List[Dict[str, Any]] = []
            for item in items[:safe_limit]:
                if not isinstance(item, dict):
                    continue
                result = _normalize_bocha_result(item)
                if result is not None:
                    normalized.append(result)
            return normalized
        elif provider == "custom":
            if not endpoint:
                logger.warning("General web search skipped: missing endpoint for provider '%s'", provider)
                return []
            if not api_key:
                logger.warning("General web search skipped: missing API key for provider '%s'", provider)
                return []
            url = f"{endpoint}?{urlencode({'q': query, 'limit': safe_limit})}"
            headers = {"Accept": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                headers["X-API-Key"] = api_key
            payload = _safe_request_json(url, headers=headers)
            items = (
                payload.get("results")
                or payload.get("items")
                or payload.get("organic_results")
                or payload.get("data")
                or []
            )
        else:
            return []

        normalized: List[Dict[str, Any]] = []
        for item in items[:safe_limit]:
            if not isinstance(item, dict):
                continue
            result = _normalize_general_web_result(item, fallback_source=provider or "web")
            if result is not None:
                normalized.append(result)
        return normalized
    except Exception as exc:
        logger.warning("General web search failed: %s", exc)
        return []


def _sync_fetch_openalex_works(query: str, limit: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if not api_key:
        return []

    params = {
        "search": query,
        "per-page": max(1, min(limit, 10)),
        "api_key": api_key,
    }
    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto

    url = f"{OPENALEX_WORKS_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "agentic-rag-openalex/1.0"})
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("results") or []


async def openalex_search_tool(input_data: OpenAlexSearchInput) -> List[Dict[str, Any]]:
    """
    Search OpenAlex for external academic works.

    Recommended use cases:
    - 查找本地知识库之外的论文
    - 推荐相关论文
    - 查找最新相关工作
    - 补充 related work
    - 对比本地论文与外部论文

    Guidance:
    - 当用户已开启联网搜索且问题涉及上述需求时，应优先考虑使用该工具。
    - 若未配置 OPENALEX_API_KEY，本工具会安全返回空结果，不影响本地知识库问答。
    """
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        works = await asyncio.to_thread(
            _sync_fetch_openalex_works,
            input_data.query,
            input_data.limit,
        )
        results: List[Dict[str, Any]] = []
        for work in works:
            openalex_id = str(work.get("id") or "")
            title = str(work.get("display_name") or "").strip()
            if not title:
                continue

            doi = work.get("doi")
            landing_page_url = (
                (work.get("primary_location") or {}).get("landing_page_url")
                or (work.get("ids") or {}).get("openalex")
                or openalex_id
            )
            pdf_url = _extract_pdf_url(work)
            abstract = _decode_openalex_abstract(work.get("abstract_inverted_index"))
            open_access = work.get("open_access") or {}
            authors = _extract_authors(work)
            year = work.get("publication_year")
            venue = _extract_venue(work)

            result = {
                "title": title,
                "authors": authors,
                "year": year,
                "source": venue,
                "cited_by_count": work.get("cited_by_count"),
                "doi": doi,
                "landing_page_url": landing_page_url,
                "pdf_url": pdf_url,
                "abstract": abstract,
                "openalex_id": openalex_id,
                "source_kind": "openalex",
                "is_oa": bool(open_access.get("is_oa")),
                "has_pdf": bool(pdf_url),
                "has_fulltext": bool(pdf_url or abstract),
            }
            results.append(result)
        return results
    except Exception as e:
        logger.warning(f"OpenAlex search failed: {e}")
        return []


async def web_search_tool(input_data: WebSearchInput) -> List[Dict[str, Any]]:
    if not is_general_web_search_enabled():
        return []
    try:
        return await asyncio.to_thread(
            _sync_general_web_search,
            input_data.query,
            input_data.limit,
        )
    except Exception as e:
        logger.warning(f"General web search failed: {e}")
        return []


async def vector_search_tool(input_data: VectorSearchInput) -> List[ChunkResult]:
    try:
        embedding = await generate_embedding(input_data.query)
    except Exception as e:
        logger.exception("Vector search embedding failed: %s", e)
        raise
    try:
        results = await asyncio.wait_for(
            vector_search(embedding=embedding, limit=input_data.limit),
            timeout=LOCAL_SEARCH_TIMEOUT_SECONDS,
        )
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["similarity"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"],
            )
            for r in results
        ]
    except Exception as e:
        logger.exception("Vector search failed: %s", e)
        raise


async def hybrid_search_tool(input_data: HybridSearchInput) -> List[ChunkResult]:
    try:
        embedding = await generate_embedding(input_data.query)
    except Exception as e:
        logger.exception("Hybrid search embedding failed: %s", e)
        raise
    try:
        results = await asyncio.wait_for(
            hybrid_search(
                embedding=embedding,
                query_text=input_data.query,
                limit=input_data.limit,
                text_weight=input_data.text_weight,
            ),
            timeout=LOCAL_SEARCH_TIMEOUT_SECONDS,
        )
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["combined_score"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"],
            )
            for r in results
        ]
    except Exception as e:
        logger.exception("Hybrid search failed: %s", e)
        raise


async def section_search_tool(input_data: SectionSearchInput) -> List[ChunkResult]:
    try:
        results = await asyncio.wait_for(
            section_search(
                query_text=input_data.query,
                section_query=input_data.section_query,
                document_id=input_data.document_id,
                limit=input_data.limit,
            ),
            timeout=LOCAL_SEARCH_TIMEOUT_SECONDS,
        )
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=float(r["combined_score"]),
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"],
            )
            for r in results
        ]
    except Exception as e:
        logger.exception("Section search failed: %s", e)
        raise


async def get_document_tool(input_data: DocumentInput) -> Optional[Dict[str, Any]]:
    try:
        document = await get_document(input_data.document_id)
        if document:
            chunks = await get_document_chunks(input_data.document_id)
            document["chunks"] = chunks
        return document
    except Exception as e:
        logger.exception("Document retrieval failed: %s", e)
        raise


async def list_documents_tool(input_data: DocumentListInput) -> List[DocumentMetadata]:
    try:
        documents = await list_documents(limit=input_data.limit, offset=input_data.offset)
        return [
            DocumentMetadata(
                id=d["id"],
                title=d["title"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                chunk_count=d.get("chunk_count"),
            )
            for d in documents
        ]
    except Exception as e:
        logger.exception("Document listing failed: %s", e)
        raise
