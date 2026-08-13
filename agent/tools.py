import asyncio
import hashlib
import json
import logging
import os
import re
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
    artifact_search,
    get_document,
    list_documents,
    get_document_chunks,
)
from .models import ChunkResult, DocumentMetadata
from .providers import build_embedding_request_kwargs, get_embedding_client, get_embedding_model
from .embedding_runtime import EmbeddingLanguage, get_embedding_client_for_route, get_embedding_route
from .cache_utils import cache_get_json, cache_set_json, make_cache_key
from .query_translation_runtime import translate_query_to_english

load_dotenv()
logger = logging.getLogger(__name__)

embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()
EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "20"))
LOCAL_SEARCH_TIMEOUT_SECONDS = float(os.getenv("LOCAL_SEARCH_TIMEOUT_SECONDS", "30"))
EMBEDDING_CACHE_TTL_SECONDS = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "86400"))
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_GENERAL_WEB_ENDPOINTS = {
    "tavily": "https://api.tavily.com/search",
    "serpapi": "https://serpapi.com/search.json",
    "brave": "https://api.search.brave.com/res/v1/web/search",
    "bocha": "https://api.bocha.cn/v1/web-search",
}


async def generate_embedding(text: str) -> List[float]:
    use_cache = _as_bool_env("ENABLE_REDIS_CACHE", True)
    cache_key = None
    if use_cache:
        query_hash = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
        cache_key = make_cache_key("embedding", EMBEDDING_MODEL, query_hash)
        cached = await cache_get_json(cache_key)
        if isinstance(cached, list) and cached:
            return cached

    try:
        response = await asyncio.wait_for(
            embedding_client.embeddings.create(
                **build_embedding_request_kwargs(
                    model=EMBEDDING_MODEL,
                    input_value=text,
                    encoding_format="float",
                )
            ),
            timeout=EMBEDDING_TIMEOUT_SECONDS,
        )
        embedding = response.data[0].embedding
        if use_cache and cache_key:
            await cache_set_json(cache_key, embedding, EMBEDDING_CACHE_TTL_SECONDS)
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


def _normalize_embedding_language(value: Optional[str]) -> Optional[EmbeddingLanguage]:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"zh", "en"} else None


async def generate_routed_embedding(
    text: str,
    embedding_language: Optional[str] = None,
) -> tuple[List[float], str]:
    """Embed a query with the same language-specific model used at ingestion."""
    route = get_embedding_route(text, language=_normalize_embedding_language(embedding_language))
    query_hash = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
    cache_key = make_cache_key("embedding", route.model, query_hash)
    cached = await cache_get_json(cache_key)
    if isinstance(cached, list) and cached:
        return cached, route.language
    response = await asyncio.wait_for(
        get_embedding_client_for_route(route).embeddings.create(
            **build_embedding_request_kwargs(model=route.model, input_value=text, encoding_format="float")
        ),
        timeout=EMBEDDING_TIMEOUT_SECONDS,
    )
    embedding = response.data[0].embedding
    await cache_set_json(cache_key, embedding, EMBEDDING_CACHE_TTL_SECONDS)
    return embedding, route.language


async def generate_retrieval_routes(
    query: str,
    embedding_language: Optional[str] = None,
) -> List[tuple[List[float], str, str]]:
    """Return the primary route plus an English bridge for Chinese free-text queries."""
    embedding, resolved_language = await generate_routed_embedding(query, embedding_language)
    routes = [(embedding, resolved_language, query)]
    if embedding_language is not None or resolved_language != "zh":
        return routes

    translated_query = await translate_query_to_english(query)
    if not translated_query:
        return routes
    english_embedding, _ = await generate_routed_embedding(translated_query, "en")
    return routes + [(english_embedding, "en", translated_query)]


def merge_chunk_results(result_sets: List[List[ChunkResult]], limit: int) -> List[ChunkResult]:
    """Deduplicate multi-route retrieval results while retaining their backend score."""
    unique: Dict[str, ChunkResult] = {}
    for results in result_sets:
        for result in results:
            existing = unique.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                unique[result.chunk_id] = result
    return sorted(unique.values(), key=lambda item: item.score, reverse=True)[:limit]


def _to_chunk_results(rows: List[Dict[str, Any]], score_key: str) -> List[ChunkResult]:
    return [
        ChunkResult(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            content=row["content"],
            score=float(row.get(score_key, row.get("score", 0.0))),
            metadata=row["metadata"],
            document_title=row["document_title"],
            document_source=row["document_source"],
        )
        for row in rows
    ]


def _expand_section_queries(section_query: str) -> List[str]:
    """Split multi-section requests so each persisted section can be matched."""
    value = str(section_query or "").strip()
    if not value:
        return [""]
    candidates = [value]
    normalized = value.lower()
    known_sections = ("abstract", "introduction", "method", "algorithm", "experiment", "evaluation", "result", "conclusion", "reference")
    candidates.extend(section for section in known_sections if re.search(rf"\b{section}\b", normalized))
    candidates.extend(part.strip() for part in re.split(r"\s+(?:and|or)\s+|[,;/]", value, flags=re.IGNORECASE) if part.strip())

    # Publishers use different front-matter labels for equivalent paper parts.
    aliases = {
        "abstract": ("summary", "executive summary"),
        "introduction": ("motivation", "background"),
        "method": ("methodology", "approach", "modeling"),
        "experiment": ("evaluation", "results", "validation"),
        "evaluation": ("experiment", "results", "validation"),
        "result": ("results", "evaluation", "validation"),
        "conclusion": ("conclusions", "discussion", "summary"),
    }
    for candidate in tuple(candidates):
        candidate_normalized = candidate.strip().lower()
        for canonical, equivalents in aliases.items():
            if canonical in candidate_normalized:
                candidates.extend(equivalents)
    return list(dict.fromkeys(candidates))


class VectorSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document UUIDs to restrict search")
    embedding_language: Optional[str] = Field(default=None, description="Corpus embedding language: zh or en")


class HybridSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    text_weight: float = Field(default=0.3, description="Weight for text similarity (0-1)")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document UUIDs to restrict search")
    embedding_language: Optional[str] = Field(default=None, description="Corpus embedding language: zh or en")


class SectionSearchInput(BaseModel):
    query: str = Field(..., description="Content query within the section")
    section_query: str = Field(..., description="Section title/path keyword, e.g. Method, Experiments, References")
    document_id: Optional[str] = Field(default=None, description="Optional document UUID to restrict search")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document UUIDs to restrict search")
    limit: int = Field(default=10, ge=1, le=50)


class ArtifactSearchInput(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=50)
    artifact_types: Optional[List[str]] = None
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    text_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    embedding_language: Optional[str] = None


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
    best_oa = work.get("best_oa_location") or {}
    primary_location = work.get("primary_location") or {}

    for candidate in [
        best_oa.get("pdf_url"),
        primary_location.get("pdf_url"),
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
        routes = await generate_retrieval_routes(
            input_data.query,
            input_data.embedding_language,
        )
    except Exception as e:
        logger.exception("Vector search embedding failed: %s", e)
        raise
    try:
        result_sets = []
        for embedding, embedding_language, _ in routes:
            rows = await asyncio.wait_for(
                vector_search(embedding=embedding, limit=input_data.limit, embedding_language=embedding_language, document_ids=input_data.document_ids),
                timeout=LOCAL_SEARCH_TIMEOUT_SECONDS,
            )
            result_sets.append(_to_chunk_results(rows, "similarity"))
        return merge_chunk_results(result_sets, input_data.limit)
    except Exception as e:
        logger.exception("Vector search failed: %s", e)
        raise


async def hybrid_search_tool(input_data: HybridSearchInput) -> List[ChunkResult]:
    try:
        routes = await generate_retrieval_routes(
            input_data.query,
            input_data.embedding_language,
        )
    except Exception as e:
        logger.exception("Hybrid search embedding failed: %s", e)
        raise
    try:
        result_sets = []
        for embedding, embedding_language, query_text in routes:
            rows = await asyncio.wait_for(
                hybrid_search(embedding=embedding, query_text=query_text, limit=input_data.limit, text_weight=input_data.text_weight, embedding_language=embedding_language, document_ids=input_data.document_ids),
                timeout=LOCAL_SEARCH_TIMEOUT_SECONDS,
            )
            result_sets.append(_to_chunk_results(rows, "combined_score"))
        return merge_chunk_results(result_sets, input_data.limit)
    except Exception as e:
        logger.exception("Hybrid search failed: %s", e)
        raise


async def section_search_tool(input_data: SectionSearchInput) -> List[ChunkResult]:
    try:
        result_sets = []
        for section_query in _expand_section_queries(input_data.section_query):
            rows = await asyncio.wait_for(
                section_search(query_text=input_data.query, section_query=section_query, document_id=input_data.document_id, document_ids=input_data.document_ids, limit=input_data.limit),
                timeout=LOCAL_SEARCH_TIMEOUT_SECONDS,
            )
            result_sets.append(_to_chunk_results(rows, "combined_score"))
        return merge_chunk_results(result_sets, input_data.limit)
    except Exception as e:
        logger.exception("Section search failed: %s", e)
        raise


async def artifact_search_tool(input_data: ArtifactSearchInput) -> List[ChunkResult]:
    allowed_types = {"table", "figure", "algorithm"}
    normalized_types = []
    for t in (input_data.artifact_types or []):
        value = str(t or "").strip().lower()
        if value in allowed_types:
            normalized_types.append(value)
    if not normalized_types:
        normalized_types = ["table", "figure", "algorithm"]

    try:
        routes = await generate_retrieval_routes(
            input_data.query,
            input_data.embedding_language,
        )
    except Exception as e:
        logger.exception("Artifact search embedding failed: %s", e)
        raise

    try:
        result_sets = []
        for embedding, embedding_language, query_text in routes:
            rows = await asyncio.wait_for(
                artifact_search(embedding=embedding, query_text=query_text, limit=input_data.limit, artifact_types=normalized_types, document_id=input_data.document_id, document_ids=input_data.document_ids, text_weight=input_data.text_weight, embedding_language=embedding_language),
                timeout=LOCAL_SEARCH_TIMEOUT_SECONDS,
            )
            result_sets.append(_to_chunk_results(rows, "combined_score"))
        return merge_chunk_results(result_sets, input_data.limit)
    except Exception as e:
        logger.exception("Artifact search failed: %s", e)
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
