import logging
from typing import Dict, Any, List, Optional

from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT
from .providers import get_llm_model
from .agent_runtime import (
    AgentDependencies,
    _resolve_default_search_type,
)
from .tool_payloads import (
    run_get_document_payload,
    run_hybrid_search_payload,
    run_list_documents_payload,
    run_openalex_payload,
    run_section_search_payload,
    run_vector_search_payload,
)

load_dotenv()
logger = logging.getLogger(__name__)


rag_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT,
)


@rag_agent.tool
async def search_knowledge_base(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Search uploaded/local papers for evidence from the local knowledge base."""
    search_type = _resolve_default_search_type(ctx.deps)
    default_limit = int((ctx.deps.search_preferences or {}).get("default_limit", 10) or 10)
    effective_limit = max(1, min(int(limit or default_limit), 50))
    if search_type == "vector":
        return await vector_search(ctx, query=query, limit=effective_limit)
    return await hybrid_search(ctx, query=query, limit=effective_limit)


@rag_agent.tool
async def vector_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    return await run_vector_search_payload(deps=ctx.deps, query=query, limit=limit)


@rag_agent.tool
async def hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
    text_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    return await run_hybrid_search_payload(
        deps=ctx.deps,
        query=query,
        limit=limit,
        text_weight=text_weight,
    )


@rag_agent.tool
async def section_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    section_query: str,
    limit: int = 10,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search chunks within a specific section using section metadata."""
    return await run_section_search_payload(
        deps=ctx.deps,
        query=query,
        section_query=section_query,
        limit=max(1, min(int(limit or 10), 50)),
        document_id=document_id,
    )


@rag_agent.tool
async def get_document(
    ctx: RunContext[AgentDependencies],
    document_id: str,
) -> Optional[Dict[str, Any]]:
    return await run_get_document_payload(deps=ctx.deps, document_id=document_id)


@rag_agent.tool
async def list_documents(
    ctx: RunContext[AgentDependencies],
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    return await run_list_documents_payload(limit=limit, offset=offset)


@rag_agent.tool
async def search_openalex_papers(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search OpenAlex for paper discovery and metadata (authors/year/DOI/venue/OA links)."""
    if not ctx.deps.use_web_search:
        return []

    return await run_openalex_payload(deps=ctx.deps, query=query, limit=limit)
