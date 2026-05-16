import os
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import re
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from .agent_runtime import AgentDependencies
from .agent_runner import (
    run_agent,
    iter_agent,
    is_model_request_node,
    get_agent_backend,
    get_stream_backend,
)
from .agent_langgraph import run_langgraph_analysis
from .agent_langchain import (
    GENERATION_RETRY_FAILED_MESSAGE,
    stream_langchain_agent,
    iter_langchain_agent_stream,
    get_langchain_chat_model,
    retry_langchain_agent_after_degenerate,
)
from .openalex_router import (
    _is_openalex_enabled,
    _is_explicit_web_paper_request,
    _run_openalex_first_if_needed,
    _split_openalex_stream_chunks,
)
from .routing import (
    _is_general_algorithm_question,
    _is_local_kb_question,
    _may_need_general_web_search,
    _run_local_kb_preflight_if_needed,
    _dedupe_sources,
    _build_format_instruction,
    _build_tool_choice_instruction,
    has_unverified_web_citations,
    is_degenerate_answer,
)
from .sse_utils import sse_event, stream_response
from .db_utils import (
    execute_init_sql,
    initialize_database,
    close_database,
    create_session,
    get_session,
    add_message,
    get_session_messages,
    test_connection,
    refresh_session_metadata,
    list_recent_sessions,
    delete_session,
    get_session_memory_metadata,
    update_session_memory_metadata,
)
from .app_config import get_rabbitmq_url
from .ingestion_tasks_db import get_ingestion_task
from .models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    ErrorResponse,
    HealthStatus,
    ToolCall,
    EvidenceSource,
    SessionListResponse,
    SessionListItem,
    SessionMessagesResponse,
    ChatMessageItem,
    IngestionTaskResponse,
)
from .tools import (
    vector_search_tool,
    hybrid_search_tool,
    list_documents_tool,
    VectorSearchInput,
    HybridSearchInput,
    DocumentListInput,
    is_general_web_search_enabled,
    get_general_web_search_provider,
)
from .prompts import SYSTEM_PROMPT
from .providers import test_llm_connection
from .memory_utils import (
    TOKEN_LIMIT,
    normalize_memory_state,
    should_trigger_compression,
    get_messages_for_next_compaction,
    build_summary_update_prompt,
    build_context_without_compaction,
    sanitize_history_messages,
)
from .ingestion_jobs import (
    add_openalex_file_to_kb,
    run_sync_upload_ingestion,
    start_upload_ingestion_job,
    get_upload_ingestion_job,
    cancel_upload_ingestion_job,
    submit_async_ingestion_task,
)
from .stream_registry import (
    register_stream_run,
    unregister_stream_run,
    cancel_stream_run,
    get_stream_run,
)
from .warning_text import clean_legacy_warning_text

load_dotenv()

logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
STREAM_PREPARE_TIMEOUT_SECONDS = float(os.getenv("STREAM_PREPARE_TIMEOUT_SECONDS", "35"))
LLM_FIRST_TOKEN_TIMEOUT_SECONDS = float(os.getenv("LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "25"))
LLM_STREAM_TOTAL_TIMEOUT_SECONDS = float(os.getenv("LLM_STREAM_TOTAL_TIMEOUT_SECONDS", "75"))
LANGGRAPH_ANALYSIS_TIMEOUT_SECONDS = float(os.getenv("LANGGRAPH_ANALYSIS_TIMEOUT_SECONDS", "90"))
NON_STREAM_FALLBACK_TIMEOUT_SECONDS = float(os.getenv("NON_STREAM_FALLBACK_TIMEOUT_SECONDS", "35"))
RABBITMQ_URL = get_rabbitmq_url()


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if APP_ENV == "development":
    logger.setLevel(logging.DEBUG)

REACT_RUNTIME_INSTRUCTION = (
    "Deep analysis is enabled for this turn. Internally run a lightweight ReAct flow: "
    "identify the task type, choose suitable tools such as hybrid_search, vector_search, "
    "get_document, list_documents, search_openalex_papers, or search_web when needed, "
    "then verify that key claims are supported by evidence. "
    "If evidence is insufficient, state uncertainty. Output only the final answer."
)

GENERAL_WEB_UNAVAILABLE_INSTRUCTION = (
    "General web search is not configured. If the user explicitly asks for web page "
    "sources, say that web browsing is unavailable. Do not invent URLs, DOIs, or sources."
)


def _append_react_instruction(full_prompt: str, enabled: bool) -> str:
    if not enabled:
        return full_prompt
    return f"{full_prompt}\n\n[Runtime deep-analysis instruction]\n{REACT_RUNTIME_INSTRUCTION}"


def _is_explicit_general_web_request(message: str) -> bool:
    text = str(message or "").lower()
    keywords = [
        "web",
        "internet",
        "online",
        "search",
        "source",
        "latest",
        "recent",
        "\u8054\u7f51",
        "\u7f51\u4e0a",
        "\u641c\u7d22",
        "\u8d44\u6599",
        "\u6765\u6e90",
        "\u6700\u65b0",
        "\u51c6\u786e",
    ]
    return any(keyword in text for keyword in keywords)


def _should_force_openalex(message: str) -> bool:
    text = str(message or "").lower()
    if not _is_explicit_web_paper_request(message):
        return False
    paper_keywords = [
        "openalex",
        "paper",
        "doi",
        "author",
        "year",
        "related work",
        "\u8bba\u6587",
        "\u4f5c\u8005",
        "\u5e74\u4efd",
        "\u63a8\u8350\u8bba\u6587",
        "\u6700\u65b0\u8bba\u6587",
        "\u68c0\u7d22\u4e00\u7bc7\u8bba\u6587",
        "\u641c\u7d22\u4e00\u7bc7\u8bba\u6587",
        "\u968f\u673a\u68c0\u7d22",
        "\u6765\u6e90\u94fe\u63a5",
        "\u77e5\u8bc6\u5e93\u5916",
    ]
    return any(keyword in text for keyword in paper_keywords)


def _normalize_web_unavailable_reply(
    response: str,
    *,
    requested_web: bool,
    sources: List[EvidenceSource],
) -> str:
    if not requested_web or is_general_web_search_enabled():
        return response
    has_web_sources = any(str(getattr(source, "source_type", "") or "").lower() == "web" for source in sources)
    if has_web_sources:
        return response
    if response == GENERATION_RETRY_FAILED_MESSAGE:
        return "General web search is not configured, so reliable web page sources are unavailable. Remove the web-source requirement or configure GENERAL_WEB_SEARCH_* and retry."
    if re.search(r"(source|web|internet|search|\u6765\u6e90|\u8054\u7f51|\u7f51\u9875|\u641c\u7d22)", response, flags=re.IGNORECASE):
        return "General web search is not configured, so reliable web page sources are unavailable. Remove the web-source requirement or configure GENERAL_WEB_SEARCH_* and retry."
    return response


def _get_pydantic_stream_event_types():
    from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta

    return PartStartEvent, PartDeltaEvent, TextPartDelta


def _should_retry_stream_answer(
    full_response: str,
    sources: List[EvidenceSource],
    *,
    is_local_question: bool,
    explicit_web_request: bool,
    has_retrieved_sources: bool,
) -> tuple[bool, str]:
    text = str(full_response or "")
    stripped = text.strip()

    if not stripped:
        return True, "empty_response"

    lowered = stripped.lower()
    # 只有严重重复乱码或内部对象泄露才触发 retry
    severe_internal_patterns = [
        "aimessage(",
        "humanmessage(",
        "toolmessage(",
        "toolcall(",
        '"role": "tool"',
        '"messages":',
        "raw messages",
    ]
    if any(pattern in lowered for pattern in severe_internal_patterns):
        return True, "internal_object_leak"

    # 出现大量重复片段 token（严重退化）
    if re.search(r"(.{2,12})\1{10,}", stripped):
        return True, "repeated_token_noise"

    # 1. 如果 explicit_web_request=True，保留原逻辑，不强行改写
    if explicit_web_request:
        if not sources and has_unverified_web_citations(stripped):
            return True, "unverified_web_citation"
        return is_degenerate_answer(stripped), "degenerate_explicit_web"

    # 2. 如果 is_local_question=True 且 sources 或 deps.retrieved_sources 非空
    # 3. 如果 is_local_question=True 且已有本地 sources：不要因为 is_degenerate_answer 单独触发 retry
    has_any_sources = bool(sources) or bool(has_retrieved_sources)
    if is_local_question and has_any_sources:
        # 本地知识库问题有证据时，优先保留原回答，不要因为排版较差就重试。
        return False, "local_with_evidence_keep"

    # 4. 如果 not sources 且 has_unverified_web_citations(full_response)，可以 retry
    if not sources and has_unverified_web_citations(stripped):
        return True, "unverified_web_citation"

    # 兜底：如果没有来源且触发了退化检测，进行 retry
    if is_degenerate_answer(stripped):
        return True, "degenerate_answer"

    return False, "no_retry"


def clean_markdown_spacing(text: str) -> str:
    """Conservative markdown spacing cleanup without table reflow."""
    if not text:
        return text

    # Normalize heading forms like ##1. / ###1. / ##1
    text = re.sub(r"(?m)^##(\d+\.)", r"## \1", text)
    text = re.sub(r"(?m)^###(\d+\.)", r"### \1", text)
    text = re.sub(r"(?m)^##(\d+)(?![\d.])", r"## \1", text)

    return text


async def _next_stream_event_with_timeout(
    stream_iter: Any,
    timeout_seconds: float,
) -> Any:
    return await asyncio.wait_for(stream_iter.__anext__(), timeout=timeout_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up system...")
    try:
        await initialize_database()
        await execute_init_sql("sql/schema.sql")
        logger.info("Database initialized")
        db_ok = await test_connection()
        if not db_ok:
            logger.error("Database connection failed")
        logger.info("System startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    logger.info("Shutting down system...")
    try:
        await close_database()
        logger.info("Connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


app = FastAPI(
    title="Agentic RAG",
    description="AI agent combining vector search",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_or_create_session(request: ChatRequest) -> str:
    if request.session_id:
        session = await get_session(request.session_id)
        if session:
            return request.session_id
    return await create_session(user_id=request.user_id, metadata=request.metadata)


async def get_conversation_context(
    session_id: str,
    max_messages: Optional[int] = None,
) -> List[Dict[str, str]]:
    messages = await get_session_messages(session_id, limit=max_messages)
    return sanitize_history_messages(messages)


async def _summarize_for_memory(
    session_id: str,
    user_id: Optional[str],
    old_summary: str,
    messages_to_compact: List[Dict[str, str]],
) -> str:
    def _extract_langchain_text(response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                    continue
                if isinstance(item, dict):
                    text = str(item.get("text") or item.get("content") or "").strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content).strip()

    summary_prompt = build_summary_update_prompt(
        old_summary=old_summary,
        messages_to_compact=messages_to_compact,
    )
    model = get_langchain_chat_model()
    response = await model.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "You are performing an internal conversation memory compression task. "
                    "Do not call tools or search documents. Summarize only from the provided history. "
                    "Output Simplified Chinese."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ]
    )
    summary_text = _extract_langchain_text(response).strip()
    if summary_text:
        return summary_text

    compact_preview = " ".join(
        msg.get("content", "").strip() for msg in messages_to_compact if msg.get("content")
    )
    compact_preview = " ".join(compact_preview.split())
    if len(compact_preview) > 300:
        compact_preview = compact_preview[:300].rstrip() + "..."
    return old_summary or compact_preview


async def _prepare_agent_prompt(
    session_id: str,
    user_id: Optional[str],
    user_message: str,
) -> Dict[str, Any]:
    history_messages = await get_conversation_context(session_id=session_id, max_messages=None)
    memory_metadata = await get_session_memory_metadata(session_id)
    memory_state = normalize_memory_state(memory_metadata)

    compression_needed = should_trigger_compression(
        system_prompt=SYSTEM_PROMPT,
        history_messages=history_messages,
        current_question=user_message,
        latest_summary=memory_state.latest_summary,
        token_limit=TOKEN_LIMIT,
    )

    summary_updated = False
    if compression_needed:
        messages_to_compact = get_messages_for_next_compaction(
            history_messages=history_messages,
            compacted_message_count=memory_state.compacted_message_count,
        )
        if messages_to_compact:
            updated_summary = await _summarize_for_memory(
                session_id=session_id,
                user_id=user_id,
                old_summary=memory_state.latest_summary,
                messages_to_compact=messages_to_compact,
            )
            memory_state.latest_summary = updated_summary
            memory_state.compression_count += 1
            memory_state.compacted_message_count = max(
                memory_state.compacted_message_count,
                max(0, len(history_messages) - 6),
            )
            summary_updated = True

    context_result = build_context_without_compaction(
        history_messages=history_messages,
        current_question=user_message,
        memory_state=memory_state,
    )
    context_result.summary_updated = summary_updated
    context_result.latest_summary = memory_state.latest_summary
    context_result.compression_count = memory_state.compression_count
    context_result.compacted_message_count = memory_state.compacted_message_count

    return {
        "full_prompt": context_result.full_prompt,
        "compression_used": context_result.compression_used,
        "summary_updated": context_result.summary_updated,
        "latest_summary": context_result.latest_summary,
        "compression_count": context_result.compression_count,
        "compacted_message_count": context_result.compacted_message_count,
        "compression_needed": compression_needed,
    }


def extract_tool_calls(result) -> List[ToolCall]:
    tools_used: List[ToolCall] = []
    try:
        messages = result.all_messages()
        for message in messages:
            if not hasattr(message, "parts"):
                continue
            for part in message.parts:
                if part.__class__.__name__ != "ToolCallPart":
                    continue
                try:
                    tool_name = str(part.tool_name) if hasattr(part, "tool_name") else "unknown"
                    tool_args: Dict[str, Any] = {}
                    if hasattr(part, "args") and part.args is not None:
                        if isinstance(part.args, str):
                            try:
                                tool_args = json.loads(part.args)
                            except json.JSONDecodeError:
                                tool_args = {}
                        elif isinstance(part.args, dict):
                            tool_args = part.args
                    tool_call_id = (
                        str(part.tool_call_id) if hasattr(part, "tool_call_id") and part.tool_call_id else None
                    )
                    tools_used.append(
                        ToolCall(tool_name=tool_name, args=tool_args, tool_call_id=tool_call_id)
                    )
                except Exception as e:
                    logger.debug(f"Failed to parse tool call part: {e}")
                    continue
    except Exception as e:
        logger.warning(f"Failed to extract tool calls: {e}")
    return tools_used


def extract_evidence_sources(
    deps: AgentDependencies,
    local_limit: int = 3,
    web_limit: int = 3,
) -> List[EvidenceSource]:
    all_sources = list(getattr(deps, "retrieved_sources", []) or [])

    normalized: List[EvidenceSource] = []
    for source in all_sources:
        source_type = str(getattr(source, "source_type", "") or "").lower()
        if source_type not in {"local", "web"}:
            meta_type = str((source.metadata or {}).get("source_type") or "").lower()
            source_type = "web" if meta_type == "web" else "local"
            source.source_type = source_type
        normalized.append(source)

    local_sources = [s for s in normalized if s.source_type == "local"]
    web_sources = [s for s in normalized if s.source_type == "web"]

    local_sources.sort(key=lambda s: (s.score is not None, s.score if s.score is not None else -1), reverse=True)
    web_sources.sort(
        key=lambda s: ((s.metadata or {}).get("cited_by_count") or -1),
        reverse=True,
    )

    return local_sources[:local_limit] + web_sources[:web_limit]


async def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_metadata: Optional[Dict[str, Any]] = None,
    assistant_metadata: Optional[Dict[str, Any]] = None,
):
    await add_message(
        session_id=session_id,
        role="user",
        content=user_message,
        metadata=user_metadata if user_metadata is not None else (metadata or {}),
    )
    await add_message(
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        metadata=assistant_metadata if assistant_metadata is not None else (metadata or {}),
    )
    await refresh_session_metadata(session_id)


def _resolve_search_type(search_type: Any) -> str:
    value = str(search_type).lower().strip()
    return "vector" if value == "vector" else "hybrid"


def _is_langchain_agent_result(result: Any) -> bool:
    return (
        hasattr(result, "message")
        and hasattr(result, "tools_used")
        and hasattr(result, "sources")
    )


@dataclass
class ChatRuntime:
    session_id: str
    deps: AgentDependencies
    requested_search_type: str
    effective_search_type: str
    explicit_web_request: bool
    effective_use_web_search: bool
    use_react: bool
    full_prompt: str
    langgraph_context_prompt: str
    compression_used: bool
    context_payload: Dict[str, Any]
    is_general_question: bool
    may_need_general_web_search: bool
    explicit_general_web_request: bool
    is_local_question: bool
    has_local_evidence: bool
    workflow_metadata: Dict[str, Any] = field(default_factory=dict)


async def prepare_chat_runtime(request: ChatRequest) -> ChatRuntime:
    requested_search_type = _resolve_search_type(request.search_type)
    explicit_web_request = _should_force_openalex(request.message)
    effective_use_web_search = bool(bool(request.use_web_search) or explicit_web_request)
    request_metadata = request.metadata or {}
    allow_web_search = bool(request_metadata.get("allow_web_search", bool(request.use_web_search)))
    allow_openalex_search = bool(request_metadata.get("allow_openalex_search", True))
    deps = AgentDependencies(
        session_id=request.session_id or "",
        user_id=request.user_id,
        use_web_search=effective_use_web_search,
        search_preferences={
            "default_search_type": requested_search_type,
            "default_limit": 10,
            "allow_web_search": allow_web_search,
            "allow_openalex_search": allow_openalex_search,
        },
    )
    context_payload = await _prepare_agent_prompt(
        session_id=request.session_id or "",
        user_id=request.user_id,
        user_message=request.message,
    )
    is_general_question = _is_general_algorithm_question(request.message)
    may_need_general_web_search = _may_need_general_web_search(request.message)
    explicit_general_web_request = _is_explicit_general_web_request(request.message)
    is_local_question = _is_local_kb_question(request.message)

    local_context = ""
    if is_local_question:
        local_context = await _run_local_kb_preflight_if_needed(request.message, deps)
    has_local_evidence = bool(local_context)

    format_instruction = _build_format_instruction(
        has_local_evidence=has_local_evidence,
        is_general_question=is_general_question,
    )
    tool_choice_instruction = _build_tool_choice_instruction(
        is_general_question=is_general_question,
        may_need_web=may_need_general_web_search,
        has_local_evidence=has_local_evidence,
    )

    base_context_prompt = context_payload["full_prompt"]
    langgraph_context_prompt = base_context_prompt
    full_prompt = base_context_prompt
    full_prompt = _append_react_instruction(full_prompt, bool(request.use_react))
    if local_context:
        full_prompt = f"{full_prompt}\n\n{local_context}"
    full_prompt = (
        f"{full_prompt}\n\n[Tool selection guidance]\n{tool_choice_instruction}"
        f"\n\n[Output format requirements]\n{format_instruction}"
    )
    if may_need_general_web_search and not is_general_web_search_enabled():
        full_prompt = f"{full_prompt}\n\n[Web capability notice]\n{GENERAL_WEB_UNAVAILABLE_INSTRUCTION}"

    return ChatRuntime(
        session_id=request.session_id or "",
        deps=deps,
        requested_search_type=requested_search_type,
        effective_search_type=str((deps.search_preferences or {}).get("default_search_type", requested_search_type)),
        explicit_web_request=explicit_web_request,
        effective_use_web_search=effective_use_web_search,
        use_react=bool(request.use_react),
        full_prompt=full_prompt,
        langgraph_context_prompt=langgraph_context_prompt,
        compression_used=bool(context_payload["compression_used"]),
        context_payload=context_payload,
        is_general_question=is_general_question,
        may_need_general_web_search=may_need_general_web_search,
        explicit_general_web_request=explicit_general_web_request,
        is_local_question=is_local_question,
        has_local_evidence=has_local_evidence,
        workflow_metadata={},
    )


async def execute_prepared_chat_runtime(
    message: str,
    runtime: ChatRuntime,
    *,
    save_conversation: bool = True,
) -> tuple[str, List[ToolCall], bool, List[EvidenceSource], str, str, Dict[str, Any]]:
    try:
        backend = get_agent_backend()
        response_backend = backend
        workflow_metadata: Dict[str, Any] = {}
        deps = runtime.deps
        compression_used = runtime.compression_used
        effective_search_type = runtime.effective_search_type

        if runtime.context_payload["summary_updated"]:
            await update_session_memory_metadata(
                session_id=runtime.session_id,
                latest_summary=runtime.context_payload["latest_summary"],
                compression_count=runtime.context_payload["compression_count"],
                compacted_message_count=runtime.context_payload["compacted_message_count"],
            )

        openalex_first_result = None
        if runtime.explicit_web_request:
            openalex_first_result = await _run_openalex_first_if_needed(message=message, deps=deps)
        if openalex_first_result is not None:
            response, tools_used, sources, workflow_metadata = openalex_first_result
            response_backend = "openalex_first"
        else:
            if runtime.use_react and backend == "langchain":
                graph_result = await run_langgraph_analysis(
                    question=message,
                    deps=deps,
                    context_prompt=runtime.langgraph_context_prompt,
                )
                response = str(getattr(graph_result, "message", "") or "")
                tools_used = list(getattr(graph_result, "tools_used", []) or [])
                sources = list(getattr(graph_result, "sources", []) or [])
                workflow_metadata = dict(getattr(graph_result, "metadata", {}) or {})
                response_backend = "langgraph"
            else:
                result = await run_agent(runtime.full_prompt, deps=deps)
                if _is_langchain_agent_result(result):
                    response = str(getattr(result, "message", "") or "")
                    tools_used = list(getattr(result, "tools_used", []) or [])
                    sources = list(getattr(result, "sources", []) or [])
                    if is_degenerate_answer(response):
                        logger.warning("Detected degenerate answer in /chat normal path; retrying once.")
                        retry_result = await retry_langchain_agent_after_degenerate(runtime.full_prompt, deps)
                        response = str(getattr(retry_result, "message", "") or "")
                        tools_used = list(getattr(retry_result, "tools_used", []) or [])
                        sources = list(getattr(retry_result, "sources", []) or [])
                else:
                    response = str(result.output)
                    tools_used = extract_tool_calls(result)
                    sources = extract_evidence_sources(deps)

        strict_langgraph_scope = (
            response_backend == "langgraph"
            and str((workflow_metadata or {}).get("scope_policy") or "") == "strict_target"
        )
        if not sources and not strict_langgraph_scope:
            sources = list(getattr(deps, "retrieved_sources", []) or [])
        sources = _dedupe_sources(sources)
        response = _normalize_web_unavailable_reply(
            response,
            requested_web=runtime.effective_use_web_search,
            sources=sources,
        )
        response = clean_legacy_warning_text(
            clean_markdown_spacing(response),
            drop_warning=bool(workflow_metadata.get("retrieval_skipped_by_planner") and workflow_metadata.get("direct_answer_allowed")),
        )
        sources_dict = [source.model_dump() for source in sources]

        safe_workflow_metadata = {
            k: v
            for k, v in workflow_metadata.items()
            if k
            not in {
                "requested_search_type",
                "effective_search_type",
                "compression_used",
                "use_web_search",
                "use_react",
                "agent_backend",
                "sources",
                "tool_calls",
            }
        }
        retrieval_error = (deps.search_preferences or {}).get("retrieval_error")
        if retrieval_error:
            safe_workflow_metadata["retrieval_error"] = retrieval_error

        if save_conversation:
            if str(response or "").strip():
                await save_conversation_turn(
                    session_id=runtime.session_id,
                    user_message=message,
                    assistant_message=response,
                    user_metadata={
                        "user_id": deps.user_id,
                        "compression_used": compression_used,
                        "requested_search_type": runtime.requested_search_type,
                        "effective_search_type": effective_search_type,
                        "use_web_search": deps.use_web_search,
                        "use_react": runtime.use_react,
                        "agent_backend": response_backend,
                        **safe_workflow_metadata,
                    },
                    assistant_metadata={
                        "tool_calls": len(tools_used),
                        "compression_used": compression_used,
                        "requested_search_type": runtime.requested_search_type,
                        "effective_search_type": effective_search_type,
                        "sources": sources_dict,
                        "use_web_search": deps.use_web_search,
                        "use_react": runtime.use_react,
                        "agent_backend": response_backend,
                        **safe_workflow_metadata,
                    },
                )
            else:
                await add_message(
                    session_id=runtime.session_id,
                    role="user",
                    content=message,
                    metadata={
                        "user_id": deps.user_id,
                        "compression_used": compression_used,
                        "requested_search_type": runtime.requested_search_type,
                        "effective_search_type": effective_search_type,
                        "use_web_search": deps.use_web_search,
                        "use_react": runtime.use_react,
                        "agent_backend": response_backend,
                        **safe_workflow_metadata,
                    },
                )
                await refresh_session_metadata(runtime.session_id)

        return (
            response,
            tools_used,
            compression_used,
            sources,
            effective_search_type,
            response_backend,
            safe_workflow_metadata,
        )
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        error_response = f"I encountered an error while processing your request: {str(e)}"

        if save_conversation:
            await save_conversation_turn(
                session_id=runtime.session_id,
                user_message=message,
                assistant_message=error_response,
                user_metadata={
                    "user_id": runtime.deps.user_id,
                    "compression_used": False,
                    "requested_search_type": runtime.requested_search_type,
                    "effective_search_type": runtime.effective_search_type,
                    "agent_backend": get_agent_backend(),
                },
                assistant_metadata={
                    "error": str(e),
                    "compression_used": False,
                    "requested_search_type": runtime.requested_search_type,
                    "effective_search_type": runtime.effective_search_type,
                    "sources": [],
                    "agent_backend": get_agent_backend(),
                },
            )

        return (
            error_response,
            [],
            False,
            [],
            runtime.effective_search_type,
            get_agent_backend(),
            {},
        )


async def execute_agent(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
    search_type: str = "hybrid",
    use_web_search: bool = False,
    use_react: bool = False,
    save_conversation: bool = True,
) -> tuple[str, List[ToolCall], bool, List[EvidenceSource], str, str, Dict[str, Any]]:
    request = ChatRequest(
        message=message,
        session_id=session_id,
        user_id=user_id,
        search_type=search_type,
        use_web_search=use_web_search,
        use_react=use_react,
    )
    runtime = await prepare_chat_runtime(request)
    return await execute_prepared_chat_runtime(
        message,
        runtime,
        save_conversation=save_conversation,
    )


@app.get("/health", response_model=HealthStatus)
async def health_check():
    try:
        db_status = await test_connection()
        llm_ok, llm_error = await test_llm_connection()
        status = "healthy" if (db_status and llm_ok) else "unhealthy"
        if llm_error:
            logger.warning("LLM health check failed: %s", llm_error)
        return HealthStatus(
            status=status,
            database=db_status,
            llm_connection=llm_ok,
            version="1.1.0",
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/health/live")
async def health_live():
    return {
        "status": "ok",
        "version": "1.1.0",
        "timestamp": datetime.now(),
    }


@app.get("/openalex/status")
async def openalex_status():
    return {"enabled": _is_openalex_enabled()}


@app.get("/web-search/status")
async def web_search_status():
    return {
        "enabled": is_general_web_search_enabled(),
        "provider": get_general_web_search_provider(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = await get_or_create_session(request)
        request.session_id = session_id
        runtime = await prepare_chat_runtime(request)
        requested_search_type = runtime.requested_search_type
        (
            response,
            tools_used,
            compression_used,
            sources,
            effective_search_type,
            response_backend,
            workflow_metadata,
        ) = await execute_prepared_chat_runtime(
            request.message,
            runtime,
        )
        return ChatResponse(
            message=response,
            session_id=session_id,
            sources=sources,
            tools_used=tools_used,
            metadata={
                "search_type": requested_search_type,
                "requested_search_type": requested_search_type,
                "effective_search_type": effective_search_type,
                "compression_used": compression_used,
                "use_web_search": runtime.effective_use_web_search,
                "use_react": runtime.use_react,
                "openalex_enabled": _is_openalex_enabled(),
                "agent_backend": response_backend,
                **workflow_metadata,
            },
        )
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        session_id = await get_or_create_session(request)
        run_id = uuid.uuid4().hex
        logger.info("stream started: session_id=%s run_id=%s", session_id, run_id)

        async def generate_stream():
            full_response = ""
            stream_backend = get_stream_backend()
            response_backend = get_agent_backend()
            workflow_metadata: Dict[str, Any] = {}
            tools_used: List[ToolCall] = []
            sources: List[EvidenceSource] = []
            retry_attempted = False
            retry_failed = False
            retry_suppressed = False
            retry_reason: Optional[str] = None
            llm_first_token_timeout = False
            llm_stream_total_timeout = False
            llm_generation_elapsed_seconds = 0.0
            requested_search_type = _resolve_search_type(request.search_type)
            effective_search_type = requested_search_type
            compression_used = False
            deps: Optional[AgentDependencies] = None
            use_react = bool(request.use_react)
            try:
                current_task = asyncio.current_task()
                if current_task is not None:
                    await register_stream_run(
                        run_id=run_id,
                        session_id=session_id,
                        task=current_task,
                        metadata={"user_id": request.user_id or "user"},
                    )

                yield sse_event("session", session_id=session_id, run_id=run_id)
                request.session_id = session_id
                try:
                    runtime = await asyncio.wait_for(
                        prepare_chat_runtime(request),
                        timeout=STREAM_PREPARE_TIMEOUT_SECONDS,
                    )
                except TimeoutError:
                    logger.warning(
                        "Stream timeout at stage=%s session_id=%s use_react=%s agent_backend=%s timeout_seconds=%s",
                        "prepare_chat_runtime",
                        session_id,
                        bool(getattr(request, "use_react", False)),
                        get_agent_backend(),
                        STREAM_PREPARE_TIMEOUT_SECONDS,
                    )
                    yield sse_event(
                        "status",
                        content="Request preparation timed out while building runtime context.",
                    )
                    yield sse_event(
                        "error",
                        content="Request preparation timed out while building local retrieval/runtime context. Please check API logs or narrow the question.",
                    )
                    yield sse_event("end")
                    return
                explicit_web_request = _should_force_openalex(request.message)
                deps = runtime.deps
                requested_search_type = runtime.requested_search_type
                effective_search_type = runtime.effective_search_type
                use_react = runtime.use_react
                context_payload = runtime.context_payload
                is_general_question = runtime.is_general_question
                may_need_general_web_search = runtime.may_need_general_web_search
                explicit_general_web_request = runtime.explicit_general_web_request
                is_local_question = runtime.is_local_question
                has_local_evidence = runtime.has_local_evidence
                full_prompt = runtime.full_prompt
                langgraph_context_prompt = runtime.langgraph_context_prompt
                compression_used = runtime.compression_used

                if context_payload["summary_updated"]:
                    yield (
                        sse_event("status", content="Compressing conversation history...")
                    )

                if context_payload["summary_updated"]:
                    await update_session_memory_metadata(
                        session_id=session_id,
                        latest_summary=context_payload["latest_summary"],
                        compression_count=context_payload["compression_count"],
                        compacted_message_count=context_payload["compacted_message_count"],
                    )


                await add_message(
                    session_id=session_id,
                    role="user",
                    content=request.message,
                    metadata={
                        "run_id": run_id,
                        "user_id": request.user_id,
                        "compression_used": compression_used,
                        "requested_search_type": requested_search_type,
                        "effective_search_type": effective_search_type,
                        "use_web_search": deps.use_web_search,
                        "use_react": use_react,
                    },
                )

                if explicit_web_request:
                    yield sse_event("status", content="Searching OpenAlex...")
                openalex_first_result = None
                if explicit_web_request:
                    openalex_first_result = await _run_openalex_first_if_needed(
                        message=request.message,
                        deps=deps,
                    )
                if openalex_first_result is not None:
                    response_text, tools_used, sources, workflow_metadata = openalex_first_result
                    response_backend = "openalex_first"
                    stream_backend = "openalex_first"

                    for chunk in _split_openalex_stream_chunks(response_text):
                        if not chunk:
                            continue
                        yield sse_event("text", content=chunk)
                        full_response += chunk
                        await asyncio.sleep(0.02)

                    if not full_response.strip():
                        full_response = response_text
                        yield sse_event("text", content=response_text)

                    sources = _dedupe_sources(sources)
                    sources_data = [source.model_dump() for source in sources]
                    if tools_used:
                        tools_data = [
                            {
                                "tool_name": tool.tool_name,
                                "args": tool.args,
                                "tool_call_id": tool.tool_call_id,
                            }
                            for tool in tools_used
                        ]
                        yield sse_event("tools", tools=tools_data)
                    yield sse_event("sources", sources=sources_data)
                    await add_message(
                        session_id=session_id,
                        role="assistant",
                        content=full_response,
                        metadata={
                            "run_id": run_id,
                            "streamed": True,
                            "tool_calls": len(tools_used),
                            "compression_used": compression_used,
                            "requested_search_type": requested_search_type,
                            "effective_search_type": effective_search_type,
                            "sources": sources_data,
                            "use_web_search": deps.use_web_search,
                            "use_react": use_react,
                            "agent_backend": response_backend,
                            "stream_backend": stream_backend,
                            **workflow_metadata,
                        },
                    )
                    await refresh_session_metadata(session_id)
                    yield sse_event("end")
                    return

                yield sse_event(
                    "status",
                    content="正在规划回答...",
                    phase="planning",
                    user_visible=True,
                    level="info",
                )
                if use_react and get_agent_backend() == "langchain":
                    progress_queue: asyncio.Queue[Any] = asyncio.Queue()

                    async def progress_callback(msg: Any) -> None:
                        await progress_queue.put(msg)

                    graph_task = asyncio.create_task(
                        run_langgraph_analysis(
                            question=request.message,
                            deps=deps,
                            context_prompt=langgraph_context_prompt,
                            progress_callback=progress_callback,
                        )
                    )
                    graph_start = asyncio.get_running_loop().time()
                    while not graph_task.done():
                        if asyncio.get_running_loop().time() - graph_start >= LANGGRAPH_ANALYSIS_TIMEOUT_SECONDS:
                            graph_task.cancel()
                            yield sse_event("error", content="Deep analysis timed out. Please turn off deep analysis or narrow the question and retry.")
                            yield sse_event("end")
                            return
                        try:
                            msg = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                            if not msg:
                                continue
                            if isinstance(msg, dict):
                                payload = {
                                    "content": str(msg.get("content") or ""),
                                    "phase": str(msg.get("phase") or "internal"),
                                    "user_visible": bool(msg.get("user_visible", True)),
                                    "level": str(msg.get("level") or "info"),
                                }
                            else:
                                payload = {
                                    "content": str(msg),
                                    "phase": "internal",
                                    "user_visible": False,
                                    "level": "debug",
                                }
                            yield sse_event("status", **payload)
                        except asyncio.TimeoutError:
                            continue
                    graph_result = await graph_task
                    while not progress_queue.empty():
                        msg = await progress_queue.get()
                        if not msg:
                            continue
                        if isinstance(msg, dict):
                            payload = {
                                "content": str(msg.get("content") or ""),
                                "phase": str(msg.get("phase") or "internal"),
                                "user_visible": bool(msg.get("user_visible", True)),
                                "level": str(msg.get("level") or "info"),
                            }
                        else:
                            payload = {
                                "content": str(msg),
                                "phase": "internal",
                                "user_visible": False,
                                "level": "debug",
                            }
                        yield sse_event("status", **payload)
                    full_response = str(getattr(graph_result, "message", "") or "")
                    tools_used = list(getattr(graph_result, "tools_used", []) or [])
                    sources = list(getattr(graph_result, "sources", []) or [])
                    workflow_metadata = dict(getattr(graph_result, "metadata", {}) or {})
                    response_backend = "langgraph"
                    full_response = clean_legacy_warning_text(
                        full_response,
                        drop_warning=bool(workflow_metadata.get("retrieval_skipped_by_planner") and workflow_metadata.get("direct_answer_allowed")),
                    )
                    if full_response:
                        yield sse_event("text", content=full_response)
                    stream_backend = "langgraph"
                elif stream_backend == "langchain":
                    used_langchain_stream = True
                    yield sse_event("status", content="Relevant passages found. Generating answer...")
                    stream_start = asyncio.get_running_loop().time()
                    got_first_text = False
                    try:
                        stream_iter = iter_langchain_agent_stream(full_prompt, deps=deps).__aiter__()
                        while True:
                            now = asyncio.get_running_loop().time()
                            elapsed = now - stream_start
                            if elapsed >= LLM_STREAM_TOTAL_TIMEOUT_SECONDS:
                                llm_stream_total_timeout = True
                                llm_generation_elapsed_seconds = round(elapsed, 3)
                                if full_response.strip():
                                    yield sse_event("status", content="Model generation is taking too long. Keeping the partial answer.")
                                    break
                                yield sse_event("error", content="Model generation timed out with no valid answer. Retry later, switch search mode, or turn off deep analysis.")
                                yield sse_event("end")
                                return

                            timeout_seconds = LLM_FIRST_TOKEN_TIMEOUT_SECONDS if not got_first_text else min(
                                60.0,
                                max(1.0, LLM_STREAM_TOTAL_TIMEOUT_SECONDS - elapsed),
                            )
                            try:
                                event = await _next_stream_event_with_timeout(stream_iter, timeout_seconds)
                            except StopAsyncIteration:
                                llm_generation_elapsed_seconds = round(asyncio.get_running_loop().time() - stream_start, 3)
                                break
                            except asyncio.TimeoutError:
                                if not got_first_text:
                                    llm_first_token_timeout = True
                                    llm_generation_elapsed_seconds = round(asyncio.get_running_loop().time() - stream_start, 3)
                                    yield sse_event("error", content="Model first-token timeout. Retry later, switch search mode, or turn off deep analysis.")
                                    yield sse_event("end")
                                    return
                                llm_stream_total_timeout = True
                                llm_generation_elapsed_seconds = round(asyncio.get_running_loop().time() - stream_start, 3)
                                if full_response.strip():
                                    yield sse_event("status", content="Model generation is taking too long. Keeping the partial answer.")
                                    break
                                yield sse_event("error", content="Model generation timed out with no valid answer. Retry later, switch search mode, or turn off deep analysis.")
                                yield sse_event("end")
                                return

                            if event.get("type") == "text":
                                chunk = str(event.get("content") or "")
                                yield sse_event("text", content=chunk)
                                full_response += chunk
                                if chunk.strip():
                                    got_first_text = True
                            elif event.get("type") == "final":
                                tools_used = list(event.get("tools_used", []) or [])
                                sources = list(event.get("sources", []) or [])
                                if not full_response.strip():
                                    full_response = str(event.get("message") or "")
                    except Exception:
                        logger.exception("LangChain streaming iteration failed; falling back to non-stream mode.")
                        stream_result = await asyncio.wait_for(
                            stream_langchain_agent(full_prompt, deps=deps),
                            timeout=NON_STREAM_FALLBACK_TIMEOUT_SECONDS,
                        )
                        chunk_list = list(getattr(stream_result, "chunks", []) or [])
                        if not chunk_list:
                            fallback_message = str(getattr(stream_result, "message", "") or "")
                            if fallback_message.strip():
                                chunk_list = [fallback_message]
                        for chunk in chunk_list:
                            yield sse_event("text", content=chunk)
                            full_response += chunk
                        if not full_response.strip():
                            full_response = str(getattr(stream_result, "message", "") or "")
                        tools_used = list(getattr(stream_result, "tools_used", []) or [])
                        sources = list(getattr(stream_result, "sources", []) or [])
                else:
                    used_langchain_stream = False
                    PartStartEvent, PartDeltaEvent, TextPartDelta = _get_pydantic_stream_event_types()
                    async with iter_agent(full_prompt, deps=deps) as run:
                        async for node in run:
                            if is_model_request_node(node):
                                async with node.stream(run.ctx) as request_stream:
                                    async for event in request_stream:
                                        if isinstance(event, PartStartEvent) and event.part.part_kind == "text":
                                            delta_content = event.part.content
                                            yield (
                                                sse_event("text", content=delta_content)
                                            )
                                            full_response += delta_content
                                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                            delta_content = event.delta.content_delta
                                            yield (
                                                sse_event("text", content=delta_content)
                                            )
                                            full_response += delta_content

                    result = run.result
                    tools_used = extract_tool_calls(result)
                    sources = extract_evidence_sources(deps)

                should_retry = False
                if stream_backend == "langchain" and not use_react:
                    should_retry, retry_reason = _should_retry_stream_answer(
                        full_response,
                        sources,
                        is_local_question=is_local_question,
                        explicit_web_request=explicit_web_request,
                        has_retrieved_sources=bool(getattr(deps, "retrieved_sources", []) or []),
                    )
                had_user_visible_text = bool(full_response.strip())
                if should_retry:
                    already_streamed_response = full_response.strip()
                    retry_attempted = True
                    retry_result = await retry_langchain_agent_after_degenerate(full_prompt, deps)
                    retry_message = str(getattr(retry_result, "message", "") or "").strip()
                    retry_tools = list(getattr(retry_result, "tools_used", []) or [])
                    retry_sources = list(getattr(retry_result, "sources", []) or [])
                    retry_failed = retry_message == GENERATION_RETRY_FAILED_MESSAGE

                    if already_streamed_response:
                        retry_suppressed = True
                        full_response = already_streamed_response
                    else:
                        if retry_message and not retry_failed:
                            yield sse_event("text", content=retry_message)
                            full_response = retry_message
                            tools_used = retry_tools
                            if retry_sources:
                                sources = retry_sources
                        else:
                            retry_failed = True
                            full_response = GENERATION_RETRY_FAILED_MESSAGE
                            yield sse_event("text", content=full_response)

                strict_langgraph_scope = (
                    response_backend == "langgraph"
                    and str((workflow_metadata or {}).get("scope_policy") or "") == "strict_target"
                )
                if not sources and not strict_langgraph_scope:
                    sources = list(getattr(deps, "retrieved_sources", []) or [])
                sources = _dedupe_sources(sources)
                if not had_user_visible_text:
                    full_response = _normalize_web_unavailable_reply(
                        full_response,
                        requested_web=bool(request.use_web_search),
                        sources=sources,
                    )
                # Lightweight markdown post-processing
                full_response = clean_legacy_warning_text(
                    clean_markdown_spacing(full_response),
                    drop_warning=bool(workflow_metadata.get("retrieval_skipped_by_planner") and workflow_metadata.get("direct_answer_allowed")),
                )
                sources_data = [source.model_dump() for source in sources]
                safe_workflow_metadata = {
                    k: v
                    for k, v in workflow_metadata.items()
                    if k
                    not in {
                        "requested_search_type",
                        "effective_search_type",
                        "compression_used",
                        "use_web_search",
                        "use_react",
                        "agent_backend",
                        "stream_backend",
                        "sources",
                        "tool_calls",
                    }
                }
                retrieval_error = (deps.search_preferences or {}).get("retrieval_error")
                if retrieval_error:
                    safe_workflow_metadata["retrieval_error"] = retrieval_error
                if tools_used:
                    tools_data = [
                        {
                            "tool_name": tool.tool_name,
                            "args": tool.args,
                            "tool_call_id": tool.tool_call_id,
                        }
                        for tool in tools_used
                    ]
                    yield sse_event("tools", tools=tools_data)
                if use_react and stream_backend != "langgraph":
                    yield sse_event("status", content="Preparing final answer...")
                yield sse_event("sources", sources=sources_data)

                if full_response.strip():
                    await add_message(
                        session_id=session_id,
                        role="assistant",
                        content=full_response,
                        metadata={
                            "run_id": run_id,
                            "streamed": True,
                            "tool_calls": len(tools_used),
                            "compression_used": compression_used,
                            "requested_search_type": requested_search_type,
                            "effective_search_type": effective_search_type,
                            "sources": sources_data,
                            "use_web_search": deps.use_web_search,
                            "use_react": use_react,
                            "agent_backend": response_backend,
                            "stream_backend": stream_backend,
                            "retry_attempted": retry_attempted,
                            "retry_failed": retry_failed,
                            "retry_suppressed": retry_suppressed,
                            "retry_reason": retry_reason,
                            "llm_first_token_timeout": llm_first_token_timeout,
                            "llm_stream_total_timeout": llm_stream_total_timeout,
                            "llm_generation_elapsed_seconds": llm_generation_elapsed_seconds,
                            **safe_workflow_metadata,
                        },
                    )
                await refresh_session_metadata(session_id)
                yield sse_event("end")
                logger.info("stream finished normally: session_id=%s run_id=%s", session_id, run_id)

            except asyncio.CancelledError:
                cancelled_by_user = False
                run = await get_stream_run(run_id)
                if run is not None:
                    cancelled_by_user = bool(run.cancelled_by_user)
                logger.info(
                    "stream cancelled: session_id=%s run_id=%s cancelled_by_user=%s",
                    session_id,
                    run_id,
                    cancelled_by_user,
                )
                try:
                    if full_response.strip() and deps is not None:
                        sources_data = [source.model_dump() for source in _dedupe_sources(sources)]
                        await asyncio.shield(
                            add_message(
                                session_id=session_id,
                                role="assistant",
                                content=full_response,
                                metadata={
                                    "run_id": run_id,
                                    "streamed": True,
                                    "cancelled": True,
                                    "cancelled_by_user": cancelled_by_user,
                                    "partial_response": True,
                                    "tool_calls": len(tools_used),
                                    "compression_used": compression_used,
                                    "requested_search_type": requested_search_type,
                                    "effective_search_type": effective_search_type,
                                    "sources": sources_data,
                                    "use_web_search": deps.use_web_search,
                                    "use_react": use_react,
                                    "agent_backend": response_backend,
                                    "stream_backend": stream_backend,
                                    "retry_attempted": retry_attempted,
                                    "retry_failed": retry_failed,
                                    "retry_suppressed": retry_suppressed,
                                    "retry_reason": retry_reason,
                                    "llm_first_token_timeout": llm_first_token_timeout,
                                    "llm_stream_total_timeout": llm_stream_total_timeout,
                                    "llm_generation_elapsed_seconds": llm_generation_elapsed_seconds,
                                    **workflow_metadata,
                                },
                            )
                        )
                        await asyncio.shield(refresh_session_metadata(session_id))
                    if cancelled_by_user:
                        yield sse_event("cancelled", run_id=run_id, message="已停止生成")
                        yield sse_event("end")
                        return
                except Exception:
                    logger.exception("Failed to persist cancelled stream partial response")
                raise
            except Exception as e:
                logger.exception("Stream error: %s", e)
                error_type = type(e).__name__
                error_message = str(e)[:300]
                yield sse_event(
                    "error",
                    content=f"Deep analysis stream failed: {error_type}: {error_message}",
                )
                yield sse_event("end")
            finally:
                logger.info("stream finally reached: session_id=%s run_id=%s", session_id, run_id)
                await unregister_stream_run(run_id)

        return stream_response(generate_stream())

    except Exception as e:
        logger.exception("Streaming chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream/{run_id}/cancel")
async def cancel_chat_stream(run_id: str):
    try:
        result = await cancel_stream_run(run_id)
        return {
            "run_id": run_id,
            "status": result.get("status", "not_found"),
        }
    except Exception as e:
        logger.error("Stream cancel failed for run_id=%s error=%s", run_id, e)
        return {
            "run_id": run_id,
            "status": "not_found",
        }


@app.post("/search/vector")
async def search_vector(request: SearchRequest):
    try:
        input_data = VectorSearchInput(query=request.query, limit=request.limit)
        start_time = datetime.now()
        results = await vector_search_tool(input_data)
        end_time = datetime.now()
        query_time = (end_time - start_time).total_seconds() * 1000
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="vector",
            query_time_ms=query_time,
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid")
async def search_hybrid(request: SearchRequest):
    try:
        input_data = HybridSearchInput(query=request.query, limit=request.limit, text_weight=0.3)
        start_time = datetime.now()
        results = await hybrid_search_tool(input_data)
        end_time = datetime.now()
        query_time = (end_time - start_time).total_seconds() * 1000
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="hybrid",
            query_time_ms=query_time,
        )
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents_endpoint(limit: int = 20, offset: int = 0):
    try:
        input_data = DocumentListInput(limit=limit, offset=offset)
        documents = await list_documents_tool(input_data)
        return {"documents": documents, "total": len(documents), "limit": limit, "offset": offset}
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions(limit: int = 20, days: int = 7):
    try:
        sessions = await list_recent_sessions(limit=limit, days=days)
        items = [SessionListItem(**session) for session in sessions]
        return SessionListResponse(sessions=items, total=len(items))
    except Exception as e:
        logger.error(f"Session list failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages_endpoint(session_id: str):
    try:
        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = await get_session_messages(session_id)
        items = [
            ChatMessageItem(
                message_id=msg["id"],
                role=msg["role"],
                content=msg["content"],
                metadata=msg.get("metadata") or {},
                created_at=datetime.fromisoformat(msg["created_at"]),
            )
            for msg in messages
        ]
        return SessionMessagesResponse(session_id=session_id, messages=items, total=len(items))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session messages failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    try:
        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingestion/tasks/{task_id}", response_model=IngestionTaskResponse)
async def get_ingestion_task_endpoint(task_id: str):
    try:
        task = await get_ingestion_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Ingestion task not found")
        return IngestionTaskResponse(**task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion task retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    try:
        deleted = await delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/openalex/add-to-kb")
async def add_openalex_to_knowledge_base(payload: Dict[str, Any]):
    file_url = str(payload.get("pdf_url") or payload.get("content_url") or "").strip()
    if not file_url.startswith("http"):
        raise HTTPException(status_code=400, detail="No valid PDF/content URL provided")
    title = str(payload.get("title") or payload.get("openalex_id") or "openalex_paper")
    return await add_openalex_file_to_kb(file_url=file_url, title=title)


@app.post("/documents/upload")
async def upload_document_to_kb(payload: Dict[str, Any]):
    return await run_sync_upload_ingestion(payload)


@app.post("/ingestion/tasks", response_model=IngestionTaskResponse)
async def submit_ingestion_task(payload: Dict[str, Any]):
    task = await submit_async_ingestion_task(payload)
    return IngestionTaskResponse(**task)


@app.post("/documents/upload/start")
async def start_document_upload_job(payload: Dict[str, Any]):
    return await start_upload_ingestion_job(payload)


@app.get("/documents/upload/jobs/{job_id}")
async def get_document_upload_job(job_id: str):
    return await get_upload_ingestion_job(job_id)


@app.post("/documents/upload/jobs/{job_id}/cancel")
async def cancel_document_upload_job(job_id: str):
    return await cancel_upload_ingestion_job(job_id)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=str(uuid.uuid4()),
    )


if __name__ == "__main__":
    uvicorn.run(
        "agent.api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
        log_level=LOG_LEVEL.lower(),
    )

