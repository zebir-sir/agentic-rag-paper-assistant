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

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from .http_errors import register_exception_handlers
from .http_middleware import register_http_middleware
from .health_runtime import APP_VERSION, build_readiness_status
from .cache_utils import close_redis_client, startup_redis_client
from .runtime_config import build_runtime_diagnostics
from .runtime_metrics import get_runtime_metrics_snapshot
from .runtime_models import HttpMetricsSnapshot, RuntimeDiagnostics
from .chat_metrics_models import ChatMetricsSnapshot
from .chat_metrics_runtime import get_chat_metrics_snapshot, record_chat_request_metric
from .agent_runtime import AgentDependencies
from .agent_langgraph import run_langgraph_analysis
from .retrieval_harness_runtime import build_retrieval_harness_trace_payload
from .agent_langchain import (
    GENERATION_RETRY_FAILED_MESSAGE,
    run_langchain_agent,
    stream_langchain_agent,
    iter_langchain_agent_stream,
    get_langchain_chat_model,
    retry_langchain_agent_after_degenerate,
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
    set_message_memory_eligible,
    get_session_messages,
    test_connection,
    refresh_session_metadata,
    list_recent_sessions,
    delete_session,
    get_session_memory_snapshot,
    save_session_memory_snapshot,
    get_artifact,
    get_artifact_image,
    get_document_pdf,
    delete_document,
    list_document_annotations,
    create_document_annotation,
    update_document_annotation_position,
    delete_document_annotation,
)
from .pdf_page_renderer import render_cached_pdf_page_png
from .translation_error_policy import is_translation_service_unavailable
from .app_config import get_rabbitmq_url
from .openalex_router import _is_openalex_enabled
from .ingestion_tasks_db import (
    delete_ingestion_task,
    get_ingestion_task,
    list_ingestion_tasks,
    pause_ingestion_task,
    reorder_ingestion_tasks,
    resume_ingestion_task as resume_ingestion_task_record,
    update_ingestion_task_status,
)
from .models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    HealthStatus,
    ReadinessStatus,
    ToolCall,
    EvidenceSource,
    SessionListResponse,
    SessionListItem,
    SessionMessagesResponse,
    SessionMemorySnapshot,
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
    build_context,
    build_memory_update_prompt,
    memory_eligible_messages,
    messages_for_memory_update,
    normalize_memory_summary,
    normalize_memory_state,
    should_update_memory,
)
from .memory_runtime import build_session_memory_snapshot
from .dialog_policy import classify_dialog_turn
from .history_resolver import resolve_history_query
from .query_rewrite_runtime import (
    build_query_rewrite_context,
    get_query_rewrite_model,
    rewrite_query_with_conversation,
)
from .answer_review_runtime import review_generated_answer
from .retrieval_failure_policy import (
    apply_external_retrieval_failure_policy,
    build_external_retrieval_disclosure,
)
from .simple_chat_runtime import (
    SimpleChatDecision,
    choose_simple_chat_strategy,
    run_simple_chat_runtime,
)
from .ingestion_jobs import (
    add_openalex_file_to_kb,
    submit_async_ingestion_task,
    submit_async_ingestion_tasks,
)
from .rabbitmq_producer import publish_ingestion_task
from .stream_registry import (
    register_stream_run,
    unregister_stream_run,
    cancel_stream_run,
    get_stream_run,
)
from .warning_text import clean_legacy_warning_text
from .request_context import get_request_id
from .document_reader_runtime import stream_document_translation, translate_document
from .selection_translation_runtime import translate_selection
from .graph_runtime import ensure_paper_graph, get_paper_graph
from .graph_localization_runtime import schedule_pending_graph_localizations
from .graph_schema import PaperGraphResponse

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
LIGHTWEIGHT_CHAT_TIMEOUT_SECONDS = float(os.getenv("LIGHTWEIGHT_CHAT_TIMEOUT_SECONDS", "15"))
RABBITMQ_URL = get_rabbitmq_url()
AGENT_RUNTIME_BACKEND = "langchain"


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


def _build_conversation_carryover_block(
    *,
    original_query: str,
    resolved_query: str,
    topic_hint: str,
    recent_history_summary: str,
    dialog_act: str,
    carry_context: bool,
    response_style: str,
) -> str:
    if not any(
        [
            carry_context,
            resolved_query and resolved_query != original_query,
            topic_hint,
            recent_history_summary,
        ]
    ):
        return ""

    lines = [
        "[Conversation carry-over]",
        f"- dialog_act: {dialog_act or 'unknown'}",
        f"- carry_context: {'true' if carry_context else 'false'}",
        f"- response_style: {response_style or 'normal'}",
    ]
    if topic_hint:
        lines.append(f"- topic_hint: {topic_hint}")
    if resolved_query:
        lines.append(f"- resolved_query_for_retrieval: {resolved_query}")
    if recent_history_summary:
        lines.append(f"- recent_history_summary: {recent_history_summary}")
    lines.extend(
        [
            "Interpret short references such as 这个/它/表/图/算法 using the resolved query and topic hint when planning retrieval.",
            "Answer naturally to the current user turn; do not say you are using hidden conversation carry-over logic.",
        ]
    )
    return "\n".join(lines)


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


def _should_retry_stream_answer(
    full_response: str,
    sources: List[EvidenceSource],
    *,
    is_local_question: bool,
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

    # Local answers with evidence should not retry solely because of formatting.
    has_any_sources = bool(sources) or bool(has_retrieved_sources)
    if is_local_question and has_any_sources:
        # 本地知识库问题有证据时，优先保留原回答，不要因为排版较差就重试。
        return False, "local_with_evidence_keep"

    if not sources and has_unverified_web_citations(stripped):
        return True, "unverified_web_citation"

    # 兜底：如果没有来源且触发了退化检测，进行 retry
    if is_degenerate_answer(stripped):
        return True, "degenerate_answer"

    return False, "no_retry"


def _apply_external_retrieval_disclosure(
    response: str,
    deps: AgentDependencies,
    workflow_metadata: Dict[str, Any],
    *,
    allow_model_knowledge: bool = True,
) -> tuple[str, List[Dict[str, Any]], Dict[str, Any], str]:
    """Expose external-search degradation without presenting it as retrieved evidence."""
    metadata = dict(workflow_metadata or {})
    statuses = [
        dict(item)
        for item in (metadata.get("external_retrieval_statuses") or (deps.search_preferences or {}).get("external_retrieval_statuses") or [])
        if isinstance(item, dict)
    ]
    input_policy = dict(metadata.get("answer_policy") or {})
    if not allow_model_knowledge:
        input_policy["blocked_source_types"] = list(
            dict.fromkeys([*input_policy.get("blocked_source_types", []), "model_knowledge"])
        )
    policy = apply_external_retrieval_failure_policy(
        input_policy,
        statuses,
    )
    disclosure = build_external_retrieval_disclosure(statuses, policy)
    text = str(response or "").strip()
    if disclosure and disclosure not in text:
        text = f"{text}\n\n{disclosure}".strip()
    return text, statuses, policy, disclosure


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
        await startup_redis_client()
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
        await close_redis_client()
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

PDF_PAGE_RENDER_SEMAPHORE = asyncio.Semaphore(2)
register_http_middleware(app)
register_exception_handlers(app)


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
    return memory_eligible_messages(messages)


async def _summarize_for_memory(
    session_id: str,
    user_id: Optional[str],
    previous_summary: Dict[str, Any],
    messages_to_compact: List[Dict[str, Any]],
) -> Dict[str, Any]:
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

    summary_prompt = build_memory_update_prompt(previous_summary, messages_to_compact)
    model = get_langchain_chat_model()
    response = await model.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "You update structured research memory. Return JSON only. Do not call tools, "
                    "do not retrieve documents, and do not store paper facts or tool output."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ]
    )
    summary_text = _extract_langchain_text(response).strip()
    try:
        normalized_text = summary_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(normalized_text)
        return normalize_memory_summary(parsed) if isinstance(parsed, dict) else normalize_memory_summary(previous_summary)
    except json.JSONDecodeError:
        logger.warning("Structured memory model response was not valid JSON for session=%s", session_id)
        return normalize_memory_summary(previous_summary)


async def _prepare_agent_prompt(
    session_id: str,
    user_id: Optional[str],
    user_message: str,
) -> Dict[str, Any]:
    history_messages = await get_conversation_context(session_id=session_id, max_messages=None)
    memory_snapshot = await get_session_memory_snapshot(session_id)
    memory_state = normalize_memory_state(memory_snapshot)

    memory_updated = False
    if should_update_memory(history_messages, memory_state, user_message):
        messages_to_compact = messages_for_memory_update(history_messages, memory_state.covered_message_count)
        if messages_to_compact:
            updated_summary = await _summarize_for_memory(
                session_id=session_id,
                user_id=user_id,
                previous_summary=memory_state.summary,
                messages_to_compact=messages_to_compact,
            )
            memory_state.summary = updated_summary
            memory_state.covered_message_count = len(history_messages)
            memory_updated = True

    context_result = build_context(
        history_messages=history_messages,
        current_question=user_message,
        memory_state=memory_state,
    )
    context_result.memory_updated = memory_updated

    return {
        "full_prompt": context_result.full_prompt,
        "compression_used": context_result.compression_used,
        "memory_updated": context_result.memory_updated,
        "memory_state": context_result.memory_state,
        "history_messages": history_messages,
    }


def _count_source_types(sources: List[EvidenceSource]) -> tuple[int, int]:
    local_source_count = 0
    web_source_count = 0
    for source in sources:
        source_type = str(getattr(source, "source_type", "") or "").lower()
        if source_type == "web":
            web_source_count += 1
        else:
            local_source_count += 1
    return local_source_count, web_source_count


def _emit_chat_request_metric(
    *,
    request_id: Optional[str],
    session_id: str,
    route: str,
    status: str,
    response_backend: str,
    requested_search_type: str,
    effective_search_type: str,
    use_web_search: bool,
    use_react: bool,
    compression_used: bool,
    tools_used: List[ToolCall],
    sources: List[EvidenceSource],
    response_text: str,
) -> None:
    local_source_count, web_source_count = _count_source_types(sources)
    record_chat_request_metric(
        request_id=request_id,
        session_id=session_id,
        route=route,
        status=status,
        response_backend=response_backend,
        requested_search_type=requested_search_type,
        effective_search_type=effective_search_type,
        use_web_search=use_web_search,
        use_react=use_react,
        compression_used=compression_used,
        tool_call_count=len(tools_used),
        local_source_count=local_source_count,
        web_source_count=web_source_count,
        response_chars=len(str(response_text or "")),
    )


async def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_metadata: Optional[Dict[str, Any]] = None,
    assistant_metadata: Optional[Dict[str, Any]] = None,
    memory_eligible: bool = False,
):
    user_message_id = await add_message(
        session_id=session_id,
        role="user",
        content=user_message,
        metadata=user_metadata if user_metadata is not None else (metadata or {}),
    )
    assistant_message_id = await add_message(
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        metadata=assistant_metadata if assistant_metadata is not None else (metadata or {}),
    )
    if memory_eligible:
        await set_message_memory_eligible(user_message_id)
        await set_message_memory_eligible(assistant_message_id)
    await refresh_session_metadata(session_id)


def _resolve_search_type(search_type: Any) -> str:
    value = str(search_type).lower().strip()
    return "vector" if value == "vector" else "hybrid"


@dataclass
class ChatRuntime:
    session_id: str
    deps: AgentDependencies
    requested_search_type: str
    effective_search_type: str
    effective_use_web_search: bool
    use_react: bool
    retrieval_query: str
    full_prompt: str
    langgraph_context_prompt: str
    compression_used: bool
    context_payload: Dict[str, Any]
    is_general_question: bool
    may_need_general_web_search: bool
    explicit_general_web_request: bool
    is_local_question: bool
    has_local_evidence: bool
    simple_chat_decision: SimpleChatDecision = field(
        default_factory=lambda: SimpleChatDecision(
            enabled=False,
            reason="not_evaluated",
        )
    )
    workflow_metadata: Dict[str, Any] = field(default_factory=dict)


async def prepare_chat_runtime(request: ChatRequest) -> ChatRuntime:
    requested_search_type = _resolve_search_type(request.search_type)
    effective_use_web_search = bool(request.use_web_search)
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
            "use_paper_graph": bool(request_metadata.get("use_paper_graph", True)),
        },
    )
    context_payload = await _prepare_agent_prompt(
        session_id=request.session_id or "",
        user_id=request.user_id,
        user_message=request.message,
    )
    history_messages = list(context_payload.get("history_messages") or [])
    history_resolution = resolve_history_query(
        latest_query=request.message,
        history_messages=history_messages,
    )
    try:
        rewrite_model = get_query_rewrite_model()
    except Exception:
        rewrite_model = None
    memory_state = context_payload.get("memory_state")
    rewrite_context = build_query_rewrite_context(
        history_messages=history_messages,
        memory_summary=dict(getattr(memory_state, "summary", {}) or {}),
    )
    query_rewrite = await rewrite_query_with_conversation(
        original_query=request.message,
        conversation_context=rewrite_context,
        model=rewrite_model,
    )
    retrieval_query = query_rewrite.rewritten_query
    dialog_policy = classify_dialog_turn(
        latest_query=request.message,
        history_messages=history_messages,
    )
    is_general_question = _is_general_algorithm_question(request.message)
    may_need_general_web_search = _may_need_general_web_search(request.message)
    explicit_general_web_request = _is_explicit_general_web_request(request.message)
    is_local_question = _is_local_kb_question(retrieval_query)
    simple_chat_decision = choose_simple_chat_strategy(
        message=request.message,
        resolved_query=retrieval_query,
        is_local_question=is_local_question,
        use_react=bool(request.use_react),
        use_web_search=effective_use_web_search,
    )

    local_context = ""
    if is_local_question:
        local_context = await _run_local_kb_preflight_if_needed(retrieval_query, deps)
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
    carryover_block = _build_conversation_carryover_block(
        original_query=history_resolution.original_query,
        resolved_query=retrieval_query,
        topic_hint=history_resolution.topic_hint,
        recent_history_summary=history_resolution.recent_history_summary,
        dialog_act=dialog_policy.dialog_act,
        carry_context=dialog_policy.carry_context,
        response_style=dialog_policy.response_style,
    )
    if carryover_block:
        langgraph_context_prompt = f"{langgraph_context_prompt}\n\n{carryover_block}"
        full_prompt = f"{full_prompt}\n\n{carryover_block}"
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
        effective_use_web_search=effective_use_web_search,
        use_react=bool(request.use_react),
        retrieval_query=retrieval_query,
        full_prompt=full_prompt,
        langgraph_context_prompt=langgraph_context_prompt,
        compression_used=bool(context_payload["compression_used"]),
        context_payload=context_payload,
        is_general_question=is_general_question,
        may_need_general_web_search=may_need_general_web_search,
        explicit_general_web_request=explicit_general_web_request,
        is_local_question=is_local_question,
        has_local_evidence=has_local_evidence,
        simple_chat_decision=simple_chat_decision,
        workflow_metadata={
            "dialog_act": dialog_policy.dialog_act,
            "dialog_reason": dialog_policy.reason,
            "dialog_response_style": dialog_policy.response_style,
            "paper_graph_used": False,
            "paper_graph_expanded_document_count": 0,
            "carry_context": dialog_policy.carry_context,
            "resolved_query": retrieval_query,
            "history_resolution_used": history_resolution.used_history,
            "history_resolution_reason": history_resolution.reason,
            "history_topic_hint": history_resolution.topic_hint,
            "query_rewrite_model_used": query_rewrite.model_used,
            "query_rewrite_reason": query_rewrite.reason,
            "query_rewrite_context_estimated_tokens": max(1, len(rewrite_context) // 4) if rewrite_context else 0,
            "simple_chat_candidate": simple_chat_decision.enabled,
            "simple_chat_candidate_mode": simple_chat_decision.mode,
            "simple_chat_candidate_reason": simple_chat_decision.reason,
        },
    )


async def execute_prepared_chat_runtime(
    message: str,
    runtime: ChatRuntime,
    *,
    save_conversation: bool = True,
) -> tuple[str, List[ToolCall], bool, List[EvidenceSource], str, str, Dict[str, Any]]:
    try:
        response_backend = AGENT_RUNTIME_BACKEND
        workflow_metadata: Dict[str, Any] = {}
        deps = runtime.deps
        compression_used = runtime.compression_used
        effective_search_type = runtime.effective_search_type

        if runtime.context_payload["memory_updated"]:
            state = runtime.context_payload["memory_state"]
            await save_session_memory_snapshot(
                session_id=runtime.session_id,
                covered_message_count=state.covered_message_count,
                summary=state.summary,
            )

        simple_chat_result = None
        if runtime.simple_chat_decision.enabled:
            simple_chat_result = await run_simple_chat_runtime(
                deps=deps,
                user_message=message,
                decision=runtime.simple_chat_decision,
                response_style=str(runtime.workflow_metadata.get("dialog_response_style") or "normal"),
            )
        if simple_chat_result is not None:
            response = simple_chat_result.message
            tools_used = list(simple_chat_result.tools_used or [])
            sources = list(simple_chat_result.sources or [])
            workflow_metadata = dict(simple_chat_result.metadata or {})
            response_backend = "simple_chat_runtime"
        else:
            if runtime.use_react:
                graph_result = await run_langgraph_analysis(
                    question=runtime.retrieval_query,
                    deps=deps,
                    context_prompt=runtime.langgraph_context_prompt,
                )
                response = str(getattr(graph_result, "message", "") or "")
                tools_used = list(getattr(graph_result, "tools_used", []) or [])
                sources = list(getattr(graph_result, "sources", []) or [])
                workflow_metadata = dict(getattr(graph_result, "metadata", {}) or {})
                response_backend = "langgraph"
            else:
                result = await run_langchain_agent(runtime.full_prompt, deps=deps)
                response = str(getattr(result, "message", "") or "")
                tools_used = list(getattr(result, "tools_used", []) or [])
                sources = list(getattr(result, "sources", []) or [])
                if is_degenerate_answer(response):
                    logger.warning("Detected degenerate answer in /chat normal path; retrying once.")
                    retry_result = await retry_langchain_agent_after_degenerate(runtime.full_prompt, deps)
                    response = str(getattr(retry_result, "message", "") or "")
                    tools_used = list(getattr(retry_result, "tools_used", []) or [])
                    sources = list(getattr(retry_result, "sources", []) or [])

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
        response, external_retrieval_statuses, external_fallback_policy, external_disclosure = _apply_external_retrieval_disclosure(
            response,
            deps,
            workflow_metadata,
            allow_model_knowledge=not (runtime.is_local_question and not sources),
        )
        review_result = review_generated_answer(
            answer=response,
            sources=sources,
            is_local_question=runtime.is_local_question,
        )
        response = review_result.revised_answer
        sources_dict = [source.model_dump() for source in sources]

        safe_workflow_metadata = {
            k: v
            for k, v in {**runtime.workflow_metadata, **workflow_metadata}.items()
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
        if external_retrieval_statuses:
            safe_workflow_metadata["external_retrieval_statuses"] = external_retrieval_statuses
            safe_workflow_metadata["external_retrieval_fallback_active"] = bool(
                external_fallback_policy.get("external_fallback_mode")
            )
            safe_workflow_metadata["external_retrieval_fallback_disclosure"] = external_disclosure
        safe_workflow_metadata["answer_review_reviewed"] = review_result.reviewed
        safe_workflow_metadata["answer_review_action"] = review_result.review_action
        safe_workflow_metadata["answer_review_risk"] = review_result.unsupported_claim_risk
        safe_workflow_metadata["answer_review_reason"] = review_result.reason
        safe_workflow_metadata["answer_review_note_count"] = len(review_result.unsupported_claim_notes or [])

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
                    memory_eligible=True,
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
                    "agent_backend": AGENT_RUNTIME_BACKEND,
                },
                assistant_metadata={
                    "error": str(e),
                    "compression_used": False,
                    "requested_search_type": runtime.requested_search_type,
                    "effective_search_type": runtime.effective_search_type,
                    "sources": [],
                    "agent_backend": AGENT_RUNTIME_BACKEND,
                },
            )

        return (
            error_response,
            [],
            False,
            [],
            runtime.effective_search_type,
            AGENT_RUNTIME_BACKEND,
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
            version=APP_VERSION,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/health/live")
async def health_live():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "timestamp": datetime.now(),
    }


@app.get("/health/ready", response_model=ReadinessStatus)
async def health_ready():
    return ReadinessStatus(**(await build_readiness_status()))


@app.get("/openalex/status")
async def openalex_status():
    return {"enabled": _is_openalex_enabled()}


@app.get("/web-search/status")
async def web_search_status():
    return {
        "enabled": is_general_web_search_enabled(),
        "provider": get_general_web_search_provider(),
    }


@app.get("/system/runtime", response_model=RuntimeDiagnostics)
async def system_runtime():
    return RuntimeDiagnostics(**build_runtime_diagnostics())


@app.get("/system/metrics", response_model=HttpMetricsSnapshot)
async def system_metrics():
    return HttpMetricsSnapshot(**get_runtime_metrics_snapshot())


@app.get("/system/chat-metrics", response_model=ChatMetricsSnapshot)
async def system_chat_metrics():
    return ChatMetricsSnapshot(**get_chat_metrics_snapshot())


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    request_id = get_request_id()
    session_id = request.session_id or ""
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
        _emit_chat_request_metric(
            request_id=request_id,
            session_id=session_id,
            route="/chat",
            status="success",
            response_backend=response_backend,
            requested_search_type=requested_search_type,
            effective_search_type=effective_search_type,
            use_web_search=runtime.effective_use_web_search,
            use_react=runtime.use_react,
            compression_used=compression_used,
            tools_used=tools_used,
            sources=sources,
            response_text=response,
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
        _emit_chat_request_metric(
            request_id=request_id,
            session_id=session_id,
            route="/chat",
            status="error",
            response_backend=AGENT_RUNTIME_BACKEND,
            requested_search_type=_resolve_search_type(request.search_type),
            effective_search_type=_resolve_search_type(request.search_type),
            use_web_search=bool(request.use_web_search),
            use_react=bool(request.use_react),
            compression_used=False,
            tools_used=[],
            sources=[],
            response_text="",
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        session_id = await get_or_create_session(request)
        request_id = get_request_id()
        run_id = uuid.uuid4().hex
        logger.info("stream started: session_id=%s run_id=%s", session_id, run_id)

        async def generate_stream():
            full_response = ""
            stream_backend = AGENT_RUNTIME_BACKEND
            response_backend = AGENT_RUNTIME_BACKEND
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
            user_message_id: Optional[str] = None
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
                        AGENT_RUNTIME_BACKEND,
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

                if context_payload["memory_updated"]:
                    yield sse_event("status", content="Updating structured research memory...")
                    state = context_payload["memory_state"]
                    await save_session_memory_snapshot(
                        session_id=session_id,
                        covered_message_count=state.covered_message_count,
                        summary=state.summary,
                    )


                user_message_id = await add_message(
                    session_id=session_id,
                    role="user",
                    content=request.message,
                    metadata={
                        "run_id": run_id,
                        "memory_eligible": False,
                        "user_id": request.user_id,
                        "compression_used": compression_used,
                        "requested_search_type": requested_search_type,
                        "effective_search_type": effective_search_type,
                        "use_web_search": deps.use_web_search,
                        "use_react": use_react,
                    },
                )

                if runtime.simple_chat_decision.enabled:
                    is_conversation = runtime.simple_chat_decision.mode == "conversation"
                    yield sse_event(
                        "status",
                        content="正在回复..." if is_conversation else "正在检索相关论文片段...",
                    )
                    try:
                        simple_chat_result = await asyncio.wait_for(
                            run_simple_chat_runtime(
                                deps=deps,
                                user_message=request.message,
                                decision=runtime.simple_chat_decision,
                                response_style=str(runtime.workflow_metadata.get("dialog_response_style") or "normal"),
                            ),
                            timeout=LIGHTWEIGHT_CHAT_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        if is_conversation:
                            simple_chat_result = None
                            yield sse_event(
                                "text",
                                content="我在。刚才回复稍微慢了一点，但这个问题不需要走论文检索。你可以再问我一次。",
                            )
                            yield sse_event("sources", sources=[])
                            yield sse_event("end")
                            return
                        raise
                    if is_conversation and simple_chat_result is None:
                        yield sse_event(
                            "text",
                            content="我在。这个问题不需要查论文，你可以直接继续和我聊。",
                        )
                        yield sse_event("sources", sources=[])
                        yield sse_event("end")
                        return
                    if simple_chat_result is not None:
                        response_backend = "simple_chat_runtime"
                        stream_backend = "simple_chat_runtime"
                        full_response = simple_chat_result.message
                        tools_used = list(simple_chat_result.tools_used or [])
                        sources = _dedupe_sources(list(simple_chat_result.sources or []))
                        workflow_metadata = dict(simple_chat_result.metadata or {})
                        full_response = clean_legacy_warning_text(clean_markdown_spacing(full_response))
                        review_result = review_generated_answer(
                            answer=full_response,
                            sources=sources,
                            is_local_question=is_local_question,
                        )
                        full_response = review_result.revised_answer
                        yield sse_event("text", content=full_response)
                        sources_data = [source.model_dump() for source in sources]
                        safe_workflow_metadata = {
                            k: v
                            for k, v in {**runtime.workflow_metadata, **workflow_metadata}.items()
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
                        safe_workflow_metadata["answer_review_reviewed"] = review_result.reviewed
                        safe_workflow_metadata["answer_review_action"] = review_result.review_action
                        safe_workflow_metadata["answer_review_risk"] = review_result.unsupported_claim_risk
                        safe_workflow_metadata["answer_review_reason"] = review_result.reason
                        safe_workflow_metadata["answer_review_note_count"] = len(review_result.unsupported_claim_notes or [])
                        if tools_used:
                            yield sse_event(
                                "tools",
                                tools=[
                                    {
                                        "tool_name": tool.tool_name,
                                        "args": tool.args,
                                        "tool_call_id": tool.tool_call_id,
                                    }
                                    for tool in tools_used
                                ],
                            )
                        yield sse_event("sources", sources=sources_data)
                        assistant_message_id = await add_message(
                            session_id=session_id,
                            role="assistant",
                            content=full_response,
                            metadata={
                                "run_id": run_id,
                                "streamed": True,
                                "memory_eligible": True,
                                "tool_calls": len(tools_used),
                                "compression_used": compression_used,
                                "requested_search_type": requested_search_type,
                                "effective_search_type": effective_search_type,
                                "sources": sources_data,
                                "use_web_search": deps.use_web_search,
                                "use_react": use_react,
                                "agent_backend": response_backend,
                                "stream_backend": stream_backend,
                                **safe_workflow_metadata,
                            },
                        )
                        if user_message_id:
                            await set_message_memory_eligible(user_message_id)
                        await refresh_session_metadata(session_id)
                        _emit_chat_request_metric(
                            request_id=request_id,
                            session_id=session_id,
                            route="/chat/stream",
                            status="success",
                            response_backend=response_backend,
                            requested_search_type=requested_search_type,
                            effective_search_type=effective_search_type,
                            use_web_search=bool(deps.use_web_search),
                            use_react=use_react,
                            compression_used=compression_used,
                            tools_used=tools_used,
                            sources=sources,
                            response_text=full_response,
                        )
                        yield sse_event("end")
                        return

                yield sse_event(
                    "status",
                    content="正在规划回答...",
                    phase="planning",
                    user_visible=True,
                    level="info",
                )
                if use_react:
                    progress_queue: asyncio.Queue[Any] = asyncio.Queue()
                    streamed_answer_parts: List[str] = []
                    answer_delta_count = 0
                    last_answer_delta_at: Optional[float] = None
                    max_answer_delta_gap_seconds = 0.0

                    async def progress_callback(msg: Any) -> None:
                        await progress_queue.put(msg)

                    async def answer_callback(content: str) -> None:
                        nonlocal answer_delta_count, last_answer_delta_at, max_answer_delta_gap_seconds
                        now = asyncio.get_running_loop().time()
                        if last_answer_delta_at is not None:
                            gap_seconds = now - last_answer_delta_at
                            max_answer_delta_gap_seconds = max(max_answer_delta_gap_seconds, gap_seconds)
                            if gap_seconds >= 2.0:
                                logger.info(
                                    "stream answer delta gap: session_id=%s run_id=%s gap_ms=%.0f",
                                    session_id,
                                    run_id,
                                    gap_seconds * 1000,
                                )
                        last_answer_delta_at = now
                        answer_delta_count += 1
                        await progress_queue.put({"type": "answer_delta", "content": content})

                    graph_task = asyncio.create_task(
                        run_langgraph_analysis(
                            question=runtime.retrieval_query,
                            deps=deps,
                            context_prompt=langgraph_context_prompt,
                            progress_callback=progress_callback,
                            answer_callback=answer_callback,
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
                                if msg.get("type") == "answer_delta":
                                    delta = str(msg.get("content") or "")
                                    if delta:
                                        streamed_answer_parts.append(delta)
                                        yield sse_event("text", content=delta)
                                    continue
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
                            if msg.get("type") == "answer_delta":
                                delta = str(msg.get("content") or "")
                                if delta:
                                    streamed_answer_parts.append(delta)
                                    yield sse_event("text", content=delta)
                                continue
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
                    workflow_metadata["stream_answer_delta_count"] = answer_delta_count
                    workflow_metadata["stream_max_answer_delta_gap_ms"] = round(max_answer_delta_gap_seconds * 1000)
                    response_backend = "langgraph"
                    full_response = clean_legacy_warning_text(
                        full_response,
                        drop_warning=bool(workflow_metadata.get("retrieval_skipped_by_planner") and workflow_metadata.get("direct_answer_allowed")),
                    )
                    streamed_answer = "".join(streamed_answer_parts).strip()
                    if full_response and not streamed_answer:
                        yield sse_event("text", content=full_response)
                    elif full_response.startswith(streamed_answer) and len(full_response) > len(streamed_answer):
                        yield sse_event("text", content=full_response[len(streamed_answer):])
                    stream_backend = "langgraph"
                else:
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
                should_retry = False
                if stream_backend == "langchain" and not use_react:
                    should_retry, retry_reason = _should_retry_stream_answer(
                        full_response,
                        sources,
                        is_local_question=is_local_question,
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
                response_before_disclosure = full_response
                full_response, external_retrieval_statuses, external_fallback_policy, external_disclosure = _apply_external_retrieval_disclosure(
                    full_response,
                    deps,
                    workflow_metadata,
                    allow_model_knowledge=not (is_local_question and not sources),
                )
                if full_response.startswith(response_before_disclosure) and len(full_response) > len(response_before_disclosure):
                    yield sse_event("text", content=full_response[len(response_before_disclosure):])
                review_result = review_generated_answer(
                    answer=full_response,
                    sources=sources,
                    is_local_question=is_local_question,
                )
                reviewed_response = review_result.revised_answer
                if reviewed_response != full_response and reviewed_response.startswith(full_response):
                    appended_suffix = reviewed_response[len(full_response):]
                    if appended_suffix.strip():
                        yield sse_event("text", content=appended_suffix)
                full_response = reviewed_response
                sources_data = [source.model_dump() for source in sources]
                safe_workflow_metadata = {
                    k: v
                    for k, v in {**runtime.workflow_metadata, **workflow_metadata}.items()
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
                if external_retrieval_statuses:
                    safe_workflow_metadata["external_retrieval_statuses"] = external_retrieval_statuses
                    safe_workflow_metadata["external_retrieval_fallback_active"] = bool(
                        external_fallback_policy.get("external_fallback_mode")
                    )
                    safe_workflow_metadata["external_retrieval_fallback_disclosure"] = external_disclosure
                safe_workflow_metadata["answer_review_reviewed"] = review_result.reviewed
                safe_workflow_metadata["answer_review_action"] = review_result.review_action
                safe_workflow_metadata["answer_review_risk"] = review_result.unsupported_claim_risk
                safe_workflow_metadata["answer_review_reason"] = review_result.reason
                safe_workflow_metadata["answer_review_note_count"] = len(review_result.unsupported_claim_notes or [])
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
                yield sse_event(
                    "workflow",
                    retrieval_harness=build_retrieval_harness_trace_payload(safe_workflow_metadata),
                )
                yield sse_event("sources", sources=sources_data)

                if full_response.strip():
                    assistant_message_id = await add_message(
                        session_id=session_id,
                        role="assistant",
                        content=full_response,
                        metadata={
                            "run_id": run_id,
                            "streamed": True,
                            "memory_eligible": True,
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
                    if user_message_id:
                        await set_message_memory_eligible(user_message_id)
                    await refresh_session_metadata(session_id)
                _emit_chat_request_metric(
                    request_id=request_id,
                    session_id=session_id,
                    route="/chat/stream",
                    status="success",
                    response_backend=response_backend,
                    requested_search_type=requested_search_type,
                    effective_search_type=effective_search_type,
                    use_web_search=bool(deps.use_web_search) if deps is not None else False,
                    use_react=use_react,
                    compression_used=compression_used,
                    tools_used=tools_used,
                    sources=sources,
                    response_text=full_response,
                )
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
                        _emit_chat_request_metric(
                            request_id=request_id,
                            session_id=session_id,
                            route="/chat/stream",
                            status="cancelled",
                            response_backend=response_backend,
                            requested_search_type=requested_search_type,
                            effective_search_type=effective_search_type,
                            use_web_search=bool(deps.use_web_search) if deps is not None else False,
                            use_react=use_react,
                            compression_used=compression_used,
                            tools_used=tools_used,
                            sources=sources,
                            response_text=full_response,
                        )
                        yield sse_event("cancelled", run_id=run_id, message="已停止生成")
                        yield sse_event("end")
                        return
                except Exception:
                    logger.exception("Failed to persist cancelled stream partial response")
                raise
            except Exception as e:
                logger.exception("Stream error: %s", e)
                _emit_chat_request_metric(
                    request_id=request_id,
                    session_id=session_id,
                    route="/chat/stream",
                    status="error",
                    response_backend=response_backend,
                    requested_search_type=requested_search_type,
                    effective_search_type=effective_search_type,
                    use_web_search=bool(deps.use_web_search) if deps is not None else False,
                    use_react=use_react,
                    compression_used=compression_used,
                    tools_used=tools_used,
                    sources=sources,
                    response_text=full_response,
                )
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


@app.get("/paper-graph", response_model=PaperGraphResponse)
async def get_paper_graph_endpoint():
    try:
        await ensure_paper_graph()
        await schedule_pending_graph_localizations()
        return await get_paper_graph()
    except Exception as exc:
        logger.error("Paper graph retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail="Paper graph is temporarily unavailable")


@app.get("/artifacts/{artifact_id}")
async def get_artifact_endpoint(artifact_id: str):
    try:
        artifact = await get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return artifact
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Artifact retrieval failed for %s: %s", artifact_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artifacts/{artifact_id}/image")
async def get_artifact_image_endpoint(artifact_id: str):
    try:
        image = await get_artifact_image(artifact_id)
        if not image:
            raise HTTPException(status_code=404, detail="Artifact image not found")
        return Response(content=image["content"], media_type=image["media_type"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Artifact image retrieval failed for %s: %s", artifact_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}/pdf")
async def get_document_pdf_endpoint(document_id: str):
    pdf = await get_document_pdf(document_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF source is unavailable for this document")
    return Response(content=pdf["content"], media_type=pdf["media_type"])


@app.get("/documents/{document_id}/pdf/pages/{page_number}/image")
async def get_document_pdf_page_image_endpoint(document_id: str, page_number: int):
    """Serve a stable raster page while the browser keeps PDF.js text selection."""
    pdf = await get_document_pdf(document_id)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF source is unavailable for this document")
    try:
        async with PDF_PAGE_RENDER_SEMAPHORE:
            image = await asyncio.to_thread(
                render_cached_pdf_page_png,
                pdf["content"],
                document_id,
                page_number,
            )
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF page rasterization failed for %s page %s", document_id, page_number)
        raise HTTPException(status_code=500, detail="PDF page rendering failed") from exc
    return Response(content=image, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.post("/documents/{document_id}/translations/{target_language}")
async def translate_document_endpoint(document_id: str, target_language: str):
    try:
        return await translate_document(document_id, target_language)
    except LookupError:
        raise HTTPException(status_code=404, detail="Document not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Document translation failed for %s", document_id)
        if is_translation_service_unavailable(exc):
            raise HTTPException(status_code=503, detail="翻译模型服务暂时不可用，请稍后重试。") from exc
        raise HTTPException(status_code=500, detail="翻译处理失败") from exc


@app.post("/documents/{document_id}/selection-translations/{target_language}")
async def translate_selection_endpoint(document_id: str, target_language: str, payload: Dict[str, Any]):
    try:
        return await translate_selection(
            document_id,
            target_language,
            str(payload.get("selection") or ""),
            str(payload.get("context_before") or ""),
            str(payload.get("context_after") or ""),
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="Document not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Selection translation failed for %s", document_id)
        if is_translation_service_unavailable(exc):
            raise HTTPException(status_code=503, detail="翻译模型服务暂时不可用，请稍后重试。") from exc
        raise HTTPException(status_code=500, detail="翻译处理失败") from exc


@app.post("/documents/{document_id}/translations/{target_language}/stream")
async def stream_document_translation_endpoint(document_id: str, target_language: str):
    async def events():
        try:
            async for event in stream_document_translation(document_id, target_language):
                event_type = str(event.pop("type"))
                yield sse_event(event_type, **event)
            yield sse_event("end")
        except LookupError:
            yield sse_event("error", content="Document not found")
            yield sse_event("end")
        except ValueError as exc:
            yield sse_event("error", content=str(exc))
            yield sse_event("end")
        except Exception as exc:
            logger.error("Document translation stream failed for %s: %s", document_id, exc)
            yield sse_event("error", content=str(exc))
            yield sse_event("end")

    return stream_response(events())


@app.get("/documents/{document_id}/annotations")
async def list_document_annotations_endpoint(document_id: str):
    return {"annotations": await list_document_annotations(document_id)}


@app.post("/documents/{document_id}/annotations")
async def create_document_annotation_endpoint(document_id: str, payload: Dict[str, Any]):
    if not str(payload.get("note") or "").strip():
        raise HTTPException(status_code=400, detail="note is required")
    return await create_document_annotation(document_id, payload)


@app.patch("/documents/{document_id}/annotations/{annotation_id}")
async def update_document_annotation_endpoint(document_id: str, annotation_id: str, payload: Dict[str, Any]):
    try:
        updated = await update_document_annotation_position(document_id, annotation_id, payload)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="page_x and page_y must be numbers between 0 and 1")
    if not updated:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return updated


@app.delete("/documents/{document_id}/annotations/{annotation_id}")
async def delete_document_annotation_endpoint(document_id: str, annotation_id: str):
    if not await delete_document_annotation(document_id, annotation_id):
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"status": "deleted", "annotation_id": annotation_id}


@app.delete("/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    if not await delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "document_id": document_id}


@app.post("/documents/{document_id}/upgrade-full")
async def upgrade_document_to_full_endpoint(document_id: str):
    """Supplement a fast ingestion with the complete evidence extraction pipeline."""
    from ingestion.ingest import DocumentIngestionPipeline
    from .models import IngestionConfig

    pipeline = DocumentIngestionPipeline(IngestionConfig(), include_images=True, include_tables=True)
    try:
        await pipeline.initialize()
        result = await pipeline.upgrade_document_to_full(document_id)
        return {"status": "succeeded", **result.model_dump()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        await pipeline.close()


@app.get("/sessions/{session_id}/memory", response_model=SessionMemorySnapshot)
async def get_session_memory_endpoint(session_id: str):
    try:
        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        memory_snapshot = await get_session_memory_snapshot(session_id)
        messages = await get_session_messages(session_id)
        snapshot = build_session_memory_snapshot(
            session_id=session_id,
            memory_snapshot=memory_snapshot,
            messages=messages,
        )
        return SessionMemorySnapshot(**snapshot)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session memory snapshot failed: {e}")
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


@app.get("/ingestion/tasks", response_model=List[IngestionTaskResponse])
async def list_ingestion_tasks_endpoint(limit: int = 100):
    return [IngestionTaskResponse(**task) for task in await list_ingestion_tasks(limit)]


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
    return await add_openalex_file_to_kb(file_url=file_url, title=title, fast=bool(payload.get("fast", False)))


@app.post("/ingestion/tasks", response_model=IngestionTaskResponse)
async def submit_ingestion_task(payload: Dict[str, Any]):
    task = await submit_async_ingestion_task(payload)
    return IngestionTaskResponse(**task)


@app.post("/ingestion/task-batches", response_model=List[IngestionTaskResponse])
async def submit_ingestion_task_batch(payload: Dict[str, Any]):
    files = payload.get("files")
    if not isinstance(files, list):
        raise HTTPException(status_code=400, detail="files must be an array")
    tasks = await submit_async_ingestion_tasks(files)
    return [IngestionTaskResponse(**task) for task in tasks]


@app.post("/ingestion/tasks/{task_id}/resume", response_model=IngestionTaskResponse)
async def resume_ingestion_task(task_id: str):
    task = await resume_ingestion_task_record(task_id)
    if not task:
        raise HTTPException(status_code=409, detail="Only a paused ingestion task can be resumed")
    try:
        await publish_ingestion_task(
            task_id=task["task_id"],
            document_id=task.get("document_id"),
            file_path=task["file_path"],
            fast=bool(task.get("fast", False)),
        )
    except Exception as exc:
        await update_ingestion_task_status(
            task_id=task_id,
            status="paused_quota",
            error_message=f"额度已恢复但任务重新投递失败：{str(exc)[:360]}",
        )
        raise HTTPException(status_code=503, detail="Failed to requeue ingestion task") from exc
    return IngestionTaskResponse(**task)


@app.post("/ingestion/tasks/{task_id}/pause", response_model=IngestionTaskResponse)
async def pause_ingestion_task_endpoint(task_id: str):
    task = await pause_ingestion_task(task_id)
    if not task:
        raise HTTPException(status_code=409, detail="Only queued or processing tasks can be paused")
    return IngestionTaskResponse(**task)


@app.delete("/ingestion/tasks/{task_id}", response_model=IngestionTaskResponse)
async def delete_ingestion_task_endpoint(task_id: str):
    task = await delete_ingestion_task(task_id)
    if not task:
        raise HTTPException(status_code=409, detail="Completed tasks belong to the library and cannot be removed here")
    return IngestionTaskResponse(**task)


@app.put("/ingestion/tasks/order", response_model=List[IngestionTaskResponse])
async def reorder_ingestion_task_queue(payload: Dict[str, Any]):
    task_ids = payload.get("task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(task_id, str) for task_id in task_ids):
        raise HTTPException(status_code=400, detail="task_ids must be an array of task IDs")
    return [IngestionTaskResponse(**task) for task in await reorder_ingestion_tasks(task_ids)]


if __name__ == "__main__":
    uvicorn.run(
        "agent.api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
        log_level=LOG_LEVEL.lower(),
    )

