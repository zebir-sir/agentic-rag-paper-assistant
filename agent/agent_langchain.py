import os
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .agent_runtime import AgentDependencies
from .langchain_tools import build_langchain_tools
from .models import EvidenceSource, ToolCall
from .prompts import SYSTEM_PROMPT
from .routing import has_unverified_web_citations, is_degenerate_answer

logger = logging.getLogger(__name__)


@dataclass
class LangChainAgentResult:
    message: str
    raw_result: Any
    tools_used: List[ToolCall]
    sources: List[EvidenceSource]


@dataclass
class LangChainStreamResult:
    message: str
    chunks: List[str]
    raw_events: List[Any]
    tools_used: List[ToolCall]
    sources: List[EvidenceSource]


GENERATION_RETRY_FAILED_MESSAGE = "这次回答生成出现异常。请重试，或尝试指定论文标题/文件名后再问。"


_DIRTY_TEXT_MARKERS = (
    "AIMessage(",
    "HumanMessage(",
    "ToolMessage(",
    "{'messages':",
    '{"messages":',
    "{'model':",
    '{"model":',
    "additional_kwargs=",
    "response_metadata=",
    "usage_metadata=",
)


def get_langchain_chat_model() -> ChatOpenAI:
    request_timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
    return ChatOpenAI(
        model=os.getenv("LLM_CHOICE", "Qwen/Qwen2.5-7B-Instruct"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
        timeout=request_timeout_seconds,
        request_timeout=request_timeout_seconds,
    )


def build_langchain_agent(deps: AgentDependencies):
    model = get_langchain_chat_model()
    tools = build_langchain_tools(deps)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


def _clone_deps_for_regeneration(deps: AgentDependencies) -> AgentDependencies:
    return AgentDependencies(
        session_id=deps.session_id,
        user_id=deps.user_id,
        use_web_search=deps.use_web_search,
        search_preferences=dict(deps.search_preferences or {}),
    )


def _build_regeneration_prompt(full_prompt: str) -> str:
    repair_instruction = (
        "刚才的回答出现重复或格式异常。请重新回答用户问题。要求："
        "直接回答问题；不要重复 token；不要输出乱码；"
        "算法名如 `RRT*` 用反引号保护；"
        "如果原问题是本地论文/知识库问题，请优先使用已有本地检索上下文和本地知识库工具；"
        "不要要求联网搜索作为默认手段；"
        "如果缺少某篇论文证据，只说明“当前检索片段不足”；"
        "不要输出伪造链接、工具名、tool call 占位符或 `search_web(...)` 这类字符串。"
    )
    return f"{full_prompt}\n\n[生成修复指令]\n{repair_instruction}"


def _message_content_to_text(content: Any) -> str:
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
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").lower() == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
                continue
            text = str(item.get("text") or item.get("content") or "").strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def extract_langchain_final_text(raw_result: Any) -> str:
    try:
        if isinstance(raw_result, dict):
            messages = raw_result.get("messages")
            if isinstance(messages, list) and messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if content is None and isinstance(last, dict):
                    content = last.get("content")
                text = _message_content_to_text(content)
                if text:
                    return text
        content = getattr(raw_result, "content", None)
        if content is not None:
            text = _message_content_to_text(content)
            if text:
                return text
        return str(raw_result)
    except Exception:
        return str(raw_result)


def _normalize_tool_call(candidate: Any) -> ToolCall | None:
    try:
        if isinstance(candidate, dict):
            name = candidate.get("name") or candidate.get("tool_name")
            args = candidate.get("args") or {}
            call_id = candidate.get("id") or candidate.get("tool_call_id")
        else:
            name = getattr(candidate, "name", None) or getattr(candidate, "tool_name", None)
            args = getattr(candidate, "args", None) or {}
            call_id = getattr(candidate, "id", None) or getattr(candidate, "tool_call_id", None)
        if not name:
            return None
        if not isinstance(args, dict):
            args = {}
        return ToolCall(
            tool_name=str(name),
            args=args,
            tool_call_id=str(call_id) if call_id is not None else None,
        )
    except Exception:
        return None


def extract_langchain_tool_calls(raw_result: Any) -> List[ToolCall]:
    tool_calls: List[ToolCall] = []
    try:
        messages = []
        if isinstance(raw_result, dict):
            maybe_messages = raw_result.get("messages")
            if isinstance(maybe_messages, list):
                messages = maybe_messages
        elif isinstance(raw_result, list):
            messages = raw_result

        for message in messages:
            calls = getattr(message, "tool_calls", None)
            if calls is None and isinstance(message, dict):
                calls = message.get("tool_calls")

            if isinstance(calls, list):
                for call in calls:
                    normalized = _normalize_tool_call(call)
                    if normalized is not None:
                        tool_calls.append(normalized)

            additional_kwargs = getattr(message, "additional_kwargs", None)
            if additional_kwargs is None and isinstance(message, dict):
                additional_kwargs = message.get("additional_kwargs")
            if isinstance(additional_kwargs, dict):
                legacy_calls = additional_kwargs.get("tool_calls")
                if isinstance(legacy_calls, list):
                    for call in legacy_calls:
                        function_payload = call.get("function") if isinstance(call, dict) else None
                        candidate = function_payload if function_payload else call
                        normalized = _normalize_tool_call(candidate)
                        if normalized is not None:
                            tool_calls.append(normalized)
    except Exception:
        return tool_calls
    return tool_calls


async def _run_langchain_agent_once(
    full_prompt: str,
    deps: AgentDependencies,
) -> LangChainAgentResult:
    agent = build_langchain_agent(deps)
    raw_result = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": full_prompt},
            ]
        }
    )
    message = extract_langchain_final_text(raw_result)
    tools_used = extract_langchain_tool_calls(raw_result)
    sources = list(deps.retrieved_sources)

    return LangChainAgentResult(
        message=message,
        raw_result=raw_result,
        tools_used=tools_used,
        sources=sources,
    )


async def run_langchain_agent(
    full_prompt: str,
    deps: AgentDependencies,
    *,
    retry_on_degenerate: bool = True,
) -> LangChainAgentResult:
    result = await _run_langchain_agent_once(full_prompt, deps)
    needs_retry = is_degenerate_answer(result.message) or (
        not result.sources and has_unverified_web_citations(result.message)
    )
    if not retry_on_degenerate or not needs_retry:
        return result

    logger.warning("Detected low-quality or unverified-citation answer in LangChain normal generation; retrying once.")
    retry_deps = _clone_deps_for_regeneration(deps)
    retry_result = await _run_langchain_agent_once(_build_regeneration_prompt(full_prompt), retry_deps)
    retry_failed = is_degenerate_answer(retry_result.message) or (
        not retry_result.sources and has_unverified_web_citations(retry_result.message)
    )
    if retry_failed:
        logger.warning("Low-quality answer persisted after LangChain retry; returning safe fallback.")
        return LangChainAgentResult(
            message=GENERATION_RETRY_FAILED_MESSAGE,
            raw_result=retry_result.raw_result,
            tools_used=[],
            sources=[],
        )
    return retry_result


async def retry_langchain_agent_after_degenerate(
    full_prompt: str,
    deps: AgentDependencies,
) -> LangChainAgentResult:
    logger.warning("Detected degenerate streamed answer; regenerating once with repair prompt.")
    retry_deps = _clone_deps_for_regeneration(deps)
    retry_result = await _run_langchain_agent_once(_build_regeneration_prompt(full_prompt), retry_deps)
    retry_failed = is_degenerate_answer(retry_result.message) or (
        not retry_result.sources and has_unverified_web_citations(retry_result.message)
    )
    if retry_failed:
        logger.warning("Low-quality answer persisted after streamed retry; returning safe fallback.")
        return LangChainAgentResult(
            message=GENERATION_RETRY_FAILED_MESSAGE,
            raw_result=retry_result.raw_result,
            tools_used=[],
            sources=[],
        )
    return retry_result


def _extract_text_chunks_from_event(event: Any) -> List[str]:
    chunks: List[str] = []
    if not isinstance(event, dict):
        return chunks
    if str(event.get("event") or "").strip() != "on_chat_model_stream":
        return chunks
    data = event.get("data")
    if not isinstance(data, dict):
        return chunks
    chunk_obj = data.get("chunk")
    content = getattr(chunk_obj, "content", None)

    if isinstance(content, str):
        raw = content
        cleaned = raw.strip()
        if cleaned and cleaned != "None":
            chunks.append(raw)
        return chunks

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if str(item.get("type") or "").lower() == "text":
                    raw = str(item.get("text") or "")
                    cleaned = raw.strip()
                    if cleaned and cleaned != "None":
                        chunks.append(raw)
        return chunks
    return chunks


def _is_clean_text_chunk(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text == "None":
        return False
    return not any(marker in text for marker in _DIRTY_TEXT_MARKERS)


def extract_langchain_tool_calls_from_events(raw_events: List[Any]) -> List[ToolCall]:
    tool_calls: List[ToolCall] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue

        candidates: List[Any] = []
        for key in ("output", "chunk", "message"):
            value = data.get(key)
            if value is not None:
                candidates.append(value)

        messages = data.get("messages")
        if isinstance(messages, list):
            candidates.extend(messages)

        for candidate in candidates:
            direct_calls = getattr(candidate, "tool_calls", None)
            if direct_calls is None and isinstance(candidate, dict):
                direct_calls = candidate.get("tool_calls")
            if isinstance(direct_calls, list):
                for call in direct_calls:
                    normalized = _normalize_tool_call(call)
                    if normalized is not None:
                        tool_calls.append(normalized)

            additional_kwargs = getattr(candidate, "additional_kwargs", None)
            if additional_kwargs is None and isinstance(candidate, dict):
                additional_kwargs = candidate.get("additional_kwargs")
            if isinstance(additional_kwargs, dict):
                legacy_calls = additional_kwargs.get("tool_calls")
                if isinstance(legacy_calls, list):
                    for call in legacy_calls:
                        function_payload = call.get("function") if isinstance(call, dict) else None
                        normalized = _normalize_tool_call(function_payload if function_payload else call)
                        if normalized is not None:
                            tool_calls.append(normalized)

    return tool_calls


async def iter_langchain_agent_stream(
    full_prompt: str,
    deps: AgentDependencies,
) -> AsyncIterator[Dict[str, Any]]:
    agent = build_langchain_agent(deps)
    raw_events: List[Any] = []
    text_chunks: List[str] = []

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": full_prompt}]},
        version="v2",
    ):
        raw_events.append(event)
        for chunk in _extract_text_chunks_from_event(event):
            if _is_clean_text_chunk(chunk):
                text_chunks.append(chunk)
                yield {"type": "text", "content": chunk}

    final_text = "".join(text_chunks).strip()
    tools_used = extract_langchain_tool_calls_from_events(raw_events)
    yield {
        "type": "final",
        "message": final_text,
        "tools_used": tools_used,
        "sources": list(deps.retrieved_sources),
        "raw_events": raw_events,
    }


async def stream_langchain_agent(
    full_prompt: str,
    deps: AgentDependencies,
) -> LangChainStreamResult:
    agent = build_langchain_agent(deps)
    chunks: List[str] = []
    raw_events: List[Any] = []

    try:
        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": full_prompt}]},
            version="v2",
        ):
            raw_events.append(event)
            for chunk in _extract_text_chunks_from_event(event):
                if _is_clean_text_chunk(chunk):
                    chunks.append(chunk)
    except Exception:
        fallback_result = await run_langchain_agent(full_prompt, deps)
        fallback_message = fallback_result.message.strip()
        fallback_chunks = [fallback_message] if _is_clean_text_chunk(fallback_message) else []
        return LangChainStreamResult(
            message=fallback_message,
            chunks=fallback_chunks,
            raw_events=raw_events,
            tools_used=fallback_result.tools_used,
            sources=fallback_result.sources,
        )

    if chunks:
        message = "".join(chunks).strip()
        tools_used = extract_langchain_tool_calls_from_events(raw_events)
    else:
        fallback_result = await run_langchain_agent(full_prompt, deps)
        message = fallback_result.message.strip()
        chunks = [message] if _is_clean_text_chunk(message) else []
        tools_used = fallback_result.tools_used
        sources = fallback_result.sources
        return LangChainStreamResult(
            message=message,
            chunks=chunks,
            raw_events=raw_events,
            tools_used=tools_used,
            sources=sources,
        )
    sources = list(deps.retrieved_sources)

    return LangChainStreamResult(
        message=message,
        chunks=chunks,
        raw_events=raw_events,
        tools_used=tools_used,
        sources=sources,
    )
