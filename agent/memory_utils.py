from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

TOKEN_LIMIT = 16000
RECENT_MESSAGE_COUNT = 6
PLANNER_DEBUG_METADATA_KEYS = {
    "intent_plan",
    "planned_retrieval_steps",
    "tools_planned",
    "tools_executed",
    "filtered_unavailable_tools",
    "planner_capabilities",
    "available_tools",
    "raw_model_content_preview",
    "fallback_reason",
    "fallback_decision",
    "retrieval_skipped_by_planner",
}
_DEBUG_CONTENT_MARKERS = {
    "intent_plan",
    "planned_retrieval_steps",
    "tools_planned",
    "tools_executed",
    "filtered_unavailable_tools",
    "planner_capabilities",
    "available_tools",
    "raw_model_content_preview",
    "fallback_reason",
    "fallback_decision",
    "retrieval_skipped_by_planner",
    "tool_call_id",
    "chunk_id",
    "document_title",
    "document_source",
}
_DEBUG_TOOL_MARKERS = {
    "search_knowledge_base",
    "hybrid_search",
    "artifact_search",
    "vector_search",
    "section_search",
    "openalex_search",
    "web_search",
}


@dataclass
class MemoryState:
    latest_summary: str = ""
    compression_count: int = 0
    compacted_message_count: int = 0


@dataclass
class ContextBuildResult:
    full_prompt: str
    compression_used: bool
    summary_updated: bool
    latest_summary: str
    compression_count: int
    compacted_message_count: int


def estimate_tokens(text: str) -> int:
    """Use a stable approximation to estimate token count without extra deps."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _looks_like_debug_payload(text: str) -> bool:
    content = str(text or "").strip()
    if not content:
        return False
    lowered = content.lower()
    if any(marker in lowered for marker in _DEBUG_CONTENT_MARKERS):
        return True
    json_like = lowered.startswith("{") or lowered.startswith("[") or "\n{" in lowered or "\n[" in lowered
    if json_like and any(marker in lowered for marker in _DEBUG_TOOL_MARKERS):
        return True
    return False


def sanitize_message_for_context(message: Dict[str, Any]) -> Optional[Dict[str, str]]:
    role = str(message.get("role") or "").strip().lower()
    if role not in {"user", "assistant"}:
        return None

    content = str(message.get("content") or "").strip()
    if not content:
        return None
    if _looks_like_debug_payload(content):
        return None
    return {"role": role, "content": content}


def sanitize_history_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sanitized: List[Dict[str, str]] = []
    for message in messages or []:
        cleaned = sanitize_message_for_context(message)
        if cleaned is not None:
            sanitized.append(cleaned)
    return sanitized


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    sanitized = sanitize_history_messages(messages)
    return "\n".join(
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in sanitized
    )


def _build_full_history_prompt(
    history_messages: List[Dict[str, str]],
    current_question: str
) -> str:
    sanitized_history = sanitize_history_messages(history_messages)
    if not sanitized_history:
        return current_question
    context_str = _messages_to_text(sanitized_history)
    return f"Previous conversation:\n{context_str}\n\nCurrent question: {current_question}"


def _build_summary_prompt(
    latest_summary: str,
    recent_messages: List[Dict[str, str]],
    current_question: str
) -> str:
    sanitized_recent = sanitize_history_messages(recent_messages)
    parts: List[str] = []
    if latest_summary.strip():
        parts.append(f"Conversation summary:\n{latest_summary.strip()}")
    if sanitized_recent:
        parts.append(f"Recent conversation:\n{_messages_to_text(sanitized_recent)}")
    parts.append(f"Current question: {current_question}")
    return "\n\n".join(parts)


def _estimate_context_tokens(
    system_prompt: str,
    history_messages: List[Dict[str, str]],
    current_question: str,
    latest_summary: str = "",
    use_summary_context: bool = False
) -> int:
    if use_summary_context:
        recent_messages = history_messages[-RECENT_MESSAGE_COUNT:]
        body = _build_summary_prompt(latest_summary, recent_messages, current_question)
    else:
        body = _build_full_history_prompt(history_messages, current_question)
    return estimate_tokens(system_prompt) + estimate_tokens(body)


def should_trigger_compression(
    system_prompt: str,
    history_messages: List[Dict[str, str]],
    current_question: str,
    latest_summary: str = "",
    token_limit: int = TOKEN_LIMIT
) -> bool:
    sanitized_history = sanitize_history_messages(history_messages)
    use_summary_context = bool(latest_summary.strip())
    total_tokens = _estimate_context_tokens(
        system_prompt=system_prompt,
        history_messages=sanitized_history,
        current_question=current_question,
        latest_summary=latest_summary,
        use_summary_context=use_summary_context,
    )
    return total_tokens > token_limit


def build_summary_update_prompt(
    old_summary: str,
    messages_to_compact: List[Dict[str, str]]
) -> str:
    compact_text = _messages_to_text(messages_to_compact)
    common_instruction = (
        "你在执行内部会话记忆压缩任务。不要调用任何工具，不要检索文档，只根据我提供的内容生成更新后的滚动摘要。\n"
        "请输出简体中文，短而高信息密度，建议 220~320 字。\n"
        "必须按以下小节输出（每节 1~2 句）：\n"
        "1) 当前讨论对象：论文标题/文档名/用户指定对象；没有则写“未明确”。\n"
        "2) 用户约束：章节范围、禁止范围、来源限制（如只基于本地知识库/不要联网/只用 OpenAlex）、回答风格要求。\n"
        "3) 已确认信息：仅记录对话中已明确出现的信息，不新增事实。\n"
        "4) 用户关注点：研究关注方向、后续分析目标、偏好的比较维度。\n"
        "5) 待继续问题：用户可能继续追问的关键问题。\n"
        "6) 不确定或缺失信息：证据不足/未检索到/尚未确认的点。\n\n"
        "合并规则：\n"
        "- 若已有旧摘要，保留仍然有效的用户约束与上下文，不要被新消息无故覆盖。\n"
        "- 若新消息明确改变讨论对象或约束，以新消息为准。\n\n"
        "严格禁止：\n"
        "- 不要编造论文结论、实验数字、作者、年份、DOI。\n"
        "- 不要把任何 debug payload、tool metadata、raw_model_content_preview、tools_executed、intent_plan、planner/runtime 调试字段写入摘要。\n"
    )
    if old_summary.strip():
        return (
            f"{common_instruction}\n"
            f"旧摘要：\n{old_summary.strip()}\n\n"
            f"新增待压缩历史消息：\n{compact_text}\n\n"
            "请返回更新后的 latest_summary："
        )
    return (
        f"{common_instruction}\n"
        f"待压缩历史消息：\n{compact_text}\n\n"
        "请返回 latest_summary："
    )


def get_messages_for_next_compaction(
    history_messages: List[Dict[str, str]],
    compacted_message_count: int
) -> List[Dict[str, str]]:
    sanitized_history = sanitize_history_messages(history_messages)
    compactable_end = max(0, len(sanitized_history) - RECENT_MESSAGE_COUNT)
    start = max(0, compacted_message_count)
    if start >= compactable_end:
        return []
    return sanitized_history[start:compactable_end]


def build_context_without_compaction(
    history_messages: List[Dict[str, str]],
    current_question: str,
    memory_state: MemoryState
) -> ContextBuildResult:
    sanitized_history = sanitize_history_messages(history_messages)
    if memory_state.latest_summary.strip():
        recent_messages = sanitized_history[-RECENT_MESSAGE_COUNT:]
        prompt = _build_summary_prompt(memory_state.latest_summary, recent_messages, current_question)
        return ContextBuildResult(
            full_prompt=prompt,
            compression_used=True,
            summary_updated=False,
            latest_summary=memory_state.latest_summary,
            compression_count=memory_state.compression_count,
            compacted_message_count=memory_state.compacted_message_count,
        )

    return ContextBuildResult(
        full_prompt=_build_full_history_prompt(sanitized_history, current_question),
        compression_used=False,
        summary_updated=False,
        latest_summary=memory_state.latest_summary,
        compression_count=memory_state.compression_count,
        compacted_message_count=memory_state.compacted_message_count,
    )


def normalize_memory_state(metadata: Optional[Dict[str, Any]]) -> MemoryState:
    data = metadata or {}
    return MemoryState(
        latest_summary=str(data.get("latest_summary") or ""),
        compression_count=int(data.get("compression_count") or 0),
        compacted_message_count=int(data.get("compacted_message_count") or 0),
    )
