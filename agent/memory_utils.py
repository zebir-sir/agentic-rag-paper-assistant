from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

TOKEN_LIMIT = 16000
RECENT_MESSAGE_COUNT = 6


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


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join(
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages
    )


def _build_full_history_prompt(
    history_messages: List[Dict[str, str]],
    current_question: str
) -> str:
    if not history_messages:
        return current_question
    context_str = _messages_to_text(history_messages)
    return f"Previous conversation:\n{context_str}\n\nCurrent question: {current_question}"


def _build_summary_prompt(
    latest_summary: str,
    recent_messages: List[Dict[str, str]],
    current_question: str
) -> str:
    parts: List[str] = []
    if latest_summary.strip():
        parts.append(f"Conversation summary:\n{latest_summary.strip()}")
    if recent_messages:
        parts.append(f"Recent conversation:\n{_messages_to_text(recent_messages)}")
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
    use_summary_context = bool(latest_summary.strip())
    total_tokens = _estimate_context_tokens(
        system_prompt=system_prompt,
        history_messages=history_messages,
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
    if old_summary.strip():
        return (
            "你在执行内部会话记忆压缩任务。不要调用任何工具，不要检索文档，只根据我提供的内容生成更新后的滚动摘要。\n"
            "请输出简体中文，控制在 220~320 字，尽量精炼，必须覆盖：\n"
            "1) 当前主要讨论主题；2) 已明确结论；3) 重要方法点；4) 创新点/局限性；5) 用户关注方向；6) 待继续问题。\n\n"
            f"旧摘要：\n{old_summary.strip()}\n\n"
            f"新增待压缩历史消息：\n{compact_text}\n\n"
            "请返回更新后的 latest_summary："
        )
    return (
        "你在执行内部会话记忆压缩任务。不要调用任何工具，不要检索文档，只根据我提供的内容生成滚动摘要。\n"
        "请输出简体中文，控制在 220~320 字，尽量精炼，必须覆盖：\n"
        "1) 当前主要讨论主题；2) 已明确结论；3) 重要方法点；4) 创新点/局限性；5) 用户关注方向；6) 待继续问题。\n\n"
        f"待压缩历史消息：\n{compact_text}\n\n"
        "请返回 latest_summary："
    )


def get_messages_for_next_compaction(
    history_messages: List[Dict[str, str]],
    compacted_message_count: int
) -> List[Dict[str, str]]:
    compactable_end = max(0, len(history_messages) - RECENT_MESSAGE_COUNT)
    start = max(0, compacted_message_count)
    if start >= compactable_end:
        return []
    return history_messages[start:compactable_end]


def build_context_without_compaction(
    history_messages: List[Dict[str, str]],
    current_question: str,
    memory_state: MemoryState
) -> ContextBuildResult:
    if memory_state.latest_summary.strip():
        recent_messages = history_messages[-RECENT_MESSAGE_COUNT:]
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
        full_prompt=_build_full_history_prompt(history_messages, current_question),
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
