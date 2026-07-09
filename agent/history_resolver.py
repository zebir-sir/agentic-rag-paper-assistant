from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


_RESOLUTION_CUES = (
    "这个",
    "这个方法",
    "这个模型",
    "这个算法",
    "这个结果",
    "这个实验",
    "这个图",
    "这个表",
    "这篇",
    "这篇论文",
    "它",
    "它的",
    "其",
    "其中",
    "那",
    "那个",
    "那篇",
    "那篇论文",
    "该",
    "该方法",
    "该模型",
    "该算法",
    "表",
    "图",
    "算法",
)

_QUESTION_CUES = (
    "?",
    "？",
    "为什么",
    "怎么",
    "如何",
    "区别",
    "差别",
    "意思",
    "说明",
    "代表",
    "看待",
    "理解",
    "总结",
    "展开",
    "继续",
    "细说",
    "对比",
)

_TOPICLESS_ACKS = {
    "好",
    "好的",
    "收到",
    "明白了",
    "知道了",
    "行",
    "可以",
    "ok",
    "okay",
    "thanks",
    "thank you",
}


@dataclass
class HistoryResolutionResult:
    original_query: str
    resolved_query: str
    used_history: bool
    topic_hint: str
    recent_history_summary: str
    reason: str


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _is_substantive_turn(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if normalized.lower() in _TOPICLESS_ACKS:
        return False
    return len(normalized) >= 6 or any(cue in normalized for cue in _QUESTION_CUES)


def _needs_history_resolution(query: str) -> bool:
    normalized = _normalize_text(query)
    if not normalized:
        return False
    lowered = normalized.lower()
    if lowered in _TOPICLESS_ACKS:
        return False
    if any(cue in normalized for cue in _RESOLUTION_CUES):
        return True
    if len(normalized) <= 18 and any(cue in normalized for cue in _QUESTION_CUES):
        return True
    return False


def _format_turn(role: str, content: str) -> str:
    role_name = "用户" if role == "user" else "助手"
    return f"{role_name}：{_normalize_text(content)}"


def _build_recent_history_summary(history_messages: List[Dict[str, str]], max_turns: int = 4) -> str:
    turns = [
        _format_turn(str(message.get("role") or ""), str(message.get("content") or ""))
        for message in history_messages[-max_turns:]
        if _normalize_text(str(message.get("content") or ""))
    ]
    return " | ".join(turns)


def _extract_topic_hint(history_messages: List[Dict[str, str]]) -> str:
    for message in reversed(history_messages):
        role = str(message.get("role") or "").strip().lower()
        content = _normalize_text(str(message.get("content") or ""))
        if role != "user":
            continue
        if not _is_substantive_turn(content):
            continue
        if _needs_history_resolution(content) and len(content) <= 18:
            continue
        return content

    for message in reversed(history_messages):
        role = str(message.get("role") or "").strip().lower()
        content = _normalize_text(str(message.get("content") or ""))
        if role != "assistant":
            continue
        if _is_substantive_turn(content):
            return content
    return ""


def resolve_history_query(
    *,
    latest_query: str,
    history_messages: List[Dict[str, str]],
) -> HistoryResolutionResult:
    normalized_query = _normalize_text(latest_query)
    history = list(history_messages or [])
    recent_history_summary = _build_recent_history_summary(history)

    if not normalized_query:
        return HistoryResolutionResult(
            original_query="",
            resolved_query="",
            used_history=False,
            topic_hint="",
            recent_history_summary=recent_history_summary,
            reason="empty_query",
        )

    if not history:
        return HistoryResolutionResult(
            original_query=normalized_query,
            resolved_query=normalized_query,
            used_history=False,
            topic_hint="",
            recent_history_summary="",
            reason="no_history",
        )

    if not _needs_history_resolution(normalized_query):
        return HistoryResolutionResult(
            original_query=normalized_query,
            resolved_query=normalized_query,
            used_history=False,
            topic_hint="",
            recent_history_summary=recent_history_summary,
            reason="query_is_self_contained",
        )

    topic_hint = _extract_topic_hint(history)
    if not topic_hint:
        return HistoryResolutionResult(
            original_query=normalized_query,
            resolved_query=normalized_query,
            used_history=False,
            topic_hint="",
            recent_history_summary=recent_history_summary,
            reason="history_topic_not_found",
        )

    resolved_query = f"{topic_hint}；当前追问：{normalized_query}"
    return HistoryResolutionResult(
        original_query=normalized_query,
        resolved_query=resolved_query,
        used_history=True,
        topic_hint=topic_hint,
        recent_history_summary=recent_history_summary,
        reason="history_topic_attached",
    )
