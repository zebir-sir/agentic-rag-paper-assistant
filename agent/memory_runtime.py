from __future__ import annotations

import re
from typing import Any, Dict, List

from .memory_utils import (
    RECENT_MESSAGE_COUNT,
    estimate_tokens,
    sanitize_history_messages,
)


_SUMMARY_SECTION_PATTERNS = {
    "current_topic": r"(?:^|\n)\s*(?:1[\)\.、:\s-]*)?当前讨论对象[：:]\s*(.*?)(?=\n\s*(?:2[\)\.、:\s-]*)?用户约束[：:]|\Z)",
    "user_constraints": r"(?:^|\n)\s*(?:2[\)\.、:\s-]*)?用户约束[：:]\s*(.*?)(?=\n\s*(?:3[\)\.、:\s-]*)?已确认信息[：:]|\Z)",
    "confirmed_facts": r"(?:^|\n)\s*(?:3[\)\.、:\s-]*)?已确认信息[：:]\s*(.*?)(?=\n\s*(?:4[\)\.、:\s-]*)?用户关注点[：:]|\Z)",
    "user_focus": r"(?:^|\n)\s*(?:4[\)\.、:\s-]*)?用户关注点[：:]\s*(.*?)(?=\n\s*(?:5[\)\.、:\s-]*)?待继续问题[：:]|\Z)",
    "pending_questions": r"(?:^|\n)\s*(?:5[\)\.、:\s-]*)?待继续问题[：:]\s*(.*?)(?=\n\s*(?:6[\)\.、:\s-]*)?(?:不确定或缺失信息|不确定信息|缺失信息)[：:]|\Z)",
    "unknowns": r"(?:^|\n)\s*(?:6[\)\.、:\s-]*)?(?:不确定或缺失信息|不确定信息|缺失信息)[：:]\s*(.*?)(?=\Z)",
}


def _clean_section_text(value: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text


def parse_memory_summary_sections(summary: str) -> Dict[str, str]:
    text = str(summary or "").strip()
    if not text:
        return {
            "current_topic": "",
            "user_constraints": "",
            "confirmed_facts": "",
            "user_focus": "",
            "pending_questions": "",
            "unknowns": "",
        }

    parsed: Dict[str, str] = {}
    for key, pattern in _SUMMARY_SECTION_PATTERNS.items():
        match = re.search(pattern, text, flags=re.DOTALL)
        parsed[key] = _clean_section_text(match.group(1)) if match else ""
    return parsed


def build_session_memory_snapshot(
    *,
    session_id: str,
    memory_metadata: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    latest_summary = str(memory_metadata.get("latest_summary") or "")
    sanitized_history = sanitize_history_messages(messages)
    recent_messages = sanitized_history[-RECENT_MESSAGE_COUNT:]
    recent_preview = [
        {
            "role": message.get("role", ""),
            "content": str(message.get("content") or "").strip()[:160],
        }
        for message in recent_messages
    ]

    full_history_text = "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}" for message in sanitized_history
    )
    summary_context_parts = []
    if latest_summary.strip():
        summary_context_parts.append(latest_summary.strip())
    if recent_messages:
        summary_context_parts.append(
            "\n".join(
                f"{message.get('role', '')}: {message.get('content', '')}" for message in recent_messages
            )
        )
    summary_context_text = "\n\n".join(summary_context_parts).strip()

    return {
        "session_id": session_id,
        "latest_summary": latest_summary,
        "compression_count": int(memory_metadata.get("compression_count") or 0),
        "compacted_message_count": int(memory_metadata.get("compacted_message_count") or 0),
        "summary_updated_at": memory_metadata.get("summary_updated_at"),
        "history_message_count": len(messages or []),
        "sanitized_history_count": len(sanitized_history),
        "recent_message_count": len(recent_messages),
        "using_summary_context": bool(latest_summary.strip()),
        "full_history_estimated_tokens": estimate_tokens(full_history_text),
        "summary_context_estimated_tokens": estimate_tokens(summary_context_text),
        "summary_sections": parse_memory_summary_sections(latest_summary),
        "recent_messages_preview": recent_preview,
    }
