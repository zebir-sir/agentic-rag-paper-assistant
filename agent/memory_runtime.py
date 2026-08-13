"""Read-model helpers for the structured database-backed session memory."""

from __future__ import annotations

from typing import Any, Dict, List

from .memory_utils import RECENT_MESSAGE_COUNT, estimate_tokens, memory_eligible_messages, normalize_memory_state


def build_session_memory_snapshot(*, session_id: str, memory_snapshot: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    state = normalize_memory_state(memory_snapshot)
    eligible = memory_eligible_messages(messages)
    recent = eligible[-RECENT_MESSAGE_COUNT:]
    context_text = "\n".join(item["content"] for item in recent)
    return {
        "session_id": session_id,
        "version": state.version,
        "covered_message_count": state.covered_message_count,
        "summary": state.summary,
        "updated_at": memory_snapshot.get("updated_at"),
        "eligible_message_count": len(eligible),
        "recent_message_count": len(recent),
        "using_structured_memory": bool(state.version),
        "recent_context_estimated_tokens": estimate_tokens(context_text),
        "recent_messages_preview": [{"role": item["role"], "content": item["content"][:160]} for item in recent],
    }
