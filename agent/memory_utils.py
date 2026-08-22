"""Scoped, structured conversation memory built only from approved final turns."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


MEMORY_CONTEXT_BUDGET_TOKENS = int(os.getenv("MEMORY_CONTEXT_BUDGET_TOKENS", "28000"))
MEMORY_COMPACTION_TRIGGER_TOKENS = int(os.getenv("MEMORY_COMPACTION_TRIGGER_TOKENS", "30000"))
RECENT_MESSAGE_COUNT = int(os.getenv("MEMORY_RECENT_TURNS", "8"))
MEMORY_UPDATE_TURN_INTERVAL = int(os.getenv("MEMORY_UPDATE_TURN_INTERVAL", "8"))
MEMORY_SUMMARY_KEYS = (
    "goal",
    "constraints",
    "confirmed_decisions",
    "open_questions",
    "uncertainties",
)


@dataclass
class SessionMemoryState:
    version: int = 0
    covered_message_count: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBuildResult:
    full_prompt: str
    compression_used: bool
    memory_updated: bool
    memory_state: SessionMemoryState


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def empty_memory_summary() -> Dict[str, Any]:
    return {
        "goal": "",
        "constraints": [],
        "confirmed_decisions": [],
        "open_questions": [],
        "uncertainties": [],
    }


def normalize_memory_summary(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    result = empty_memory_summary()
    result["goal"] = str(source.get("goal") or "").strip()[:600]
    for key in ("constraints", "confirmed_decisions", "open_questions", "uncertainties"):
        raw_items = source.get(key, [])
        if not isinstance(raw_items, list):
            raw_items = []
        result[key] = [" ".join(str(item).split())[:360] for item in raw_items if str(item).strip()][:12]
    return result


def normalize_memory_state(snapshot: Optional[Dict[str, Any]]) -> SessionMemoryState:
    value = snapshot or {}
    return SessionMemoryState(
        version=max(0, int(value.get("version") or 0)),
        covered_message_count=max(0, int(value.get("covered_message_count") or 0)),
        summary=normalize_memory_summary(value.get("summary")),
    )


def is_memory_eligible_message(message: Dict[str, Any]) -> bool:
    role = str(message.get("role") or "").strip().lower()
    content = str(message.get("content") or "").strip()
    metadata = message.get("metadata") or {}
    return role in {"user", "assistant"} and bool(metadata.get("memory_eligible")) and bool(content)


def memory_eligible_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    eligible: List[Dict[str, Any]] = []
    for message in messages or []:
        if not is_memory_eligible_message(message):
            continue
        item: Dict[str, Any] = {
            "role": str(message["role"]),
            "content": str(message["content"]).strip(),
        }
        eligible.append(item)
    return eligible


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in messages:
        lines.append(f"{item['role']}: {item['content']}")
    return "\n".join(lines)


def build_memory_update_prompt(
    previous_summary: Dict[str, Any],
    turns: List[Dict[str, str]],
) -> str:
    return (
        "Update a structured research conversation memory. Return JSON only. "
        "Schema: {goal:string, constraints:string[], "
        "confirmed_decisions:string[], open_questions:string[], uncertainties:string[]}. "
        "Only keep durable conversation coordination: user goal, "
        "explicit constraints, user-confirmed decisions, follow-up questions, and uncertainty boundaries. "
        "Never store paper claims, numerical results, citations, source snippets, retrieval results, "
        "tool calls, model reasoning, debug data, errors, or assistant guesses as memory. "
        "When a new explicit constraint replaces an old one, replace it. Omit empty or stale items.\n\n"
        f"Previous structured memory:\n{json.dumps(normalize_memory_summary(previous_summary), ensure_ascii=False)}\n\n"
        f"Eligible completed turns:\n{_messages_to_text(turns)}"
    )


def should_update_memory(
    history_messages: List[Dict[str, str]],
    memory_state: SessionMemoryState,
    current_question: str,
) -> bool:
    unprocessed = len(history_messages) - memory_state.covered_message_count
    projected = _messages_to_text(history_messages + [{"role": "user", "content": current_question}])
    return unprocessed >= MEMORY_UPDATE_TURN_INTERVAL or estimate_tokens(projected) >= MEMORY_COMPACTION_TRIGGER_TOKENS


def messages_for_memory_update(
    history_messages: List[Dict[str, str]],
    covered_message_count: int,
) -> List[Dict[str, str]]:
    return history_messages[max(0, covered_message_count) :]


def build_context(
    history_messages: List[Dict[str, str]],
    current_question: str,
    memory_state: SessionMemoryState,
) -> ContextBuildResult:
    recent = history_messages[-RECENT_MESSAGE_COUNT:]
    blocks: List[str] = []
    use_snapshot = bool(memory_state.version and any(memory_state.summary.values()))
    if use_snapshot:
        blocks.append(f"Structured conversation memory:\n{json.dumps(memory_state.summary, ensure_ascii=False)}")
    if recent:
        blocks.append(f"Recent eligible conversation:\n{_messages_to_text(recent)}")
    blocks.append(f"Current question: {current_question}")
    prompt = "\n\n".join(blocks)
    if estimate_tokens(prompt) > MEMORY_CONTEXT_BUDGET_TOKENS:
        bounded_recent: List[Dict[str, Any]] = []
        for item in reversed(recent):
            candidate = [item] + bounded_recent
            candidate_blocks = []
            if use_snapshot:
                candidate_blocks.append(
                    f"Structured conversation memory:\n{json.dumps(memory_state.summary, ensure_ascii=False)}"
                )
            candidate_blocks.append(f"Recent eligible conversation:\n{_messages_to_text(candidate)}")
            candidate_blocks.append(f"Current question: {current_question}")
            if estimate_tokens("\n\n".join(candidate_blocks)) > MEMORY_CONTEXT_BUDGET_TOKENS:
                break
            bounded_recent = candidate
        blocks = []
        if use_snapshot:
            blocks.append(f"Structured conversation memory:\n{json.dumps(memory_state.summary, ensure_ascii=False)}")
        if bounded_recent:
            blocks.append(f"Recent eligible conversation:\n{_messages_to_text(bounded_recent)}")
        blocks.append(f"Current question: {current_question}")
        prompt = "\n\n".join(blocks)
    return ContextBuildResult(
        full_prompt=prompt,
        compression_used=use_snapshot,
        memory_updated=False,
        memory_state=memory_state,
    )
