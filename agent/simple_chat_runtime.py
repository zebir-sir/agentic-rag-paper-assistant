from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import os

from langchain_openai import ChatOpenAI

from .agent_runtime import AgentDependencies
from .models import EvidenceSource, ToolCall
from .routing import is_degenerate_answer
from .tool_payloads import (
    run_artifact_search_payload,
    run_hybrid_search_payload,
    run_section_search_payload,
)


_COMPLEXITY_CUES = (
    "对比",
    "比较",
    "区别",
    "差别",
    "创新点",
    "局限",
    "为什么",
    "如何评价",
    "详细",
    "深入",
    "全面",
    "多篇",
    "related work",
    "综述",
    "复现",
    "路线",
    "推导",
)

_SECTION_CUES = {
    "abstract": "Abstract",
    "摘要": "Abstract",
    "introduction": "Introduction",
    "引言": "Introduction",
    "method": "Method",
    "methods": "Method",
    "方法": "Method",
    "experiment": "Experiments",
    "experiments": "Experiments",
    "实验": "Experiments",
    "result": "Results",
    "results": "Results",
    "结果": "Results",
    "conclusion": "Conclusion",
    "结论": "Conclusion",
}

_ARTIFACT_CUES = {
    "table": "table",
    "表": "table",
    "figure": "figure",
    "图": "figure",
    "algorithm": "algorithm",
    "算法": "algorithm",
    "伪代码": "algorithm",
}


def get_simple_chat_model() -> ChatOpenAI:
    request_timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
    return ChatOpenAI(
        model=os.getenv("LLM_CHOICE", "Qwen/Qwen2.5-7B-Instruct"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
        timeout=request_timeout_seconds,
        request_timeout=request_timeout_seconds,
    )


@dataclass
class SimpleChatDecision:
    enabled: bool
    mode: str = ""
    query: str = ""
    section_query: str = ""
    artifact_types: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class SimpleChatResult:
    message: str
    sources: List[EvidenceSource]
    tools_used: List[ToolCall]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _extract_message_text(response: Any) -> str:
    content = getattr(response, "content", response)
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
    return str(content or "").strip()


def _detect_section_query(message: str) -> str:
    text = str(message or "").lower()
    for cue, section_name in _SECTION_CUES.items():
        if cue in text:
            return section_name
    return ""


def _detect_artifact_types(message: str) -> List[str]:
    text = str(message or "").lower()
    types: List[str] = []
    for cue, artifact_type in _ARTIFACT_CUES.items():
        if cue in text and artifact_type not in types:
            types.append(artifact_type)
    return types


def choose_simple_chat_strategy(
    *,
    message: str,
    resolved_query: str,
    is_local_question: bool,
    use_react: bool,
    use_web_search: bool,
) -> SimpleChatDecision:
    normalized_message = _normalize_text(message)
    if not is_local_question:
        return SimpleChatDecision(enabled=False, reason="not_local_question")
    if use_react:
        return SimpleChatDecision(enabled=False, reason="deep_analysis_enabled")
    if use_web_search:
        return SimpleChatDecision(enabled=False, reason="external_search_enabled")
    if not normalized_message:
        return SimpleChatDecision(enabled=False, reason="empty_message")
    if len(normalized_message) > 48:
        return SimpleChatDecision(enabled=False, reason="message_too_long")
    lowered = normalized_message.lower()
    if any(cue in lowered for cue in _COMPLEXITY_CUES):
        return SimpleChatDecision(enabled=False, reason="complex_question")

    artifact_types = _detect_artifact_types(normalized_message)
    if artifact_types:
        return SimpleChatDecision(
            enabled=True,
            mode="artifact",
            query=resolved_query or normalized_message,
            artifact_types=artifact_types,
            reason="artifact_focused_local_question",
        )

    section_query = _detect_section_query(normalized_message)
    if section_query:
        return SimpleChatDecision(
            enabled=True,
            mode="section",
            query=resolved_query or normalized_message,
            section_query=section_query,
            reason="section_focused_local_question",
        )

    return SimpleChatDecision(
        enabled=True,
        mode="hybrid",
        query=resolved_query or normalized_message,
        reason="short_local_question",
    )


def _build_evidence_block(sources: List[EvidenceSource], limit: int = 3) -> str:
    lines: List[str] = []
    for index, source in enumerate(sources[:limit], start=1):
        metadata = source.metadata or {}
        section_name = str(metadata.get("section_path_text") or metadata.get("section_title") or "").strip()
        artifact_type = str(metadata.get("artifact_type") or "").strip()
        header_parts = [f"[{index}] {source.document_title}"]
        if section_name:
            header_parts.append(section_name)
        if artifact_type:
            header_parts.append(f"artifact={artifact_type}")
        lines.append(" | ".join(header_parts))
        lines.append(str(source.snippet or "").strip())
    return "\n".join(lines).strip()


def _build_generation_prompt(
    *,
    user_message: str,
    resolved_query: str,
    response_style: str,
    evidence_block: str,
) -> str:
    style_hint = "回答保持简洁。" if response_style != "respect_constraints" else "优先尊重用户刚刚补充的范围或限制，回答保持简洁。"
    return (
        f"用户当前问题：{user_message}\n"
        f"用于检索的 resolved query：{resolved_query}\n"
        f"回答风格提示：{response_style or 'normal'}\n\n"
        f"本地证据片段：\n{evidence_block}\n\n"
        "请只基于这些片段回答，不要调用工具，不要编造未出现的实验数字、机制或外部事实。"
        "如果某个细节当前片段未明确说明，就直接说“当前检索片段未明确说明”。"
        f"{style_hint}"
    )


async def run_simple_chat_runtime(
    *,
    deps: AgentDependencies,
    user_message: str,
    decision: SimpleChatDecision,
    response_style: str = "normal",
) -> Optional[SimpleChatResult]:
    if not decision.enabled:
        return None

    if decision.mode == "artifact":
        await run_artifact_search_payload(
            deps=deps,
            query=decision.query,
            limit=4,
            artifact_types=decision.artifact_types,
        )
        tool_name = "artifact_search"
        tool_args: Dict[str, Any] = {
            "query": decision.query,
            "limit": 4,
            "artifact_types": decision.artifact_types,
        }
    elif decision.mode == "section":
        await run_section_search_payload(
            deps=deps,
            query=decision.query,
            section_query=decision.section_query,
            limit=4,
        )
        tool_name = "section_search"
        tool_args = {
            "query": decision.query,
            "section_query": decision.section_query,
            "limit": 4,
        }
    else:
        await run_hybrid_search_payload(
            deps=deps,
            query=decision.query,
            limit=4,
        )
        tool_name = "hybrid_search"
        tool_args = {"query": decision.query, "limit": 4}

    sources = list(getattr(deps, "retrieved_sources", []) or [])
    if not sources:
        return None

    evidence_block = _build_evidence_block(sources)
    if not evidence_block:
        return None

    model = get_simple_chat_model()
    response = await model.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "你是一名科研论文阅读助手。当前任务是对简单本地论文问题做轻量回答。"
                    "只使用提供的本地证据片段，不要调用工具，不要输出检索过程。"
                ),
            },
            {
                "role": "user",
                "content": _build_generation_prompt(
                    user_message=user_message,
                    resolved_query=decision.query,
                    response_style=response_style,
                    evidence_block=evidence_block,
                ),
            },
        ]
    )
    message = _extract_message_text(response)
    if not message or is_degenerate_answer(message):
        return None

    return SimpleChatResult(
        message=message,
        sources=sources,
        tools_used=[ToolCall(tool_name=tool_name, args=tool_args)],
        metadata={
            "simple_chat_used": True,
            "simple_chat_mode": decision.mode,
            "simple_chat_reason": decision.reason,
            "simple_chat_result_count": len(sources),
        },
    )
