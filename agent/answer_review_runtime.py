from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from evals.judges.answer_rubric import judge_answer_with_rubric


_ALREADY_CAUTIONED_MARKERS = (
    "当前检索片段未明确说明",
    "仍需回到原文确认",
    "仅基于当前检索片段",
    "以上判断基于当前检索片段",
)


@dataclass
class AnswerReviewRuntimeResult:
    reviewed: bool
    approved: bool
    revised_answer: str
    review_action: str
    unsupported_claim_risk: int
    unsupported_claim_notes: List[Dict[str, str]] = field(default_factory=list)
    reason: str = ""


def _normalize_source(source: Any) -> Dict[str, Any]:
    if hasattr(source, "model_dump"):
        payload = source.model_dump()
        return payload if isinstance(payload, dict) else {}
    if isinstance(source, dict):
        return dict(source)
    return {}


def _collect_caveat_fragments(notes: List[Dict[str, str]]) -> List[str]:
    claim_types = {str(note.get("claim_type") or "") for note in notes}
    fragments: List[str] = []
    if "unsupported_numeric_claim" in claim_types:
        fragments.append("具体数字/年份")
    if "unsupported_mechanism_claim" in claim_types:
        fragments.append("具体机制细节")
    if "unsupported_external_fact" in claim_types:
        fragments.append("外部来源或论文元数据")
    if "assertion" in claim_types and not fragments:
        fragments.append("部分结论表述")
    return fragments


def _build_runtime_caveat(notes: List[Dict[str, str]]) -> str:
    fragments = _collect_caveat_fragments(notes)
    if not fragments:
        return "注：以上判断基于当前检索片段，仍需回到原文进一步确认。"
    joined = "、".join(fragments)
    return f"注：以上回答基于当前检索片段；其中涉及{joined}的内容，仍需回到原文进一步确认。"


def _has_existing_caveat(answer: str) -> bool:
    text = str(answer or "").strip()
    return any(marker in text for marker in _ALREADY_CAUTIONED_MARKERS)


def review_generated_answer(
    *,
    answer: str,
    sources: List[Any],
    is_local_question: bool,
) -> AnswerReviewRuntimeResult:
    text = str(answer or "").strip()
    normalized_sources = [_normalize_source(source) for source in list(sources or [])]
    normalized_sources = [source for source in normalized_sources if source]

    if not text:
        return AnswerReviewRuntimeResult(
            reviewed=False,
            approved=True,
            revised_answer=text,
            review_action="skip_empty_answer",
            unsupported_claim_risk=0,
            reason="回答为空，跳过运行时审核。",
        )

    if not is_local_question:
        return AnswerReviewRuntimeResult(
            reviewed=False,
            approved=True,
            revised_answer=text,
            review_action="skip_non_local_question",
            unsupported_claim_risk=0,
            reason="当前不是本地论文证据型问题，跳过运行时审核。",
        )

    if not normalized_sources:
        return AnswerReviewRuntimeResult(
            reviewed=False,
            approved=True,
            revised_answer=text,
            review_action="skip_no_sources",
            unsupported_claim_risk=0,
            reason="当前没有证据 sources，避免对直答场景误判。",
        )

    rubric = judge_answer_with_rubric(case={"question": ""}, answer=text, sources=normalized_sources)
    risk = int(rubric.get("unsupported_claim_risk") or 0)
    notes = list(rubric.get("unsupported_claim_notes") or [])

    if risk <= 0:
        return AnswerReviewRuntimeResult(
            reviewed=True,
            approved=True,
            revised_answer=text,
            review_action="keep",
            unsupported_claim_risk=risk,
            unsupported_claim_notes=notes,
            reason="未发现明显未支撑断言风险。",
        )

    if _has_existing_caveat(text):
        return AnswerReviewRuntimeResult(
            reviewed=True,
            approved=True,
            revised_answer=text,
            review_action="keep_existing_caveat",
            unsupported_claim_risk=risk,
            unsupported_claim_notes=notes,
            reason="回答已包含证据边界提醒，不再重复追加。",
        )

    revised = f"{text}\n\n{_build_runtime_caveat(notes)}"
    return AnswerReviewRuntimeResult(
        reviewed=True,
        approved=True,
        revised_answer=revised,
        review_action="append_caveat",
        unsupported_claim_risk=risk,
        unsupported_claim_notes=notes,
        reason="检测到未支撑断言风险，已追加运行时证据边界提醒。",
    )
