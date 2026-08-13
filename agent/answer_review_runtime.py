from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from evals.judges.answer_rubric import judge_answer_with_rubric


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

    return AnswerReviewRuntimeResult(
        reviewed=True,
        approved=True,
        revised_answer=text,
        review_action="retain_with_metadata",
        unsupported_claim_risk=risk,
        unsupported_claim_notes=notes,
        reason="检测到未支撑断言风险，已记录到会话元数据供依据面板与调试使用。",
    )
