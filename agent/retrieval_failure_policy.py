"""Policy for explicit model-knowledge fallback after external retrieval failure."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .external_retrieval_resilience import FAILURE_STATES


_SOURCE_LABELS = {
    "general_web": "通用网页检索",
    "external_academic": "OpenAlex 学术检索",
}


def _unique_statuses(statuses: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[tuple[str, str], Dict[str, Any]] = {}
    for raw in statuses:
        if not isinstance(raw, dict):
            continue
        source_type = str(raw.get("source_type") or "").strip()
        state = str(raw.get("state") or "").strip()
        if not source_type or state not in FAILURE_STATES:
            continue
        unique[(source_type, state)] = {
            "source_type": source_type,
            "provider": str(raw.get("provider") or source_type).strip(),
            "state": state,
            "retry_count": max(0, int(raw.get("retry_count") or 0)),
            "retry_after_seconds": raw.get("retry_after_seconds"),
        }
    return list(unique.values())


def apply_external_retrieval_failure_policy(
    answer_policy: Dict[str, Any],
    statuses: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Permit a disclosed model-knowledge fallback without source impersonation."""
    policy = dict(answer_policy or {})
    failures = _unique_statuses(statuses)
    if not failures:
        return policy

    blocked = {str(value).strip() for value in policy.get("blocked_source_types") or [] if str(value).strip()}
    allowed = {str(value).strip() for value in policy.get("allowed_source_types") or [] if str(value).strip()}
    model_knowledge_allowed = "model_knowledge" not in blocked
    if model_knowledge_allowed:
        allowed.add("model_knowledge")
        policy["mode"] = "answer_with_disclosure"
        policy["answer_boundary"] = "external_retrieval_failed_model_knowledge_fallback"
        policy["guidance_to_answer_agent"] = (
            "External retrieval failed. You may provide a clearly qualified answer from model knowledge, "
            "but must not present it as verified web or OpenAlex evidence."
        )
        policy["external_fallback_mode"] = "model_knowledge_with_disclosure"
    else:
        policy["external_fallback_mode"] = "disclosure_only"
    policy["allowed_source_types"] = sorted(allowed)
    policy["must_disclose_limitations"] = True
    policy["external_retrieval_failures"] = failures
    return policy


def build_external_retrieval_disclosure(
    statuses: Iterable[Dict[str, Any]],
    answer_policy: Dict[str, Any],
) -> str:
    failures = _unique_statuses(statuses)
    if not failures:
        return ""

    source_labels = list(
        dict.fromkeys(_SOURCE_LABELS.get(item["source_type"], item["source_type"]) for item in failures)
    )
    sources = "、".join(source_labels)
    model_knowledge_allowed = str(answer_policy.get("external_fallback_mode") or "") == "model_knowledge_with_disclosure"
    if model_knowledge_allowed:
        return (
            f"说明：本轮{sources}暂时不可用；未由成功来源支撑的补充内容基于模型通用知识，"
            "不等同于已核验的外部资料，请自行辨别并核验。"
        )
    return f"说明：本轮{sources}暂时不可用，当前无法用模型知识替代所需的受限证据。"
