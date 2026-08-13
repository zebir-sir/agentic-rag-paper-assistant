"""Pure retrieval-harness policy functions.

The LangGraph workflow remains the execution owner. This module only compiles
the existing planner policy into a stable contract and validates proposals
against it, so tool selection cannot silently cross source boundaries.
"""

from typing import Any, Dict, Iterable, List, Tuple

from .intent_planner import IntentPlan, RetrievalStep
from .retrieval_harness_schema import (
    RetrievalContract,
    RetrievalContractEvaluation,
    RetrievalPlanEnforcement,
)
from .tool_specs import get_tool_source_type


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value or "").strip() for value in values if str(value or "").strip()))


def compile_retrieval_contract(
    plan: IntentPlan,
    *,
    scope_policy: str,
    target_document_ids: List[str],
    allow_supplemental: bool,
    max_retrieval_rounds: int,
) -> RetrievalContract:
    """Compile planner source policy into a per-run retrieval contract."""
    source_requirements = dict(plan.source_requirements or {})
    answer_policy = dict(plan.answer_policy or {})
    planned_sources = _unique(
        get_tool_source_type(str(step.tool)) or "" for step in plan.retrieval_steps
    )
    required = _unique(source_requirements.get("required_source_types") or [])
    preferred = _unique(source_requirements.get("preferred_source_types") or planned_sources)
    blocked = _unique(answer_policy.get("blocked_source_types") or source_requirements.get("forbidden_source_types") or [])
    # Use the task contract rather than every capability exposed by the runtime.
    # This prevents a retry from silently widening a local-paper task into web search.
    allowed = _unique([*required, *preferred, *planned_sources])
    if not allowed:
        allowed = _unique(source for source in planned_sources if source not in blocked)

    return RetrievalContract(
        required_source_types=required,
        preferred_source_types=preferred,
        allowed_source_types=allowed,
        blocked_source_types=blocked,
        unavailable_required_sources=_unique(answer_policy.get("unavailable_required_sources") or []),
        scope_policy=str(scope_policy or "broad_kb"),
        target_document_ids=_unique(target_document_ids),
        allow_supplemental=bool(allow_supplemental),
        citation_required=bool(source_requirements.get("citation_required")),
        freshness_required=bool(source_requirements.get("freshness_required")),
        max_tool_calls_per_round=min(2, max(0, int(plan.max_tools))),
        max_retrieval_rounds=max(1, int(max_retrieval_rounds)),
        must_disclose_limitations=bool(answer_policy.get("must_disclose_limitations")),
        answer_boundary=str(answer_policy.get("answer_boundary") or "use_available_sources_only"),
    )


def enforce_retrieval_contract(
    plan: IntentPlan,
    contract: RetrievalContract,
) -> Tuple[IntentPlan, RetrievalPlanEnforcement]:
    """Filter a planner proposal without replacing it with another source type."""
    enforced = plan.model_copy(deep=True)
    kept: List[RetrievalStep] = []
    filtered: List[Dict[str, Any]] = []
    allowed = set(contract.allowed_source_types)
    blocked = set(contract.blocked_source_types)

    for step in list(enforced.retrieval_steps or []):
        source_type = get_tool_source_type(str(step.tool))
        if not source_type:
            filtered.append({"tool": str(step.tool), "reason": "unknown_tool_source"})
            continue
        if source_type in blocked:
            filtered.append({"tool": str(step.tool), "source_type": source_type, "reason": "blocked_source_type"})
            continue
        if allowed and source_type not in allowed:
            filtered.append({"tool": str(step.tool), "source_type": source_type, "reason": "outside_contract"})
            continue
        if len(kept) >= contract.max_tool_calls_per_round:
            filtered.append({"tool": str(step.tool), "source_type": source_type, "reason": "tool_budget_exceeded"})
            continue
        kept.append(step)

    enforced.retrieval_steps = kept
    enforced.max_tools = min(enforced.max_tools, contract.max_tool_calls_per_round, len(kept))
    enforced.needs_retrieval = bool(kept)
    if filtered:
        enforced.warnings = _unique([*enforced.warnings, "retrieval_contract_filtered_steps"])
    return enforced, RetrievalPlanEnforcement(
        kept_tools=[str(step.tool) for step in kept],
        filtered_tools=filtered,
        reason="planner proposal constrained by retrieval contract",
    )


def evaluate_retrieval_contract(
    contract: RetrievalContract,
    results: List[Dict[str, Any]],
    tools_executed: List[Dict[str, Any]],
) -> RetrievalContractEvaluation:
    """Check required source coverage using normalized evidence envelopes."""
    evidence_source_types = _unique(
        dict(hit.get("metadata") or {}).get("source_type") for hit in results if isinstance(hit, dict)
    )
    executed_source_types = _unique(
        item.get("source_type") for item in tools_executed if isinstance(item, dict)
    )
    missing = [source for source in contract.required_source_types if source not in evidence_source_types]
    if contract.unavailable_required_sources:
        missing = _unique([*missing, *contract.unavailable_required_sources])
    satisfied = not missing
    reason = "required_source_coverage_satisfied" if satisfied else f"missing_required_sources:{','.join(missing)}"
    return RetrievalContractEvaluation(
        required_sources_satisfied=satisfied,
        missing_required_source_types=missing,
        evidence_source_types=evidence_source_types,
        executed_source_types=executed_source_types,
        result_count=len(results),
        reason=reason,
    )
