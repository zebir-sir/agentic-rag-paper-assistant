"""Presentation-safe retrieval harness metadata for API and UI consumers."""

from typing import Any, Dict, List


def build_retrieval_harness_trace_payload(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small, query-free trace payload suitable for an SSE client."""
    metadata = dict(metadata or {})
    contract = dict(metadata.get("retrieval_contract") or {})
    evaluation = dict(metadata.get("retrieval_contract_evaluation") or {})
    if not contract and not evaluation:
        return {"available": False}

    tools: List[str] = []
    for record in list(metadata.get("retrieval_execution_records") or []):
        if isinstance(record, dict):
            tool = str(record.get("tool") or "").strip()
            if tool and tool not in tools:
                tools.append(tool)

    return {
        "available": True,
        "scope_policy": str(contract.get("scope_policy") or metadata.get("scope_policy") or "broad_kb"),
        "required_source_types": list(contract.get("required_source_types") or []),
        "allowed_source_types": list(contract.get("allowed_source_types") or []),
        "citation_required": bool(contract.get("citation_required")),
        "freshness_required": bool(contract.get("freshness_required")),
        "max_tool_calls_per_round": contract.get("max_tool_calls_per_round"),
        "retrieval_attempt_count": int(metadata.get("retrieval_attempt_count") or 0),
        "tools_executed": tools,
        "required_sources_satisfied": evaluation.get("required_sources_satisfied"),
        "missing_required_source_types": list(evaluation.get("missing_required_source_types") or []),
        "evidence_source_types": list(evaluation.get("evidence_source_types") or []),
        "reason": str(evaluation.get("reason") or metadata.get("retrieval_insufficient_reason") or ""),
    }
