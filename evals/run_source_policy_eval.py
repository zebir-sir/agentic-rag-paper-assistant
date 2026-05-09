from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.intent_planner import PlannerCapabilities, plan_user_intent_debug
from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8


def _tools(payload: Dict[str, Any]) -> List[str]:
    steps = ((payload.get("normalized_plan") or {}).get("retrieval_steps") or [])
    return [str(s.get("tool")) for s in steps if isinstance(s, dict) and s.get("tool")]


async def run_suite(cases: List[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    stats = {"intent_ok":0,"need_ok":0,"tool_ok":0,"no_retrieval_ok":0,"boundary_expected":0,"source_violation":0,"filtered_unavailable":0}

    sample = cases[:limit] if limit > 0 else cases
    for c in sample:
        caps = PlannerCapabilities(**dict(c.get("capabilities") or {}))
        payload = await plan_user_intent_debug(question=str(c.get("question") or ""), model=None, capabilities=caps)
        plan = payload.get("normalized_plan") or {}
        tools = _tools(payload)
        warnings = list(plan.get("warnings") or [])
        policy = dict(plan.get("answer_policy") or {})

        intent_ok = str(plan.get("intent")) in set(c.get("expected_intent") or [])
        need_ok = bool(plan.get("needs_retrieval")) == bool(c.get("expected_needs_retrieval"))
        required_any = set(c.get("required_tools_any_of") or [])
        forbidden = set(c.get("forbidden_tools") or [])
        tool_ok = (not required_any or any(t in required_any for t in tools)) and not any(t in forbidden for t in tools)
        no_retrieval_ok = (not bool(c.get("expected_needs_retrieval"))) == (len(tools) == 0)

        boundary_exp = bool(c.get("boundary_disclosure_expected", False))
        if boundary_exp:
            stats["boundary_expected"] += 1
        source_violation = 1 if (boundary_exp and not bool(policy.get("must_disclose_limitations"))) else 0

        filtered_unavailable = sum(1 for w in warnings if str(w).startswith("filtered_unavailable_tools:"))

        stats["intent_ok"] += 1 if intent_ok else 0
        stats["need_ok"] += 1 if need_ok else 0
        stats["tool_ok"] += 1 if tool_ok else 0
        stats["no_retrieval_ok"] += 1 if no_retrieval_ok else 0
        stats["source_violation"] += source_violation
        stats["filtered_unavailable"] += filtered_unavailable

        rows.append({"id":c.get("id"),"intent":plan.get("intent"),"needs_retrieval":plan.get("needs_retrieval"),"tools":tools,"warnings":warnings,"policy":policy,"intent_ok":intent_ok,"needs_retrieval_ok":need_ok,"tool_plan_ok":tool_ok,"no_retrieval_ok":no_retrieval_ok,"source_violation":source_violation})

    n = len(rows) or 1
    summary = {
        "total_cases": len(rows),
        "intent_accuracy": stats["intent_ok"]/n,
        "needs_retrieval_accuracy": stats["need_ok"]/n,
        "tool_plan_accuracy": stats["tool_ok"]/n,
        "no_retrieval_accuracy": stats["no_retrieval_ok"]/n,
        "boundary_disclosure_expected_count": stats["boundary_expected"],
        "source_violation_count": stats["source_violation"],
        "unavailable_tool_filtered_count": stats["filtered_unavailable"],
    }
    return {"summary":summary,"cases":rows}


def to_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = ["# Source Policy Eval","",f"- total_cases: {s['total_cases']}",f"- intent_accuracy: {s['intent_accuracy']:.3f}",f"- needs_retrieval_accuracy: {s['needs_retrieval_accuracy']:.3f}",f"- tool_plan_accuracy: {s['tool_plan_accuracy']:.3f}",f"- no_retrieval_accuracy: {s['no_retrieval_accuracy']:.3f}",f"- boundary_disclosure_expected_count: {s['boundary_disclosure_expected_count']}",f"- source_violation_count: {s['source_violation_count']}",f"- unavailable_tool_filtered_count: {s['unavailable_tool_filtered_count']}","","| Case | Intent | Needs Retrieval | Tools | Intent OK | Need OK | Tool OK | Violation |","|---|---|---:|---|---:|---:|---:|---:|"]
    for c in report.get("cases",[]):
        lines.append(f"| {c['id']} | {c['intent']} | {1 if c['needs_retrieval'] else 0} | {','.join(c['tools'])} | {1 if c['intent_ok'] else 0} | {1 if c['needs_retrieval_ok'] else 0} | {1 if c['tool_plan_ok'] else 0} | {c['source_violation']} |")
    return "\n".join(lines)+"\n"


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="evals/cases/source_policy_cases.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output-dir", default="evals/results")
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    report = await run_suite(read_json_robust(Path(a.cases)), int(a.limit or 0))
    write_json_utf8(out / "source_policy_eval.json", report, indent=2)
    write_text_utf8(out / "source_policy_eval.md", to_markdown(report))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
