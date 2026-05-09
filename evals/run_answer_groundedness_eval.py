from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_langgraph import run_langgraph_analysis
from agent.agent_runtime import AgentDependencies
from agent.db_utils import close_database, initialize_database
from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8
from evals.baselines.naive_hybrid_rag import run_naive_hybrid_rag
from evals.judges.answer_rubric import judge_answer_with_rubric
from evals.judges.source_boundary_judge import judge_source_boundary


def _source_dicts(items: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


def _extract_judge_capabilities(metadata: Dict[str, Any] | None) -> Dict[str, bool]:
    payload = metadata if isinstance(metadata, dict) else {}
    planner_caps = payload.get("planner_capabilities") if isinstance(payload.get("planner_capabilities"), dict) else {}
    intent_plan = payload.get("intent_plan") if isinstance(payload.get("intent_plan"), dict) else {}
    answer_policy = intent_plan.get("answer_policy") if isinstance(intent_plan.get("answer_policy"), dict) else {}
    unavailable_required = {
        str(item or "").strip().lower()
        for item in list(answer_policy.get("unavailable_required_sources") or [])
        if str(item or "").strip()
    }
    tools_executed = {
        str(item or "").strip().lower()
        for item in list(payload.get("tools_executed") or [])
        if str(item or "").strip()
    }

    web_enabled = bool(planner_caps.get("web_search_enabled", False))
    openalex_enabled = bool(planner_caps.get("openalex_search_enabled", False))

    if not web_enabled and any(tool in tools_executed for tool in {"web_search", "search_web", "web"}):
        web_enabled = True
    if not openalex_enabled and any(tool in tools_executed for tool in {"openalex_search", "search_openalex_papers"}):
        openalex_enabled = True

    if "general_web" in unavailable_required:
        web_enabled = False
    if "external_academic" in unavailable_required:
        openalex_enabled = False

    return {
        "web_search_enabled": web_enabled,
        "openalex_search_enabled": openalex_enabled,
    }


def _select_cases(cases: List[Dict[str, Any]], case_id: str | None, limit: int) -> List[Dict[str, Any]]:
    if case_id:
        selected = [case for case in cases if str(case.get("id") or "") == case_id]
        if not selected:
            raise ValueError(f"case_id not found: {case_id}")
        return selected
    return cases[:limit] if limit > 0 else cases


async def run_suite(cases: List[Dict[str, Any]], limit: int, timeout_seconds: int, case_id: str | None = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in _select_cases(cases, case_id=case_id, limit=limit):
        q = str(case.get("question") or "")
        case_row: Dict[str, Any] = {"id": case.get("id"), "question": q, "error": None}
        try:
            baseline = await run_naive_hybrid_rag(q, top_k=5)
            deps = AgentDependencies(session_id=f"eval-answer-{uuid.uuid4().hex[:10]}", user_id="eval")
            full = await asyncio.wait_for(run_langgraph_analysis(question=q, deps=deps, context_prompt=""), timeout=max(1, timeout_seconds))

            baseline_sources = list(baseline.get("retrieval_hits") or [])
            full_sources = _source_dicts(list(full.sources or []))
            baseline_caps = _extract_judge_capabilities(dict(baseline.get("metadata") or {}))
            full_caps = _extract_judge_capabilities(dict(full.metadata or {}))

            b_score = judge_answer_with_rubric(case, str(baseline.get("answer") or ""), baseline_sources)
            f_score = judge_answer_with_rubric(case, str(full.message or ""), full_sources)
            b_boundary = judge_source_boundary(case, str(baseline.get("answer") or ""), baseline_caps)
            f_boundary = judge_source_boundary(case, str(full.message or ""), full_caps)

            # 提取详细指标
            notes = f_score.get("unsupported_claim_notes", [])
            numeric_claims = sum(1 for n in notes if n.get("claim_type") == "unsupported_numeric_claim")
            mechanism_claims = sum(1 for n in notes if n.get("claim_type") == "unsupported_mechanism_claim")
            external_fact_claims = sum(1 for n in notes if n.get("claim_type") == "unsupported_external_fact")
            
            answer_text = str(full.message or "")
            inference_labeling = 1 if "基于现有片段可推断" in answer_text else 0
            evidence_gap_disclosed = 1 if "未明确说明" in answer_text or "无法提供" in answer_text or "边界" in answer_text else 0
            
            # Audit Status 逻辑
            boundary_vio = int(f_boundary.get("source_boundary_violation", 0))
            if boundary_vio == 1 or numeric_claims > 0 or external_fact_claims > 0:
                audit_status = "FAIL"
                audit_reason = "Critical grounding violation (boundary or factual claim)"
            elif mechanism_claims > 0 or f_score.get("unsupported_claim_risk", 0) > 0:
                audit_status = "WARN"
                audit_reason = "Mechanism or assertion risk"
            else:
                audit_status = "PASS"
                audit_reason = "Compliant with grounding rules"

            case_row.update(
                {
                    "audit": {
                        "status": audit_status,
                        "reason": audit_reason,
                        "source_boundary_violation": boundary_vio,
                        "unsupported_numeric_claim_count": numeric_claims,
                        "unsupported_mechanism_claim_count": mechanism_claims,
                        "unsupported_external_fact_count": external_fact_claims,
                        "inference_labeling": inference_labeling,
                        "evidence_gap_disclosed": evidence_gap_disclosed,
                        "answer_usefulness": f_score.get("completeness_score", 0) + f_score.get("clarity_score", 0)
                    },
                    "baseline": {"answer": baseline.get("answer"), "score": b_score, "boundary": b_boundary},
                    "full_system": {"answer": full.message, "score": f_score, "boundary": f_boundary, "metadata": full.metadata},
                }
            )
        except Exception as exc:
            case_row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(case_row)

    valid = [r for r in rows if not r.get("error") and r.get("audit")]
    n = len(valid) or 1
    
    return {
        "summary": {
            "total_cases": len(rows),
            "valid_cases": len(valid),
            "pass_rate": sum(1 for r in valid if r["audit"]["status"] == "PASS") / n,
            "warn_rate": sum(1 for r in valid if r["audit"]["status"] == "WARN") / n,
            "fail_rate": sum(1 for r in valid if r["audit"]["status"] == "FAIL") / n,
            "avg_unsupported_numeric": sum(r["audit"]["unsupported_numeric_claim_count"] for r in valid) / n,
            "avg_unsupported_mechanism": sum(r["audit"]["unsupported_mechanism_claim_count"] for r in valid) / n,
            "inference_labeling_rate": sum(r["audit"]["inference_labeling"] for r in valid) / n,
            "gap_disclosure_rate": sum(r["audit"]["evidence_gap_disclosed"] for r in valid) / n,
            "note": "Focus on responsibility validation and evidence-faithful auditing.",
        },
        "cases": rows,
    }


def to_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Answer Groundedness Audit",
        "",
        "- 目标：审核回答是否严格遵守证据锚定规则，是否如实说明证据差距。",
        "",
        f"- total_cases: {s['total_cases']}",
        f"- audit pass_rate: {s['pass_rate']:.3f}",
        f"- avg_unsupported_numeric_claims: {s['avg_unsupported_numeric']:.2f}",
        f"- inference_labeling_rate: {s['inference_labeling_rate']:.3f}",
        f"- gap_disclosure_rate: {s['gap_disclosure_rate']:.3f}",
        "",
        "## Audit Details",
        "",
        "| Case | Status | Reason | Num. Claim | Mech. Claim | Gap Disclosed | Usefulness |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for c in report.get("cases", []):
        if c.get("error"):
            lines.append(f"| {c.get('id')} | ERROR | {c.get('error')} | - | - | - | - |")
            continue
        a = c["audit"]
        lines.append(f"| {c.get('id')} | {a['status']} | {a['reason']} | {a['unsupported_numeric_claim_count']} | {a['unsupported_mechanism_claim_count']} | {a['evidence_gap_disclosed']} | {a['answer_usefulness']} |")
    return "\n".join(lines) + "\n"


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="evals/cases/answer_groundedness_cases.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--case-id", default="")
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--output-dir", default="evals/results")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = read_json_robust(Path(args.cases))

    await initialize_database()
    try:
        t0 = time.time()
        report = await run_suite(
            cases=cases,
            limit=int(args.limit or 0),
            timeout_seconds=int(args.timeout_seconds or 120),
            case_id=str(args.case_id or "").strip() or None,
        )
        report["summary"]["runtime_seconds"] = round(time.time() - t0, 3)
    finally:
        await close_database()

    write_json_utf8(out / "answer_groundedness_eval.json", report, indent=2)
    write_text_utf8(out / "answer_groundedness_eval.md", to_markdown(report))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
