from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.db_utils import close_database, initialize_database
from agent.tools import HybridSearchInput, hybrid_search_tool


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return any(_norm(k) in t for k in keywords if str(k or "").strip())


def _doc_text(hit: Dict[str, Any]) -> str:
    return f"{hit.get('document_title','')} {hit.get('document_source','')}"


def _section_text(hit: Dict[str, Any]) -> str:
    md = hit.get("metadata") or {}
    if not isinstance(md, dict):
        return ""
    return f"{md.get('section_title','')} {md.get('section_path_text','')}"


def _keyword_recall(results: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    keys = [k for k in (expected_keywords or []) if str(k or "").strip()]
    if not keys:
        return 0.0
    blob = "\n".join(str(r.get("content") or "") for r in results).lower()
    hit = sum(1 for k in keys if _norm(k) in blob)
    return hit / len(keys)


def _safe_case_fields(case: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    return (
        list(case.get("expected_document_keywords") or []),
        list(case.get("expected_section_keywords") or []),
        list(case.get("expected_content_keywords") or []),
    )


async def run_eval(cases: List[Dict[str, Any]], limit: int, text_weight: float) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    section_denom = 0
    doc_hit_at_1_sum = 0
    doc_hit_at_k_sum = 0
    section_hit_at_k_sum = 0
    keyword_recall_sum = 0.0

    for case in cases:
        cid = case.get("id") or "unknown"
        query = str(case.get("query") or "").strip()
        expected_docs, expected_sections, expected_content = _safe_case_fields(case)

        case_result: Dict[str, Any] = {
            "id": cid,
            "query": query,
            "limit": limit,
            "doc_hit_at_1": False,
            "doc_hit_at_k": False,
            "section_hit_at_k": None,
            "keyword_recall_at_k": 0.0,
            "result_count": 0,
            "error": None,
        }

        try:
            tool_rows = await hybrid_search_tool(
                HybridSearchInput(query=query, limit=limit, text_weight=text_weight)
            )
            rows = [
                {
                    "document_title": r.document_title,
                    "document_source": r.document_source,
                    "content": r.content,
                    "metadata": r.metadata,
                    "score": r.score,
                }
                for r in tool_rows
            ]
            case_result["result_count"] = len(rows)

            if rows:
                case_result["doc_hit_at_1"] = _contains_any(_doc_text(rows[0]), expected_docs)
                case_result["doc_hit_at_k"] = any(_contains_any(_doc_text(r), expected_docs) for r in rows)
            else:
                case_result["doc_hit_at_1"] = False
                case_result["doc_hit_at_k"] = False

            if expected_sections:
                section_denom += 1
                case_result["section_hit_at_k"] = any(
                    _contains_any(_section_text(r), expected_sections) for r in rows
                )
            else:
                case_result["section_hit_at_k"] = None

            case_result["keyword_recall_at_k"] = _keyword_recall(rows, expected_content)

        except Exception as exc:
            case_result["error"] = f"{type(exc).__name__}: {exc}"

        doc_hit_at_1_sum += 1 if case_result["doc_hit_at_1"] else 0
        doc_hit_at_k_sum += 1 if case_result["doc_hit_at_k"] else 0
        if case_result["section_hit_at_k"] is True:
            section_hit_at_k_sum += 1
        keyword_recall_sum += float(case_result["keyword_recall_at_k"] or 0.0)
        results.append(case_result)

    total = len(results)
    summary = {
        "total_cases": total,
        "doc_hit_at_1": (doc_hit_at_1_sum / total) if total else 0.0,
        "doc_hit_at_k": (doc_hit_at_k_sum / total) if total else 0.0,
        "section_hit_at_k": (section_hit_at_k_sum / section_denom) if section_denom else None,
        "avg_keyword_recall_at_k": (keyword_recall_sum / total) if total else 0.0,
        "limit": limit,
        "text_weight": text_weight,
    }
    return {"summary": summary, "cases": results}


def to_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = []
    section_hit = "N/A" if s["section_hit_at_k"] is None else f"{s['section_hit_at_k']:.3f}"
    lines.append("# Retrieval Eval Report")
    lines.append("")
    lines.append(f"- Total Cases: {s['total_cases']}")
    lines.append(f"- Doc Hit@1: {s['doc_hit_at_1']:.3f}")
    lines.append(f"- Doc Hit@K: {s['doc_hit_at_k']:.3f}")
    lines.append(f"- Section Hit@K: {section_hit}")
    lines.append(f"- Avg Keyword Recall@K: {s['avg_keyword_recall_at_k']:.3f}")
    lines.append("")
    lines.append("| Case ID | Doc@1 | Doc@K | Section@K | KeywordRecall@K | Count | Error |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for c in report["cases"]:
        sec = "N/A" if c["section_hit_at_k"] is None else ("1" if c["section_hit_at_k"] else "0")
        lines.append(
            f"| {c['id']} | {1 if c['doc_hit_at_1'] else 0} | {1 if c['doc_hit_at_k'] else 0} | {sec} | {float(c['keyword_recall_at_k']):.3f} | {c['result_count']} | {c['error'] or ''} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight retrieval eval")
    parser.add_argument("--cases", default="evals/retrieval_cases.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--text-weight", type=float, default=0.3)
    parser.add_argument("--out-dir", default="evals/results")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads(cases_path.read_text(encoding="utf-8-sig"))

    await initialize_database()
    try:
        report = await run_eval(cases=cases, limit=max(1, args.limit), text_weight=args.text_weight)
    finally:
        await close_database()

    json_path = out_dir / "retrieval_eval.json"
    md_path = out_dir / "retrieval_eval.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print("[retrieval-eval] done")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"json: {json_path}")
    print(f"md:   {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
