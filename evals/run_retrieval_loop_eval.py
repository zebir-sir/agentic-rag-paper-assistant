from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_langgraph import run_langgraph_analysis
from agent.agent_runtime import AgentDependencies
from agent.db_utils import close_database, initialize_database


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return any(_norm(k) in t for k in keywords if str(k or "").strip())


def _source_to_dict(item: Any) -> Dict[str, Any]:
    if hasattr(item, "model_dump"):
        payload = item.model_dump()
    elif isinstance(item, dict):
        payload = dict(item)
    else:
        payload = {
            "document_title": getattr(item, "document_title", ""),
            "document_source": getattr(item, "document_source", ""),
            "snippet": getattr(item, "snippet", ""),
            "metadata": getattr(item, "metadata", {}),
        }
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    payload["metadata"] = metadata
    payload["document_title"] = str(payload.get("document_title") or "")
    payload["document_source"] = str(payload.get("document_source") or "")
    payload["snippet"] = str(payload.get("snippet") or "")
    return payload


def _keyword_recall_from_sources(sources: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    keys = [k for k in (expected_keywords or []) if str(k or "").strip()]
    if not keys:
        return 0.0
    blob = "\n".join(str(s.get("snippet") or "") for s in sources).lower()
    hit = sum(1 for k in keys if _norm(k) in blob)
    return hit / len(keys)


def _extract_float(metadata: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_int(metadata: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def _extract_rewrite_used(metadata: Dict[str, Any]) -> Optional[bool]:
    rewritten = metadata.get("rewritten_queries")
    if isinstance(rewritten, list):
        return len(rewritten) > 0
    retry_count = metadata.get("retrieval_retry_count")
    if isinstance(retry_count, (int, float)):
        return float(retry_count) > 0
    return None


async def run_eval(
    cases: List[Dict[str, Any]],
    timeout_seconds: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    case_results: List[Dict[str, Any]] = []
    doc_hit_sum = 0
    keyword_sum = 0.0
    rewrite_sum = 0
    rewrite_denom = 0
    attempts_sum = 0.0
    attempts_denom = 0
    conf_sum = 0.0
    conf_denom = 0

    total_cases = len(cases)
    for idx_case, case in enumerate(cases, 1):
        cid = str(case.get("id") or "unknown")
        query = str(case.get("query") or "").strip()
        expected_docs = list(case.get("expected_document_keywords") or [])
        expected_content = list(case.get("expected_content_keywords") or [])
        notes = str(case.get("notes") or "")

        item: Dict[str, Any] = {
            "id": cid,
            "query": query,
            "notes": notes,
            "doc_hit_at_k": False,
            "keyword_recall_at_k": 0.0,
            "rewrite_used": None,
            "retrieval_attempt_count": None,
            "retrieval_confidence": None,
            "retrieval_sufficient": None,
            "rewritten_queries": None,
            "final_query": None,
            "source_titles": [],
            "source_snippets_preview": [],
            "error": None,
        }

        deps = AgentDependencies(session_id=f"eval-loop-{uuid.uuid4().hex[:12]}", user_id="eval")
        if verbose:
            print(f"[loop-eval] running {idx_case}/{total_cases} {cid} ...")
        try:
            result = await asyncio.wait_for(
                run_langgraph_analysis(question=query, deps=deps, context_prompt=""),
                timeout=max(1, int(timeout_seconds)),
            )
            metadata = dict(result.metadata or {})
            sources = [_source_to_dict(s) for s in list(result.sources or [])]
            item["source_titles"] = sorted(
                {
                    str(s.get("document_title") or s.get("document_source") or "").strip()
                    for s in sources
                    if str(s.get("document_title") or s.get("document_source") or "").strip()
                }
            )
            item["source_snippets_preview"] = [str(s.get("snippet") or "")[:180] for s in sources[:5]]

            doc_blob = " ".join(item["source_titles"])
            item["doc_hit_at_k"] = _contains_any(doc_blob, expected_docs)
            item["keyword_recall_at_k"] = _keyword_recall_from_sources(sources, expected_content)

            item["retrieval_attempt_count"] = _extract_int(metadata, "retrieval_attempt_count")
            item["retrieval_confidence"] = _extract_float(
                metadata, "retrieval_confidence", "retrieval_top_score"
            )
            rs = metadata.get("retrieval_sufficient")
            item["retrieval_sufficient"] = bool(rs) if isinstance(rs, bool) else None
            item["rewritten_queries"] = metadata.get("rewritten_queries")
            item["rewrite_used"] = _extract_rewrite_used(metadata)
            final_query = metadata.get("current_query")
            if not isinstance(final_query, str):
                final_query = None
            item["final_query"] = final_query
            if verbose:
                print(
                    f"[loop-eval] done {cid} "
                    f"doc_hit={item['doc_hit_at_k']} "
                    f"keyword_recall={float(item['keyword_recall_at_k'] or 0.0):.2f} "
                    f"attempts={item.get('retrieval_attempt_count')}"
                )

        except asyncio.TimeoutError:
            item["timeout"] = max(1, int(timeout_seconds))
            item["error"] = f"TimeoutError: case exceeded {max(1, int(timeout_seconds))}s"
            if verbose:
                print(f"[loop-eval] timeout {cid} after {max(1, int(timeout_seconds))}s")
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[loop-eval] failed {cid}: {item['error']}")

        doc_hit_sum += 1 if item["doc_hit_at_k"] else 0
        keyword_sum += float(item["keyword_recall_at_k"] or 0.0)

        if isinstance(item["rewrite_used"], bool):
            rewrite_denom += 1
            rewrite_sum += 1 if item["rewrite_used"] else 0

        if isinstance(item["retrieval_attempt_count"], int):
            attempts_denom += 1
            attempts_sum += float(item["retrieval_attempt_count"])

        if isinstance(item["retrieval_confidence"], (int, float)):
            conf_denom += 1
            conf_sum += float(item["retrieval_confidence"])

        case_results.append(item)

    total = len(case_results)
    summary = {
        "total_cases": total,
        "doc_hit_at_k": (doc_hit_sum / total) if total else 0.0,
        "avg_keyword_recall_at_k": (keyword_sum / total) if total else 0.0,
        "rewrite_used_rate": (rewrite_sum / rewrite_denom) if rewrite_denom else None,
        "avg_retrieval_attempts": (attempts_sum / attempts_denom) if attempts_denom else None,
        "avg_retrieval_confidence": (conf_sum / conf_denom) if conf_denom else None,
    }
    return {"summary": summary, "cases": case_results}


def to_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines: List[str] = []
    rewrite_used_rate = s.get("rewrite_used_rate")
    avg_attempts = s.get("avg_retrieval_attempts")
    avg_conf = s.get("avg_retrieval_confidence")
    lines.append("# Retrieval Loop Eval Report")
    lines.append("")
    lines.append(f"- Total Cases: {s.get('total_cases', 0)}")
    lines.append(f"- Doc Hit@K: {float(s.get('doc_hit_at_k') or 0.0):.3f}")
    lines.append(f"- Avg Keyword Recall@K: {float(s.get('avg_keyword_recall_at_k') or 0.0):.3f}")
    lines.append(f"- Rewrite Used Rate: {'N/A' if rewrite_used_rate is None else f'{float(rewrite_used_rate):.3f}'}")
    lines.append(f"- Avg Retrieval Attempts: {'N/A' if avg_attempts is None else f'{float(avg_attempts):.3f}'}")
    lines.append(f"- Avg Retrieval Confidence: {'N/A' if avg_conf is None else f'{float(avg_conf):.3f}'}")
    lines.append("")
    lines.append("| Case ID | Query | Doc Hit | Keyword Recall | Rewrite Used | Attempts | Confidence | Notes | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
    for c in report.get("cases", []):
        lines.append(
            "| {id} | {q} | {dh} | {kr:.3f} | {rw} | {att} | {conf} | {notes} | {err} |".format(
                id=str(c.get("id") or ""),
                q=str(c.get("query") or "").replace("|", "/"),
                dh=1 if c.get("doc_hit_at_k") else 0,
                kr=float(c.get("keyword_recall_at_k") or 0.0),
                rw="N/A" if c.get("rewrite_used") is None else (1 if c.get("rewrite_used") else 0),
                att="N/A" if c.get("retrieval_attempt_count") is None else c.get("retrieval_attempt_count"),
                conf=(
                    "N/A"
                    if c.get("retrieval_confidence") is None
                    else f"{float(c.get('retrieval_confidence')):.3f}"
                ),
                notes=str(c.get("notes") or "").replace("|", "/"),
                err=str(c.get("error") or "").replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangGraph retrieval-loop eval")
    parser.add_argument("--cases", default="evals/retrieval_loop_cases.json")
    parser.add_argument("--out-dir", default="evals/results")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cases = json.loads(cases_path.read_text(encoding="utf-8-sig"))
    max_cases = int(args.max_cases or 0)
    cases = all_cases if max_cases <= 0 else list(all_cases)[:max_cases]

    print(
        f"[loop-eval] loaded total_cases={len(all_cases)} "
        f"running_cases={len(cases)} timeout={max(1, int(args.timeout_seconds))}s"
    )

    await initialize_database()
    try:
        report = await run_eval(
            cases=cases,
            timeout_seconds=max(1, int(args.timeout_seconds)),
            verbose=bool(args.verbose),
        )
    finally:
        await close_database()

    json_path = out_dir / "retrieval_loop_eval.json"
    md_path = out_dir / "retrieval_loop_eval.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print("[retrieval-loop-eval] done")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"json: {json_path}")
    print(f"md:   {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
