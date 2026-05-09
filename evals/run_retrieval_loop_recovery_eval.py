from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_langgraph import run_langgraph_analysis
from agent.agent_runtime import AgentDependencies
from agent.db_utils import close_database, initialize_database
from agent.tools import HybridSearchInput, hybrid_search_tool
from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(_norm(keyword) in _norm(text) for keyword in keywords if str(keyword or "").strip())


def _kw_recall(blob: str, keys: List[str]) -> float:
    return (sum(1 for key in keys if _norm(key) in _norm(blob)) / len(keys)) if keys else 0.0


def _to_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _snippet_of(payload: Dict[str, Any]) -> str:
    return str(payload.get("snippet") or payload.get("content") or "").strip()


def _source_to_dict(item: Any) -> Dict[str, Any]:
    if hasattr(item, "model_dump"):
        payload = item.model_dump()
    elif isinstance(item, dict):
        payload = dict(item)
    else:
        payload = {
            "source_type": getattr(item, "source_type", "local"),
            "document_id": getattr(item, "document_id", None),
            "document_title": getattr(item, "document_title", ""),
            "document_source": getattr(item, "document_source", ""),
            "chunk_id": getattr(item, "chunk_id", None),
            "snippet": getattr(item, "snippet", ""),
            "score": getattr(item, "score", None),
            "metadata": getattr(item, "metadata", {}),
        }
    metadata = _to_metadata(payload)
    return {
        "source_type": str(payload.get("source_type") or metadata.get("source_type") or "local"),
        "document_id": str(payload.get("document_id") or metadata.get("document_id") or "") or None,
        "document_title": str(payload.get("document_title") or ""),
        "document_source": str(payload.get("document_source") or ""),
        "chunk_id": str(payload.get("chunk_id") or metadata.get("chunk_id") or "") or None,
        "snippet": _snippet_of(payload),
        "content": str(payload.get("content") or payload.get("snippet") or ""),
        "score": float(payload.get("score")) if isinstance(payload.get("score"), (int, float)) else None,
        "metadata": metadata,
    }


def _hit_to_dict(item: Any) -> Dict[str, Any]:
    if hasattr(item, "model_dump"):
        payload = item.model_dump()
    elif isinstance(item, dict):
        payload = dict(item)
    else:
        payload = {
            "document_id": getattr(item, "document_id", None),
            "document_title": getattr(item, "document_title", ""),
            "document_source": getattr(item, "document_source", ""),
            "chunk_id": getattr(item, "chunk_id", None),
            "content": getattr(item, "content", ""),
            "snippet": getattr(item, "snippet", ""),
            "score": getattr(item, "score", None),
            "metadata": getattr(item, "metadata", {}),
        }
    metadata = _to_metadata(payload)
    return {
        "document_id": str(payload.get("document_id") or metadata.get("document_id") or "") or None,
        "document_title": str(payload.get("document_title") or ""),
        "document_source": str(payload.get("document_source") or ""),
        "chunk_id": str(payload.get("chunk_id") or metadata.get("chunk_id") or "") or None,
        "snippet": _snippet_of(payload),
        "content": str(payload.get("content") or payload.get("snippet") or ""),
        "score": float(payload.get("score")) if isinstance(payload.get("score"), (int, float)) else None,
        "metadata": metadata,
    }


def _source_from_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_type": str((hit.get("metadata") or {}).get("source_type") or "local"),
        "document_id": hit.get("document_id"),
        "document_title": str(hit.get("document_title") or ""),
        "document_source": str(hit.get("document_source") or ""),
        "chunk_id": hit.get("chunk_id"),
        "snippet": str(hit.get("snippet") or ""),
        "content": str(hit.get("content") or hit.get("snippet") or ""),
        "score": hit.get("score"),
        "metadata": dict(hit.get("metadata") or {}),
    }


def _evaluate_items(items: List[Dict[str, Any]], expected_docs: List[str], expected_keywords: List[str], top_k: int) -> Dict[str, Any]:
    selected = list(items[: max(1, top_k)])
    doc_blob = "\n".join(
        f"{item.get('document_title', '')} {item.get('document_source', '')}".strip()
        for item in selected
    )
    kw_blob = "\n".join(str(item.get("content") or item.get("snippet") or "") for item in selected)
    return {
        "doc_hit": 1 if _contains_any(doc_blob, expected_docs) else 0,
        "keyword_recall": _kw_recall(kw_blob, expected_keywords),
        "titles": [
            str(item.get("document_title") or item.get("document_source") or "").strip()
            for item in selected
            if str(item.get("document_title") or item.get("document_source") or "").strip()
        ],
        "count": len(selected),
    }


def _select_cases(cases: List[Dict[str, Any]], case_id: str | None, limit: int) -> List[Dict[str, Any]]:
    if case_id:
        selected = [case for case in cases if str(case.get("id") or "") == case_id]
        if not selected:
            raise ValueError(f"case_id not found: {case_id}")
        return selected
    return cases[:limit] if limit > 0 else cases


def _unique(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = _norm(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _extract_target_cues(question: str, expected_docs: List[str]) -> Dict[str, List[str]]:
    title_cues = _unique(list(expected_docs))
    acronym_cues = _unique(
        re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9*]+)+\b|\b[A-Z]{2,}\b", question + " " + " ".join(expected_docs))
    )
    filename_cues = _unique(re.findall(r"\b[\w\-]+\.pdf\b", question, flags=re.IGNORECASE))
    domain_candidates = [
        "usv",
        "ship",
        "ocean environment",
        "marine",
        "path planning",
        "robot",
        "maritime",
        "narrow channel",
        "narrow",
    ]
    method_candidates = [
        "adaptive strategy",
        "adaptive sampling",
        "path smoothing",
        "hybrid sampling",
        "method",
        "algorithm",
        "framework",
        "sampling",
    ]
    section_candidates = [
        "abstract",
        "introduction",
        "results",
        "conclusion",
        "figure",
        "table",
        "algorithm",
        "section",
    ]
    domain_entity_cues = _unique([item for item in domain_candidates if item in _norm(question)])
    method_name_cues = _unique([item for item in method_candidates if item in _norm(question)])
    section_cues = _unique([item for item in section_candidates if item in _norm(question)])
    time_constraint_cues = _unique(
        re.findall(r"\b20\d{2}\b|\b\d+\s*(?:次迭代|iterations?|s|seconds?)\b|time cost", question, flags=re.IGNORECASE)
    )
    return {
        "title": title_cues,
        "acronym": acronym_cues,
        "filename": filename_cues,
        "domain_entity": domain_entity_cues,
        "method_name": method_name_cues,
        "section": section_cues,
        "time_constraint": time_constraint_cues,
    }


def _preserved_cues(cues: Dict[str, List[str]], rewritten_queries: List[str]) -> Dict[str, List[str]]:
    if not rewritten_queries:
        return {key: [] for key in cues}
    blob = _norm("\n".join(rewritten_queries))
    return {
        key: [cue for cue in values if _norm(cue) in blob]
        for key, values in cues.items()
    }


def _cue_preservation_ratio(original_cues: Dict[str, List[str]], preserved_cues: Dict[str, List[str]]) -> float:
    total = sum(len(values) for values in original_cues.values())
    if total == 0:
        return 1.0
    preserved = sum(len(preserved_cues.get(key, [])) for key in original_cues)
    return preserved / total


async def _baseline(question: str, top_k: int) -> Dict[str, Any]:
    rows = await hybrid_search_tool(HybridSearchInput(query=question, limit=top_k))
    raw_hits = [_hit_to_dict(row) for row in rows]
    sources = [_source_from_hit(hit) for hit in raw_hits]
    return {
        "initial_query": question,
        "rewritten_queries": [],
        "raw_hits": raw_hits,
        "sources": sources,
        "attempts": 1,
        "confidence": (raw_hits[0]["score"] if raw_hits else 0.0),
    }


async def _full(question: str, timeout_seconds: int) -> Dict[str, Any]:
    deps = AgentDependencies(session_id=f"eval-loop-v2-{uuid.uuid4().hex[:10]}", user_id="eval")
    result = await asyncio.wait_for(
        run_langgraph_analysis(question=question, deps=deps, context_prompt=""),
        timeout=max(1, timeout_seconds),
    )
    metadata = dict(result.metadata or {})
    sources = [_source_to_dict(item) for item in list(result.sources or [])]
    retrieval_results = [_hit_to_dict(item) for item in list(metadata.get("retrieval_results") or [])]
    retrieval_attempts = [dict(item) for item in list(metadata.get("retrieval_attempts") or []) if isinstance(item, dict)]
    initial_query = str((retrieval_attempts[0] or {}).get("query") or question) if retrieval_attempts else question
    return {
        "initial_query": initial_query,
        "rewritten_queries": [str(item) for item in list(metadata.get("rewritten_queries") or []) if str(item or "").strip()],
        "raw_hits": retrieval_results,
        "sources": sources,
        "retrieval_attempts": retrieval_attempts,
        "attempts": int(metadata.get("retrieval_attempt_count") or 1),
        "rewrite_used": bool((metadata.get("rewritten_queries") or [])),
        "confidence": float(metadata.get("retrieval_top_score") or 0.0),
        "metadata": metadata,
    }


async def run_suite(
    cases: List[Dict[str, Any]],
    limit: int,
    timeout_seconds: int,
    top_k: int,
    case_id: str | None = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    rewrite_triggered_count = 0
    cue_preservation_sum = 0.0
    target_doc_preserved_count = 0
    initial_doc_hit_count = 0
    final_doc_hit_count = 0
    initial_sufficient_count = 0
    rewrite_necessary_count = 0
    rewrite_unnecessary_triggered_count = 0
    rewrite_cue_drop_count = 0
    attempts_sum = 0.0
    timeout_count = 0

    for case in _select_cases(cases, case_id=case_id, limit=limit):
        question = str(case.get("question") or "")
        expected_docs = list(case.get("expected_document_keywords") or [])
        expected_keywords = list(case.get("expected_content_keywords") or [])
        original_target_cues = _extract_target_cues(question, expected_docs)

        baseline = await _baseline(question, top_k)
        baseline_hits_eval = _evaluate_items(baseline["raw_hits"], expected_docs, expected_keywords, top_k)
        initial_doc_hit = int(baseline_hits_eval["doc_hit"])
        initial_kw_recall = float(baseline_hits_eval["keyword_recall"])
        initial_retrieval_sufficient = 1 if (initial_doc_hit == 1 and initial_kw_recall >= 0.6) else 0
        rewrite_necessary = 0 if initial_retrieval_sufficient == 1 else 1
        initial_sufficient_count += initial_retrieval_sufficient
        rewrite_necessary_count += rewrite_necessary
        initial_doc_hit_count += initial_doc_hit

        failure_reason = None
        try:
            full = await _full(question, timeout_seconds)
            full_hits_eval = _evaluate_items(full["raw_hits"], expected_docs, expected_keywords, top_k)
            full_sources_eval = _evaluate_items(full["sources"], expected_docs, expected_keywords, top_k)
            preserved_cues = _preserved_cues(original_target_cues, full["rewritten_queries"])
            cue_preservation_ratio = _cue_preservation_ratio(original_target_cues, preserved_cues)
            full_doc_hit = int(full_hits_eval["doc_hit"])
            final_doc_hit_count += full_doc_hit

            rewrite_triggered = 1 if full["rewrite_used"] else 0
            rewrite_triggered_count += rewrite_triggered
            if full["rewrite_used"]:
                cue_preservation_sum += cue_preservation_ratio
                if cue_preservation_ratio < 1.0:
                    rewrite_cue_drop_count += 1
            if rewrite_triggered == 1 and rewrite_necessary == 0:
                rewrite_unnecessary_triggered_count += 1
            
            # Target doc preserved: 如果初始命中了，改写后是否依然命中
            target_doc_preserved = 1 if (initial_doc_hit == 1 and full_doc_hit == 1) else (1 if initial_doc_hit == 0 else 0)
            if initial_doc_hit == 1 and full_doc_hit == 1:
                target_doc_preserved_count += 1
            elif initial_doc_hit == 0:
                # 如果初始没命中，不计入 preserved 统计的分母，或者另行处理
                pass

            attempts_sum += float(full["attempts"])
            
            if full_doc_hit == 0:
                failure_reason = "No target document hit after retrieval loop"

            rows.append(
                {
                    "id": case.get("id"),
                    "question": question,
                    "initial_retrieval_sufficient": initial_retrieval_sufficient,
                    "rewrite_necessary": rewrite_necessary,
                    "rewrite_triggered": rewrite_triggered,
                    "rewritten_query_dropped_target_cues": 1 if (rewrite_triggered == 1 and cue_preservation_ratio < 1.0) else 0,
                    "cue_preservation_ratio": cue_preservation_ratio,
                    "initial_doc_hit": initial_doc_hit,
                    "final_doc_hit": full_doc_hit,
                    "target_doc_preserved": 1 if (initial_doc_hit == 1 and full_doc_hit == 1) else 0,
                    "attempts": full["attempts"],
                    "failure_reason": failure_reason,
                    "rewritten_queries": full["rewritten_queries"],
                    # 诊断数据
                    "retrieval_diagnostic": {
                        "raw_hits": full["raw_hits"][:top_k],
                        "kw_recall": full_hits_eval["keyword_recall"]
                    },
                    "answer_stage_diagnostic": {
                        "sources": full["sources"][:top_k],
                        "kw_recall": full_sources_eval["keyword_recall"]
                    }
                }
            )
        except Exception as exc:
            timeout_count += 1 if "Timeout" in type(exc).__name__ else 0
            rows.append(
                {
                    "id": case.get("id"),
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                    "failure_reason": "Execution error or timeout"
                }
            )

    total = max(1, len(rows))
    return {
        "summary": {
            "total_cases": len(rows),
            "rewrite_triggered_rate": rewrite_triggered_count / total,
            "avg_cue_preservation_ratio": cue_preservation_sum / max(1, rewrite_triggered_count),
            "target_doc_retention_rate": target_doc_preserved_count / max(1, initial_doc_hit_count),
            "initial_retrieval_sufficient_rate": initial_sufficient_count / total,
            "rewrite_necessary_rate": rewrite_necessary_count / total,
            "rewrite_unnecessary_triggered_rate": rewrite_unnecessary_triggered_count / total,
            "rewrite_cue_drop_rate": rewrite_cue_drop_count / max(1, rewrite_triggered_count),
            "final_success_rate": final_doc_hit_count / total,
            "avg_attempts": attempts_sum / total,
            "timeout_rate": timeout_count / total,
        },
        "cases": rows,
    }


def to_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Retrieval Loop Diagnostics",
        "",
        "- 目标：诊断检索循环（重写/重试）是否有效保留了原始意图，并观察其对召回的影响。",
        "",
        f"- total_cases: {s['total_cases']}",
        f"- rewrite_triggered_rate: {s['rewrite_triggered_rate']:.3f}",
        f"- avg_cue_preservation_ratio: {s['avg_cue_preservation_ratio']:.3f}",
        f"- target_doc_retention_rate: {s['target_doc_retention_rate']:.3f}",
        f"- initial_retrieval_sufficient_rate: {s['initial_retrieval_sufficient_rate']:.3f}",
        f"- rewrite_necessary_rate: {s['rewrite_necessary_rate']:.3f}",
        f"- rewrite_unnecessary_triggered_rate: {s['rewrite_unnecessary_triggered_rate']:.3f}",
        f"- rewrite_cue_drop_rate: {s['rewrite_cue_drop_rate']:.3f}",
        f"- final_success_rate: {s['final_success_rate']:.3f}",
        f"- avg_attempts: {s['avg_attempts']:.3f}",
        "",
        "## Diagnostic Details",
        "",
        "| ID | Init Suff. | Rewrite Need | Rewrite | Cue Drop | Cue Pres. | Init Hit | Final Hit | Retained | Attempts | Failure Reason |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in report.get("cases", []):
        if "error" in c:
            lines.append(f"| {c['id']} | - | - | ERROR | - | - | - | - | - | - | {c['failure_reason']} |")
        else:
            lines.append(
                f"| {c['id']} | {c['initial_retrieval_sufficient']} | {c['rewrite_necessary']} | {c['rewrite_triggered']} | {c['rewritten_query_dropped_target_cues']} | {c['cue_preservation_ratio']:.3f} | {c['initial_doc_hit']} | {c['final_doc_hit']} | {c['target_doc_preserved']} | {c['attempts']} | {c['failure_reason'] or 'None'} |"
            )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evals/cases/retrieval_loop_recovery_cases.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output-dir", default="evals/results")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    await initialize_database()
    try:
        report = await run_suite(
            read_json_robust(Path(args.cases)),
            int(args.limit or 0),
            int(args.timeout_seconds or 120),
            int(args.top_k or 5),
            case_id=str(args.case_id or "").strip() or None,
        )
    finally:
        await close_database()

    write_json_utf8(out / "retrieval_loop_recovery_eval.json", report, indent=2)
    write_text_utf8(out / "retrieval_loop_recovery_eval.md", to_markdown(report))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
