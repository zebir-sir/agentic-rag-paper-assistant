from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8
from evals.run_corpus_readiness_eval import percentile, request_json


def norm(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def keyword_recall(results: list[dict[str, Any]], keywords: list[str]) -> float:
    expected = [norm(item) for item in keywords if norm(item)]
    if not expected:
        return 1.0
    blob = "\n".join(norm(result.get("content")) for result in results)
    return sum(1 for keyword in expected if keyword in blob) / len(expected)


def section_hit(results: list[dict[str, Any]], keywords: list[str]) -> bool:
    expected = [norm(item) for item in keywords if norm(item)]
    if not expected:
        return True
    blob = "\n".join(norm((result.get("metadata") or {}).get("section_path_text")) for result in results)
    return any(keyword in blob for keyword in expected)


def run_cases(api_url: str, cases: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        response = request_json(f"{api_url}/search/hybrid", {"query": str(case["question"]), "limit": top_k})
        latency_ms = (time.perf_counter() - started) * 1000.0
        rows = list(response.get("results") or [])
        ids: list[str] = []
        for row in rows:
            document_id = str(row.get("document_id") or "")
            if document_id and document_id not in ids:
                ids.append(document_id)
        expected_ids = [str(item) for item in case.get("expected_document_ids") or []]
        rank = next((ids.index(item) + 1 for item in expected_ids if item in ids), None)
        latencies.append(latency_ms)
        outputs.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_document_ids": expected_ids,
                "expected_document_title": case.get("expected_document_title"),
                "rationale": case.get("rationale"),
                "rank": rank,
                "document_hit": rank is not None,
                "section_hint_hit": section_hit(rows, list(case.get("expected_section_keywords") or [])),
                "content_keyword_recall": keyword_recall(rows, list(case.get("expected_content_keywords") or [])),
                "retrieved_document_ids": ids,
                "latency_ms": round(latency_ms, 1),
                "api_query_time_ms": round(float(response.get("query_time_ms") or 0.0), 1),
            }
        )

    total = len(outputs)
    return {
        "protocol": {
            "annotation_type": "single_expert_constructed",
            "retrieval_mode": "hybrid",
            "top_k": top_k,
            "metrics": "Document Hit@K, MRR@K, section-hint hit rate, and evidence-keyword recall.",
            "limit": "No inter-annotator agreement is available. The set measures retrieval relevance only, not final answer correctness.",
        },
        "summary": {
            "cases": total,
            "document_hit_at_1": sum(1 for item in outputs if item["rank"] == 1) / max(1, total),
            "document_hit_at_k": sum(1 for item in outputs if item["document_hit"]) / max(1, total),
            "mrr_at_k": sum(1 / int(item["rank"]) for item in outputs if item["rank"]) / max(1, total),
            "section_hint_hit_rate": sum(1 for item in outputs if item["section_hint_hit"]) / max(1, total),
            "mean_content_keyword_recall": sum(float(item["content_keyword_recall"]) for item in outputs) / max(1, total),
            "latency_ms_p50": percentile(latencies, 0.5),
            "latency_ms_p90": percentile(latencies, 0.9),
        },
        "cases": outputs,
    }


def to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    top_k = report["protocol"]["top_k"]
    lines = [
        "# Expert-Gold Retrieval Evaluation",
        "",
        "- Annotation type: single-expert constructed gold set.",
        "- Scope: Chinese research questions against the existing 46-paper local corpus.",
        "- Boundary: retrieval relevance only. It is not a multi-annotator benchmark or an answer-correctness score.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Cases | {summary['cases']} |",
        f"| Document Hit@1 | {summary['document_hit_at_1']:.1%} |",
        f"| Document Hit@{top_k} | {summary['document_hit_at_k']:.1%} |",
        f"| MRR@{top_k} | {summary['mrr_at_k']:.3f} |",
        f"| Section-hint Hit@{top_k} | {summary['section_hint_hit_rate']:.1%} |",
        f"| Evidence-keyword Recall@{top_k} | {summary['mean_content_keyword_recall']:.1%} |",
        f"| Latency P50 / P90 | {summary['latency_ms_p50']:.0f} / {summary['latency_ms_p90']:.0f} ms |",
        "",
        "## Case Audit",
        "",
        "| Case | Target | Rank | Section Hint | Keyword Recall |",
        "|---|---|---:|---:|---:|",
    ]
    for item in report["cases"]:
        rank = item["rank"] if item["rank"] is not None else "miss"
        lines.append(f"| {item['id']} | {item['expected_document_title']} | {rank} | {'yes' if item['section_hint_hit'] else 'no'} | {item['content_keyword_recall']:.1%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-expert gold retrieval evaluation.")
    parser.add_argument("--api-url", default="http://localhost:8059")
    parser.add_argument("--cases", default="evals/cases/corpus_46_expert_gold_cases.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="evals/results/corpus_46")
    args = parser.parse_args()
    payload = read_json_robust(Path(args.cases))
    cases = list(payload.get("cases") or [])
    report = run_cases(str(args.api_url).rstrip("/"), cases, max(1, int(args.top_k)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_utf8(output_dir / "expert_gold_retrieval_eval.json", report, indent=2)
    write_text_utf8(output_dir / "expert_gold_retrieval_eval.md", to_markdown(report))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
