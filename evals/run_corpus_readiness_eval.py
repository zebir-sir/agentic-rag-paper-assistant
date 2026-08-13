from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}: {exc.read().decode('utf-8', errors='replace')[:300]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def compact_title(title: str) -> str:
    return " ".join(str(title or "").replace("â", "").replace("*", " ").split())


def run_title_retrieval(api_url: str, documents: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    latencies: list[float] = []
    for index, document in enumerate(documents, start=1):
        title = compact_title(str(document.get("title") or ""))
        if len(title) < 8:
            cases.append({"document_id": document.get("id"), "title": title, "status": "skipped_short_title"})
            continue
        started = time.perf_counter()
        try:
            response = request_json(
                f"{api_url}/search/hybrid",
                {"query": title, "limit": top_k},
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            ranks: list[str] = []
            for row in response.get("results") or []:
                result_id = str(row.get("document_id") or "")
                if result_id and result_id not in ranks:
                    ranks.append(result_id)
            expected_id = str(document.get("id") or "")
            rank = ranks.index(expected_id) + 1 if expected_id in ranks else None
            cases.append(
                {
                    "document_id": expected_id,
                    "title": title,
                    "status": "ok",
                    "rank": rank,
                    "retrieved_document_ids": ranks,
                    "latency_ms": round(latency_ms, 1),
                    "api_query_time_ms": round(float(response.get("query_time_ms") or 0.0), 1),
                }
            )
        except RuntimeError as exc:
            cases.append({"document_id": document.get("id"), "title": title, "status": "error", "error": str(exc)})

    completed = [item for item in cases if item.get("status") == "ok"]
    ranks = [int(item["rank"]) for item in completed if item.get("rank")]
    return {
        "protocol": {
            "name": "known_document_title_retrieval",
            "description": "For each indexed paper, use its stored title as a known-item query and check whether hybrid retrieval returns that document.",
            "scope": "Measures document localization in this exact 46-paper corpus. It does not measure open-ended answer correctness or cross-paper reasoning.",
            "top_k": top_k,
        },
        "summary": {
            "attempted": len(cases),
            "completed": len(completed),
            "errors": sum(1 for item in cases if item.get("status") == "error"),
            "skipped": sum(1 for item in cases if item.get("status", "").startswith("skipped")),
            "hit_at_1": sum(1 for item in completed if item.get("rank") == 1) / max(1, len(completed)),
            "hit_at_k": sum(1 for item in completed if item.get("rank")) / max(1, len(completed)),
            "mrr_at_k": sum(1 / int(item["rank"]) for item in completed if item.get("rank")) / max(1, len(completed)),
            "latency_ms_p50": percentile(latencies, 0.5),
            "latency_ms_p90": percentile(latencies, 0.9),
        },
        "cases": cases,
    }


def markdown(report: dict[str, Any]) -> str:
    corpus = report["corpus"]
    ingestion = report.get("ingestion") or {}
    retrieval = report["retrieval"]
    summary = retrieval["summary"]
    lines = [
        "# 46-Paper Corpus Readiness Evaluation",
        "",
        f"- generated_at_utc: {report['generated_at_utc']}",
        "- evaluation_scope: existing indexed corpus only; no document was re-ingested.",
        "",
        "## Corpus And Graph",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Indexed documents | {corpus['documents']} |",
        f"| Indexed chunks | {corpus['chunks']} |",
        f"| Mean chunks per paper | {corpus['mean_chunks_per_document']:.1f} |",
        f"| Graph node coverage | {corpus['graph_node_coverage']:.1%} |",
        f"| Graph edges | {corpus['graph_edges']} |",
        f"| Mean graph degree | {corpus['mean_graph_degree']:.1f} |",
        "",
        "## Ingestion Integrity",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Section metadata coverage | {float(ingestion.get('section_metadata_coverage') or 0):.1%} |",
        f"| Line metadata coverage | {float(ingestion.get('line_metadata_coverage') or 0):.1%} |",
        f"| Empty chunks | {int(ingestion.get('empty_chunk_count') or 0)} |",
        f"| Artifact chunks | {int(ingestion.get('artifact_chunk_count') or 0)} |",
        f"| Figure / table / algorithm chunks | {int(ingestion.get('figure_chunk_count') or 0)} / {int(ingestion.get('table_chunk_count') or 0)} / {int(ingestion.get('algorithm_chunk_count') or 0)} |",
        "",
        "## Known-Document Retrieval",
        "",
        f"Protocol: each paper title is issued as a hybrid-search query against the same 46-paper corpus; the target is its persisted document ID. This is a document-localization metric, not an open-ended QA benchmark.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Completed / attempted | {summary['completed']} / {summary['attempted']} |",
        f"| Hit@1 | {summary['hit_at_1']:.1%} |",
        f"| Hit@{retrieval['protocol']['top_k']} | {summary['hit_at_k']:.1%} |",
        f"| MRR@{retrieval['protocol']['top_k']} | {summary['mrr_at_k']:.3f} |",
        f"| End-to-end latency P50 / P90 | {summary['latency_ms_p50']:.0f} / {summary['latency_ms_p90']:.0f} ms |",
        "",
        "## Interview-Safe Interpretation",
        "",
        "- The corpus has been evaluated as an operational evidence store: structured chunks, artifacts, a connected paper graph, and repeatable known-item retrieval.",
        "- Do not present Hit@K as answer accuracy. Open-ended answers still need a manually curated question-answer set with source-level relevance labels and groundedness review.",
        "- The JSON companion contains every query, expected document ID, returned document IDs, rank, and latency for audit and reruns.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the existing paper corpus without re-ingestion.")
    parser.add_argument("--api-url", default="http://localhost:8059")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ingestion-report", default="evals/results/corpus_46/ingestion_quality_eval.json")
    parser.add_argument("--output-dir", default="evals/results/corpus_46")
    args = parser.parse_args()

    api_url = str(args.api_url).rstrip("/")
    documents_response = request_json(f"{api_url}/documents?limit=100")
    documents = list(documents_response.get("documents") or [])
    graph = request_json(f"{api_url}/paper-graph")
    ingestion_path = Path(args.ingestion_report)
    ingestion_payload = read_json_robust(ingestion_path) if ingestion_path.exists() else {}
    ingestion = dict(ingestion_payload.get("summary") or {})
    document_count = len(documents)
    chunk_count = sum(int(document.get("chunk_count") or 0) for document in documents)
    graph_nodes = list(graph.get("nodes") or [])
    graph_edges = list(graph.get("edges") or [])
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "documents": document_count,
            "chunks": chunk_count,
            "mean_chunks_per_document": chunk_count / max(1, document_count),
            "graph_nodes": len(graph_nodes),
            "graph_edges": len(graph_edges),
            "graph_node_coverage": len(graph_nodes) / max(1, document_count),
            "mean_graph_degree": (2 * len(graph_edges)) / max(1, len(graph_nodes)),
        },
        "ingestion": ingestion,
        "retrieval": run_title_retrieval(api_url, documents, max(1, args.top_k)),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_utf8(output_dir / "corpus_readiness_eval.json", report, indent=2)
    write_text_utf8(output_dir / "corpus_readiness_eval.md", markdown(report))
    print(json.dumps({"corpus": report["corpus"], "retrieval": report["retrieval"]["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
