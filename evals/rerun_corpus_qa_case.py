"""Rerun one existing corpus-QA case and update its recorded evaluation result."""
from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from pathlib import Path

from agent.agent_langgraph import run_langgraph_analysis
from agent.agent_runtime import AgentDependencies
from common.encoding_utils import read_json_robust, write_json_utf8
from evals.corpus_qa_eval_runtime import _as_dict, _score_case


async def rerun_case(output_dir: Path, case_id: str, timeout_seconds: int) -> dict:
    progress_path = output_dir / "qa_100_progress.json"
    progress = read_json_robust(progress_path)
    rows = list(progress.get("cases") or [])
    normalized_case_id = str(case_id).strip().lower().replace("-", "_")
    target = next(
        (row for row in rows if str(row.get("id")).strip().lower().replace("-", "_") == normalized_case_id),
        None,
    )
    if target is None:
        raise ValueError(f"case not found: {case_id}")

    started = time.perf_counter()
    replacement = {key: value for key, value in target.items() if key not in {"answer", "sources", "tools_executed", "metadata", "score", "error", "latency_seconds"}}
    try:
        result = await asyncio.wait_for(
            run_langgraph_analysis(
                str(target["question"]),
                AgentDependencies(
                    session_id=f"corpus-qa-rerun-{case_id}-{uuid.uuid4().hex[:8]}",
                    user_id="evaluation",
                ),
                context_prompt="",
            ),
            timeout=timeout_seconds,
        )
        sources = [_as_dict(source) for source in result.sources]
        metadata = dict(result.metadata or {})
        tools = [str(item) for item in metadata.get("tools_executed") or []]
        replacement.update(
            {
                "answer": str(result.message or ""),
                "sources": sources,
                "tools_executed": tools,
                "metadata": metadata,
                "score": _score_case(target, result.message, sources, tools, metadata),
                "error": None,
            }
        )
    except Exception as exc:
        replacement["error"] = f"{type(exc).__name__}: {exc}"
    replacement["latency_seconds"] = round(time.perf_counter() - started, 3)

    progress["cases"] = [replacement if row is target else row for row in rows]
    progress["completed"] = sum(not row.get("error") for row in progress["cases"])
    write_json_utf8(progress_path, progress, indent=2)
    return replacement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=150)
    args = parser.parse_args()
    result = asyncio.run(rerun_case(Path(args.output_dir), args.case_id, args.timeout_seconds))
    print({"id": result["id"], "error": result.get("error"), "score": result.get("score"), "latency_seconds": result["latency_seconds"]})


if __name__ == "__main__":
    main()
