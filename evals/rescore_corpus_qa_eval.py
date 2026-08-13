"""Recompute QA contracts after an evaluator-only scoring-rule correction."""
from __future__ import annotations

import argparse
from pathlib import Path

from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8
from evals.corpus_qa_eval_runtime import _score_case, markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    progress_path = output_dir / "qa_100_progress.json"
    progress = read_json_robust(progress_path)
    cases = list(progress.get("cases") or [])
    for row in cases:
        if row.get("error"):
            continue
        row["score"] = _score_case(
            row,
            str(row.get("answer") or ""),
            list(row.get("sources") or []),
            list(row.get("tools_executed") or []),
            dict(row.get("metadata") or {}),
        )
    progress["cases"] = cases
    write_json_utf8(progress_path, progress, indent=2)
    summary = {
        "cases": len(cases),
        "completed": sum(not row.get("error") for row in cases),
        "errors": sum(bool(row.get("error")) for row in cases),
        "answer_contract_pass_rate": sum(bool((row.get("score") or {}).get("contract_pass")) for row in cases) / max(1, len(cases)),
    }
    report = {
        "protocol": {"annotation": "rescored after evaluator-only web-boundary rule correction"},
        "summary": summary,
        "cases": cases,
    }
    write_json_utf8(output_dir / "qa_100_answer_eval.json", report, indent=2)
    write_text_utf8(output_dir / "qa_100_answer_eval.md", markdown(report))
    print(summary)


if __name__ == "__main__":
    main()
