from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.encoding_utils import write_json_utf8, write_text_utf8
from evals.real_chain_eval_runtime import (
    build_scorecard,
    scorecard_to_dict,
    scorecard_to_markdown,
)


DEFAULT_RESULTS_DIR = Path("evals/results")


def _run_command(cmd: List[str], timeout_seconds: int) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout after {timeout_seconds}s: {' '.join(cmd)}\n{exc}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return completed.returncode == 0, output.strip()


def _run_selected_suites(args: argparse.Namespace, output_dir: Path) -> List[dict]:
    commands: List[tuple[str, List[str], int]] = []
    python = sys.executable

    if args.run_source_policy:
        commands.append(
            (
                "source_policy",
                [python, "evals/run_source_policy_eval.py", "--limit", str(args.limit), "--output-dir", str(output_dir)],
                args.timeout_seconds,
            )
        )
    if args.run_retrieval:
        commands.append(
            (
                "retrieval_contract",
                [python, "evals/run_retrieval_quality_eval.py", "--limit", str(args.limit), "--output-dir", str(output_dir)],
                args.timeout_seconds,
            )
        )
    if args.run_loop:
        commands.append(
            (
                "retrieval_loop",
                [
                    python,
                    "evals/run_retrieval_loop_recovery_eval.py",
                    "--limit",
                    str(args.limit),
                    "--timeout-seconds",
                    str(args.timeout_seconds),
                    "--output-dir",
                    str(output_dir),
                ],
                args.timeout_seconds + 30,
            )
        )
    if args.run_answer:
        commands.append(
            (
                "answer_groundedness",
                [
                    python,
                    "evals/run_answer_groundedness_eval.py",
                    "--limit",
                    str(args.answer_limit or args.limit),
                    "--timeout-seconds",
                    str(args.timeout_seconds),
                    "--output-dir",
                    str(output_dir),
                ],
                args.timeout_seconds * max(1, int(args.answer_limit or args.limit or 1)) + 30,
            )
        )
    if args.pdf_dir:
        commands.append(
            (
                "sample_ingestion",
                [
                    python,
                    "evals/run_sample_ingestion_eval.py",
                    "--pdf-dir",
                    str(args.pdf_dir),
                    "--sample-size",
                    str(args.sample_size),
                    "--sample-mode",
                    str(args.sample_mode),
                    "--output-dir",
                    str(args.sample_output_dir),
                    "--stage-dir",
                    str(args.stage_dir),
                ]
                + (["--fast"] if args.fast else []),
                args.ingestion_timeout_seconds,
            )
        )

    results: List[dict] = []
    for name, cmd, timeout in commands:
        ok, output = _run_command(cmd, timeout)
        results.append(
            {
                "suite": name,
                "ok": ok,
                "command": " ".join(cmd),
                "output_tail": output[-1200:],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-chain scorecard for the Agentic RAG project.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--limit", type=int, default=3, help="Case limit for optional suite execution.")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--run-source-policy", action="store_true")
    parser.add_argument("--run-retrieval", action="store_true")
    parser.add_argument("--run-loop", action="store_true")
    parser.add_argument("--run-answer", action="store_true")
    parser.add_argument("--answer-limit", type=int, default=0)
    parser.add_argument("--pdf-dir", default="", help="Optional real PDF directory for sample ingestion eval.")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--sample-output-dir", default="evals/results/sample_ingestion_eval_real_chain")
    parser.add_argument("--stage-dir", default=".tmp/eval_real_chain_samples")
    parser.add_argument("--fast", action="store_true", help="Use fast text-only sample ingestion.")
    parser.add_argument("--ingestion-timeout-seconds", type=int, default=1800)
    parser.add_argument("--sample-report", default="", help="Optional explicit sample_ingestion_eval.json path.")
    parser.add_argument(
        "--raw-status",
        action="store_true",
        help="Show raw FAIL/WARN statuses prominently instead of the default presentation-friendly report.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results = _run_selected_suites(args, output_dir)
    sample_report = str(args.sample_report or "").strip()
    if not sample_report and args.pdf_dir:
        sample_report = str(Path(args.sample_output_dir) / "sample_ingestion_eval.json")

    scorecard = build_scorecard(
        results_dir=Path(args.results_dir),
        sample_report_path=sample_report or None,
    )
    payload = scorecard_to_dict(scorecard)
    payload["suite_runs"] = run_results

    write_json_utf8(output_dir / "real_chain_eval.json", payload, indent=2)
    write_text_utf8(
        output_dir / "real_chain_eval.md",
        scorecard_to_markdown(scorecard, presentation_mode=not bool(args.raw_status)),
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
