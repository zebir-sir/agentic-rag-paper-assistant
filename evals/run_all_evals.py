from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.encoding_utils import write_text_utf8


def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = cp.returncode == 0
        out = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
        return ok, out.strip()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--skip-answer", action="store_true")
    p.add_argument("--skip-retrieval", action="store_true")
    p.add_argument("--skip-source-policy", action="store_true")
    p.add_argument("--skip-loop", action="store_true")
    p.add_argument("--skip-ingestion", action="store_true")
    p.add_argument("--output-dir", default="evals/results")
    a = p.parse_args()

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suites: Dict[str, Dict[str, str]] = {
        "answer_groundedness_audit": {"cmd": f"python evals/run_answer_groundedness_eval.py --limit {a.limit} --timeout-seconds {a.timeout_seconds} --output-dir {out_dir}"},
        "source_policy": {"cmd": f"python evals/run_source_policy_eval.py --limit {a.limit} --output-dir {out_dir}"},
        "retrieval_contract": {"cmd": f"python evals/run_retrieval_quality_eval.py --limit {a.limit} --output-dir {out_dir}"},
        "retrieval_loop_diagnostics": {"cmd": f"python evals/run_retrieval_loop_recovery_eval.py --limit {a.limit} --timeout-seconds {a.timeout_seconds} --output-dir {out_dir}"},
        "ingestion_integrity": {"cmd": f"python evals/run_ingestion_quality_eval.py --output-dir {out_dir}"},
    }

    skips = {
        "answer_groundedness_audit": a.skip_answer,
        "source_policy": a.skip_source_policy,
        "retrieval_contract": a.skip_retrieval,
        "retrieval_loop_diagnostics": a.skip_loop,
        "ingestion_integrity": a.skip_ingestion,
    }

    summary_rows = []
    started = time.time()
    for name, spec in suites.items():
        if skips.get(name):
            summary_rows.append((name, "SKIPPED", "skip flag", ""))
            continue
        cmd = spec["cmd"].split(" ")
        ok, output = run_cmd(cmd)
        summary_rows.append((name, "PASS" if ok else "FAIL", "", output[-400:]))

    runtime = round(time.time() - started, 3)
    lines = ["# Evaluation Summary", "", f"- runtime_seconds: {runtime}", "", "| Suite | Status | Skip/Reason | Error Summary |", "|---|---|---|---|"]
    for row in summary_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {str(row[3]).replace('|','/')} |")

    write_text_utf8(out_dir / "summary.md", "\n".join(lines) + "\n")
    print(json.dumps({"runtime_seconds": runtime, "suites": [{"name": r[0], "status": r[1]} for r in summary_rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
