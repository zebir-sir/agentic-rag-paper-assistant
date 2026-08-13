from __future__ import annotations
import argparse, asyncio
from pathlib import Path
from agent.db_utils import close_database, initialize_database
from common.encoding_utils import write_json_utf8, write_text_utf8
from evals.corpus_qa_eval_runtime import build_cases, markdown, run_cases

async def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--output-dir', default='evals/results/overnight_20260813'); parser.add_argument('--timeout-seconds', type=int, default=150); parser.add_argument('--limit', type=int, default=0); parser.add_argument('--categories', default=''); args = parser.parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    await initialize_database()
    try:
        cases = await build_cases()
        selected_categories = {value.strip() for value in str(args.categories).split(',') if value.strip()}
        if selected_categories:
            cases = [case for case in cases if case['category'] in selected_categories]
        if args.limit > 0:
            cases = cases[:args.limit]
        write_json_utf8(output_dir / 'qa_100_gold_cases.json', {'cases': cases}, indent=2)
        report = await run_cases(cases, output_dir, args.timeout_seconds)
    finally:
        await close_database()
    write_json_utf8(output_dir / 'qa_100_answer_eval.json', report, indent=2); write_text_utf8(output_dir / 'qa_100_answer_eval.md', markdown(report))
    print(report['summary'])

if __name__ == '__main__': asyncio.run(main())
