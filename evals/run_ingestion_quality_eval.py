from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.db_utils import close_database, initialize_database
from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8
from evals.ingestion_eval_runtime import quality_report_to_markdown, run_ingestion_quality_suite


def _load_document_ids(path_value: str | None) -> list[str] | None:
    if not path_value:
        return None
    payload = read_json_robust(Path(path_value))
    if not isinstance(payload, list):
        raise ValueError("document ids file must be a JSON array")
    return [str(item).strip() for item in payload if str(item).strip()]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="evals/cases/ingestion_quality_manifest.json")
    parser.add_argument("--output-dir", default="evals/results")
    parser.add_argument(
        "--document-ids-file",
        default="",
        help="Optional JSON file containing a document id array for scoped ingestion evaluation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json_robust(Path(args.manifest))
    document_ids = _load_document_ids(args.document_ids_file)

    await initialize_database()
    try:
        report = await run_ingestion_quality_suite(manifest=manifest, document_ids=document_ids)
    finally:
        await close_database()

    write_json_utf8(output_dir / "ingestion_quality_eval.json", report, indent=2)
    write_text_utf8(output_dir / "ingestion_quality_eval.md", quality_report_to_markdown(report))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
