from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.encoding_utils import read_json_robust
from evals.ingestion_eval_runtime import run_sample_ingestion_evaluation


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a scoped ingestion evaluation on sampled PDFs.")
    parser.add_argument("--pdf-dir", required=True, help="Source directory containing PDFs to sample.")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of PDFs to sample.")
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="first",
        help="How to select the sample PDFs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when sample-mode=random.")
    parser.add_argument("--manifest", default="evals/cases/ingestion_quality_manifest.json")
    parser.add_argument("--output-dir", default="evals/results/sample_ingestion_eval")
    parser.add_argument("--stage-dir", default=".tmp/eval_ingestion_samples")
    parser.add_argument("--chunk-size", type=int, default=850)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast text-only ingestion: disables semantic chunking, image extraction, and table extraction.",
    )
    args = parser.parse_args()

    manifest = read_json_robust(Path(args.manifest))
    report = await run_sample_ingestion_evaluation(
        pdf_dir=args.pdf_dir,
        sample_size=args.sample_size,
        sample_mode=args.sample_mode,
        seed=args.seed,
        manifest=manifest,
        output_dir=args.output_dir,
        stage_root=args.stage_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        fast=bool(args.fast),
    )
    print(json.dumps(report["quality"]["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
