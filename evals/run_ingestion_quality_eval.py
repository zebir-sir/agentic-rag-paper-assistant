from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.db_utils import close_database, db_pool, initialize_database
from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8


async def run_suite(manifest: Dict[str, Any]) -> Dict[str, Any]:
    tiny_threshold = int(manifest.get("tiny_chunk_threshold", 80))
    async with db_pool.acquire() as conn:
        total_documents = await conn.fetchval("SELECT COUNT(*) FROM documents")
        total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        cov = await conn.fetchrow("SELECT AVG(CASE WHEN COALESCE(metadata->>'section_title','')<>'' THEN 1.0 ELSE 0.0 END) AS section_cov, AVG(CASE WHEN (metadata ? 'section_start_line') AND (metadata ? 'section_end_line') THEN 1.0 ELSE 0.0 END) AS line_cov FROM chunks")
        art = await conn.fetchrow("SELECT COUNT(*) FILTER (WHERE COALESCE(metadata->>'content_type','')='artifact') AS artifact_chunk_count, COUNT(*) FILTER (WHERE COALESCE(metadata->>'artifact_type','')='table') AS table_chunk_count, COUNT(*) FILTER (WHERE COALESCE(metadata->>'artifact_type','')='figure') AS figure_chunk_count, COUNT(*) FILTER (WHERE COALESCE(metadata->>'artifact_type','')='algorithm') AS algorithm_chunk_count, COUNT(*) FILTER (WHERE COALESCE(metadata->>'content_type','')='artifact' AND COALESCE(metadata->>'context_before','')<>'' AND COALESCE(metadata->>'context_after','')<>'') AS artifact_context_coverage_count FROM chunks")
        size = await conn.fetchrow("SELECT COUNT(*) FILTER (WHERE LENGTH(TRIM(COALESCE(content,'')))=0) AS empty_chunk_count, COUNT(*) FILTER (WHERE LENGTH(TRIM(COALESCE(content,''))) < $1) AS tiny_chunk_count, percentile_cont(0.5) WITHIN GROUP (ORDER BY LENGTH(COALESCE(content,''))) AS p50, percentile_cont(0.9) WITHIN GROUP (ORDER BY LENGTH(COALESCE(content,''))) AS p90, MAX(LENGTH(COALESCE(content,''))) AS max_len FROM chunks", tiny_threshold)
        largest_rows = await conn.fetch(
            """
            SELECT
                c.id::text AS chunk_id,
                d.title AS document_title,
                d.source AS document_source,
                LENGTH(COALESCE(c.content, '')) AS chunk_size,
                COALESCE(c.metadata->>'section_title', '') AS section_title,
                COALESCE(c.metadata->>'section_path_text', '') AS section_path_text,
                COALESCE(c.metadata->>'artifact_type', '') AS artifact_type,
                COALESCE(c.metadata->>'chunk_method', '') AS chunk_method
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY LENGTH(COALESCE(c.content, '')) DESC, c.id ASC
            LIMIT 5
            """
        )

    total_chunks = int(total_chunks or 0)
    tiny = int(size["tiny_chunk_count"] or 0)
    report = {
        "summary": {
            "total_documents": int(total_documents or 0),
            "total_chunks": total_chunks,
            "section_metadata_coverage": float(cov["section_cov"] or 0.0),
            "line_metadata_coverage": float(cov["line_cov"] or 0.0),
            "artifact_chunk_count": int(art["artifact_chunk_count"] or 0),
            "table_chunk_count": int(art["table_chunk_count"] or 0),
            "figure_chunk_count": int(art["figure_chunk_count"] or 0),
            "algorithm_chunk_count": int(art["algorithm_chunk_count"] or 0),
            "artifact_context_coverage": (int(art["artifact_context_coverage_count"] or 0) / max(1, int(art["artifact_chunk_count"] or 0))),
            "empty_chunk_count": int(size["empty_chunk_count"] or 0),
            "tiny_chunk_rate": (tiny / total_chunks) if total_chunks else 0.0,
            "chunk_size_p50": float(size["p50"] or 0.0),
            "chunk_size_p90": float(size["p90"] or 0.0),
            "chunk_size_max": int(size["max_len"] or 0),
            "tiny_chunk_threshold": tiny_threshold,
        },
        "largest_chunks": [
            {
                "chunk_id": str(row["chunk_id"] or ""),
                "document_title": str(row["document_title"] or ""),
                "document_source": str(row["document_source"] or ""),
                "chunk_size": int(row["chunk_size"] or 0),
                "section_title": str(row["section_title"] or ""),
                "section_path_text": str(row["section_path_text"] or ""),
                "artifact_type": str(row["artifact_type"] or ""),
                "chunk_method": str(row["chunk_method"] or ""),
            }
            for row in largest_rows
        ],
    }
    return report


def to_markdown(report: Dict[str, Any]) -> str:
    s=report["summary"]
    lines=["# Ingestion Integrity Eval","",f"- total_documents: {s['total_documents']}",f"- total_chunks: {s['total_chunks']}",f"- section_metadata_coverage: {s['section_metadata_coverage']:.3f}",f"- line_metadata_coverage: {s['line_metadata_coverage']:.3f}",f"- artifact_chunk_count: {s['artifact_chunk_count']}",f"- table_chunk_count: {s['table_chunk_count']}",f"- figure_chunk_count: {s['figure_chunk_count']}",f"- algorithm_chunk_count: {s['algorithm_chunk_count']}",f"- artifact_context_coverage: {s['artifact_context_coverage']:.3f}",f"- empty_chunk_count: {s['empty_chunk_count']}",f"- tiny_chunk_rate: {s['tiny_chunk_rate']:.3f}",f"- chunk_size_p50 / p90 / max: {s['chunk_size_p50']:.1f} / {s['chunk_size_p90']:.1f} / {s['chunk_size_max']}"]
    largest = list(report.get("largest_chunks") or [])
    if largest:
        lines.extend(
            [
                "",
                "## Largest Chunks Top 5",
                "",
                "| Chunk ID | Document Title | Document Source | Chunk Size | Section Title | Section Path | Artifact Type | Chunk Method |",
                "|---|---|---|---:|---|---|---|---|",
            ]
        )
        for item in largest:
            lines.append(
                "| {chunk_id} | {document_title} | {document_source} | {chunk_size} | {section_title} | {section_path_text} | {artifact_type} | {chunk_method} |".format(
                    chunk_id=str(item.get("chunk_id") or "").replace("|", "/"),
                    document_title=str(item.get("document_title") or "").replace("|", "/"),
                    document_source=str(item.get("document_source") or "").replace("|", "/"),
                    chunk_size=int(item.get("chunk_size") or 0),
                    section_title=str(item.get("section_title") or "").replace("|", "/"),
                    section_path_text=str(item.get("section_path_text") or "").replace("|", "/"),
                    artifact_type=str(item.get("artifact_type") or "").replace("|", "/"),
                    chunk_method=str(item.get("chunk_method") or "").replace("|", "/"),
                )
            )
    return "\n".join(lines)+"\n"

async def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--manifest", default="evals/cases/ingestion_quality_manifest.json"); p.add_argument("--output-dir", default="evals/results"); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest=read_json_robust(Path(a.manifest))
    await initialize_database()
    try: report=await run_suite(manifest)
    finally: await close_database()
    write_json_utf8(out/"ingestion_quality_eval.json",report,indent=2); write_text_utf8(out/"ingestion_quality_eval.md",to_markdown(report)); print(json.dumps(report["summary"],ensure_ascii=False,indent=2))

if __name__=="__main__": asyncio.run(main())
