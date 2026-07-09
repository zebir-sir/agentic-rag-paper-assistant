from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence
import random

from agent.db_utils import close_database, db_pool, initialize_database
from agent.models import IngestionConfig
from common.encoding_utils import write_json_utf8, write_text_utf8


def select_pdf_files(
    pdf_dir: str | Path,
    sample_size: int,
    sample_mode: str = "first",
    seed: int = 42,
) -> List[Path]:
    root = Path(pdf_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"PDF directory not found or not a directory: {root}")

    pdfs = sorted(path for path in root.rglob("*.pdf") if path.is_file())
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under: {root}")

    safe_size = max(1, int(sample_size))
    if sample_mode == "random":
        rng = random.Random(seed)
        if safe_size >= len(pdfs):
            return pdfs
        return sorted(rng.sample(pdfs, safe_size))

    return pdfs[:safe_size]


def stage_pdf_files(
    selected_files: Sequence[Path],
    source_root: str | Path,
    stage_root: str | Path,
) -> List[Path]:
    source_base = Path(source_root).resolve()
    target_base = Path(stage_root).resolve()
    staged: List[Path] = []

    for file_path in selected_files:
        resolved = Path(file_path).resolve()
        try:
            relative_path = resolved.relative_to(source_base)
        except ValueError:
            relative_path = Path(resolved.name)

        target_path = target_base / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, target_path)
        staged.append(target_path)

    return staged


def _empty_quality_summary(tiny_threshold: int) -> Dict[str, Any]:
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "section_metadata_coverage": 0.0,
        "line_metadata_coverage": 0.0,
        "artifact_chunk_count": 0,
        "table_chunk_count": 0,
        "figure_chunk_count": 0,
        "algorithm_chunk_count": 0,
        "artifact_context_coverage": 0.0,
        "empty_chunk_count": 0,
        "tiny_chunk_rate": 0.0,
        "chunk_size_p50": 0.0,
        "chunk_size_p90": 0.0,
        "chunk_size_max": 0,
        "tiny_chunk_threshold": tiny_threshold,
    }


def _build_scope_sql(document_ids: Sequence[str] | None) -> tuple[str, List[Any]]:
    if document_ids is None:
        return "", []

    normalized_ids = [str(doc_id).strip() for doc_id in list(document_ids) if str(doc_id).strip()]
    if not normalized_ids:
        return "WHERE 1 = 0", []
    return "WHERE d.id = ANY($1::uuid[])", [normalized_ids]


async def run_ingestion_quality_suite(
    manifest: Dict[str, Any],
    document_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    tiny_threshold = int(manifest.get("tiny_chunk_threshold", 80))
    scope_sql, scope_params = _build_scope_sql(document_ids)

    async with db_pool.acquire() as conn:
        total_documents = await conn.fetchval(
            f"SELECT COUNT(*) FROM documents d {scope_sql}",
            *scope_params,
        )

        if not int(total_documents or 0):
            return {
                "summary": _empty_quality_summary(tiny_threshold),
                "largest_chunks": [],
            }

        total_chunks = await conn.fetchval(
            f"""
            SELECT COUNT(*)
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {scope_sql}
            """,
            *scope_params,
        )
        cov = await conn.fetchrow(
            f"""
            SELECT
                AVG(CASE WHEN COALESCE(c.metadata->>'section_title','')<>'' THEN 1.0 ELSE 0.0 END) AS section_cov,
                AVG(CASE WHEN (c.metadata ? 'section_start_line') AND (c.metadata ? 'section_end_line') THEN 1.0 ELSE 0.0 END) AS line_cov
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {scope_sql}
            """,
            *scope_params,
        )
        art = await conn.fetchrow(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE COALESCE(c.metadata->>'content_type','')='artifact') AS artifact_chunk_count,
                COUNT(*) FILTER (WHERE COALESCE(c.metadata->>'artifact_type','')='table') AS table_chunk_count,
                COUNT(*) FILTER (WHERE COALESCE(c.metadata->>'artifact_type','')='figure') AS figure_chunk_count,
                COUNT(*) FILTER (WHERE COALESCE(c.metadata->>'artifact_type','')='algorithm') AS algorithm_chunk_count,
                COUNT(*) FILTER (
                    WHERE COALESCE(c.metadata->>'content_type','')='artifact'
                      AND COALESCE(c.metadata->>'context_before','')<>''
                      AND COALESCE(c.metadata->>'context_after','')<>''
                ) AS artifact_context_coverage_count
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {scope_sql}
            """,
            *scope_params,
        )

        size_params = [*scope_params, tiny_threshold]
        threshold_idx = len(size_params)
        size = await conn.fetchrow(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE LENGTH(TRIM(COALESCE(c.content,'')))=0) AS empty_chunk_count,
                COUNT(*) FILTER (WHERE LENGTH(TRIM(COALESCE(c.content,''))) < ${threshold_idx}) AS tiny_chunk_count,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY LENGTH(COALESCE(c.content,''))) AS p50,
                percentile_cont(0.9) WITHIN GROUP (ORDER BY LENGTH(COALESCE(c.content,''))) AS p90,
                MAX(LENGTH(COALESCE(c.content,''))) AS max_len
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {scope_sql}
            """,
            *size_params,
        )
        largest_rows = await conn.fetch(
            f"""
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
            {scope_sql}
            ORDER BY LENGTH(COALESCE(c.content, '')) DESC, c.id ASC
            LIMIT 5
            """,
            *scope_params,
        )

    total_chunks_int = int(total_chunks or 0)
    tiny_chunk_count = int(size["tiny_chunk_count"] or 0)
    artifact_chunk_count = int(art["artifact_chunk_count"] or 0)
    report = {
        "summary": {
            "total_documents": int(total_documents or 0),
            "total_chunks": total_chunks_int,
            "section_metadata_coverage": float(cov["section_cov"] or 0.0),
            "line_metadata_coverage": float(cov["line_cov"] or 0.0),
            "artifact_chunk_count": artifact_chunk_count,
            "table_chunk_count": int(art["table_chunk_count"] or 0),
            "figure_chunk_count": int(art["figure_chunk_count"] or 0),
            "algorithm_chunk_count": int(art["algorithm_chunk_count"] or 0),
            "artifact_context_coverage": (
                int(art["artifact_context_coverage_count"] or 0) / max(1, artifact_chunk_count)
            ),
            "empty_chunk_count": int(size["empty_chunk_count"] or 0),
            "tiny_chunk_rate": (tiny_chunk_count / total_chunks_int) if total_chunks_int else 0.0,
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


def quality_report_to_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Ingestion Integrity Eval",
        "",
        f"- total_documents: {summary['total_documents']}",
        f"- total_chunks: {summary['total_chunks']}",
        f"- section_metadata_coverage: {summary['section_metadata_coverage']:.3f}",
        f"- line_metadata_coverage: {summary['line_metadata_coverage']:.3f}",
        f"- artifact_chunk_count: {summary['artifact_chunk_count']}",
        f"- table_chunk_count: {summary['table_chunk_count']}",
        f"- figure_chunk_count: {summary['figure_chunk_count']}",
        f"- algorithm_chunk_count: {summary['algorithm_chunk_count']}",
        f"- artifact_context_coverage: {summary['artifact_context_coverage']:.3f}",
        f"- empty_chunk_count: {summary['empty_chunk_count']}",
        f"- tiny_chunk_rate: {summary['tiny_chunk_rate']:.3f}",
        f"- chunk_size_p50 / p90 / max: {summary['chunk_size_p50']:.1f} / {summary['chunk_size_p90']:.1f} / {summary['chunk_size_max']}",
    ]

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

    return "\n".join(lines) + "\n"


def sample_eval_report_to_markdown(report: Dict[str, Any]) -> str:
    run_info = report.get("run") or {}
    ingestion = report.get("ingestion") or {}
    quality = report.get("quality") or {}
    quality_summary = quality.get("summary") or {}
    documents = list(ingestion.get("documents") or [])

    lines = [
        "# Sample Ingestion Eval",
        "",
        f"- source_pdf_dir: {run_info.get('source_pdf_dir')}",
        f"- sample_mode: {run_info.get('sample_mode')}",
        f"- sample_size_requested: {run_info.get('sample_size_requested')}",
        f"- sample_size_selected: {run_info.get('sample_size_selected')}",
        f"- fast_mode: {run_info.get('fast_mode')}",
        f"- elapsed_seconds: {run_info.get('elapsed_seconds')}",
        f"- successful_documents: {ingestion.get('successful_documents')}",
        f"- failed_documents: {ingestion.get('failed_documents')}",
        f"- total_chunks_created: {ingestion.get('total_chunks_created')}",
        "",
        "## Quality Summary",
        "",
        f"- section_metadata_coverage: {quality_summary.get('section_metadata_coverage', 0.0):.3f}",
        f"- line_metadata_coverage: {quality_summary.get('line_metadata_coverage', 0.0):.3f}",
        f"- artifact_chunk_count: {quality_summary.get('artifact_chunk_count', 0)}",
        f"- table_chunk_count: {quality_summary.get('table_chunk_count', 0)}",
        f"- figure_chunk_count: {quality_summary.get('figure_chunk_count', 0)}",
        f"- algorithm_chunk_count: {quality_summary.get('algorithm_chunk_count', 0)}",
        f"- tiny_chunk_rate: {quality_summary.get('tiny_chunk_rate', 0.0):.3f}",
        f"- chunk_size_p50 / p90 / max: {quality_summary.get('chunk_size_p50', 0.0):.1f} / {quality_summary.get('chunk_size_p90', 0.0):.1f} / {quality_summary.get('chunk_size_max', 0)}",
    ]

    if documents:
        lines.extend(
            [
                "",
                "## Document Results",
                "",
                "| Title | Document ID | Chunks | Processing Time (ms) | Source PDF |",
                "|---|---|---:|---:|---|",
            ]
        )
        for item in documents:
            lines.append(
                "| {title} | {document_id} | {chunks_created} | {processing_time_ms:.1f} | {source_pdf} |".format(
                    title=str(item.get("title") or "").replace("|", "/"),
                    document_id=str(item.get("document_id") or "").replace("|", "/"),
                    chunks_created=int(item.get("chunks_created") or 0),
                    processing_time_ms=float(item.get("processing_time_ms") or 0.0),
                    source_pdf=str(item.get("source_pdf") or "").replace("|", "/"),
                )
            )

    return "\n".join(lines) + "\n"


async def run_sample_ingestion_evaluation(
    pdf_dir: str | Path,
    sample_size: int,
    sample_mode: str,
    seed: int,
    manifest: Dict[str, Any],
    output_dir: str | Path,
    stage_root: str | Path,
    chunk_size: int = 850,
    chunk_overlap: int = 150,
    fast: bool = False,
) -> Dict[str, Any]:
    from ingestion.ingest import DocumentIngestionPipeline

    selected_files = select_pdf_files(pdf_dir=pdf_dir, sample_size=sample_size, sample_mode=sample_mode, seed=seed)
    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_dir = Path(stage_root) / f"sample_ingestion_eval_{run_label}"
    staged_files = stage_pdf_files(selected_files=selected_files, source_root=pdf_dir, stage_root=stage_dir)

    config = IngestionConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_semantic_chunking=not fast,
    )
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=str(stage_dir),
        include_images=not fast,
        include_tables=not fast,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = perf_counter()
    try:
        results = await pipeline.ingest_documents()
    finally:
        await pipeline.close()

    elapsed_seconds = round(perf_counter() - wall_start, 2)
    successful_results = [result for result in results if str(result.document_id or "").strip()]
    document_ids = [str(result.document_id).strip() for result in successful_results if str(result.document_id).strip()]

    await initialize_database()
    try:
        quality_report = await run_ingestion_quality_suite(manifest=manifest, document_ids=document_ids)
    finally:
        await close_database()

    report = {
        "run": {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "source_pdf_dir": str(Path(pdf_dir).resolve()),
            "stage_dir": str(stage_dir.resolve()),
            "sample_mode": sample_mode,
            "sample_size_requested": int(sample_size),
            "sample_size_selected": len(selected_files),
            "seed": int(seed),
            "fast_mode": bool(fast),
        },
        "selected_files": [
            {
                "source_pdf": str(source.resolve()),
                "staged_pdf": str(staged.resolve()),
            }
            for source, staged in zip(selected_files, staged_files)
        ],
        "ingestion": {
            "successful_documents": len(successful_results),
            "failed_documents": len(results) - len(successful_results),
            "total_chunks_created": sum(int(result.chunks_created or 0) for result in results),
            "document_ids": document_ids,
            "documents": [
                {
                    "title": result.title,
                    "document_id": result.document_id,
                    "chunks_created": int(result.chunks_created or 0),
                    "processing_time_ms": float(result.processing_time_ms or 0.0),
                    "source_pdf": str(selected_files[index].resolve()),
                }
                for index, result in enumerate(results)
            ],
        },
        "quality": quality_report,
    }

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    write_json_utf8(output_base / "sample_ingestion_eval.json", report, indent=2)
    write_text_utf8(output_base / "sample_ingestion_eval.md", sample_eval_report_to_markdown(report))
    write_json_utf8(output_base / "sample_ingestion_quality_eval.json", quality_report, indent=2)
    write_text_utf8(output_base / "sample_ingestion_quality_eval.md", quality_report_to_markdown(quality_report))
    return report
