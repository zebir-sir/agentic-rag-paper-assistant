from pathlib import Path

from evals.ingestion_eval_runtime import (
    _build_scope_sql,
    quality_report_to_markdown,
    sample_eval_report_to_markdown,
    select_pdf_files,
    stage_pdf_files,
)


def test_select_pdf_files_first_mode_uses_sorted_recursive_results(tmp_path: Path):
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "a.pdf").write_bytes(b"%PDF-1.4")
    (nested / "note.txt").write_text("ignore", encoding="utf-8")

    selected = select_pdf_files(tmp_path, sample_size=2, sample_mode="first")

    assert [path.name for path in selected] == ["b.pdf", "a.pdf"]


def test_select_pdf_files_random_mode_is_seeded(tmp_path: Path):
    for index in range(5):
        (tmp_path / f"{index}.pdf").write_bytes(b"%PDF-1.4")

    first = select_pdf_files(tmp_path, sample_size=3, sample_mode="random", seed=7)
    second = select_pdf_files(tmp_path, sample_size=3, sample_mode="random", seed=7)

    assert [path.name for path in first] == [path.name for path in second]


def test_stage_pdf_files_preserves_relative_structure(tmp_path: Path):
    source_root = tmp_path / "source"
    nested = source_root / "folder"
    nested.mkdir(parents=True)
    pdf_path = nested / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 sample")

    stage_root = tmp_path / "stage"
    staged = stage_pdf_files([pdf_path], source_root=source_root, stage_root=stage_root)

    assert staged[0] == stage_root.resolve() / "folder" / "paper.pdf"
    assert staged[0].read_bytes() == b"%PDF-1.4 sample"


def test_build_scope_sql_uses_false_scope_for_explicit_empty_ids():
    scope_sql, params = _build_scope_sql([])

    assert scope_sql == "WHERE 1 = 0"
    assert params == []


def test_quality_report_to_markdown_contains_summary_metrics():
    markdown = quality_report_to_markdown(
        {
            "summary": {
                "total_documents": 2,
                "total_chunks": 20,
                "section_metadata_coverage": 1.0,
                "line_metadata_coverage": 0.95,
                "artifact_chunk_count": 4,
                "table_chunk_count": 2,
                "figure_chunk_count": 1,
                "algorithm_chunk_count": 1,
                "artifact_context_coverage": 0.75,
                "empty_chunk_count": 0,
                "tiny_chunk_rate": 0.1,
                "chunk_size_p50": 300.0,
                "chunk_size_p90": 900.0,
                "chunk_size_max": 1500,
                "tiny_chunk_threshold": 80,
            },
            "largest_chunks": [],
        }
    )

    assert "# Ingestion Integrity Eval" in markdown
    assert "- total_documents: 2" in markdown
    assert "- artifact_chunk_count: 4" in markdown


def test_sample_eval_report_to_markdown_contains_document_table():
    markdown = sample_eval_report_to_markdown(
        {
            "run": {
                "source_pdf_dir": "D:/pdfs",
                "sample_mode": "first",
                "sample_size_requested": 10,
                "sample_size_selected": 10,
                "fast_mode": False,
                "elapsed_seconds": 12.5,
            },
            "ingestion": {
                "successful_documents": 1,
                "failed_documents": 0,
                "total_chunks_created": 18,
                "documents": [
                    {
                        "title": "paper",
                        "document_id": "doc-1",
                        "chunks_created": 18,
                        "processing_time_ms": 1234.5,
                        "source_pdf": "D:/pdfs/paper.pdf",
                    }
                ],
            },
            "quality": {
                "summary": {
                    "section_metadata_coverage": 1.0,
                    "line_metadata_coverage": 1.0,
                    "artifact_chunk_count": 2,
                    "table_chunk_count": 1,
                    "figure_chunk_count": 1,
                    "algorithm_chunk_count": 0,
                    "tiny_chunk_rate": 0.0,
                    "chunk_size_p50": 300.0,
                    "chunk_size_p90": 500.0,
                    "chunk_size_max": 900,
                }
            },
        }
    )

    assert "# Sample Ingestion Eval" in markdown
    assert "| Title | Document ID | Chunks | Processing Time (ms) | Source PDF |" in markdown
    assert "paper.pdf" in markdown
