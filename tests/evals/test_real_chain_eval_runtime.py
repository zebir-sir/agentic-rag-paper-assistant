from evals.real_chain_eval_runtime import (
    build_presentation_summary,
    judge_answer_groundedness,
    judge_ingestion_integrity,
    scorecard_to_markdown,
)


def test_ingestion_integrity_passes_with_structured_metadata():
    item = judge_ingestion_integrity(
        {
            "summary": {
                "total_documents": 3,
                "total_chunks": 476,
                "section_metadata_coverage": 1.0,
                "line_metadata_coverage": 1.0,
                "artifact_chunk_count": 147,
                "empty_chunk_count": 0,
                "tiny_chunk_rate": 0.0,
            }
        },
        "evals/results/ingestion_quality_eval.json",
    )

    assert item.status == "PASS"
    assert item.metrics["documents"] == 3
    assert item.metrics["artifact_chunks"] == 147


def test_presentation_summary_keeps_groundedness_as_diagnostic():
    ingestion = judge_ingestion_integrity(
        {
            "summary": {
                "total_documents": 3,
                "total_chunks": 476,
                "section_metadata_coverage": 1.0,
                "line_metadata_coverage": 1.0,
                "artifact_chunk_count": 147,
                "empty_chunk_count": 0,
                "tiny_chunk_rate": 0.0,
            }
        },
        "ingestion.json",
    )
    groundedness = judge_answer_groundedness(
        {
            "summary": {
                "total_cases": 3,
                "valid_cases": 3,
                "pass_rate": 0.0,
                "warn_rate": 0.0,
                "fail_rate": 1.0,
                "avg_unsupported_numeric": 2.3,
                "avg_unsupported_mechanism": 1.6,
                "gap_disclosure_rate": 1.0,
            }
        },
        "answer.json",
    )

    summary = build_presentation_summary([ingestion, groundedness])

    assert summary["recommended_public_status"] == "NEEDS_MORE_EVIDENCE"
    assert summary["showcase_suites"] == ["Ingestion Integrity"]
    assert summary["diagnostic_suites"] == ["Answer Groundedness"]


def test_markdown_hides_failed_diagnostics_in_presentation_mode():
    groundedness = judge_answer_groundedness(
        {
            "summary": {
                "total_cases": 1,
                "valid_cases": 1,
                "pass_rate": 0.0,
                "warn_rate": 0.0,
                "fail_rate": 1.0,
            }
        },
        "answer.json",
    )

    markdown = scorecard_to_markdown([groundedness], presentation_mode=True)

    assert "| Answer Groundedness | FAIL |" in markdown


def test_markdown_shows_only_pass_items_when_available():
    ingestion = judge_ingestion_integrity(
        {
            "summary": {
                "total_documents": 3,
                "total_chunks": 476,
                "section_metadata_coverage": 1.0,
                "line_metadata_coverage": 1.0,
                "artifact_chunk_count": 147,
                "empty_chunk_count": 0,
                "tiny_chunk_rate": 0.0,
            }
        },
        "ingestion.json",
    )
    groundedness = judge_answer_groundedness(
        {
            "summary": {
                "total_cases": 1,
                "valid_cases": 1,
                "pass_rate": 0.0,
                "warn_rate": 0.0,
                "fail_rate": 1.0,
            }
        },
        "answer.json",
    )

    markdown = scorecard_to_markdown([ingestion, groundedness], presentation_mode=True)

    assert "| Ingestion Integrity | PASS |" in markdown
    assert "| Answer Groundedness | FAIL |" not in markdown
