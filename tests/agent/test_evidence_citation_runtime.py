from agent.evidence_citation_runtime import (
    build_evidence_references,
    extract_citation_ids,
    format_evidence_references_for_prompt,
    review_answer_citations,
)


def test_build_evidence_references_numbers_hits_for_prompt():
    refs = build_evidence_references(
        [
            {
                "document_id": "d1",
                "chunk_id": "c1",
                "document_title": "Hybrid-RRT",
                "content": "The method improves success rate by 40.83%.",
                "score": 0.88,
                "metadata": {"section_path_text": "Abstract", "artifact_type": "table"},
            }
        ]
    )

    assert len(refs) == 1
    assert refs[0].marker == "[1]"
    assert refs[0].section == "Abstract"
    prompt = format_evidence_references_for_prompt(refs)
    assert "[Evidence 1]" in prompt
    assert "citation_marker: [1]" in prompt
    assert "artifact_type: table" in prompt


def test_extract_citation_ids_accepts_plain_and_evidence_markers():
    assert extract_citation_ids("结论来自 [1] 和 [Evidence 2]，补充见 [E3]。") == [1, 2, 3]


def test_review_answer_citations_flags_invalid_ref_and_uncited_claim():
    refs = build_evidence_references(
        [
            {
                "document_id": "d1",
                "chunk_id": "c1",
                "document_title": "Hybrid-RRT",
                "content": "The method improves success rate by 40.83%.",
                "score": 0.88,
                "metadata": {},
            }
        ]
    )
    result = review_answer_citations(
        answer="该方法提升了 40.83% [2]。实验结果表明它优于 baseline。",
        references=refs,
    )

    assert result.reviewed is True
    assert result.risk == 2
    assert result.invalid_ref_ids == [2]
    assert result.missing_citation_claims


def test_review_answer_citations_flags_claim_that_drifts_from_cited_evidence():
    refs = build_evidence_references(
        [
            {
                "document_id": "d1",
                "chunk_id": "c1",
                "document_title": "Hybrid-RRT",
                "content": "The method improves success rate by 40.83%.",
                "score": 0.88,
                "metadata": {},
            }
        ]
    )

    result = review_answer_citations(
        answer="The method improves success rate by 99% [1].",
        references=refs,
    )

    assert result.invalid_ref_ids == []
    assert result.unsupported_citation_claims == ["The method improves success rate by 99% [1]."]
    assert result.risk == 2


def test_review_answer_citations_accepts_supported_cited_claim():
    refs = build_evidence_references(
        [
            {
                "document_id": "d1",
                "chunk_id": "c1",
                "document_title": "Hybrid-RRT",
                "content": "The method improves success rate by 40.83%.",
                "score": 0.88,
                "metadata": {},
            }
        ]
    )

    result = review_answer_citations(
        answer="The method improves success rate by 40.83% [1].",
        references=refs,
    )

    assert result.unsupported_citation_claims == []
