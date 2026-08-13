from agent.graph_relation_policy import select_graph_relations
from agent.graph_relation_runtime import GraphCandidate, extract_evidence_backed_relations


def test_reference_title_match_creates_directed_citation_with_evidence():
    relations = extract_evidence_backed_relations(
        "paper-a",
        [{"id": "chunk-1", "content": "[7] Karaman and Frazzoli. Sampling-based Algorithms for Optimal Motion Planning.", "metadata": {"section_title": "References"}}],
        [GraphCandidate(document_id="paper-b", title="Sampling-based Algorithms for Optimal Motion Planning")],
    )
    assert len(relations) == 1
    assert relations[0]["relation_type"] == "cites"
    assert relations[0]["target_document_id"] == "paper-b"
    assert relations[0]["evidence"]["source_section"] == "References"


def test_method_lineage_requires_method_section_and_explicit_cue():
    relations = extract_evidence_backed_relations(
        "paper-a",
        [{"id": "chunk-2", "content": "Our method extends Sampling-based Algorithms for Optimal Motion Planning.", "metadata": {"section_path_text": "3 Method"}}],
        [GraphCandidate(document_id="paper-b", title="Sampling-based Algorithms for Optimal Motion Planning")],
    )
    assert relations[0]["relation_type"] == "method_lineage"
    assert relations[0]["evidence"]["cue"] == "extends"


def test_method_lineage_does_not_infer_from_untagged_prose():
    relations = extract_evidence_backed_relations(
        "paper-a",
        [{"id": "chunk-3", "content": "We extend Sampling-based Algorithms for Optimal Motion Planning.", "metadata": {"section_title": "Introduction"}}],
        [GraphCandidate(document_id="paper-b", title="Sampling-based Algorithms for Optimal Motion Planning")],
    )
    assert relations == []


def test_relation_policy_uses_directed_lineage_for_origin_and_evolution_questions():
    assert select_graph_relations("这个方法基于什么工作？").direction == "outgoing"
    assert select_graph_relations("这个方法后续如何演进？").direction == "incoming"
    assert select_graph_relations("有哪些相似工作？").relation_types[0] == "semantic_similarity"
