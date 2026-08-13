from agent.graph_abbreviation import build_title_abbreviation
from pathlib import Path


def test_build_title_abbreviation_keeps_explicit_acronym():
    assert build_title_abbreviation("Hybrid-Aware RRT for Motion Planning") == "HA-RRT"


def test_build_title_abbreviation_uses_deterministic_initials():
    assert build_title_abbreviation("Informed RRT for Motion Planning") == "IRRT"


def test_build_title_abbreviation_handles_empty_title():
    assert build_title_abbreviation("") == "PAPER"


def test_graph_query_groups_its_created_at_ordering_field():
    source = (Path(__file__).resolve().parents[2] / "agent" / "graph_runtime.py").read_text(encoding="utf-8")
    assert "GROUP BY n.document_id,d.title,d.source,d.created_at" in source
