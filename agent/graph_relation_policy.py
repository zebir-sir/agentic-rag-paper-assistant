"""Conservative relationship routing for paper-graph retrieval expansion."""

from dataclasses import dataclass
from typing import List, Literal


GraphDirection = Literal["both", "outgoing", "incoming"]


@dataclass(frozen=True)
class GraphRelationSelection:
    relation_types: List[str]
    direction: GraphDirection = "both"


def select_graph_relations(question: str) -> GraphRelationSelection:
    """Choose graph edges as candidate-scope aids, never as answer evidence."""
    text = str(question or "").strip().lower()
    lineage_cues = ("演进", "后续", "发展", "改进", "extends", "improves", "evolution")
    origin_cues = ("来源", "起源", "基础", "基于什么", "based on", "origin", "foundation")
    citation_cues = ("引用", "被引", "reference", "citation")
    if any(cue in text for cue in lineage_cues):
        return GraphRelationSelection(["method_lineage", "cites", "semantic_similarity"], "incoming")
    if any(cue in text for cue in origin_cues):
        return GraphRelationSelection(["method_lineage", "cites", "semantic_similarity"], "outgoing")
    if any(cue in text for cue in citation_cues):
        return GraphRelationSelection(["cites", "semantic_similarity"], "both")
    return GraphRelationSelection(["semantic_similarity", "method_lineage", "cites"], "both")
