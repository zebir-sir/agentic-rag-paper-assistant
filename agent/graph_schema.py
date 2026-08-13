from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PaperGraphNode(BaseModel):
    document_id: str
    title: str
    abbreviation: str
    source: str = ""
    chunk_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    title_zh: str = ""
    localization_status: str = "pending"
    research_card: Dict[str, Any] = Field(default_factory=dict)


class PaperGraphEdge(BaseModel):
    source_document_id: str
    target_document_id: str
    relation_type: str
    score: float
    evidence: Dict[str, Any] = Field(default_factory=dict)


class PaperGraphResponse(BaseModel):
    version: int = 0
    nodes: List[PaperGraphNode] = Field(default_factory=list)
    edges: List[PaperGraphEdge] = Field(default_factory=list)
    updated_at: Optional[str] = None
