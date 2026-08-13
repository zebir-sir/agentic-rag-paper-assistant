"""Stable data contracts for retrieval policy enforcement and observability."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RetrievalContract(BaseModel):
    """The non-LLM control plane for one knowledge-base answer run."""

    version: str = "v1"
    required_source_types: List[str] = Field(default_factory=list)
    preferred_source_types: List[str] = Field(default_factory=list)
    allowed_source_types: List[str] = Field(default_factory=list)
    blocked_source_types: List[str] = Field(default_factory=list)
    unavailable_required_sources: List[str] = Field(default_factory=list)
    scope_policy: str = "broad_kb"
    target_document_ids: List[str] = Field(default_factory=list)
    allow_supplemental: bool = True
    citation_required: bool = False
    freshness_required: bool = False
    max_tool_calls_per_round: int = Field(default=2, ge=0, le=2)
    max_retrieval_rounds: int = Field(default=2, ge=1)
    must_disclose_limitations: bool = False
    answer_boundary: str = "use_available_sources_only"


class RetrievalContractEvaluation(BaseModel):
    """Deterministic evidence coverage result for a completed retrieval round."""

    required_sources_satisfied: bool
    missing_required_source_types: List[str] = Field(default_factory=list)
    evidence_source_types: List[str] = Field(default_factory=list)
    executed_source_types: List[str] = Field(default_factory=list)
    result_count: int = 0
    reason: str = ""


class RetrievalPlanEnforcement(BaseModel):
    """Audit record of how a planner proposal was constrained by its contract."""

    kept_tools: List[str] = Field(default_factory=list)
    filtered_tools: List[Dict[str, Any]] = Field(default_factory=list)
    reason: str = ""
