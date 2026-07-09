from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequestMetricItem(BaseModel):
    occurred_at: datetime
    request_id: Optional[str] = None
    session_id: str
    route: str
    status: Literal["success", "error", "cancelled"]
    response_backend: str
    requested_search_type: str
    effective_search_type: str
    use_web_search: bool
    use_react: bool
    compression_used: bool
    tool_call_count: int
    source_count: int
    local_source_count: int
    web_source_count: int
    source_mix: str
    response_chars: int


class ChatMetricsSnapshot(BaseModel):
    started_at: datetime
    total_requests: int
    stream_requests: int
    avg_response_chars: float
    avg_tool_call_count: float
    status_counts: Dict[str, int] = Field(default_factory=dict)
    backend_counts: Dict[str, int] = Field(default_factory=dict)
    route_counts: Dict[str, int] = Field(default_factory=dict)
    effective_search_type_counts: Dict[str, int] = Field(default_factory=dict)
    source_mix_counts: Dict[str, int] = Field(default_factory=dict)
    recent_requests: List[ChatRequestMetricItem] = Field(default_factory=list)
