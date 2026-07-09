from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RuntimeFeatureStatus(BaseModel):
    enabled: bool
    configured: bool
    detail: str


class RuntimeWarningItem(BaseModel):
    code: str
    message: str


class RuntimeDiagnostics(BaseModel):
    status: Literal["ok", "warning"]
    app_env: str
    request_id_header: str
    features: Dict[str, RuntimeFeatureStatus] = Field(default_factory=dict)
    warnings: List[RuntimeWarningItem] = Field(default_factory=list)


class HttpMetricsSnapshot(BaseModel):
    started_at: datetime
    uptime_seconds: float
    requests_in_flight: int
    total_requests: int
    avg_duration_ms: float
    max_duration_ms: float
    status_counts: Dict[str, int] = Field(default_factory=dict)
    method_counts: Dict[str, int] = Field(default_factory=dict)
    path_counts: Dict[str, int] = Field(default_factory=dict)
    last_request_at: Optional[datetime] = None
