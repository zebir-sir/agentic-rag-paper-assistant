from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from enum import Enum
from urllib.parse import urlparse


class SearchType(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_title: str
    document_source: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class SearchResponse(BaseModel):
    results: List[ChunkResult] = Field(default_factory=list)
    total_results: int = 0
    search_type: SearchType
    query_time_ms: float


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search to perform")
    use_web_search: bool = Field(default=False, description="Enable optional web academic search")
    use_react: bool = Field(default=False, description="Enable deep analysis mode")
    model_config = ConfigDict(use_enum_values=True)


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    model_config = ConfigDict(use_enum_values=True)


class DocumentMetadata(BaseModel):
    id: str
    title: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None


class ToolCall(BaseModel):
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class EvidenceSource(BaseModel):
    source_type: str = "local"
    document_id: Optional[str] = None
    document_title: str
    document_source: str
    chunk_id: Optional[str] = None
    snippet: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    message: str
    session_id: str
    sources: List[EvidenceSource] = Field(default_factory=list)
    tools_used: List[ToolCall] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionListItem(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    message_count: int = 0
    last_message_preview: Optional[str] = None
    recoverable: bool = True


class SessionListResponse(BaseModel):
    sessions: List[SessionListItem] = Field(default_factory=list)
    total: int = 0


class ChatMessageItem(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: List[ChatMessageItem] = Field(default_factory=list)
    total: int = 0


class SessionMemoryPreviewItem(BaseModel):
    role: str
    content: str


class SessionMemorySummary(BaseModel):
    goal: str = ""
    constraints: List[str] = Field(default_factory=list)
    confirmed_decisions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)


class SessionMemorySnapshot(BaseModel):
    session_id: str
    version: int = 0
    covered_message_count: int = 0
    summary: SessionMemorySummary = Field(default_factory=SessionMemorySummary)
    updated_at: Optional[str] = None
    eligible_message_count: int = 0
    recent_message_count: int = 0
    using_structured_memory: bool = False
    recent_context_estimated_tokens: int = 0
    recent_messages_preview: List[SessionMemoryPreviewItem] = Field(default_factory=list)


class IngestionTaskResponse(BaseModel):
    task_id: str
    document_id: Optional[str] = None
    file_path: str
    filename: str = ""
    fast: bool = False
    status: str
    queue_order: int = 0
    progress_percent: int = 0
    progress_stage: str = "等待入库"
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class OpenAlexIngestionRequest(BaseModel):
    """A verified OpenAlex PDF selected from an external search result."""

    title: str = Field(default="openalex_paper", min_length=1, max_length=500)
    pdf_url: Optional[str] = Field(default=None, min_length=1, max_length=4096)
    content_url: Optional[str] = Field(default=None, min_length=1, max_length=4096)
    openalex_id: Optional[str] = Field(default=None, max_length=512)
    fast: bool = False

    @field_validator("title", "openalex_id", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        if value is None:
            return None
        return " ".join(str(value).split()).strip()

    @field_validator("pdf_url", "content_url")
    @classmethod
    def validate_pdf_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        url = str(value or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("pdf_url must be an absolute http(s) URL")
        return url

    @model_validator(mode="after")
    def normalize_legacy_content_url(self):
        if not self.pdf_url and self.content_url:
            self.pdf_url = self.content_url
        if not self.pdf_url:
            raise ValueError("pdf_url is required")
        return self


class IngestionConfig(BaseModel):
    chunk_size: int = Field(default=850, ge=100, le=5000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)
    use_semantic_chunking: bool = True

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


class IngestionResult(BaseModel):
    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float


class ErrorResponse(BaseModel):
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


class HealthStatus(BaseModel):
    status: Literal["healthy", "unhealthy"]
    database: bool
    llm_connection: bool
    version: str
    timestamp: datetime


class ComponentStatus(BaseModel):
    enabled: bool
    healthy: bool
    detail: str


class ReadinessStatus(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    version: str
    timestamp: datetime
    components: Dict[str, ComponentStatus] = Field(default_factory=dict)
