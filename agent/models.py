from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


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


class IngestionTaskResponse(BaseModel):
    task_id: str
    document_id: Optional[str] = None
    file_path: str
    status: str
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


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
