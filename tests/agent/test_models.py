"""
Pydantic 模型测试。
"""

import pytest
from datetime import datetime
from uuid import uuid4

from agent.models import (
    ChatRequest,
    SearchRequest,
    DocumentMetadata,
    ChunkResult,
    SearchResponse,
    ChatResponse,
    EvidenceSource,
    IngestionConfig,
    IngestionResult,
    ErrorResponse,
    HealthStatus,
    SearchType
)


class TestRequestModels:
    """测试请求模型。"""
    
    def test_chat_request_valid(self):
        """测试有效的聊天请求。"""
        request = ChatRequest(
            message="What are Google's AI initiatives?",
            session_id="test-session",
            user_id="test-user",
            search_type=SearchType.HYBRID
        )
        
        assert request.message == "What are Google's AI initiatives?"
        assert request.session_id == "test-session"
        assert request.user_id == "test-user"
        assert request.search_type == SearchType.HYBRID
        assert request.metadata == {}
    
    def test_chat_request_minimal(self):
        """测试最小化聊天请求。"""
        request = ChatRequest(message="Hello")
        
        assert request.message == "Hello"
        assert request.session_id is None
        assert request.user_id is None
        assert request.search_type == SearchType.HYBRID
        assert request.metadata == {}
    
    def test_search_request_valid(self):
        """测试有效的搜索请求。"""
        request = SearchRequest(
            query="Microsoft AI",
            search_type=SearchType.VECTOR,
            limit=20
        )
        
        assert request.query == "Microsoft AI"
        assert request.search_type == SearchType.VECTOR
        assert request.limit == 20
        assert request.filters == {}
    
    def test_search_request_limit_validation(self):
        """测试搜索请求的 limit 校验。"""
        # 测试最小限制值
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=0)
        
        # 测试最大限制值
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=100)
        
        # 测试有效限制值
        request = SearchRequest(query="test", limit=1)
        assert request.limit == 1
        
        request = SearchRequest(query="test", limit=50)
        assert request.limit == 50


class TestResponseModels:
    """测试响应模型。"""
    
    def test_document_metadata(self):
        """测试文档元数据模型。"""
        now = datetime.now()
        metadata = DocumentMetadata(
            id="doc-123",
            title="Test Document",
            source="test.md",
            metadata={"topic": "AI"},
            created_at=now,
            updated_at=now,
            chunk_count=5
        )
        
        assert metadata.id == "doc-123"
        assert metadata.title == "Test Document"
        assert metadata.source == "test.md"
        assert metadata.metadata == {"topic": "AI"}
        assert metadata.chunk_count == 5
    
    def test_chunk_result(self):
        """测试分块结果模型。"""
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=0.85,
            metadata={"index": 0},
            document_title="Test Doc",
            document_source="test.md"
        )
        
        assert chunk.chunk_id == "chunk-123"
        assert chunk.document_id == "doc-123"
        assert chunk.content == "Test content"
        assert chunk.score == 0.85
        assert chunk.document_title == "Test Doc"
    
    def test_chunk_result_score_validation(self):
        """测试分块结果分数校验。"""
        # 测试通过校验器对分数进行截断
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=1.5,  # > 1.0，应被截断为 1.0
            document_title="Test Doc",
            document_source="test.md"
        )
        assert chunk.score == 1.0
        
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=-0.5,  # < 0.0，应被截断为 0.0
            document_title="Test Doc",
            document_source="test.md"
        )
        assert chunk.score == 0.0
        
        # 测试有效分数
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=0.85,  # 有效分数
            document_title="Test Doc",
            document_source="test.md"
        )
        assert chunk.score == 0.85
    
    
    def test_search_response(self):
        """测试搜索响应模型。"""
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=0.85,
            document_title="Test Doc",
            document_source="test.md"
        )
        
        response = SearchResponse(
            results=[chunk],
            total_results=1,
            search_type=SearchType.VECTOR,
            query_time_ms=150.5
        )
        
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.search_type == SearchType.VECTOR
        assert response.query_time_ms == 150.5
    
    def test_chat_response(self):
        """测试聊天响应模型。"""
        source = EvidenceSource(
            source_type="local",
            document_id="doc-123",
            document_title="Test Document",
            document_source="test.md",
            chunk_id="chunk-123",
            snippet="Relevant passage",
            score=0.85,
        )
        
        response = ChatResponse(
            message="Google is working on AI",
            session_id="session-123",
            sources=[source],
            metadata={"tokens": 100}
        )
        
        assert response.message == "Google is working on AI"
        assert response.session_id == "session-123"
        assert len(response.sources) == 1
        assert response.metadata["tokens"] == 100

class TestConfigurationModels:
    """测试配置模型。"""
    
    def test_ingestion_config(self):
        """测试导入配置。"""
        config = IngestionConfig(
            chunk_size=1000,
            chunk_overlap=200,
            max_chunk_size=2000,
            use_semantic_chunking=True
        )
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_chunk_size == 2000
        assert config.use_semantic_chunking is True
    
    def test_ingestion_config_validation(self):
        """测试导入配置校验。"""
        # 测试无效的重叠值（>= chunk_size）
        with pytest.raises(ValueError, match="Chunk overlap .* must be less than chunk size"):
            IngestionConfig(
                chunk_size=1000,
                chunk_overlap=1000  # 与 chunk_size 相同
            )
        
        # 测试有效配置
        config = IngestionConfig(
            chunk_size=1000,
            chunk_overlap=200
        )
        assert config.chunk_overlap == 200
    
    def test_ingestion_result(self):
        """测试导入结果模型。"""
        result = IngestionResult(
            document_id="doc-123",
            title="Test Document",
            chunks_created=10,
            processing_time_ms=1500.0
        )
        
        assert result.document_id == "doc-123"
        assert result.title == "Test Document"
        assert result.chunks_created == 10
        assert result.processing_time_ms == 1500.0


    
    def test_error_response(self):
        """测试错误响应模型。"""
        error = ErrorResponse(
            error="Something went wrong",
            error_type="ValueError",
            details={"code": 400},
            request_id="req-123"
        )
        
        assert error.error == "Something went wrong"
        assert error.error_type == "ValueError"
        assert error.details == {"code": 400}
        assert error.request_id == "req-123"
    
    def test_health_status(self):
        """测试健康状态模型。"""
        now = datetime.now()
        health = HealthStatus(
            status="healthy",
            database=True,
            llm_connection=True,
            version="0.1.0",
            timestamp=now
        )
        
        assert health.status == "healthy"
        assert health.database is True
        assert health.llm_connection is True
        assert health.version == "0.1.0"
        assert health.timestamp == now
