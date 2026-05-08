"""
数据库工具测试。
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

from agent.db_utils import (
    DatabasePool,
    create_session,
    get_session,
    add_message,
    get_session_messages,
    get_document,
    list_documents,
    vector_search,
    hybrid_search,
    artifact_search,
    get_document_chunks,
    test_connection as db_test_connection
)


class TestDatabasePool:
    """测试数据库连接池管理。"""
    
    def test_init_with_url(self):
        """测试使用数据库 URL 初始化。"""
        url = "postgresql://user:pass@host:5432/db"
        pool = DatabasePool(url)
        assert pool.database_url == url
    
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """测试连接池初始化。"""
        pool = DatabasePool("postgresql://test")
        
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
            mock_pool = Mock()
            mock_create_pool.return_value = mock_pool
            
            await pool.initialize()
            
            assert pool.pool == mock_pool
            mock_create_pool.assert_called_once_with(
                "postgresql://test",
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
    
    @pytest.mark.asyncio
    async def test_close(self):
        """测试连接池关闭。"""
        pool = DatabasePool("postgresql://test")
        mock_pool = AsyncMock()
        pool.pool = mock_pool
        
        await pool.close()
        
        mock_pool.close.assert_called_once()
        assert pool.pool is None
    
    @pytest.mark.asyncio
    async def test_acquire_context_manager(self):
        """测试连接获取。"""
        pool = DatabasePool("postgresql://test")
        
        mock_connection = Mock()
        
        # 创建一个直接返回上下文管理器的模拟对象
        class MockContextManager:
            async def __aenter__(self):
                return mock_connection
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
        
        mock_pool = Mock()
        mock_pool.acquire = Mock(return_value=MockContextManager())
        
        pool.pool = mock_pool
        
        async with pool.acquire() as conn:
            assert conn == mock_connection


class TestSessionManagement:
    """测试会话管理函数。"""
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """测试会话创建。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {"id": "session-123"}
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            session_id = await create_session(
                user_id="user-123",
                metadata={"client": "web"},
                timeout_minutes=30
            )
            
            assert session_id == "session-123"
            mock_conn.fetchrow.assert_called_once()
            
            # 检查 SQL 调用
            call_args = mock_conn.fetchrow.call_args
            assert "INSERT INTO sessions" in call_args[0][0]
            assert call_args[0][1] == "user-123"  # user_id 字段
            metadata = json.loads(call_args[0][2])
            assert metadata["client"] == "web"
            assert metadata["title"] == "New Chat"
            assert metadata["title_generated"] is False
            assert metadata["latest_summary"] == ""
            assert metadata["compression_count"] == 0
            assert metadata["compacted_message_count"] == 0
            assert metadata["last_message_at"] is None
            assert metadata["summary_updated_at"] is None
    
    @pytest.mark.asyncio
    async def test_get_session_exists(self):
        """测试获取已存在的会话。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_result = {
                "id": "session-123",
                "user_id": "user-123",
                "metadata": '{"client": "web"}',
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=1)
            }
            mock_conn.fetchrow.return_value = mock_result
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_pool.acquire.return_value = mock_context_manager
            
            session = await get_session("session-123")
            
            assert session is not None
            assert session["id"] == "session-123"
            assert session["user_id"] == "user-123"
            assert session["metadata"] == {"client": "web"}
    
    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        """测试获取不存在的会话。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = None
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            session = await get_session("nonexistent")
            
            assert session is None
    

class TestMessageManagement:
    """测试消息管理函数。"""
    
    @pytest.mark.asyncio
    async def test_add_message(self):
        """测试添加消息。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {"id": "message-123"}
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            message_id = await add_message(
                session_id="session-123",
                role="user",
                content="Hello",
                metadata={"client": "web"}
            )
            
            assert message_id == "message-123"
            mock_conn.fetchrow.assert_called_once()
            
            # 检查 SQL 调用
            call_args = mock_conn.fetchrow.call_args
            assert "INSERT INTO messages" in call_args[0][0]
            assert call_args[0][2] == "user"  # role 字段
            assert call_args[0][3] == "Hello"  # content 字段
    
    @pytest.mark.asyncio
    async def test_get_session_messages(self):
        """测试获取会话消息。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_messages = [
                {
                    "id": "msg-1",
                    "role": "user",
                    "content": "Hello",
                    "metadata": '{}',
                    "created_at": datetime.now(timezone.utc)
                },
                {
                    "id": "msg-2",
                    "role": "assistant",
                    "content": "Hi there!",
                    "metadata": '{}',
                    "created_at": datetime.now(timezone.utc)
                }
            ]
            mock_conn.fetch.return_value = mock_messages
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            messages = await get_session_messages("session-123", limit=10)
            
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            mock_conn.fetch.assert_called_once()


class TestDocumentManagement:
    """测试文档管理函数。"""
    
    @pytest.mark.asyncio
    async def test_get_document(self):
        """测试获取文档。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_result = {
                "id": "doc-123",
                "title": "Test Document",
                "source": "test.md",
                "content": "Test content",
                "metadata": '{"author": "test"}',
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            mock_conn.fetchrow.return_value = mock_result
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_pool.acquire.return_value = mock_context_manager
            
            document = await get_document("doc-123")
            
            assert document is not None
            assert document["id"] == "doc-123"
            assert document["title"] == "Test Document"
            assert document["metadata"] == {"author": "test"}
    
    @pytest.mark.asyncio
    async def test_list_documents(self):
        """测试列出文档。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_results = [
                {
                    "id": "doc-1",
                    "title": "Document 1",
                    "source": "doc1.md",
                    "metadata": '{}',
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "chunk_count": 5
                },
                {
                    "id": "doc-2",
                    "title": "Document 2",
                    "source": "doc2.md",
                    "metadata": '{}',
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "chunk_count": 3
                }
            ]
            mock_conn.fetch.return_value = mock_results
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            documents = await list_documents(limit=10, offset=0)
            
            assert len(documents) == 2
            assert documents[0]["title"] == "Document 1"
            assert documents[1]["title"] == "Document 2"


class TestVectorSearch:
    """测试向量搜索函数。"""
    
    @pytest.mark.asyncio
    async def test_vector_search(self):
        """测试向量相似度搜索。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_results = [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "Test content 1",
                    "similarity": 0.95,
                    "metadata": '{}',
                    "document_title": "Test Doc",
                    "document_source": "test.md"
                }
            ]
            mock_conn.fetch.return_value = mock_results
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            embedding = [0.1] * 1536  # 模拟嵌入向量
            results = await vector_search(embedding, limit=5)
            
            assert len(results) == 1
            assert results[0]["chunk_id"] == "chunk-1"
            assert results[0]["similarity"] == 0.95
            
            # 检查是否调用了 match_chunks 函数
            mock_conn.fetch.assert_called_once()
            call_args = mock_conn.fetch.call_args
            assert "match_chunks" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        """测试混合搜索。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_results = [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "Test content",
                    "combined_score": 0.90,
                    "vector_similarity": 0.85,
                    "text_similarity": 0.70,
                    "metadata": '{}',
                    "document_title": "Test Doc",
                    "document_source": "test.md"
                }
            ]
            mock_conn.fetch.return_value = mock_results
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            embedding = [0.1] * 1536
            results = await hybrid_search(
                embedding=embedding,
                query_text="test query",
                limit=5,
                text_weight=0.3
            )
            
            assert len(results) == 1
            assert results[0]["combined_score"] == 0.90
            assert results[0]["vector_similarity"] == 0.85
            assert results[0]["text_similarity"] == 0.70
    
    @pytest.mark.asyncio
    async def test_get_document_chunks(self):
        """测试获取文档分块。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_results = [
                {
                    "chunk_id": "chunk-1",
                    "content": "First chunk",
                    "chunk_index": 0,
                    "metadata": '{}'
                },
                {
                    "chunk_id": "chunk-2",
                    "content": "Second chunk",
                    "chunk_index": 1,
                    "metadata": '{}'
                }
            ]
            mock_conn.fetch.return_value = mock_results
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            chunks = await get_document_chunks("doc-123")
            
            assert len(chunks) == 2
            assert chunks[0]["chunk_index"] == 0
            assert chunks[1]["chunk_index"] == 1


class TestUtilityFunctions:
    """测试工具函数。"""
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """测试连接检测成功的情况。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 1
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await db_test_connection()
            
            assert result is True
            mock_conn.fetchval.assert_called_once_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """测试连接检测失败的情况。"""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_pool.acquire.side_effect = Exception("Connection failed")
            
            result = await db_test_connection()
            
            assert result is False


class TestArtifactSearch:
    @pytest.mark.asyncio
    async def test_artifact_search_includes_artifact_filter(self):
        with patch("agent.db_utils.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = []
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            await artifact_search(embedding=[0.1] * 8, query_text="performance", limit=5)

            call_args = mock_conn.fetch.call_args
            sql = call_args[0][0]
            assert "content_type" in sql
            assert "'artifact'" in sql

    @pytest.mark.asyncio
    async def test_artifact_search_filters_artifact_type(self):
        with patch("agent.db_utils.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = []
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            await artifact_search(
                embedding=[0.1] * 8,
                query_text="table summary",
                artifact_types=["table"],
                limit=5,
            )

            call_args = mock_conn.fetch.call_args
            sql = call_args[0][0]
            params = call_args[0][1:]
            assert "artifact_type" in sql
            assert any(isinstance(p, list) and p == ["table"] for p in params)

    @pytest.mark.asyncio
    async def test_artifact_search_filters_document_id(self):
        with patch("agent.db_utils.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = []
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            await artifact_search(
                embedding=[0.1] * 8,
                query_text="algorithm",
                document_id="11111111-1111-1111-1111-111111111111",
                limit=5,
            )

            call_args = mock_conn.fetch.call_args
            sql = call_args[0][0]
            params = call_args[0][1:]
            assert "d.id = $" in sql
            assert "11111111-1111-1111-1111-111111111111" in params

    @pytest.mark.asyncio
    async def test_artifact_search_returns_expected_fields(self):
        with patch("agent.db_utils.db_pool") as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = [
                {
                    "chunk_id": "chunk-a",
                    "document_id": "doc-a",
                    "content": "artifact content",
                    "combined_score": 0.9,
                    "vector_similarity": 0.8,
                    "text_similarity": 0.7,
                    "metadata": '{"artifact_type":"table"}',
                    "document_title": "Doc A",
                    "document_source": "doc-a.md",
                }
            ]
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            rows = await artifact_search(embedding=[0.1] * 8, query_text="table", limit=5)
            assert len(rows) == 1
            row = rows[0]
            assert row["chunk_id"] == "chunk-a"
            assert row["document_id"] == "doc-a"
            assert row["content"] == "artifact content"
            assert row["metadata"]["artifact_type"] == "table"
            assert row["document_title"] == "Doc A"
            assert row["document_source"] == "doc-a.md"
            assert "combined_score" in row
