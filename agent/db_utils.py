
import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
import logging

import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class DatabasePool:
    """管理 PostgreSQL 连接池。"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        初始化数据库连接池。
        
        Args:
            database_url: PostgreSQL 连接 URL
        """
        USER = os.getenv("DB_USER", "postgres")
        PASSWORD = os.getenv("DB_PASSWORD", "postgres")
        HOST = os.getenv("DB_HOST", "postgres")
        PORT = os.getenv("DB_PORT", 5432)
        DBNAME = os.getenv("DB_NAME", "postgres")

        self.database_url = database_url  or f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.pool: Optional[Pool] = None
    
    async def initialize(self):
        """创建连接池。"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
    
    async def close(self):
        """关闭连接池。"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """从连接池获取一个连接。"""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection

# 全局数据库连接池实例
db_pool = DatabasePool()

async def initialize_database():
    """初始化数据库连接池。"""
    await db_pool.initialize()

async def close_database():
    """关闭数据库连接池。"""
    await db_pool.close()

async def execute_init_sql(sql_path: str):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = 'documents'
            ) AS exists
        """)
        
        if row["exists"]:
            logger.info("Schema already initialized, skipping.")
            return

        with open(sql_path, 'r') as file:
            sql = file.read()
            await conn.execute(sql)
            logger.info("Schema created successfully.")

# 会话管理函数
async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_minutes: int = 10080
) -> str:
    """
    创建一个新会话。
    
    Args:
        user_id: 可选的用户标识
        metadata: 可选的会话元数据
        timeout_minutes: 会话超时时长（分钟）
    
    Returns:
        会话 ID
    """
    async with db_pool.acquire() as conn:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
        session_metadata = {
            "title": "New Chat",
            "title_generated": False,
            "last_message_at": None,
            "latest_summary": "",
            "compression_count": 0,
            "compacted_message_count": 0,
            "summary_updated_at": None,
            **(metadata or {})
        }
        
        result = await conn.fetchrow(
            """
            INSERT INTO sessions (user_id, metadata, expires_at)
            VALUES ($1, $2, $3)
            RETURNING id::text
            """,
            user_id,
            json.dumps(session_metadata),
            expires_at
        )
        
        return result["id"]


def _generate_session_title(first_user_message: Optional[str], max_len: int = 22) -> str:
    """Generate a short session title from first user message."""
    if not first_user_message:
        return "New Chat"

    normalized = " ".join(first_user_message.split()).strip()
    if not normalized:
        return "New Chat"

    if len(normalized) <= max_len:
        return normalized

    return normalized[:max_len].rstrip() + "..."

async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    根据 ID 获取会话。
    
    Args:
        session_id: 会话 UUID
    
    Returns:
        会话数据；如果未找到或已过期则返回 None
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                user_id,
                metadata,
                created_at,
                updated_at,
                expires_at
            FROM sessions
            WHERE id = $1::uuid
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id
        )
        
        if result:
            return {
                "id": result["id"],
                "user_id": result["user_id"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "expires_at": result["expires_at"].isoformat() if result["expires_at"] else None
            }
        
        return None


async def update_session_metadata(
    session_id: str,
    metadata_updates: Dict[str, Any]
) -> bool:
    """Merge updates into session metadata."""
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
            SET metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb
            WHERE id = $1::uuid
              AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id,
            json.dumps(metadata_updates)
        )
        return result != "UPDATE 0"


async def get_session_memory_metadata(session_id: str) -> Dict[str, Any]:
    """Read memory-related metadata fields for a session."""
    session = await get_session(session_id)
    if not session:
        return {
            "latest_summary": "",
            "compression_count": 0,
            "compacted_message_count": 0,
            "summary_updated_at": None,
        }

    metadata = session.get("metadata") or {}
    return {
        "latest_summary": str(metadata.get("latest_summary") or ""),
        "compression_count": int(metadata.get("compression_count") or 0),
        "compacted_message_count": int(metadata.get("compacted_message_count") or 0),
        "summary_updated_at": metadata.get("summary_updated_at"),
    }


async def update_session_memory_metadata(
    session_id: str,
    latest_summary: str,
    compression_count: int,
    compacted_message_count: int,
    summary_updated_at: Optional[str] = None,
) -> bool:
    """Merge memory metadata into existing session metadata."""
    updates = {
        "latest_summary": latest_summary,
        "compression_count": compression_count,
        "compacted_message_count": compacted_message_count,
        "summary_updated_at": summary_updated_at or datetime.now(timezone.utc).isoformat(),
    }
    return await update_session_metadata(session_id, updates)


async def refresh_session_metadata(session_id: str) -> None:
    """Refresh title and last_message_at metadata for a session."""
    async with db_pool.acquire() as conn:
        current = await conn.fetchrow(
            """
            SELECT metadata
            FROM sessions
            WHERE id = $1::uuid
              AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id
        )
        if not current:
            return

        metadata = json.loads(current["metadata"]) if current["metadata"] else {}
        updates: Dict[str, Any] = {}

        last_message_at = await conn.fetchval(
            """
            SELECT MAX(created_at)
            FROM messages
            WHERE session_id = $1::uuid
            """,
            session_id
        )
        if last_message_at:
            updates["last_message_at"] = last_message_at.isoformat()

        if not metadata.get("title_generated"):
            first_user_message = await conn.fetchval(
                """
                SELECT content
                FROM messages
                WHERE session_id = $1::uuid
                  AND role = 'user'
                  AND btrim(content) <> ''
                ORDER BY created_at ASC
                LIMIT 1
                """,
                session_id
            )
            updates["title"] = _generate_session_title(first_user_message)
            updates["title_generated"] = True

        if updates:
            await conn.execute(
                """
                UPDATE sessions
                SET metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb
                WHERE id = $1::uuid
                """,
                session_id,
                json.dumps(updates)
            )


async def list_recent_sessions(limit: int = 20, days: int = 7) -> List[Dict[str, Any]]:
    """List recent visible sessions with lightweight stats."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                s.id::text AS session_id,
                s.metadata,
                s.created_at,
                s.updated_at,
                s.expires_at,
                COALESCE(mc.message_count, 0) AS message_count,
                lm.content AS last_message_preview
            FROM sessions s
            LEFT JOIN LATERAL (
                SELECT COUNT(*)::int AS message_count
                FROM messages m
                WHERE m.session_id = s.id
            ) mc ON TRUE
            LEFT JOIN LATERAL (
                SELECT m.content
                FROM messages m
                WHERE m.session_id = s.id
                ORDER BY m.created_at DESC
                LIMIT 1
            ) lm ON TRUE
            WHERE (s.expires_at IS NULL OR s.expires_at > CURRENT_TIMESTAMP)
              AND s.updated_at >= (CURRENT_TIMESTAMP - make_interval(days => $2::int))
            ORDER BY s.updated_at DESC
            LIMIT $1
            """,
            limit,
            days
        )

        sessions: List[Dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            title = metadata.get("title") or "New Chat"
            preview = row["last_message_preview"] or ""
            preview = " ".join(preview.split()).strip()
            if len(preview) > 80:
                preview = preview[:80].rstrip() + "..."

            sessions.append(
                {
                    "session_id": row["session_id"],
                    "title": title,
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
                    "message_count": row["message_count"],
                    "last_message_preview": preview,
                    "recoverable": True
                }
            )

        return sessions

# 消息管理函数
async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    向会话中添加一条消息。
    
    Args:
        session_id: 会话 UUID
        role: 消息角色（user/assistant/system）
        content: 消息内容
        metadata: 可选的消息元数据
    
    Returns:
        消息 ID
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO messages (session_id, role, content, metadata)
            VALUES ($1::uuid, $2, $3, $4)
            RETURNING id::text
            """,
            session_id,
            role,
            content,
            json.dumps(metadata or {})
        )
        
        return result["id"]

async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    获取某个会话的消息。
    
    Args:
        session_id: 会话 UUID
        limit: 返回消息的最大数量
    
    Returns:
        按创建时间排序的消息列表
    """
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                id::text,
                role,
                content,
                metadata,
                created_at
            FROM messages
            WHERE session_id = $1::uuid
            ORDER BY created_at
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = await conn.fetch(query, session_id)
        
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]

# 文档管理函数
async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """
    根据 ID 获取文档。
    
    Args:
        document_id: 文档 UUID
    
    Returns:
        文档数据；如果未找到则返回 None
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                title,
                source,
                content,
                metadata,
                created_at,
                updated_at
            FROM documents
            WHERE id = $1::uuid
            """,
            document_id
        )
        
        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        
        return None


async def list_documents(
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    列出文档，并可选择按条件过滤。
    
    Args:
        limit: 返回文档的最大数量
        offset: 需要跳过的文档数量
        metadata_filter: 可选的元数据过滤条件
    
    Returns:
        文档列表
    """
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                d.id::text,
                d.title,
                d.source,
                d.metadata,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
        """
        
        params = []
        conditions = []
        
        if metadata_filter:
            conditions.append(f"d.metadata @> ${len(params) + 1}::jsonb")
            params.append(json.dumps(metadata_filter))
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
            GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
            ORDER BY d.created_at DESC
            LIMIT $%d OFFSET $%d
        """ % (len(params) + 1, len(params) + 2)
        
        params.extend([limit, offset])
        
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "chunk_count": row["chunk_count"]
            }
            for row in results
        ]

# 向量搜索函数
async def vector_search(
    embedding: List[float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    执行向量相似度搜索。
    
    Args:
        embedding: 查询对应的嵌入向量
        limit: 返回结果的最大数量
    
    Returns:
        按相似度排序的匹配分块列表（最相关的在前）
    """
    async with db_pool.acquire() as conn:
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        results = await conn.fetch(
            "SELECT * FROM match_chunks($1::vector, $2)",
            embedding_str,
            limit
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "similarity": row["similarity"],
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]

async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    执行混合搜索（向量 + 关键词）。
    
    Args:
        embedding: 查询对应的嵌入向量
        query_text: 用于关键词搜索的查询文本
        limit: 返回结果的最大数量
        text_weight: 文本相似度权重（0-1）
    
    Returns:
        按综合分数排序的匹配分块列表（最相关的在前）
    """
    async with db_pool.acquire() as conn:
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        results = await conn.fetch(
            "SELECT * FROM hybrid_search($1::vector, $2, $3, $4)",
            embedding_str,
            query_text,
            limit,
            text_weight
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "combined_score": row["combined_score"],
                "vector_similarity": row["vector_similarity"],
                "text_similarity": row["text_similarity"],
                "metadata": json.loads(row["metadata"]),
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


async def section_search(
    query_text: str,
    limit: int = 10,
    document_id: Optional[str] = None,
    section_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search chunks within section-aware metadata (section_title/section_path_text).
    This does not change vector/hybrid behavior and is a standalone retrieval path.
    """
    async with db_pool.acquire() as conn:
        safe_limit = max(1, min(int(limit or 10), 50))
        query_value = str(query_text or "").strip()
        section_value = str(section_query or "").strip()
        normalized_section_value = re.sub(r"[\s\W_]+", "", section_value.lower(), flags=re.UNICODE)

        conditions: List[str] = []
        params: List[Any] = []
        idx = 1

        # Prefer section-aware chunks for this tool.
        conditions.append(
            "("
            "COALESCE(c.metadata->>'section_title','') <> '' "
            "OR COALESCE(c.metadata->>'section_path_text','') <> ''"
            ")"
        )

        if section_value:
            section_like = f"%{section_value}%"
            normalized_section_like = f"%{normalized_section_value}%"
            conditions.append(
                "("
                f"LOWER(COALESCE(c.metadata->>'section_title','')) LIKE LOWER(${idx}) "
                f"OR LOWER(COALESCE(c.metadata->>'section_path_text','')) LIKE LOWER(${idx}) "
                f"OR regexp_replace(lower(COALESCE(c.metadata->>'section_title','')), '[[:space:][:punct:]_]+', '', 'g') LIKE ${idx + 1} "
                f"OR regexp_replace(lower(COALESCE(c.metadata->>'section_path_text','')), '[[:space:][:punct:]_]+', '', 'g') LIKE ${idx + 1}"
                ")"
            )
            params.append(section_like)
            params.append(normalized_section_like)
            idx += 2

        if document_id:
            conditions.append(f"d.id = ${idx}::uuid")
            params.append(document_id)
            idx += 1

        score_param_idx: Optional[int] = None
        if query_value and not section_value:
            query_like = f"%{query_value}%"
            conditions.append(
                "("
                f"c.content ILIKE ${idx} "
                f"OR COALESCE(c.metadata->>'section_title','') ILIKE ${idx} "
                f"OR COALESCE(c.metadata->>'section_path_text','') ILIKE ${idx}"
                ")"
            )
            params.append(query_like)
            score_param_idx = idx
            idx += 1
        elif query_value:
            query_like = f"%{query_value}%"
            params.append(query_like)
            score_param_idx = idx
            idx += 1

        where_sql = " AND ".join(conditions) if conditions else "TRUE"

        # query_text only affects light scoring when section_query/document_id already matched.
        if score_param_idx is not None:
            score_expr = (
                f"CASE WHEN c.content ILIKE ${score_param_idx} "
                f"OR COALESCE(c.metadata->>'section_title','') ILIKE ${score_param_idx} "
                f"OR COALESCE(c.metadata->>'section_path_text','') ILIKE ${score_param_idx} "
                "THEN 1.0 ELSE 0.5 END"
            )
        else:
            score_expr = "0.5"

        sql = f"""
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                c.content AS content,
                {score_expr}::float8 AS combined_score,
                NULL::float8 AS vector_similarity,
                {score_expr}::float8 AS text_similarity,
                c.metadata AS metadata,
                d.title AS document_title,
                d.source AS document_source
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_sql}
            ORDER BY
                COALESCE((c.metadata->>'section_start_line')::int, 2147483647) ASC,
                COALESCE((c.metadata->>'section_chunk_index')::int, 2147483647) ASC,
                COALESCE(c.chunk_index, 2147483647) ASC
            LIMIT ${idx}
        """
        params.append(safe_limit)
        rows = await conn.fetch(sql, *params)

        out: List[Dict[str, Any]] = []
        for row in rows:
            metadata_raw = row["metadata"]
            if isinstance(metadata_raw, str):
                try:
                    metadata_value = json.loads(metadata_raw)
                except Exception:
                    metadata_value = {}
            elif isinstance(metadata_raw, dict):
                metadata_value = metadata_raw
            else:
                metadata_value = dict(metadata_raw or {})

            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "combined_score": float(row["combined_score"]),
                    "vector_similarity": row["vector_similarity"],
                    "text_similarity": float(row["text_similarity"]),
                    "metadata": metadata_value,
                    "document_title": row["document_title"],
                    "document_source": row["document_source"],
                }
            )
        return out


async def artifact_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    artifact_types: Optional[List[str]] = None,
    document_id: Optional[str] = None,
    text_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Search only artifact chunks (table/figure/algorithm) with vector + text relevance.
    """
    async with db_pool.acquire() as conn:
        safe_limit = max(1, min(int(limit or 10), 50))
        safe_text_weight = max(0.0, min(float(text_weight or 0.0), 1.0))
        vector_weight = 1.0 - safe_text_weight
        query_value = str(query_text or "").strip()
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        conditions: List[str] = [
            "COALESCE(c.metadata->>'content_type', '') = 'artifact'",
        ]
        params: List[Any] = [embedding_str, query_value]
        idx = 3

        normalized_types = []
        for t in (artifact_types or []):
            value = str(t or "").strip().lower()
            if value in {"table", "figure", "algorithm"}:
                normalized_types.append(value)
        if normalized_types:
            conditions.append(f"LOWER(COALESCE(c.metadata->>'artifact_type','')) = ANY(${idx}::text[])")
            params.append(normalized_types)
            idx += 1

        if document_id:
            conditions.append(f"d.id = ${idx}::uuid")
            params.append(document_id)
            idx += 1

        where_sql = " AND ".join(conditions)
        sql = f"""
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                c.content AS content,
                (
                    ({vector_weight}::float8 * GREATEST(0.0, LEAST(1.0, 1.0 - (c.embedding <=> $1::vector))))
                    +
                    ({safe_text_weight}::float8 * GREATEST(
                        ts_rank_cd(
                            setweight(to_tsvector('simple', COALESCE(c.content, '')), 'B')
                            ||
                            setweight(to_tsvector('simple', COALESCE(c.metadata->>'caption', '')), 'A')
                            ||
                            setweight(to_tsvector('simple', COALESCE(c.metadata->>'retrieval_title', '')), 'A')
                            ||
                            setweight(to_tsvector('simple', COALESCE(c.metadata->>'section_path_text', '')), 'C'),
                            plainto_tsquery('simple', $2)
                        ),
                        0.0
                    ))
                )::float8 AS combined_score,
                GREATEST(0.0, LEAST(1.0, 1.0 - (c.embedding <=> $1::vector)))::float8 AS vector_similarity,
                GREATEST(
                    ts_rank_cd(
                        setweight(to_tsvector('simple', COALESCE(c.content, '')), 'B')
                        ||
                        setweight(to_tsvector('simple', COALESCE(c.metadata->>'caption', '')), 'A')
                        ||
                        setweight(to_tsvector('simple', COALESCE(c.metadata->>'retrieval_title', '')), 'A')
                        ||
                        setweight(to_tsvector('simple', COALESCE(c.metadata->>'section_path_text', '')), 'C'),
                        plainto_tsquery('simple', $2)
                    ),
                    0.0
                )::float8 AS text_similarity,
                c.metadata AS metadata,
                d.title AS document_title,
                d.source AS document_source
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_sql}
            ORDER BY combined_score DESC, vector_similarity DESC
            LIMIT ${idx}
        """
        params.append(safe_limit)
        rows = await conn.fetch(sql, *params)

        out: List[Dict[str, Any]] = []
        for row in rows:
            metadata_raw = row["metadata"]
            if isinstance(metadata_raw, str):
                try:
                    metadata_value = json.loads(metadata_raw)
                except Exception:
                    metadata_value = {}
            elif isinstance(metadata_raw, dict):
                metadata_value = metadata_raw
            else:
                metadata_value = dict(metadata_raw or {})

            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "combined_score": float(row["combined_score"]),
                    "vector_similarity": float(row["vector_similarity"]),
                    "text_similarity": float(row["text_similarity"]),
                    "metadata": metadata_value,
                    "document_title": row["document_title"],
                    "document_source": row["document_source"],
                }
            )
        return out

# 分块管理函数
async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """
    获取某个文档的所有分块。
    
    Args:
        document_id: 文档 UUID
    
    Returns:
        按分块索引排序的分块列表
    """
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            "SELECT * FROM get_document_chunks($1::uuid)",
            document_id
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": json.loads(row["metadata"])
            }
            for row in results
        ]

async def test_connection() -> bool:
    """
    测试数据库连接。
    
    Returns:
        连接成功时返回 True
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
