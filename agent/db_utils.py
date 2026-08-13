
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
        # schema.sql only contains idempotent DDL and is also our migration path.
        # Do not skip it merely because the original documents table already exists.
        with open(sql_path, "r", encoding="utf-8") as file:
            sql = file.read()
        await conn.execute(sql)
        logger.info("Schema ensured successfully.")

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


async def get_session_memory_snapshot(session_id: str) -> Dict[str, Any]:
    """Read the structured session snapshot; messages remain the immutable source log."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT version, covered_message_count, summary, updated_at
               FROM session_memory_snapshots WHERE session_id=$1::uuid""",
            session_id,
        )
        if not row:
            return {"version": 0, "covered_message_count": 0, "summary": {}, "updated_at": None}
        return {
            "version": int(row["version"]),
            "covered_message_count": int(row["covered_message_count"]),
            "summary": json.loads(row["summary"]) if isinstance(row["summary"], str) else dict(row["summary"] or {}),
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }


async def save_session_memory_snapshot(
    session_id: str,
    covered_message_count: int,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO session_memory_snapshots(session_id, version, covered_message_count, summary)
               VALUES ($1::uuid, 1, $2, $3::jsonb)
               ON CONFLICT (session_id) DO UPDATE SET
                   version=session_memory_snapshots.version + 1,
                   covered_message_count=EXCLUDED.covered_message_count,
                   summary=EXCLUDED.summary,
                   updated_at=CURRENT_TIMESTAMP
               RETURNING version, covered_message_count, summary, updated_at""",
            session_id,
            max(0, int(covered_message_count)),
            json.dumps(summary, ensure_ascii=False),
        )
        return {
            "version": int(row["version"]),
            "covered_message_count": int(row["covered_message_count"]),
            "summary": json.loads(row["summary"]) if isinstance(row["summary"], str) else dict(row["summary"] or {}),
            "updated_at": row["updated_at"].isoformat(),
        }


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


async def delete_session(session_id: str) -> bool:
    """Delete a session and its messages. Returns True if session existed."""
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                DELETE FROM messages
                WHERE session_id = $1::uuid
                """,
                session_id,
            )
            result = await conn.execute(
                """
                DELETE FROM sessions
                WHERE id = $1::uuid
                """,
                session_id,
            )
    return str(result).endswith(" 1")

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


async def set_message_memory_eligible(message_id: str) -> bool:
    """Promote a persisted final turn into the memory white list after successful completion."""
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """UPDATE messages
               SET metadata = COALESCE(metadata, '{}'::jsonb) || '{"memory_eligible": true}'::jsonb
               WHERE id=$1::uuid""",
            message_id,
        )
        return result != "UPDATE 0"

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


async def get_document_pdf(document_id: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT pdf_blob, pdf_media_type FROM documents WHERE id = $1::uuid", document_id
        )
        if not row or row["pdf_blob"] is None:
            return None
        return {"content": bytes(row["pdf_blob"]), "media_type": str(row["pdf_media_type"] or "application/pdf")}


async def get_document_pdf_bytes(document_id: str) -> Optional[bytes]:
    pdf = await get_document_pdf(document_id)
    return pdf["content"] if pdf else None


async def delete_document(document_id: str) -> bool:
    """Delete one paper and all dependent chunks, artifacts, translations and annotations."""
    async with db_pool.acquire() as conn:
        result = await conn.execute("DELETE FROM documents WHERE id = $1::uuid", document_id)
        return result != "DELETE 0"


async def get_document_translation(document_id: str, target_language: str, source_sha256: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id::text, document_id::text, target_language, content_markdown, model, updated_at
               FROM document_translations WHERE document_id=$1::uuid AND target_language=$2 AND source_sha256=$3""",
            document_id, target_language, source_sha256,
        )
        if not row:
            return None
        return {"id": row["id"], "document_id": row["document_id"], "target_language": row["target_language"], "content_markdown": row["content_markdown"], "model": row["model"], "updated_at": row["updated_at"].isoformat()}


async def save_document_translation(*, document_id: str, target_language: str, source_sha256: str, content_markdown: str, model: str) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO document_translations (document_id,target_language,source_sha256,content_markdown,model)
               VALUES ($1::uuid,$2,$3,$4,$5)
               ON CONFLICT (document_id,target_language,source_sha256) DO UPDATE
               SET content_markdown=EXCLUDED.content_markdown, model=EXCLUDED.model, updated_at=CURRENT_TIMESTAMP
               RETURNING id::text, document_id::text, target_language, content_markdown, model, updated_at""",
            document_id, target_language, source_sha256, content_markdown, model,
        )
        return {"id": row["id"], "document_id": row["document_id"], "target_language": row["target_language"], "content_markdown": row["content_markdown"], "model": row["model"], "updated_at": row["updated_at"].isoformat()}


async def get_translation_profile(document_id: str, target_language: str, source_sha256: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT profile_json, model, updated_at FROM document_translation_profiles
               WHERE document_id=$1::uuid AND target_language=$2 AND source_sha256=$3""",
            document_id, target_language, source_sha256,
        )
    if not row:
        return None
    profile = row["profile_json"] if isinstance(row["profile_json"], dict) else json.loads(row["profile_json"] or "{}")
    return {"profile": profile, "model": row["model"], "updated_at": row["updated_at"].isoformat()}


async def save_translation_profile(*, document_id: str, target_language: str, source_sha256: str, profile: Dict[str, Any], model: str) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO document_translation_profiles(document_id,target_language,source_sha256,profile_json,model)
               VALUES($1::uuid,$2,$3,$4::jsonb,$5)
               ON CONFLICT (document_id,target_language,source_sha256) DO UPDATE
               SET profile_json=EXCLUDED.profile_json, model=EXCLUDED.model, updated_at=CURRENT_TIMESTAMP
               RETURNING profile_json, model, updated_at""",
            document_id, target_language, source_sha256, json.dumps(profile, ensure_ascii=False), model,
        )
    return {"profile": row["profile_json"], "model": row["model"], "updated_at": row["updated_at"].isoformat()}


async def get_selection_translation(document_id: str, target_language: str, source_sha256: str, selection_sha256: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id::text, source_text, translated_text, model, updated_at FROM document_selection_translations
               WHERE document_id=$1::uuid AND target_language=$2 AND source_sha256=$3 AND selection_sha256=$4""",
            document_id, target_language, source_sha256, selection_sha256,
        )
    return dict(row) if row else None


async def save_selection_translation(*, document_id: str, target_language: str, source_sha256: str, selection_sha256: str, source_text: str, translated_text: str, context_before: str, context_after: str, model: str) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO document_selection_translations(document_id,target_language,source_sha256,selection_sha256,source_text,translated_text,context_before,context_after,model)
               VALUES($1::uuid,$2,$3,$4,$5,$6,$7,$8,$9)
               ON CONFLICT (document_id,target_language,source_sha256,selection_sha256) DO UPDATE
               SET translated_text=EXCLUDED.translated_text, context_before=EXCLUDED.context_before, context_after=EXCLUDED.context_after, model=EXCLUDED.model, updated_at=CURRENT_TIMESTAMP
               RETURNING id::text, source_text, translated_text, model, updated_at""",
            document_id, target_language, source_sha256, selection_sha256, source_text, translated_text, context_before, context_after, model,
        )
    return dict(row)


async def list_document_annotations(document_id: str) -> List[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT id::text, page_number, page_x, page_y, quote, note, color, created_at, updated_at FROM document_annotations WHERE document_id=$1::uuid ORDER BY created_at", document_id)
        return [{"id": r["id"], "page_number": r["page_number"], "page_x": float(r["page_x"]), "page_y": float(r["page_y"]), "quote": r["quote"], "note": r["note"], "color": r["color"], "created_at": r["created_at"].isoformat(), "updated_at": r["updated_at"].isoformat()} for r in rows]


async def create_document_annotation(document_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        page_x = min(1.0, max(0.0, float(payload.get("page_x", 0.5))))
        page_y = min(1.0, max(0.0, float(payload.get("page_y", 0.5))))
        row = await conn.fetchrow("""INSERT INTO document_annotations(document_id,page_number,page_x,page_y,quote,note,color) VALUES($1::uuid,$2,$3,$4,$5,$6,$7) RETURNING id::text,page_number,page_x,page_y,quote,note,color,created_at,updated_at""", document_id, payload.get("page_number"), page_x, page_y, str(payload.get("quote") or ""), str(payload.get("note") or "").strip(), str(payload.get("color") or "yellow"))
        return {"id": row["id"], "page_number": row["page_number"], "page_x": float(row["page_x"]), "page_y": float(row["page_y"]), "quote": row["quote"], "note": row["note"], "color": row["color"], "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat()}


async def update_document_annotation_position(document_id: str, annotation_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    page_x = min(1.0, max(0.0, float(payload.get("page_x", 0.5))))
    page_y = min(1.0, max(0.0, float(payload.get("page_y", 0.5))))
    note = payload.get("note")
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""UPDATE document_annotations SET page_number=COALESCE($3,page_number), page_x=$4, page_y=$5,
                                   note=COALESCE($6,note), updated_at=CURRENT_TIMESTAMP
                                   WHERE id=$1::uuid AND document_id=$2::uuid
                                   RETURNING id::text,page_number,page_x,page_y,quote,note,color,created_at,updated_at""", annotation_id, document_id, payload.get("page_number"), page_x, page_y, str(note).strip() if note is not None else None)
        if not row:
            return None
        return {"id": row["id"], "page_number": row["page_number"], "page_x": float(row["page_x"]), "page_y": float(row["page_y"]), "quote": row["quote"], "note": row["note"], "color": row["color"], "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat()}


async def delete_document_annotation(document_id: str, annotation_id: str) -> bool:
    async with db_pool.acquire() as conn:
        result = await conn.execute("DELETE FROM document_annotations WHERE id=$1::uuid AND document_id=$2::uuid", annotation_id, document_id)
        return result != "DELETE 0"


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
    limit: int = 10,
    embedding_language: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
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
        normalized_document_ids = list(dict.fromkeys(str(value).strip() for value in (document_ids or []) if str(value).strip()))
        if embedding_language or normalized_document_ids:
            conditions = ["c.embedding IS NOT NULL"]
            params: List[Any] = [embedding_str]
            if embedding_language:
                conditions.append(f"c.metadata->>'embedding_language' = ${len(params) + 1}")
                params.append(embedding_language)
            if normalized_document_ids:
                conditions.append(f"c.document_id = ANY(${len(params) + 1}::uuid[])")
                params.append(normalized_document_ids)
            params.append(limit)
            results = await conn.fetch(
                f"""
                SELECT c.id::text AS chunk_id, c.document_id::text AS document_id, c.content,
                       (1 - (c.embedding <=> $1::vector))::double precision AS similarity,
                       c.metadata, d.title AS document_title, d.source AS document_source
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE {' AND '.join(conditions)}
                ORDER BY c.embedding <=> $1::vector LIMIT ${len(params)}
                """,
                *params,
            )
        else:
            results = await conn.fetch("SELECT * FROM match_chunks($1::vector, $2)", embedding_str, limit)
        
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
    text_weight: float = 0.3,
    embedding_language: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
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
        
        normalized_document_ids = list(dict.fromkeys(str(value).strip() for value in (document_ids or []) if str(value).strip()))
        if embedding_language or normalized_document_ids:
            safe_text_weight = max(0.0, min(float(text_weight), 1.0))
            vector_weight = 1.0 - safe_text_weight
            conditions = ["c.embedding IS NOT NULL"]
            params: List[Any] = [embedding_str, query_text]
            if embedding_language:
                conditions.append(f"c.metadata->>'embedding_language' = ${len(params) + 1}")
                params.append(embedding_language)
            if normalized_document_ids:
                conditions.append(f"c.document_id = ANY(${len(params) + 1}::uuid[])")
                params.append(normalized_document_ids)
            params.append(limit)
            results = await conn.fetch(
                f"""
                SELECT c.id::text AS chunk_id, c.document_id::text AS document_id, c.content,
                       ({vector_weight} * GREATEST(0.0, 1 - (c.embedding <=> $1::vector)) + {safe_text_weight} *
                        ts_rank_cd(to_tsvector('simple', c.content), plainto_tsquery('simple', $2)))::float8 AS combined_score,
                       GREATEST(0.0, 1 - (c.embedding <=> $1::vector))::float8 AS vector_similarity,
                       ts_rank_cd(to_tsvector('simple', c.content), plainto_tsquery('simple', $2))::float8 AS text_similarity,
                       c.metadata, d.title AS document_title, d.source AS document_source
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE {' AND '.join(conditions)}
                ORDER BY combined_score DESC LIMIT ${len(params)}
                """,
                *params,
            )
        else:
            results = await conn.fetch(
                "SELECT * FROM hybrid_search($1::vector, $2, $3, $4)", embedding_str, query_text, limit, text_weight
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
    document_ids: Optional[List[str]] = None,
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
        elif document_ids:
            normalized_document_ids = list(dict.fromkeys(str(value).strip() for value in document_ids if str(value).strip()))
            if normalized_document_ids:
                conditions.append(f"d.id = ANY(${idx}::uuid[])")
                params.append(normalized_document_ids)
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
    document_ids: Optional[List[str]] = None,
    text_weight: float = 0.3,
    embedding_language: Optional[str] = None,
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
        elif document_ids:
            normalized_document_ids = list(dict.fromkeys(str(value).strip() for value in document_ids if str(value).strip()))
            if normalized_document_ids:
                conditions.append(f"d.id = ANY(${idx}::uuid[])")
                params.append(normalized_document_ids)
                idx += 1

        if embedding_language:
            conditions.append(f"COALESCE(c.metadata->>'embedding_language','') = ${idx}")
            params.append(embedding_language)
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


async def get_artifact(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Load a complete figure/table evidence record without exposing its binary payload."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT a.id::text, a.document_id::text, a.artifact_type, a.caption, a.page_number,
                   a.section_path, a.context_before, a.context_after, a.raw_content,
                   a.structured_data, a.retrieval_text, a.image_media_type,
                   (a.image_blob IS NOT NULL) AS has_image, d.title AS document_title
            FROM artifacts a
            JOIN documents d ON d.id = a.document_id
            WHERE a.id = $1::uuid
            """,
            artifact_id,
        )
        if not row:
            return None
        structured_data = row["structured_data"]
        if isinstance(structured_data, str):
            structured_data = json.loads(structured_data or "{}")
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "document_title": row["document_title"],
            "artifact_type": row["artifact_type"],
            "caption": row["caption"],
            "page_number": row["page_number"],
            "section_path": row["section_path"],
            "context_before": row["context_before"],
            "context_after": row["context_after"],
            "raw_content": row["raw_content"],
            "structured_data": dict(structured_data or {}),
            "retrieval_text": row["retrieval_text"],
            "has_image": bool(row["has_image"]),
            "image_media_type": row["image_media_type"],
        }


async def get_artifact_image(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Load the original image bytes for a figure evidence record."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT image_blob, image_media_type
            FROM artifacts
            WHERE id = $1::uuid AND artifact_type = 'figure' AND image_blob IS NOT NULL
            """,
            artifact_id,
        )
        if not row:
            return None
        return {
            "content": bytes(row["image_blob"]),
            "media_type": str(row["image_media_type"] or "image/png"),
        }

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
