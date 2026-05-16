from datetime import datetime
from typing import Any, Dict, Literal, Optional

from .db_utils import db_pool

IngestionTaskStatus = Literal["queued", "processing", "done", "failed"]


def _serialize_ingestion_task(row: Any) -> Dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "document_id": row["document_id"],
        "file_path": row["file_path"],
        "status": row["status"],
        "error_message": row["error_message"],
        "retry_count": row["retry_count"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
    }


async def create_ingestion_task(
    task_id: str,
    file_path: str,
    document_id: Optional[str] = None,
    status: IngestionTaskStatus = "queued",
    error_message: Optional[str] = None,
    retry_count: int = 0,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO ingestion_tasks (
                task_id,
                document_id,
                file_path,
                status,
                error_message,
                retry_count,
                started_at,
                finished_at
            )
            VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8)
            RETURNING
                task_id,
                document_id::text,
                file_path,
                status,
                error_message,
                retry_count,
                created_at,
                updated_at,
                started_at,
                finished_at
            """,
            task_id,
            document_id,
            file_path,
            status,
            error_message,
            retry_count,
            started_at,
            finished_at,
        )
        return _serialize_ingestion_task(row)


async def get_ingestion_task(task_id: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                task_id,
                document_id::text,
                file_path,
                status,
                error_message,
                retry_count,
                created_at,
                updated_at,
                started_at,
                finished_at
            FROM ingestion_tasks
            WHERE task_id = $1
            """,
            task_id,
        )
        if not row:
            return None
        return _serialize_ingestion_task(row)


async def update_ingestion_task_status(
    task_id: str,
    status: IngestionTaskStatus,
    document_id: Optional[str] = None,
    error_message: Optional[str] = None,
    retry_count: Optional[int] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE ingestion_tasks
            SET
                status = $2,
                document_id = COALESCE($3::uuid, document_id),
                error_message = $4,
                retry_count = COALESCE($5, retry_count),
                started_at = COALESCE($6, started_at),
                finished_at = COALESCE($7, finished_at),
                updated_at = CURRENT_TIMESTAMP
            WHERE task_id = $1
            RETURNING
                task_id,
                document_id::text,
                file_path,
                status,
                error_message,
                retry_count,
                created_at,
                updated_at,
                started_at,
                finished_at
            """,
            task_id,
            status,
            document_id,
            error_message,
            retry_count,
            started_at,
            finished_at,
        )
        if not row:
            return None
        return _serialize_ingestion_task(row)
