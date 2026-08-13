from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

from .db_utils import db_pool

IngestionTaskStatus = Literal[
    "queued", "processing", "paused", "done", "failed", "paused_quota", "deleted"
]

_TASK_COLUMNS = """
    task_id, document_id::text, file_path, filename, fast, status, queue_order,
    progress_percent, progress_stage, error_message, retry_count, created_at,
    updated_at, started_at, finished_at
"""


def _serialize_ingestion_task(row: Any) -> Dict[str, Any]:
    get = row.get if hasattr(row, "get") else row.__getitem__
    return {
        "task_id": get("task_id"),
        "document_id": get("document_id"),
        "file_path": get("file_path"),
        "filename": get("filename") or Path(str(get("file_path") or "")).name,
        "fast": bool(get("fast") or False),
        "status": get("status"),
        "queue_order": int(get("queue_order") or 0),
        "progress_percent": int(get("progress_percent") or 0),
        "progress_stage": get("progress_stage") or "等待入库",
        "error_message": get("error_message"),
        "retry_count": int(get("retry_count") or 0),
        "created_at": get("created_at").isoformat() if get("created_at") else None,
        "updated_at": get("updated_at").isoformat() if get("updated_at") else None,
        "started_at": get("started_at").isoformat() if get("started_at") else None,
        "finished_at": get("finished_at").isoformat() if get("finished_at") else None,
    }


async def create_ingestion_task(
    task_id: str,
    file_path: str,
    fast: bool = False,
    document_id: Optional[str] = None,
    status: IngestionTaskStatus = "queued",
    error_message: Optional[str] = None,
    retry_count: int = 0,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            INSERT INTO ingestion_tasks (
                task_id, document_id, file_path, filename, fast, status, queue_order,
                progress_percent, progress_stage, error_message, retry_count, started_at, finished_at
            )
            VALUES (
                $1, $2::uuid, $3, $4, $5, $6,
                (SELECT COALESCE(MAX(queue_order), 0) + 1 FROM ingestion_tasks),
                0, '等待入库', $7, $8, $9, $10
            )
            RETURNING {_TASK_COLUMNS}
            """,
            task_id, document_id, file_path, Path(file_path).name, fast, status,
            error_message, retry_count, started_at, finished_at,
        )
        return _serialize_ingestion_task(row)


async def get_ingestion_task(task_id: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT {_TASK_COLUMNS} FROM ingestion_tasks WHERE task_id = $1", task_id)
        return _serialize_ingestion_task(row) if row else None


async def list_ingestion_tasks(limit: int = 100) -> Sequence[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT {_TASK_COLUMNS} FROM ingestion_tasks WHERE status <> 'deleted'
                ORDER BY CASE WHEN status='processing' THEN 0 ELSE 1 END, queue_order, created_at LIMIT $1""",
            max(1, min(limit, 500)),
        )
        return [_serialize_ingestion_task(row) for row in rows]


async def get_next_queued_ingestion_task() -> Optional[Dict[str, Any]]:
    """Return the next runnable task according to the user-managed queue order."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""SELECT {_TASK_COLUMNS} FROM ingestion_tasks WHERE status='queued'
                ORDER BY queue_order, created_at LIMIT 1"""
        )
        return _serialize_ingestion_task(row) if row else None


async def update_ingestion_task_status(
    task_id: str,
    status: IngestionTaskStatus,
    document_id: Optional[str] = None,
    error_message: Optional[str] = None,
    retry_count: Optional[int] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
    progress_percent: Optional[int] = None,
    progress_stage: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE ingestion_tasks SET status=$2, document_id=COALESCE($3::uuid, document_id),
                error_message=$4, retry_count=COALESCE($5, retry_count),
                started_at=COALESCE($6, started_at), finished_at=COALESCE($7, finished_at),
                progress_percent=COALESCE($8, progress_percent),
                progress_stage=COALESCE($9, progress_stage), updated_at=CURRENT_TIMESTAMP
            WHERE task_id=$1 RETURNING {_TASK_COLUMNS}
            """,
            task_id, status, document_id, error_message, retry_count, started_at, finished_at,
            max(0, min(100, progress_percent)) if progress_percent is not None else None, progress_stage,
        )
        return _serialize_ingestion_task(row) if row else None


async def pause_ingestion_task(task_id: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""UPDATE ingestion_tasks SET status='paused', progress_stage='已暂停，等待恢复',
                    error_message=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE task_id=$1 AND status IN ('queued', 'processing') RETURNING {_TASK_COLUMNS}""",
            task_id,
        )
        return _serialize_ingestion_task(row) if row else None


async def resume_ingestion_task(task_id: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""UPDATE ingestion_tasks SET status='queued', error_message=NULL, retry_count=0,
                    progress_stage='等待入库', finished_at=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE task_id=$1 AND status IN ('paused', 'paused_quota') RETURNING {_TASK_COLUMNS}""",
            task_id,
        )
        return _serialize_ingestion_task(row) if row else None


async def delete_ingestion_task(task_id: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""UPDATE ingestion_tasks SET status='deleted', progress_stage='已删除',
                    updated_at=CURRENT_TIMESTAMP WHERE task_id=$1 AND status <> 'done'
                RETURNING {_TASK_COLUMNS}""",
            task_id,
        )
        return _serialize_ingestion_task(row) if row else None


async def reorder_ingestion_tasks(task_ids: Sequence[str]) -> Sequence[Dict[str, Any]]:
    unique_task_ids = list(dict.fromkeys(task_ids))
    if not unique_task_ids:
        return await list_ingestion_tasks()
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            for index, task_id in enumerate(unique_task_ids, start=1):
                await conn.execute(
                    "UPDATE ingestion_tasks SET queue_order=$2, updated_at=CURRENT_TIMESTAMP "
                    "WHERE task_id=$1 AND status IN ('queued', 'paused', 'paused_quota')",
                    task_id, index,
                )
    return await list_ingestion_tasks()
