from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")

from agent.api import (
    delete_ingestion_task_endpoint,
    list_ingestion_tasks_endpoint,
    pause_ingestion_task_endpoint,
    reorder_ingestion_task_queue,
)


@pytest.mark.asyncio
async def test_list_ingestion_tasks_returns_persisted_queue():
    with patch("agent.api.list_ingestion_tasks", new_callable=AsyncMock) as mocked:
        mocked.return_value = [{
            "task_id": "task-1", "document_id": None, "file_path": "/tmp/paper.pdf",
            "filename": "paper.pdf", "fast": False, "status": "queued", "queue_order": 1,
            "progress_percent": 0, "progress_stage": "等待入库", "error_message": None,
            "retry_count": 0, "created_at": "2026-08-12T00:00:00+00:00",
            "updated_at": "2026-08-12T00:00:00+00:00", "started_at": None, "finished_at": None,
        }]
        result = await list_ingestion_tasks_endpoint()
    assert result[0].filename == "paper.pdf"
    assert result[0].progress_percent == 0


@pytest.mark.asyncio
async def test_pause_and_delete_delegate_to_task_store():
    task = {
        "task_id": "task-1", "document_id": None, "file_path": "/tmp/paper.pdf", "filename": "paper.pdf",
        "fast": False, "status": "paused", "queue_order": 1, "progress_percent": 20,
        "progress_stage": "已暂停，等待恢复", "error_message": None, "retry_count": 0,
        "created_at": "2026-08-12T00:00:00+00:00", "updated_at": "2026-08-12T00:00:00+00:00",
        "started_at": None, "finished_at": None,
    }
    with patch("agent.api.pause_ingestion_task", new_callable=AsyncMock, return_value=task):
        paused = await pause_ingestion_task_endpoint("task-1")
    with patch("agent.api.delete_ingestion_task", new_callable=AsyncMock, return_value={**task, "status": "deleted"}):
        deleted = await delete_ingestion_task_endpoint("task-1")
    assert paused.status == "paused"
    assert deleted.status == "deleted"


@pytest.mark.asyncio
async def test_reorder_requires_task_id_array():
    with pytest.raises(Exception):
        await reorder_ingestion_task_queue({"task_ids": "task-1"})
