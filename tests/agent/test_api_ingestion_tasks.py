import pytest
from unittest.mock import AsyncMock, patch

pytest.importorskip("fastapi")
from fastapi import HTTPException

from agent.api import get_ingestion_task_endpoint, submit_ingestion_task


@pytest.mark.asyncio
async def test_get_ingestion_task_endpoint_found():
    with patch("agent.api.get_ingestion_task", new_callable=AsyncMock) as mock_get_task:
        mock_get_task.return_value = {
            "task_id": "task-10",
            "document_id": None,
            "file_path": "/tmp/file.pdf",
            "status": "queued",
            "error_message": None,
            "retry_count": 0,
            "created_at": "2026-05-16T00:00:00+00:00",
            "updated_at": "2026-05-16T00:00:00+00:00",
            "started_at": None,
            "finished_at": None,
        }

        resp = await get_ingestion_task_endpoint("task-10")
        assert resp.task_id == "task-10"
        assert resp.status == "queued"


@pytest.mark.asyncio
async def test_get_ingestion_task_endpoint_not_found():
    with patch("agent.api.get_ingestion_task", new_callable=AsyncMock) as mock_get_task:
        mock_get_task.return_value = None

        with pytest.raises(HTTPException) as exc:
            await get_ingestion_task_endpoint("missing-task")

        assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_submit_ingestion_task_returns_queued():
    with patch("agent.api.submit_async_ingestion_task", new_callable=AsyncMock) as mock_submit:
        mock_submit.return_value = {
            "task_id": "task-queued",
            "document_id": None,
            "file_path": "/tmp/file.pdf",
            "status": "queued",
            "error_message": None,
            "retry_count": 0,
            "created_at": "2026-05-16T00:00:00+00:00",
            "updated_at": "2026-05-16T00:00:00+00:00",
            "started_at": None,
            "finished_at": None,
        }
        resp = await submit_ingestion_task({"filename": "paper.pdf", "content_base64": "abc"})
        assert resp.task_id == "task-queued"
        assert resp.status == "queued"
