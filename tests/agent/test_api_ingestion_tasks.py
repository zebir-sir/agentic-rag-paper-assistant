import pytest
from unittest.mock import AsyncMock, patch

pytest.importorskip("fastapi")
from fastapi import HTTPException

from agent.api import get_ingestion_task_endpoint, submit_ingestion_task, submit_openalex_ingestion
from agent.models import OpenAlexIngestionRequest


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


@pytest.mark.asyncio
async def test_get_ingestion_task_endpoint_exposes_queue_progress_fields():
    with patch("agent.api.get_ingestion_task", new_callable=AsyncMock) as mock_get_task:
        mock_get_task.return_value = {
            "task_id": "task-progress", "document_id": None, "file_path": "/tmp/paper.pdf",
            "filename": "paper.pdf", "fast": False, "status": "processing", "queue_order": 4,
            "progress_percent": 65, "progress_stage": "正在生成检索向量", "error_message": None,
            "retry_count": 0, "created_at": "2026-05-16T00:00:00+00:00",
            "updated_at": "2026-05-16T00:00:00+00:00", "started_at": None, "finished_at": None,
        }
        response = await get_ingestion_task_endpoint("task-progress")
    assert response.filename == "paper.pdf"
    assert response.queue_order == 4
    assert response.progress_percent == 65


@pytest.mark.asyncio
async def test_submit_openalex_ingestion_queues_explicit_pdf_url():
    payload = OpenAlexIngestionRequest(
        title="Open Access Planning Paper",
        pdf_url="https://repository.example/paper.pdf",
        openalex_id="https://openalex.org/W123",
    )
    task = {
        "task_id": "openalex-task", "document_id": None, "file_path": "/tmp/paper.pdf",
        "filename": "paper.pdf", "fast": False, "status": "queued", "queue_order": 1,
        "progress_percent": 0, "progress_stage": "等待入库", "error_message": None,
        "retry_count": 0, "created_at": "2026-05-16T00:00:00+00:00",
        "updated_at": "2026-05-16T00:00:00+00:00", "started_at": None, "finished_at": None,
    }
    with patch("agent.api.add_openalex_file_to_kb", new_callable=AsyncMock, return_value=task) as mock_add:
        response = await submit_openalex_ingestion(payload)

    assert response.task_id == "openalex-task"
    mock_add.assert_awaited_once_with(
        file_url="https://repository.example/paper.pdf",
        title="Open Access Planning Paper",
        fast=False,
    )


def test_openalex_ingestion_request_rejects_non_http_pdf_url():
    with pytest.raises(ValueError, match="pdf_url"):
        OpenAlexIngestionRequest(title="Paper", pdf_url="file:///tmp/paper.pdf")
