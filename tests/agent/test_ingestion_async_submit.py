from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException
from agent import ingestion_jobs


def _fake_payload():
    return {"filename": "paper.pdf", "content_base64": "ignored", "fast": True}


@pytest.mark.asyncio
async def test_submit_async_ingestion_task_publish_success(monkeypatch):
    workspace_tmp = Path("D:/agent/agentic_rag_project-main2/.tmp")
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    import_dir = workspace_tmp / "async-submit-success"
    import_dir.mkdir(parents=True, exist_ok=True)
    target_path = import_dir / "paper.pdf"

    monkeypatch.setattr(
        ingestion_jobs,
        "_validate_and_decode_upload_payload",
        lambda payload: ("paper.pdf", b"%PDF-1.7", True),
    )
    monkeypatch.setattr(
        ingestion_jobs,
        "_make_upload_paths",
        lambda _: ("paper.pdf", import_dir, target_path),
    )

    mock_create = AsyncMock(
        return_value={
            "task_id": "task-ok",
            "document_id": None,
            "file_path": str(target_path),
            "status": "queued",
            "error_message": None,
            "retry_count": 0,
            "created_at": "2026-05-16T00:00:00+00:00",
            "updated_at": "2026-05-16T00:00:00+00:00",
            "started_at": None,
            "finished_at": None,
        }
    )
    mock_publish = AsyncMock(return_value=None)
    monkeypatch.setattr(ingestion_jobs, "create_ingestion_task", mock_create)
    monkeypatch.setattr(ingestion_jobs, "publish_ingestion_task", mock_publish)

    result = await ingestion_jobs.submit_async_ingestion_task(_fake_payload())
    assert result["task_id"] == "task-ok"
    assert result["status"] == "queued"
    assert target_path.exists()
    mock_create.assert_awaited_once()
    mock_publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_async_ingestion_task_publish_failure_marks_failed(monkeypatch):
    workspace_tmp = Path("D:/agent/agentic_rag_project-main2/.tmp")
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    import_dir = workspace_tmp / "async-submit-failed"
    import_dir.mkdir(parents=True, exist_ok=True)
    target_path = import_dir / "paper.pdf"

    monkeypatch.setattr(
        ingestion_jobs,
        "_validate_and_decode_upload_payload",
        lambda payload: ("paper.pdf", b"%PDF-1.7", True),
    )
    monkeypatch.setattr(
        ingestion_jobs,
        "_make_upload_paths",
        lambda _: ("paper.pdf", import_dir, target_path),
    )

    mock_create = AsyncMock(
        return_value={
            "task_id": "task-failed",
            "document_id": None,
            "file_path": str(target_path),
            "status": "queued",
            "error_message": None,
            "retry_count": 0,
            "created_at": "2026-05-16T00:00:00+00:00",
            "updated_at": "2026-05-16T00:00:00+00:00",
            "started_at": None,
            "finished_at": None,
        }
    )
    mock_publish = AsyncMock(side_effect=RuntimeError("rabbitmq down"))
    mock_update = AsyncMock(
        return_value={
            "task_id": "task-failed",
            "status": "failed",
            "error_message": "rabbitmq down",
        }
    )
    monkeypatch.setattr(ingestion_jobs, "create_ingestion_task", mock_create)
    monkeypatch.setattr(ingestion_jobs, "publish_ingestion_task", mock_publish)
    monkeypatch.setattr(ingestion_jobs, "update_ingestion_task_status", mock_update)

    with pytest.raises(HTTPException) as exc:
        await ingestion_jobs.submit_async_ingestion_task(_fake_payload())

    assert exc.value.status_code == 503
    mock_update.assert_awaited_once()
    kwargs = mock_update.await_args.kwargs
    assert kwargs["task_id"] == "task-failed"
    assert kwargs["status"] == "failed"
