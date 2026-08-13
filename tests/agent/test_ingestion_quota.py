from unittest.mock import AsyncMock, patch

import pytest

from agent.ingestion_quota import is_quota_exhausted_error
from agent.ingestion_worker import handle_ingestion_message


def test_detects_provider_quota_errors():
    assert is_quota_exhausted_error("HTTP 429: insufficient_quota")
    assert is_quota_exhausted_error("额度不足，请充值后重试")
    assert not is_quota_exhausted_error("temporary connection reset")


@pytest.mark.asyncio
async def test_quota_failure_pauses_without_retry():
    current = {"task_id": "quota-1", "status": "queued", "retry_count": 0, "file_path": "paper.pdf", "document_id": None, "fast": False}
    with patch("agent.ingestion_worker.get_ingestion_task", new=AsyncMock(return_value=current)):
        with patch("agent.ingestion_worker.update_ingestion_task_status", new=AsyncMock(return_value={"status": "paused_quota"})) as update:
            with patch("agent.ingestion_worker.ingest_saved_pdf_file", new=AsyncMock(side_effect=RuntimeError("HTTP 429 insufficient_quota"))):
                result = await handle_ingestion_message({"task_id": "quota-1", "file_path": "paper.pdf", "fast": False})

    assert result["action"] == "paused_quota"
    assert update.await_args_list[-1].kwargs["status"] == "paused_quota"
