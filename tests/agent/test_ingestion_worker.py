from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("asyncpg")

from agent.ingestion_worker import (
    handle_ingestion_message,
    parse_ingestion_message,
    process_queue_message_and_ack,
    process_rabbitmq_message,
)
from agent.rabbitmq_producer import build_ingestion_dlq_message


def test_parse_ingestion_message_valid():
    payload = parse_ingestion_message(
        b'{"task_id":"task-1","document_id":null,"file_path":"documents/ui_uploads/a/paper.pdf"}'
    )
    assert payload["task_id"] == "task-1"
    assert payload["file_path"].endswith("paper.pdf")
    assert payload["document_id"] is None


def test_parse_ingestion_message_missing_task_id():
    with pytest.raises(ValueError):
        parse_ingestion_message(b'{"file_path":"documents/ui_uploads/a/paper.pdf"}')


@pytest.mark.asyncio
async def test_done_task_skips_ingestion():
    with patch("agent.ingestion_worker.get_ingestion_task", new_callable=AsyncMock) as mock_get:
        with patch("agent.ingestion_worker.ingest_saved_pdf_file", new_callable=AsyncMock) as mock_ingest:
            mock_get.return_value = {
                "task_id": "task-done",
                "status": "done",
                "retry_count": 0,
                "file_path": "documents/ui_uploads/a/paper.pdf",
                "document_id": "11111111-1111-1111-1111-111111111111",
            }
            result = await handle_ingestion_message(
                {"task_id": "task-done", "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None}
            )
            assert result["action"] == "skip_done"
            mock_ingest.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_max_task_skips_ingestion():
    with patch("agent.ingestion_worker.get_ingestion_max_retries", return_value=3):
        with patch("agent.ingestion_worker.get_ingestion_task", new_callable=AsyncMock) as mock_get:
            with patch("agent.ingestion_worker.ingest_saved_pdf_file", new_callable=AsyncMock) as mock_ingest:
                mock_get.return_value = {
                    "task_id": "task-fmax",
                    "status": "failed",
                    "retry_count": 3,
                    "file_path": "documents/ui_uploads/a/paper.pdf",
                    "document_id": None,
                }
                result = await handle_ingestion_message(
                    {"task_id": "task-fmax", "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None}
                )
                assert result["action"] == "skip_failed_max"
                mock_ingest.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_not_found_skips_ingestion():
    with patch("agent.ingestion_worker.get_ingestion_task", new_callable=AsyncMock) as mock_get:
        with patch("agent.ingestion_worker.ingest_saved_pdf_file", new_callable=AsyncMock) as mock_ingest:
            mock_get.return_value = None
            result = await handle_ingestion_message(
                {"task_id": "task-missing", "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None}
            )
            assert result["action"] == "skip_not_found"
            mock_ingest.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_waits_until_its_queue_turn():
    with patch("agent.ingestion_worker.get_ingestion_task", new_callable=AsyncMock) as mock_get:
        with patch("agent.ingestion_worker.list_ingestion_tasks", new_callable=AsyncMock) as mock_list:
            with patch("agent.ingestion_worker.ingest_saved_pdf_file", new_callable=AsyncMock) as mock_ingest:
                mock_get.return_value = {
                    "task_id": "task-later", "status": "queued", "retry_count": 0,
                    "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None, "queue_order": 2,
                }
                mock_list.return_value = [
                    {"task_id": "task-first", "status": "queued"},
                    {"task_id": "task-later", "status": "queued"},
                ]
                result = await handle_ingestion_message(
                    {"task_id": "task-later", "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None}
                )
    assert result["action"] == "defer_until_turn"
    mock_ingest.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_when_failure_and_retry_not_reached():
    with patch("agent.ingestion_worker.get_ingestion_max_retries", return_value=3):
        with patch("agent.ingestion_worker.get_ingestion_task", new_callable=AsyncMock) as mock_get:
            with patch("agent.ingestion_worker.update_ingestion_task_status", new_callable=AsyncMock) as mock_update:
                with patch("agent.ingestion_worker.ingest_saved_pdf_file", new_callable=AsyncMock) as mock_ingest:
                    with patch("agent.ingestion_worker.publish_ingestion_task", new_callable=AsyncMock) as mock_publish:
                        mock_get.return_value = {
                            "task_id": "task-r1",
                            "status": "queued",
                            "retry_count": 1,
                            "file_path": "documents/ui_uploads/a/paper.pdf",
                            "document_id": None,
                        }
                        mock_ingest.side_effect = RuntimeError("network error")
                        mock_update.side_effect = [
                            {"task_id": "task-r1", "status": "processing", "retry_count": 1, "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None},
                            {"task_id": "task-r1", "status": "queued", "retry_count": 2, "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None},
                        ]

                        result = await process_rabbitmq_message(
                            b'{"task_id":"task-r1","document_id":null,"file_path":"documents/ui_uploads/a/paper.pdf"}'
                        )
                        assert result["action"] == "retry"
                        assert mock_update.await_args_list[1].kwargs["retry_count"] == 2
                        mock_publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_document_id_is_not_marked_done():
    with patch("agent.ingestion_worker.get_ingestion_max_retries", return_value=3):
        with patch("agent.ingestion_worker.get_ingestion_task", new_callable=AsyncMock) as mock_get:
            with patch("agent.ingestion_worker.update_ingestion_task_status", new_callable=AsyncMock) as mock_update:
                with patch("agent.ingestion_worker.ingest_saved_pdf_file", new_callable=AsyncMock) as mock_ingest:
                    with patch("agent.ingestion_worker.publish_ingestion_task", new_callable=AsyncMock):
                        mock_get.return_value = {
                            "task_id": "task-empty-id", "status": "queued", "retry_count": 0,
                            "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None,
                        }
                        mock_ingest.return_value = {"document_id": None}
                        mock_update.side_effect = [
                            {"task_id": "task-empty-id", "status": "processing"},
                            {"task_id": "task-empty-id", "status": "queued", "retry_count": 1},
                        ]

                        result = await process_rabbitmq_message(
                            b'{"task_id":"task-empty-id","document_id":null,"file_path":"documents/ui_uploads/a/paper.pdf"}'
                        )

    assert result["action"] == "retry"
    assert "persisted document ID" in mock_update.await_args_list[1].kwargs["error_message"]


@pytest.mark.asyncio
async def test_failed_and_dlq_when_retry_reached():
    with patch("agent.ingestion_worker.get_ingestion_max_retries", return_value=3):
        with patch("agent.ingestion_worker.get_ingestion_task", new_callable=AsyncMock) as mock_get:
            with patch("agent.ingestion_worker.update_ingestion_task_status", new_callable=AsyncMock) as mock_update:
                with patch("agent.ingestion_worker.ingest_saved_pdf_file", new_callable=AsyncMock) as mock_ingest:
                    with patch("agent.ingestion_worker.publish_ingestion_dlq_message", new_callable=AsyncMock) as mock_dlq:
                        mock_get.return_value = {
                            "task_id": "task-rmax",
                            "status": "queued",
                            "retry_count": 2,
                            "file_path": "documents/ui_uploads/a/paper.pdf",
                            "document_id": None,
                        }
                        mock_ingest.side_effect = RuntimeError("final failure")
                        mock_update.side_effect = [
                            {"task_id": "task-rmax", "status": "processing", "retry_count": 2, "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None},
                            {"task_id": "task-rmax", "status": "failed", "retry_count": 3, "file_path": "documents/ui_uploads/a/paper.pdf", "document_id": None, "error_message": "final failure"},
                        ]

                        result = await process_rabbitmq_message(
                            b'{"task_id":"task-rmax","document_id":null,"file_path":"documents/ui_uploads/a/paper.pdf"}'
                        )
                        assert result["action"] == "failed_max"
                        assert mock_update.await_args_list[1].kwargs["status"] == "failed"
                        assert mock_update.await_args_list[1].kwargs["retry_count"] == 3
                        mock_dlq.assert_awaited_once()


def test_build_dlq_message_does_not_include_pdf_content_chunk_or_embedding():
    payload = build_ingestion_dlq_message(
        task_id="task-dlq",
        document_id=None,
        file_path="documents/ui_uploads/a/paper.pdf",
        retry_count=3,
        error_message="failed",
        failed_at="2026-05-16T12:00:00+00:00",
    )
    assert "content_base64" not in payload
    assert "chunk" not in payload
    assert "embedding" not in payload


@pytest.mark.asyncio
async def test_process_queue_message_and_ack_acks_on_success():
    class DummyMessage:
        def __init__(self):
            self.body = b'{"task_id":"task-a","file_path":"documents/ui_uploads/a/paper.pdf","document_id":null}'
            self.ack = AsyncMock()
            self.nack = AsyncMock()

    msg = DummyMessage()
    with patch("agent.ingestion_worker.process_rabbitmq_message", new_callable=AsyncMock) as mock_process:
        mock_process.return_value = {"action": "done"}
        await process_queue_message_and_ack(msg)
        msg.ack.assert_awaited_once()
        msg.nack.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_queue_message_and_ack_nacks_when_processing_fails():
    class DummyMessage:
        def __init__(self):
            self.body = b'{"task_id":"task-b","file_path":"documents/ui_uploads/a/paper.pdf","document_id":null}'
            self.ack = AsyncMock()
            self.nack = AsyncMock()

    msg = DummyMessage()
    with patch("agent.ingestion_worker.process_rabbitmq_message", new_callable=AsyncMock) as mock_process:
        mock_process.side_effect = RuntimeError("publish retry failed")
        await process_queue_message_and_ack(msg)
        msg.ack.assert_not_awaited()
        msg.nack.assert_awaited_once_with(requeue=True)


@pytest.mark.asyncio
async def test_process_queue_message_and_ack_dlqs_invalid_message_before_ack():
    class DummyMessage:
        def __init__(self):
            self.body = b'{"file_path":"documents/ui_uploads/a/paper.pdf"}'
            self.ack = AsyncMock()
            self.nack = AsyncMock()

    msg = DummyMessage()
    with patch("agent.ingestion_worker.process_rabbitmq_message", new_callable=AsyncMock) as mock_process:
        with patch("agent.ingestion_worker.publish_ingestion_dlq_message", new_callable=AsyncMock) as mock_dlq:
            mock_process.side_effect = ValueError("Missing required field: task_id")
            await process_queue_message_and_ack(msg)
            payload = mock_dlq.await_args.args[0]
            assert payload["status"] == "invalid_message"
            assert payload["error_message"] == "Missing required field: task_id"
            assert '"file_path"' in payload["raw_message"]
            msg.ack.assert_awaited_once()
            msg.nack.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_queue_message_and_ack_nacks_when_invalid_message_dlq_publish_fails():
    class DummyMessage:
        def __init__(self):
            self.body = b'{"file_path":"documents/ui_uploads/a/paper.pdf"}'
            self.ack = AsyncMock()
            self.nack = AsyncMock()

    msg = DummyMessage()
    with patch("agent.ingestion_worker.process_rabbitmq_message", new_callable=AsyncMock) as mock_process:
        with patch("agent.ingestion_worker.publish_ingestion_dlq_message", new_callable=AsyncMock) as mock_dlq:
            mock_process.side_effect = ValueError("Missing required field: task_id")
            mock_dlq.side_effect = RuntimeError("dlq unavailable")
            await process_queue_message_and_ack(msg)
            msg.ack.assert_not_awaited()
            msg.nack.assert_awaited_once_with(requeue=True)
