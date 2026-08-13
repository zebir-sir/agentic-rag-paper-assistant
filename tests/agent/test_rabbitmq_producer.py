from agent.rabbitmq_producer import build_ingestion_invalid_message, build_ingestion_task_message


def test_build_ingestion_task_message_shape():
    payload = build_ingestion_task_message(
        task_id="task-123",
        document_id=None,
        file_path="documents/ui_uploads/a/paper.pdf",
    )
    assert payload == {
        "task_id": "task-123",
        "document_id": None,
        "file_path": "documents/ui_uploads/a/paper.pdf",
        "fast": False,
    }


def test_build_ingestion_invalid_message_shape():
    payload = build_ingestion_invalid_message(
        error_message="Missing required field: task_id",
        failed_at="2026-07-09T12:00:00+00:00",
        raw_message='{"file_path":"documents/ui_uploads/a/paper.pdf"}',
    )
    assert payload == {
        "task_id": None,
        "document_id": None,
        "file_path": None,
        "status": "invalid_message",
        "error_message": "Missing required field: task_id",
        "failed_at": "2026-07-09T12:00:00+00:00",
        "raw_message": '{"file_path":"documents/ui_uploads/a/paper.pdf"}',
    }
