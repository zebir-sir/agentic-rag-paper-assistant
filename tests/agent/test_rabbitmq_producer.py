from agent.rabbitmq_producer import build_ingestion_task_message


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
    }
