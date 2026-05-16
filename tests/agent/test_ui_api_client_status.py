from ui.api_client import (
    cancel_chat_stream,
    clean_assistant_display_text,
    get_ingestion_task,
    map_async_ingestion_task_status,
    map_status_text,
    submit_async_ingestion_task,
    start_pdf_ingestion,
    should_display_status_event,
)


def test_status_internal_not_displayed():
    event = {"type": "status", "content": "内部调试", "phase": "internal", "user_visible": False}
    assert should_display_status_event(event) is False


def test_status_generation_displayed():
    event = {"type": "status", "content": "正在生成回答...", "phase": "generation", "user_visible": True}
    assert should_display_status_event(event) is True
    assert map_status_text(event["content"], event["phase"]) == "正在生成回答..."


def test_clean_assistant_display_text_rewrites_legacy_warning():
    legacy = "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    cleaned = clean_assistant_display_text(legacy)
    assert "以下内容" not in cleaned
    assert "本轮没有可核对的检索片段" in cleaned


def test_start_pdf_ingestion_fast_payload_true(monkeypatch):
    captured = {}

    class DummyResp:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "accepted", "job_id": "job-1", "filename": "paper.pdf"}

    def fake_post(url, json=None, timeout=0):
        captured["url"] = url
        captured["json"] = json or {}
        captured["timeout"] = timeout
        return DummyResp()

    monkeypatch.setattr("ui.api_client.requests.post", fake_post)
    ok, _, payload = start_pdf_ingestion("http://localhost:8059", "paper.pdf", b"%PDF test", fast=True)

    assert ok is True
    assert payload.get("status") == "accepted"
    assert payload.get("job_id")
    assert captured["url"].endswith("/documents/upload/start")
    assert captured["json"]["filename"] == "paper.pdf"
    assert "content_base64" in captured["json"]
    assert captured["json"]["fast"] is True


def test_start_pdf_ingestion_fast_payload_false(monkeypatch):
    captured = {}

    class DummyResp:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "accepted", "job_id": "job-2", "filename": "paper.pdf"}

    def fake_post(url, json=None, timeout=0):
        captured["url"] = url
        captured["json"] = json or {}
        captured["timeout"] = timeout
        return DummyResp()

    monkeypatch.setattr("ui.api_client.requests.post", fake_post)
    ok, _, payload = start_pdf_ingestion("http://localhost:8059", "paper.pdf", b"%PDF test", fast=False)

    assert ok is True
    assert payload.get("status") == "accepted"
    assert payload.get("job_id")
    assert captured["url"].endswith("/documents/upload/start")
    assert captured["json"]["filename"] == "paper.pdf"
    assert "content_base64" in captured["json"]
    assert captured["json"]["fast"] is False


def test_cancel_chat_stream_parses_cancelled(monkeypatch):
    class DummyResp:
        status_code = 200

        @staticmethod
        def json():
            return {"run_id": "r1", "status": "cancelled"}

    monkeypatch.setattr("ui.api_client.requests.post", lambda *args, **kwargs: DummyResp())
    ok, status, payload = cancel_chat_stream("http://localhost:8059", "r1")
    assert ok is True
    assert status == "cancelled"
    assert payload["run_id"] == "r1"


def test_cancel_chat_stream_parses_not_found(monkeypatch):
    class DummyResp:
        status_code = 200

        @staticmethod
        def json():
            return {"run_id": "r2", "status": "not_found"}

    monkeypatch.setattr("ui.api_client.requests.post", lambda *args, **kwargs: DummyResp())
    ok, status, payload = cancel_chat_stream("http://localhost:8059", "r2")
    assert ok is True
    assert status == "not_found"
    assert payload["run_id"] == "r2"


def test_submit_async_ingestion_task_payload(monkeypatch):
    captured = {}

    class DummyResp:
        status_code = 200

        @staticmethod
        def json():
            return {"task_id": "task-1", "status": "queued", "file_path": "documents/ui_uploads/a.pdf"}

    def fake_post(url, json=None, timeout=0):
        captured["url"] = url
        captured["json"] = json or {}
        captured["timeout"] = timeout
        return DummyResp()

    monkeypatch.setattr("ui.api_client.requests.post", fake_post)
    ok, _, payload = submit_async_ingestion_task("http://localhost:8059", "paper.pdf", b"%PDF test", fast=True)

    assert ok is True
    assert payload["task_id"] == "task-1"
    assert captured["url"].endswith("/ingestion/tasks")
    assert captured["json"]["filename"] == "paper.pdf"
    assert captured["json"]["fast"] is True
    assert "content_base64" in captured["json"]


def test_get_ingestion_task(monkeypatch):
    class DummyResp:
        status_code = 200

        @staticmethod
        def json():
            return {"task_id": "task-2", "status": "processing"}

    monkeypatch.setattr("ui.api_client.requests.get", lambda *args, **kwargs: DummyResp())
    ok, msg, payload = get_ingestion_task("http://localhost:8059", "task-2")

    assert ok is True
    assert msg == "ok"
    assert payload["status"] == "processing"


def test_map_async_ingestion_task_status():
    assert "queued" in map_async_ingestion_task_status("queued").lower()
    assert "processing" in map_async_ingestion_task_status("processing").lower()
    assert "done" in map_async_ingestion_task_status("done").lower()
    failed = map_async_ingestion_task_status("failed", "boom")
    assert "boom" in failed.lower()
