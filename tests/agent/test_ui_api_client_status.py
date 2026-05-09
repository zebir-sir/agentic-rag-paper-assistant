from ui.api_client import (
    cancel_chat_stream,
    clean_assistant_display_text,
    map_status_text,
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
