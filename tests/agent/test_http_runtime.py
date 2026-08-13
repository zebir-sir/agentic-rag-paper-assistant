import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from agent.http_errors import register_exception_handlers
from agent.http_middleware import register_http_middleware
from agent.runtime_metrics import get_runtime_metrics_snapshot, reset_runtime_metrics


class _EchoBody(BaseModel):
    message: str


def _build_test_app() -> FastAPI:
    app = FastAPI()
    register_http_middleware(app)
    register_exception_handlers(app)

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    @app.get("/missing")
    async def missing():
        raise HTTPException(status_code=404, detail="missing")

    @app.post("/echo")
    async def echo(body: _EchoBody):
        return body.model_dump()

    @app.post("/chat")
    async def chat():
        return {"accepted": True}

    @app.post("/ingestion/tasks")
    async def upload():
        return {"accepted": True}

    return app


def test_request_headers_and_security_headers_present(monkeypatch):
    reset_runtime_metrics()
    monkeypatch.setenv("ENABLE_SECURITY_HEADERS", "true")
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "false")
    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/ok", headers={"X-Request-ID": "req-001"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-001"
    assert "X-Process-Time-Ms" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_http_exception_returns_structured_payload(monkeypatch):
    reset_runtime_metrics()
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "false")
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/missing")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "missing"
    assert payload["error_type"] == "HTTPException"
    assert payload["request_id"]


def test_validation_exception_returns_422(monkeypatch):
    reset_runtime_metrics()
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "false")
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/echo", json={})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "Request validation error"
    assert payload["error_type"] == "RequestValidationError"
    assert "errors" in payload["details"]


def test_unhandled_exception_returns_500(monkeypatch):
    reset_runtime_metrics()
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "false")
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    payload = response.json()
    assert payload["error"] == "Internal server error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["request_id"]


def test_rate_limit_returns_429(monkeypatch):
    reset_runtime_metrics()
    async def _deny_rate_limit(*_args, **_kwargs):
        return False

    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "true")
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setattr("agent.http_middleware.check_rate_limit", _deny_rate_limit)
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/chat")

    assert response.status_code == 429
    payload = response.json()
    assert payload["error"] == "Too many requests"
    assert payload["error_type"] == "RateLimitExceeded"
    assert payload["details"]["scope"] == "chat"


def test_host_allowlist_rejects_unexpected_host(monkeypatch):
    reset_runtime_metrics()
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "false")
    monkeypatch.setenv("ALLOWED_HOSTS", "localhost,127.0.0.1")
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/ok", headers={"host": "evil.example"})

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "Invalid host header"
    assert payload["error_type"] == "HostNotAllowed"
    assert payload["details"]["host"] == "evil.example"


def test_request_size_limit_rejects_oversized_chat_request(monkeypatch):
    reset_runtime_metrics()
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "false")
    monkeypatch.setenv("ALLOWED_HOSTS", "*")
    monkeypatch.setenv("ENABLE_REQUEST_SIZE_LIMIT", "true")
    monkeypatch.setenv("CHAT_MAX_REQUEST_BODY_BYTES", "20")
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/chat", json={"message": "this body is definitely too large"})

    assert response.status_code == 413
    payload = response.json()
    assert payload["error"] == "Request body too large"
    assert payload["error_type"] == "RequestTooLarge"
    assert payload["details"]["scope"] == "chat"


def test_request_size_limit_rejects_oversized_upload_request(monkeypatch):
    reset_runtime_metrics()
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "false")
    monkeypatch.setenv("ALLOWED_HOSTS", "*")
    monkeypatch.setenv("ENABLE_REQUEST_SIZE_LIMIT", "true")
    monkeypatch.setenv("UPLOAD_MAX_REQUEST_BODY_BYTES", "25")
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/ingestion/tasks", json={"content_base64": "x" * 200})

    assert response.status_code == 413
    payload = response.json()
    assert payload["error"] == "Request body too large"
    assert payload["error_type"] == "RequestTooLarge"
    assert payload["details"]["scope"] == "upload"


def test_runtime_metrics_collect_request_summary(monkeypatch):
    reset_runtime_metrics()
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "false")
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    client.get("/ok")
    client.get("/missing")
    client.post("/echo", json={"message": "hello"})

    metrics = get_runtime_metrics_snapshot()
    assert metrics["total_requests"] == 3
    assert metrics["requests_in_flight"] == 0
    assert metrics["method_counts"]["GET"] == 2
    assert metrics["method_counts"]["POST"] == 1
    assert metrics["status_counts"]["2xx"] == 2
    assert metrics["status_counts"]["4xx"] == 1
    assert metrics["path_counts"]["/ok"] == 1
    assert metrics["path_counts"]["/missing"] == 1
    assert metrics["path_counts"]["/echo"] == 1
