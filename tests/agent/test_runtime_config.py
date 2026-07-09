from agent.runtime_config import build_runtime_diagnostics


def test_runtime_diagnostics_warns_when_rate_limit_has_no_redis(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("REQUEST_ID_HEADER", "X-Request-ID")
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "true")
    monkeypatch.setenv("ENABLE_REDIS_CACHE", "true")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_CHOICE", "gpt-4o-mini")
    monkeypatch.setenv("GENERAL_WEB_SEARCH_ENABLED", "false")
    monkeypatch.delenv("GENERAL_WEB_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GENERAL_WEB_SEARCH_ENDPOINT", raising=False)

    diagnostics = build_runtime_diagnostics()

    assert diagnostics["status"] == "warning"
    warning_codes = {item["code"] for item in diagnostics["warnings"]}
    assert "redis_cache_without_redis_url" in warning_codes
    assert "rate_limit_without_redis_url" in warning_codes
    assert diagnostics["features"]["api_rate_limit"]["enabled"] is True
    assert diagnostics["features"]["api_rate_limit"]["configured"] is False


def test_runtime_diagnostics_reports_clean_runtime(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("REQUEST_ID_HEADER", "X-Request-ID")
    monkeypatch.setenv("ENABLE_API_RATE_LIMIT", "false")
    monkeypatch.setenv("ENABLE_REDIS_CACHE", "true")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_CHOICE", "gpt-4o-mini")
    monkeypatch.setenv("GENERAL_WEB_SEARCH_ENABLED", "false")
    monkeypatch.delenv("GENERAL_WEB_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GENERAL_WEB_SEARCH_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)

    diagnostics = build_runtime_diagnostics()

    assert diagnostics["status"] == "ok"
    assert diagnostics["warnings"] == []
    assert diagnostics["features"]["llm"]["configured"] is True
    assert diagnostics["features"]["redis_cache"]["configured"] is True
