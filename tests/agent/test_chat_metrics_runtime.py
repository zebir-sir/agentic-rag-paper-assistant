from agent.chat_metrics_runtime import (
    get_chat_metrics_snapshot,
    record_chat_request_metric,
    reset_chat_metrics,
)


def test_chat_metrics_summary_tracks_backend_and_source_mix():
    reset_chat_metrics()

    record_chat_request_metric(
        request_id="req-1",
        session_id="session-1",
        route="/chat",
        status="success",
        response_backend="langgraph",
        requested_search_type="hybrid",
        effective_search_type="vector",
        use_web_search=False,
        use_react=True,
        compression_used=False,
        tool_call_count=2,
        local_source_count=2,
        web_source_count=0,
        response_chars=120,
    )
    record_chat_request_metric(
        request_id="req-2",
        session_id="session-2",
        route="/chat/stream",
        status="cancelled",
        response_backend="openalex_first",
        requested_search_type="hybrid",
        effective_search_type="hybrid",
        use_web_search=True,
        use_react=False,
        compression_used=True,
        tool_call_count=1,
        local_source_count=1,
        web_source_count=2,
        response_chars=80,
    )

    snapshot = get_chat_metrics_snapshot()

    assert snapshot["total_requests"] == 2
    assert snapshot["stream_requests"] == 1
    assert snapshot["backend_counts"]["langgraph"] == 1
    assert snapshot["backend_counts"]["openalex_first"] == 1
    assert snapshot["status_counts"]["success"] == 1
    assert snapshot["status_counts"]["cancelled"] == 1
    assert snapshot["source_mix_counts"]["local_only"] == 1
    assert snapshot["source_mix_counts"]["mixed"] == 1
    assert snapshot["avg_response_chars"] == 100.0
    assert snapshot["avg_tool_call_count"] == 1.5
    assert snapshot["recent_requests"][0]["request_id"] == "req-2"
    assert snapshot["recent_requests"][1]["request_id"] == "req-1"


def test_chat_metrics_recent_limit_uses_env(monkeypatch):
    monkeypatch.setenv("CHAT_METRICS_MAX_RECENT", "1")
    reset_chat_metrics()

    record_chat_request_metric(
        request_id="req-1",
        session_id="session-1",
        route="/chat",
        status="success",
        response_backend="langchain",
        requested_search_type="hybrid",
        effective_search_type="hybrid",
        use_web_search=False,
        use_react=False,
        compression_used=False,
        tool_call_count=0,
        local_source_count=0,
        web_source_count=0,
        response_chars=20,
    )
    record_chat_request_metric(
        request_id="req-2",
        session_id="session-2",
        route="/chat",
        status="error",
        response_backend="langchain",
        requested_search_type="hybrid",
        effective_search_type="hybrid",
        use_web_search=False,
        use_react=False,
        compression_used=False,
        tool_call_count=0,
        local_source_count=0,
        web_source_count=0,
        response_chars=10,
    )

    snapshot = get_chat_metrics_snapshot()
    assert len(snapshot["recent_requests"]) == 1
    assert snapshot["recent_requests"][0]["request_id"] == "req-2"
