import asyncio

from agent.external_retrieval_resilience import (
    external_retrieval_circuits,
    parse_external_retrieval_status,
)
from agent.tools import (
    OpenAlexSearchInput,
    WebSearchInput,
    openalex_search_tool,
    web_search_tool,
)


def test_web_tool_returns_not_configured_status_instead_of_empty_list(monkeypatch):
    monkeypatch.setenv("GENERAL_WEB_SEARCH_ENABLED", "false")

    result = asyncio.run(web_search_tool(WebSearchInput(query="status", limit=1)))

    assert len(result) == 1
    status = parse_external_retrieval_status(result[0])
    assert status is not None
    assert status.source_type == "general_web"
    assert status.state == "not_configured"


def test_openalex_tool_returns_provider_error_after_retry_exhaustion(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    monkeypatch.setenv("EXTERNAL_RETRIEVAL_MAX_RETRIES", "0")
    monkeypatch.setenv("EXTERNAL_RETRIEVAL_CIRCUIT_FAILURE_THRESHOLD", "3")
    external_retrieval_circuits.reset()

    def fail(*_args, **_kwargs):
        raise TimeoutError("provider unavailable")

    monkeypatch.setattr("agent.tools._sync_fetch_openalex_works", fail)
    result = asyncio.run(openalex_search_tool(OpenAlexSearchInput(query="paper", limit=1)))

    assert len(result) == 1
    status = parse_external_retrieval_status(result[0])
    assert status is not None
    assert status.source_type == "external_academic"
    assert status.state == "provider_error"
