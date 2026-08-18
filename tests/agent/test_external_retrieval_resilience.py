import asyncio

from agent.external_retrieval_resilience import (
    external_retrieval_circuits,
    run_external_retrieval,
)


def test_external_retrieval_retries_then_opens_circuit(monkeypatch):
    monkeypatch.setenv("EXTERNAL_RETRIEVAL_MAX_RETRIES", "1")
    monkeypatch.setenv("EXTERNAL_RETRIEVAL_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("EXTERNAL_RETRIEVAL_CIRCUIT_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("EXTERNAL_RETRIEVAL_CIRCUIT_COOLDOWN_SECONDS", "60")
    external_retrieval_circuits.reset()
    calls = 0

    async def fail():
        nonlocal calls
        calls += 1
        raise TimeoutError("provider timed out")

    async def run_scenario():
        first = await run_external_retrieval(
            source_type="general_web",
            provider="test-web",
            operation=fail,
        )
        second = await run_external_retrieval(
            source_type="general_web",
            provider="test-web",
            operation=fail,
        )
        blocked = await run_external_retrieval(
            source_type="general_web",
            provider="test-web",
            operation=fail,
        )
        return first, second, blocked

    first, second, blocked = asyncio.run(run_scenario())

    assert first.status.state == "provider_error"
    assert first.status.retry_count == 1
    assert second.status.state == "provider_error"
    assert blocked.status.state == "circuit_open"
    assert blocked.status.retry_after_seconds is not None
    assert calls == 4


def test_successful_external_retrieval_closes_existing_circuit_state(monkeypatch):
    monkeypatch.setenv("EXTERNAL_RETRIEVAL_MAX_RETRIES", "0")
    monkeypatch.setenv("EXTERNAL_RETRIEVAL_CIRCUIT_FAILURE_THRESHOLD", "3")
    external_retrieval_circuits.reset()

    async def success():
        return [{"title": "result"}]

    outcome = asyncio.run(
        run_external_retrieval(
            source_type="external_academic",
            provider="test-openalex",
            operation=success,
        )
    )

    assert outcome.status.state == "success"
    assert outcome.items == [{"title": "result"}]
