"""Retry and circuit-breaker controls for external retrieval providers."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Tuple


logger = logging.getLogger(__name__)
RETRIEVAL_STATUS_KEY = "_external_retrieval_status"
FAILURE_STATES = frozenset({"not_configured", "provider_error", "circuit_open"})


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default) or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default) or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def external_retrieval_retry_count() -> int:
    return _env_int("EXTERNAL_RETRIEVAL_MAX_RETRIES", 1, minimum=0, maximum=3)


def external_retrieval_retry_backoff_seconds() -> float:
    return _env_float("EXTERNAL_RETRIEVAL_RETRY_BACKOFF_SECONDS", 0.4, minimum=0.0, maximum=5.0)


def external_retrieval_circuit_threshold() -> int:
    return _env_int("EXTERNAL_RETRIEVAL_CIRCUIT_FAILURE_THRESHOLD", 3, minimum=1, maximum=20)


def external_retrieval_circuit_cooldown_seconds() -> float:
    return _env_float("EXTERNAL_RETRIEVAL_CIRCUIT_COOLDOWN_SECONDS", 60.0, minimum=1.0, maximum=3600.0)


@dataclass(frozen=True)
class ExternalRetrievalStatus:
    source_type: str
    provider: str
    state: str
    retry_count: int = 0
    retry_after_seconds: int | None = None
    detail: str = ""

    def model_dump(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "provider": self.provider,
            "state": self.state,
            "retry_count": self.retry_count,
            "retry_after_seconds": self.retry_after_seconds,
            "detail": self.detail,
        }

    def as_tool_payload(self) -> Dict[str, Any]:
        return {RETRIEVAL_STATUS_KEY: self.model_dump()}


@dataclass(frozen=True)
class ExternalRetrievalResult:
    items: List[Dict[str, Any]]
    status: ExternalRetrievalStatus


@dataclass
class _CircuitState:
    failure_count: int = 0
    opened_until: float = 0.0


class ExternalRetrievalCircuitRegistry:
    """Process-local circuit state shared across HTTP requests.

    This intentionally avoids requiring Redis for a protective, single-instance
    deployment safeguard. A restarted API begins with closed circuits.
    """

    def __init__(self) -> None:
        self._states: Dict[str, _CircuitState] = {}
        self._lock = threading.Lock()

    def allow(self, provider: str) -> Tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            state = self._states.get(provider)
            if state is None or state.opened_until <= now:
                return True, 0
            return False, max(1, int(state.opened_until - now + 0.999))

    def record_success(self, provider: str) -> None:
        with self._lock:
            self._states.pop(provider, None)

    def record_failure(self, provider: str) -> None:
        threshold = external_retrieval_circuit_threshold()
        cooldown = external_retrieval_circuit_cooldown_seconds()
        with self._lock:
            state = self._states.setdefault(provider, _CircuitState())
            state.failure_count += 1
            if state.failure_count >= threshold:
                state.opened_until = time.monotonic() + cooldown

    def reset(self) -> None:
        with self._lock:
            self._states.clear()


external_retrieval_circuits = ExternalRetrievalCircuitRegistry()


def is_external_retrieval_status_payload(item: Any) -> bool:
    return isinstance(item, dict) and isinstance(item.get(RETRIEVAL_STATUS_KEY), dict)


def parse_external_retrieval_status(item: Any) -> ExternalRetrievalStatus | None:
    if not is_external_retrieval_status_payload(item):
        return None
    payload = dict(item[RETRIEVAL_STATUS_KEY])
    state = str(payload.get("state") or "provider_error").strip()
    source_type = str(payload.get("source_type") or "").strip()
    provider = str(payload.get("provider") or source_type or "external").strip()
    if not source_type:
        return None
    retry_after = payload.get("retry_after_seconds")
    return ExternalRetrievalStatus(
        source_type=source_type,
        provider=provider,
        state=state,
        retry_count=max(0, int(payload.get("retry_count") or 0)),
        retry_after_seconds=max(1, int(retry_after)) if isinstance(retry_after, (int, float)) else None,
        detail=str(payload.get("detail") or "").strip(),
    )


async def run_external_retrieval(
    *,
    source_type: str,
    provider: str,
    operation: Callable[[], Awaitable[List[Dict[str, Any]]]],
) -> ExternalRetrievalResult:
    allowed, retry_after = external_retrieval_circuits.allow(provider)
    if not allowed:
        return ExternalRetrievalResult(
            items=[],
            status=ExternalRetrievalStatus(
                source_type=source_type,
                provider=provider,
                state="circuit_open",
                retry_after_seconds=retry_after,
                detail="external retrieval circuit is cooling down",
            ),
        )

    max_retries = external_retrieval_retry_count()
    backoff = external_retrieval_retry_backoff_seconds()
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            items = list(await operation() or [])
            external_retrieval_circuits.record_success(provider)
            return ExternalRetrievalResult(
                items=items,
                status=ExternalRetrievalStatus(
                    source_type=source_type,
                    provider=provider,
                    state="success" if items else "no_match",
                    retry_count=attempt,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            last_error = type(exc).__name__
            if attempt < max_retries:
                await asyncio.sleep(backoff * (2**attempt))

    external_retrieval_circuits.record_failure(provider)
    logger.warning("External retrieval failed: provider=%s error=%s", provider, last_error)
    return ExternalRetrievalResult(
        items=[],
        status=ExternalRetrievalStatus(
            source_type=source_type,
            provider=provider,
            state="provider_error",
            retry_count=max_retries,
            detail=last_error,
        ),
    )


def not_configured_external_retrieval(*, source_type: str, provider: str) -> ExternalRetrievalResult:
    return ExternalRetrievalResult(
        items=[],
        status=ExternalRetrievalStatus(
            source_type=source_type,
            provider=provider,
            state="not_configured",
            detail="external retrieval provider is not configured",
        ),
    )
