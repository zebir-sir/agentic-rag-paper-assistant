from __future__ import annotations

from collections import Counter
from datetime import datetime
from threading import Lock
from typing import Any, Dict


class _RuntimeMetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.started_at = datetime.now()
            self.last_request_at = None
            self.requests_in_flight = 0
            self.total_requests = 0
            self.total_duration_ms = 0.0
            self.max_duration_ms = 0.0
            self.status_counts: Counter[str] = Counter()
            self.method_counts: Counter[str] = Counter()
            self.path_counts: Counter[str] = Counter()

    def begin_request(self) -> None:
        with self._lock:
            self.requests_in_flight += 1

    def record_request(self, *, method: str, path: str, status_code: int, duration_ms: float) -> None:
        with self._lock:
            self.requests_in_flight = max(0, self.requests_in_flight - 1)
            self.total_requests += 1
            self.total_duration_ms += max(0.0, float(duration_ms))
            self.max_duration_ms = max(self.max_duration_ms, float(duration_ms))
            self.last_request_at = datetime.now()
            self.method_counts[str(method).upper()] += 1
            self.path_counts[str(path)] += 1
            status_family = f"{int(status_code) // 100}xx"
            self.status_counts[status_family] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self.total_requests
            avg_duration_ms = self.total_duration_ms / total_requests if total_requests else 0.0
            uptime_seconds = max(0.0, (datetime.now() - self.started_at).total_seconds())
            return {
                "started_at": self.started_at,
                "uptime_seconds": round(uptime_seconds, 3),
                "requests_in_flight": self.requests_in_flight,
                "total_requests": total_requests,
                "avg_duration_ms": round(avg_duration_ms, 3),
                "max_duration_ms": round(self.max_duration_ms, 3),
                "status_counts": dict(self.status_counts),
                "method_counts": dict(self.method_counts),
                "path_counts": dict(self.path_counts),
                "last_request_at": self.last_request_at,
            }


_STORE = _RuntimeMetricsStore()


def begin_request_metrics() -> None:
    _STORE.begin_request()


def record_request_metrics(*, method: str, path: str, status_code: int, duration_ms: float) -> None:
    _STORE.record_request(
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
    )


def get_runtime_metrics_snapshot() -> Dict[str, Any]:
    return _STORE.snapshot()


def reset_runtime_metrics() -> None:
    _STORE.reset()
