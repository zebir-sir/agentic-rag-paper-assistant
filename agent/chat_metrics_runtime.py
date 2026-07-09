from __future__ import annotations

import os
from collections import Counter, deque
from datetime import datetime
from threading import Lock
from typing import Any, Deque, Dict, Optional


def _recent_limit() -> int:
    raw = str(os.getenv("CHAT_METRICS_MAX_RECENT", "50") or "50").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 50
    return max(1, value)


def _resolve_source_mix(*, local_source_count: int, web_source_count: int) -> str:
    if local_source_count > 0 and web_source_count > 0:
        return "mixed"
    if local_source_count > 0:
        return "local_only"
    if web_source_count > 0:
        return "web_only"
    return "none"


class _ChatMetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.started_at = datetime.now()
            self.total_requests = 0
            self.stream_requests = 0
            self.total_response_chars = 0
            self.total_tool_call_count = 0
            self.status_counts: Counter[str] = Counter()
            self.backend_counts: Counter[str] = Counter()
            self.route_counts: Counter[str] = Counter()
            self.effective_search_type_counts: Counter[str] = Counter()
            self.source_mix_counts: Counter[str] = Counter()
            self.recent_requests: Deque[Dict[str, Any]] = deque(maxlen=_recent_limit())

    def record(
        self,
        *,
        request_id: Optional[str],
        session_id: str,
        route: str,
        status: str,
        response_backend: str,
        requested_search_type: str,
        effective_search_type: str,
        use_web_search: bool,
        use_react: bool,
        compression_used: bool,
        tool_call_count: int,
        local_source_count: int,
        web_source_count: int,
        response_chars: int,
    ) -> None:
        local_count = max(0, int(local_source_count))
        web_count = max(0, int(web_source_count))
        item = {
            "occurred_at": datetime.now(),
            "request_id": request_id or None,
            "session_id": str(session_id or ""),
            "route": str(route or "/chat"),
            "status": str(status or "success"),
            "response_backend": str(response_backend or "unknown"),
            "requested_search_type": str(requested_search_type or "hybrid"),
            "effective_search_type": str(effective_search_type or "hybrid"),
            "use_web_search": bool(use_web_search),
            "use_react": bool(use_react),
            "compression_used": bool(compression_used),
            "tool_call_count": max(0, int(tool_call_count)),
            "source_count": local_count + web_count,
            "local_source_count": local_count,
            "web_source_count": web_count,
            "source_mix": _resolve_source_mix(
                local_source_count=local_count,
                web_source_count=web_count,
            ),
            "response_chars": max(0, int(response_chars)),
        }
        with self._lock:
            self.total_requests += 1
            if item["route"] == "/chat/stream":
                self.stream_requests += 1
            self.total_response_chars += item["response_chars"]
            self.total_tool_call_count += item["tool_call_count"]
            self.status_counts[item["status"]] += 1
            self.backend_counts[item["response_backend"]] += 1
            self.route_counts[item["route"]] += 1
            self.effective_search_type_counts[item["effective_search_type"]] += 1
            self.source_mix_counts[item["source_mix"]] += 1
            self.recent_requests.appendleft(item)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self.total_requests
            avg_response_chars = self.total_response_chars / total_requests if total_requests else 0.0
            avg_tool_call_count = self.total_tool_call_count / total_requests if total_requests else 0.0
            return {
                "started_at": self.started_at,
                "total_requests": total_requests,
                "stream_requests": self.stream_requests,
                "avg_response_chars": round(avg_response_chars, 3),
                "avg_tool_call_count": round(avg_tool_call_count, 3),
                "status_counts": dict(self.status_counts),
                "backend_counts": dict(self.backend_counts),
                "route_counts": dict(self.route_counts),
                "effective_search_type_counts": dict(self.effective_search_type_counts),
                "source_mix_counts": dict(self.source_mix_counts),
                "recent_requests": list(self.recent_requests),
            }


_STORE = _ChatMetricsStore()


def record_chat_request_metric(
    *,
    request_id: Optional[str],
    session_id: str,
    route: str,
    status: str,
    response_backend: str,
    requested_search_type: str,
    effective_search_type: str,
    use_web_search: bool,
    use_react: bool,
    compression_used: bool,
    tool_call_count: int,
    local_source_count: int,
    web_source_count: int,
    response_chars: int,
) -> None:
    _STORE.record(
        request_id=request_id,
        session_id=session_id,
        route=route,
        status=status,
        response_backend=response_backend,
        requested_search_type=requested_search_type,
        effective_search_type=effective_search_type,
        use_web_search=use_web_search,
        use_react=use_react,
        compression_used=compression_used,
        tool_call_count=tool_call_count,
        local_source_count=local_source_count,
        web_source_count=web_source_count,
        response_chars=response_chars,
    )


def get_chat_metrics_snapshot() -> Dict[str, Any]:
    return _STORE.snapshot()


def reset_chat_metrics() -> None:
    _STORE.reset()
