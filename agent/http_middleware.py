from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .cache_utils import check_rate_limit
from .request_context import reset_request_id, set_request_id
from .runtime_metrics import begin_request_metrics, record_request_metrics


logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _request_id_header_name() -> str:
    return str(os.getenv("REQUEST_ID_HEADER", "X-Request-ID") or "X-Request-ID").strip() or "X-Request-ID"


def _security_headers_enabled() -> bool:
    return _env_bool("ENABLE_SECURITY_HEADERS", True)


def _rate_limit_enabled() -> bool:
    return _env_bool("ENABLE_API_RATE_LIMIT", False)


def _allowed_hosts() -> list[str]:
    raw = str(os.getenv("ALLOWED_HOSTS", "*") or "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _request_size_limit_enabled() -> bool:
    return _env_bool("ENABLE_REQUEST_SIZE_LIMIT", True)


def _request_size_limit_bytes(name: str, default: int) -> int:
    return max(0, _env_int(name, default))


def _resolve_rate_limit_rule(request: Request) -> Optional[Tuple[str, int, int]]:
    if request.method.upper() != "POST":
        return None

    chat_paths = {"/chat", "/chat/stream"}
    upload_paths = {
        "/ingestion/tasks",
        "/ingestion/task-batches",
        "/openalex/add-to-kb",
    }
    if request.url.path in chat_paths:
        return (
            "chat",
            max(1, _env_int("CHAT_RATE_LIMIT_PER_MINUTE", 30)),
            60,
        )
    if request.url.path in upload_paths:
        return (
            "upload",
            max(1, _env_int("UPLOAD_RATE_LIMIT_PER_MINUTE", 10)),
            60,
        )
    return None


def _resolve_request_size_limit(request: Request) -> Optional[Tuple[str, int]]:
    if request.method.upper() not in {"POST", "PUT", "PATCH"}:
        return None

    if request.url.path in {"/chat", "/chat/stream"}:
        return (
            "chat",
            _request_size_limit_bytes("CHAT_MAX_REQUEST_BODY_BYTES", 262_144),
        )

    if request.url.path in {
        "/ingestion/tasks",
        "/ingestion/task-batches",
        "/openalex/add-to-kb",
    }:
        return (
            "upload",
            _request_size_limit_bytes("UPLOAD_MAX_REQUEST_BODY_BYTES", 45 * 1024 * 1024),
        )
    return None


def _extract_client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _apply_response_headers(response, request_id: str, process_time_ms: float) -> None:
    response.headers[_request_id_header_name()] = request_id
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    if _security_headers_enabled():
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Cache-Control", "no-store")


def _build_guard_response(
    *,
    request_id: str,
    started_at: float,
    status_code: int,
    error: str,
    error_type: str,
    details: dict,
    extra_headers: Optional[dict[str, str]] = None,
):
    process_time_ms = (time.perf_counter() - started_at) * 1000
    response = JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "error_type": error_type,
            "details": details,
            "request_id": request_id,
        },
        headers=extra_headers or {},
    )
    _apply_response_headers(response, request_id, process_time_ms)
    return response, process_time_ms


def _is_host_allowed(request: Request) -> bool:
    allowed_hosts = _allowed_hosts()
    if "*" in allowed_hosts:
        return True
    host = str(request.headers.get("host", "") or "").strip().lower()
    host_without_port = host.split(":", 1)[0]
    return host_without_port in allowed_hosts


def register_http_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id_header = _request_id_header_name()
        request_id = request.headers.get(request_id_header) or uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.request_started_at = time.time()
        token = set_request_id(request_id)
        started_at = time.perf_counter()
        begin_request_metrics()

        try:
            if not _is_host_allowed(request):
                response, process_time_ms = _build_guard_response(
                    request_id=request_id,
                    started_at=started_at,
                    status_code=400,
                    error="Invalid host header",
                    error_type="HostNotAllowed",
                    details={
                        "host": str(request.headers.get("host", "") or ""),
                        "allowed_hosts": _allowed_hosts(),
                    },
                )
                record_request_metrics(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=process_time_ms,
                )
                logger.warning(
                    "Host rejected request_id=%s method=%s path=%s host=%s",
                    request_id,
                    request.method,
                    request.url.path,
                    request.headers.get("host", ""),
                )
                return response

            size_rule = _resolve_request_size_limit(request)
            if _request_size_limit_enabled() and size_rule is not None:
                scope_name, max_bytes = size_rule
                content_length_raw = str(request.headers.get("content-length", "") or "").strip()
                if max_bytes > 0 and content_length_raw:
                    try:
                        content_length = int(content_length_raw)
                    except ValueError:
                        content_length = -1
                    if content_length > max_bytes:
                        response, process_time_ms = _build_guard_response(
                            request_id=request_id,
                            started_at=started_at,
                            status_code=413,
                            error="Request body too large",
                            error_type="RequestTooLarge",
                            details={
                                "scope": scope_name,
                                "content_length": content_length,
                                "max_bytes": max_bytes,
                            },
                            extra_headers={"Retry-After": "0"},
                        )
                        record_request_metrics(
                            method=request.method,
                            path=request.url.path,
                            status_code=response.status_code,
                            duration_ms=process_time_ms,
                        )
                        logger.warning(
                            "Request body rejected request_id=%s method=%s path=%s scope=%s content_length=%s max_bytes=%s",
                            request_id,
                            request.method,
                            request.url.path,
                            scope_name,
                            content_length,
                            max_bytes,
                        )
                        return response

            rate_limit_rule = _resolve_rate_limit_rule(request)
            if _rate_limit_enabled() and rate_limit_rule is not None:
                scope_name, limit, window_seconds = rate_limit_rule
                client_id = _extract_client_id(request)
                allowed = await check_rate_limit(
                    key=f"rate_limit:{scope_name}:{client_id}",
                    limit=limit,
                    window_seconds=window_seconds,
                )
                if not allowed:
                    response, process_time_ms = _build_guard_response(
                        request_id=request_id,
                        started_at=started_at,
                        status_code=429,
                        error="Too many requests",
                        error_type="RateLimitExceeded",
                        details={
                            "scope": scope_name,
                            "limit": limit,
                            "window_seconds": window_seconds,
                        },
                        extra_headers={"Retry-After": str(window_seconds)},
                    )
                    record_request_metrics(
                        method=request.method,
                        path=request.url.path,
                        status_code=response.status_code,
                        duration_ms=process_time_ms,
                    )
                    logger.warning(
                        "Rate limit exceeded request_id=%s method=%s path=%s client_id=%s scope=%s",
                        request_id,
                        request.method,
                        request.url.path,
                        client_id,
                        scope_name,
                    )
                    return response

            response = await call_next(request)
            process_time_ms = (time.perf_counter() - started_at) * 1000
            _apply_response_headers(response, request_id, process_time_ms)
            record_request_metrics(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=process_time_ms,
            )
            logger.info(
                "Request completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                process_time_ms,
            )
            return response
        except Exception:
            process_time_ms = (time.perf_counter() - started_at) * 1000
            record_request_metrics(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=process_time_ms,
            )
            raise
        finally:
            reset_request_id(token)
