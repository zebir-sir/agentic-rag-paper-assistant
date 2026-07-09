from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)


def _build_error_payload(
    *,
    error: str,
    error_type: str,
    request_id: Optional[str],
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error": error,
        "error_type": error_type,
        "request_id": request_id,
    }
    if details:
        payload["details"] = details
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "Request validation failed request_id=%s path=%s errors=%s",
            request_id,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content=_build_error_payload(
                error="Request validation error",
                error_type="RequestValidationError",
                request_id=request_id,
                details={"errors": exc.errors()},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "HTTP exception request_id=%s path=%s status=%s detail=%s",
            request_id,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_payload(
                error=str(exc.detail),
                error_type="HTTPException",
                request_id=request_id,
            ),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception(
            "Unhandled exception request_id=%s path=%s",
            request_id,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=_build_error_payload(
                error="Internal server error",
                error_type=type(exc).__name__,
                request_id=request_id,
            ),
        )
