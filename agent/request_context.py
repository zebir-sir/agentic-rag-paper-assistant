from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional


_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def set_request_id(request_id: str) -> Token:
    return _request_id_ctx.set(str(request_id or ""))


def reset_request_id(token: Token) -> None:
    _request_id_ctx.reset(token)


def get_request_id(default: Optional[str] = None) -> Optional[str]:
    request_id = _request_id_ctx.get()
    if request_id:
        return request_id
    return default
