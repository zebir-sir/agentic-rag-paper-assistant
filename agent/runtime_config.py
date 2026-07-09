from __future__ import annotations

import os
from typing import Any, Dict, List

from .app_config import get_rabbitmq_url
from .cache_utils import get_redis_runtime_status
from .openalex_router import _is_openalex_enabled
from .tools import get_general_web_search_provider, is_general_web_search_enabled


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or default).strip()


def _build_feature(enabled: bool, configured: bool, detail: str) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "configured": configured,
        "detail": detail,
    }


def build_runtime_diagnostics() -> Dict[str, Any]:
    warnings: List[Dict[str, str]] = []

    app_env = _env_text("APP_ENV", "development")
    request_id_header = _env_text("REQUEST_ID_HEADER", "X-Request-ID") or "X-Request-ID"

    openai_api_key = _env_text("OPENAI_API_KEY")
    openai_base_url = _env_text("OPENAI_BASE_URL")
    llm_choice = _env_text("LLM_CHOICE")
    redis_url = _env_text("REDIS_URL")
    redis_runtime = get_redis_runtime_status()
    rabbitmq_url = get_rabbitmq_url()
    rate_limit_enabled = _env_bool("ENABLE_API_RATE_LIMIT", False)
    redis_cache_enabled = _env_bool("ENABLE_REDIS_CACHE", True)
    security_headers_enabled = _env_bool("ENABLE_SECURITY_HEADERS", True)
    web_enabled = is_general_web_search_enabled()
    web_provider = get_general_web_search_provider()
    openalex_enabled = _is_openalex_enabled()

    features: Dict[str, Dict[str, Any]] = {
        "llm": _build_feature(
            enabled=True,
            configured=bool(openai_api_key and openai_base_url and llm_choice),
            detail="OpenAI-compatible LLM runtime",
        ),
        "redis_cache": _build_feature(
            enabled=redis_cache_enabled,
            configured=bool(redis_url),
            detail=(
                f"Redis-backed embedding cache ({redis_runtime.get('unavailable_reason')})"
                if redis_runtime.get("unavailable_reason")
                else "Redis-backed embedding cache"
            ),
        ),
        "api_rate_limit": _build_feature(
            enabled=rate_limit_enabled,
            configured=bool(redis_url),
            detail="Redis-backed API rate limiting",
        ),
        "rabbitmq_ingestion": _build_feature(
            enabled=bool(rabbitmq_url),
            configured=bool(rabbitmq_url),
            detail="RabbitMQ async ingestion worker",
        ),
        "openalex": _build_feature(
            enabled=openalex_enabled,
            configured=openalex_enabled,
            detail="OpenAlex academic search",
        ),
        "web_search": _build_feature(
            enabled=web_enabled,
            configured=web_enabled,
            detail=f"General web search provider: {web_provider}",
        ),
        "security_headers": _build_feature(
            enabled=security_headers_enabled,
            configured=True,
            detail="API response hardening headers",
        ),
        "host_allowlist": _build_feature(
            enabled=_env_text("ALLOWED_HOSTS", "*") != "*",
            configured=True,
            detail="Host header allowlist guard",
        ),
        "request_size_limit": _build_feature(
            enabled=_env_bool("ENABLE_REQUEST_SIZE_LIMIT", True),
            configured=True,
            detail="Request body size guard for chat and upload routes",
        ),
        "chat_request_metrics": _build_feature(
            enabled=True,
            configured=True,
            detail="In-memory chat request summaries for runtime inspection",
        ),
    }

    if not openai_api_key:
        warnings.append(
            {
                "code": "missing_openai_api_key",
                "message": "OPENAI_API_KEY is empty; LLM and embedding requests will fail.",
            }
        )
    if not openai_base_url:
        warnings.append(
            {
                "code": "missing_openai_base_url",
                "message": "OPENAI_BASE_URL is empty; the API backend has no model endpoint.",
            }
        )
    if not llm_choice:
        warnings.append(
            {
                "code": "missing_llm_choice",
                "message": "LLM_CHOICE is empty; model selection falls back unpredictably.",
            }
        )

    if redis_cache_enabled and not redis_url:
        warnings.append(
            {
                "code": "redis_cache_without_redis_url",
                "message": "ENABLE_REDIS_CACHE is on but REDIS_URL is empty; cache will silently degrade.",
            }
        )
    if rate_limit_enabled and not redis_url:
        warnings.append(
            {
                "code": "rate_limit_without_redis_url",
                "message": "ENABLE_API_RATE_LIMIT is on but REDIS_URL is empty; rate limiting cannot take effect.",
            }
        )

    rabbitmq_fields = [
        _env_text("RABBITMQ_DEFAULT_USER"),
        _env_text("RABBITMQ_DEFAULT_PASS"),
        _env_text("RABBITMQ_HOST"),
        _env_text("RABBITMQ_PORT"),
    ]
    if any(rabbitmq_fields) and not rabbitmq_url:
        warnings.append(
            {
                "code": "partial_rabbitmq_config",
                "message": "RabbitMQ environment variables are partially configured; async ingestion may not connect.",
            }
        )

    if _env_text("OPENALEX_MAILTO") and not openalex_enabled:
        warnings.append(
            {
                "code": "openalex_mailto_without_api_key",
                "message": "OPENALEX_MAILTO is set but OPENALEX_API_KEY is empty; OpenAlex remains disabled.",
            }
        )

    web_provider_raw = _env_text("GENERAL_WEB_SEARCH_PROVIDER", "custom").lower()
    web_api_key = _env_text("GENERAL_WEB_SEARCH_API_KEY")
    web_endpoint = _env_text("GENERAL_WEB_SEARCH_ENDPOINT")
    if _env_bool("GENERAL_WEB_SEARCH_ENABLED", False) and not web_enabled:
        warnings.append(
            {
                "code": "general_web_search_incomplete",
                "message": "GENERAL_WEB_SEARCH_ENABLED is on but provider credentials are incomplete.",
            }
        )
    if web_provider_raw == "custom" and web_api_key and not web_endpoint:
        warnings.append(
            {
                "code": "custom_web_search_missing_endpoint",
                "message": "GENERAL_WEB_SEARCH_ENDPOINT is empty for custom web search provider.",
            }
        )

    return {
        "status": "warning" if warnings else "ok",
        "app_env": app_env,
        "request_id_header": request_id_header,
        "features": features,
        "warnings": warnings,
    }
