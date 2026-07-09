from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from .app_config import get_rabbitmq_url
from .cache_utils import get_redis_client, get_redis_runtime_status
from .openalex_router import _is_openalex_enabled
from .providers import test_llm_connection
from .tools import get_general_web_search_provider, is_general_web_search_enabled
from .db_utils import test_connection


logger = logging.getLogger(__name__)
APP_VERSION = "1.1.0"


async def _check_redis_health() -> Dict[str, Any]:
    client = get_redis_client()
    if client is None:
        runtime_status = get_redis_runtime_status()
        if runtime_status["configured"]:
            return {
                "enabled": True,
                "healthy": False,
                "detail": str(runtime_status.get("unavailable_reason") or "Redis cache unavailable"),
            }
        return {"enabled": False, "healthy": False, "detail": "Redis cache not configured"}
    try:
        await client.ping()
        return {"enabled": True, "healthy": True, "detail": "Redis cache reachable"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis readiness check failed: %s", exc)
        return {"enabled": True, "healthy": False, "detail": str(exc)}


def _build_optional_component(enabled: bool, healthy: bool, detail: str) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "healthy": healthy,
        "detail": detail,
    }


async def build_readiness_status() -> Dict[str, Any]:
    database_ok = await test_connection()
    llm_ok, llm_error = await test_llm_connection()
    redis_status = await _check_redis_health()
    rabbitmq_enabled = bool(get_rabbitmq_url())
    openalex_enabled = _is_openalex_enabled()
    web_enabled = is_general_web_search_enabled()

    required_ok = database_ok and llm_ok
    optional_failures = [not redis_status["healthy"] and redis_status["enabled"]]

    if required_ok and not any(optional_failures):
        status = "ready"
    elif required_ok:
        status = "degraded"
    else:
        status = "not_ready"

    return {
        "status": status,
        "version": APP_VERSION,
        "timestamp": datetime.now(),
        "components": {
            "database": _build_optional_component(True, database_ok, "PostgreSQL reachable" if database_ok else "PostgreSQL unavailable"),
            "llm": _build_optional_component(True, llm_ok, "LLM reachable" if llm_ok else (llm_error or "LLM unavailable")),
            "redis_cache": redis_status,
            "rabbitmq": _build_optional_component(
                rabbitmq_enabled,
                rabbitmq_enabled,
                "RabbitMQ configured" if rabbitmq_enabled else "RabbitMQ not configured",
            ),
            "openalex": _build_optional_component(
                openalex_enabled,
                openalex_enabled,
                "OpenAlex enabled" if openalex_enabled else "OpenAlex not configured",
            ),
            "web_search": _build_optional_component(
                web_enabled,
                web_enabled,
                f"Web search provider: {get_general_web_search_provider()}" if web_enabled else "General web search disabled",
            ),
        },
    }
