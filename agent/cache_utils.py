import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_redis_client: Any = None
_redis_unavailable_reason: Optional[str] = None


def make_cache_key(prefix: str, *parts: Any) -> str:
    normalized = [str(prefix or "").strip()]
    normalized.extend(str(part or "").strip() for part in parts)
    return ":".join(normalized)


def _redis_url() -> str:
    return str(os.getenv("REDIS_URL", "") or "").strip()


def get_redis_unavailable_reason() -> Optional[str]:
    return _redis_unavailable_reason


def get_redis_runtime_status() -> dict[str, Any]:
    redis_url = _redis_url()
    return {
        "configured": bool(redis_url),
        "connected": _redis_client is not None and _redis_unavailable_reason is None,
        "client_initialized": _redis_client is not None,
        "unavailable_reason": _redis_unavailable_reason,
    }


def get_redis_client() -> Optional[Any]:
    global _redis_client, _redis_unavailable_reason
    if _redis_client is not None:
        return _redis_client
    redis_url = _redis_url()
    if not redis_url:
        _redis_unavailable_reason = "REDIS_URL is empty"
        return None
    try:
        from redis.asyncio import from_url

        _redis_client = from_url(redis_url, decode_responses=True)
        _redis_unavailable_reason = None
        return _redis_client
    except Exception as exc:
        _redis_unavailable_reason = str(exc)
        logger.warning("Redis client init failed, fallback without cache: %s", exc)
        return None


async def startup_redis_client() -> bool:
    global _redis_client, _redis_unavailable_reason
    client = get_redis_client()
    if client is None:
        return False
    try:
        await client.ping()
        _redis_unavailable_reason = None
        logger.info("Redis client startup ping succeeded")
        return True
    except Exception as exc:
        _redis_unavailable_reason = str(exc)
        logger.warning("Redis startup ping failed, fallback without cache: %s", exc)
        try:
            await client.aclose()
        except Exception:
            logger.debug("Redis client close after failed startup ping also failed", exc_info=True)
        _redis_client = None
        return False


async def close_redis_client() -> None:
    global _redis_client
    client = _redis_client
    _redis_client = None
    if client is None:
        return
    try:
        await client.aclose()
    except Exception as exc:
        logger.warning("Redis client close failed: %s", exc)


async def cache_get_json(key: str) -> Optional[Any]:
    global _redis_unavailable_reason
    client = get_redis_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        if raw is None:
            return None
        _redis_unavailable_reason = None
        return json.loads(raw)
    except Exception as exc:
        _redis_unavailable_reason = str(exc)
        logger.warning("Redis GET failed, fallback without cache: %s", exc)
        return None


async def cache_set_json(key: str, value: Any, ttl: int) -> bool:
    global _redis_unavailable_reason
    client = get_redis_client()
    if client is None:
        return False
    try:
        payload = json.dumps(value, ensure_ascii=False)
        await client.set(key, payload, ex=max(1, int(ttl)))
        _redis_unavailable_reason = None
        return True
    except Exception as exc:
        _redis_unavailable_reason = str(exc)
        logger.warning("Redis SET failed, fallback without cache: %s", exc)
        return False


async def check_rate_limit(key: str, limit: int, window_seconds: int) -> bool:
    global _redis_unavailable_reason
    client = get_redis_client()
    if client is None:
        return True
    try:
        current = await client.incr(key)
        if int(current) == 1:
            await client.expire(key, max(1, int(window_seconds)))
        _redis_unavailable_reason = None
        return int(current) <= int(limit)
    except Exception as exc:
        _redis_unavailable_reason = str(exc)
        logger.warning("Redis rate limit check failed, fallback allow: %s", exc)
        return True
