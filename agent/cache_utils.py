import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_redis_client: Any = None


def make_cache_key(prefix: str, *parts: Any) -> str:
    normalized = [str(prefix or "").strip()]
    normalized.extend(str(part or "").strip() for part in parts)
    return ":".join(normalized)


def get_redis_client() -> Optional[Any]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    redis_url = str(os.getenv("REDIS_URL", "") or "").strip()
    if not redis_url:
        return None
    try:
        from redis.asyncio import from_url

        _redis_client = from_url(redis_url, decode_responses=True)
        return _redis_client
    except Exception as exc:
        logger.warning("Redis client init failed, fallback without cache: %s", exc)
        return None


async def cache_get_json(key: str) -> Optional[Any]:
    client = get_redis_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Redis GET failed, fallback without cache: %s", exc)
        return None


async def cache_set_json(key: str, value: Any, ttl: int) -> bool:
    client = get_redis_client()
    if client is None:
        return False
    try:
        payload = json.dumps(value, ensure_ascii=False)
        await client.set(key, payload, ex=max(1, int(ttl)))
        return True
    except Exception as exc:
        logger.warning("Redis SET failed, fallback without cache: %s", exc)
        return False


async def check_rate_limit(key: str, limit: int, window_seconds: int) -> bool:
    client = get_redis_client()
    if client is None:
        return True
    try:
        current = await client.incr(key)
        if int(current) == 1:
            await client.expire(key, max(1, int(window_seconds)))
        return int(current) <= int(limit)
    except Exception as exc:
        logger.warning("Redis rate limit check failed, fallback allow: %s", exc)
        return True
