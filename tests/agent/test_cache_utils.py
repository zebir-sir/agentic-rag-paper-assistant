import pytest

from agent import cache_utils


def test_make_cache_key_stable():
    key1 = cache_utils.make_cache_key("embedding", "model-x", "abc123")
    key2 = cache_utils.make_cache_key("embedding", "model-x", "abc123")
    assert key1 == "embedding:model-x:abc123"
    assert key1 == key2


@pytest.mark.asyncio
async def test_cache_get_json_redis_unavailable_no_raise(monkeypatch):
    class BrokenClient:
        async def get(self, _key):
            raise RuntimeError("redis down")

    monkeypatch.setattr(cache_utils, "get_redis_client", lambda: BrokenClient())
    value = await cache_utils.cache_get_json("k1")
    assert value is None
