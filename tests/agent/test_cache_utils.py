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


def test_get_redis_runtime_status_without_configuration(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(cache_utils, "_redis_client", None)
    monkeypatch.setattr(cache_utils, "_redis_unavailable_reason", "REDIS_URL is empty")

    status = cache_utils.get_redis_runtime_status()

    assert status["configured"] is False
    assert status["connected"] is False
    assert status["unavailable_reason"] == "REDIS_URL is empty"


@pytest.mark.asyncio
async def test_close_redis_client_clears_initialized_client(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    fake_client = FakeClient()
    monkeypatch.setattr(cache_utils, "_redis_client", fake_client)

    await cache_utils.close_redis_client()

    assert fake_client.closed is True
    assert cache_utils._redis_client is None
