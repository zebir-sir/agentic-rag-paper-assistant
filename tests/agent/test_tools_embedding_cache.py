from types import SimpleNamespace

import pytest

pytest.importorskip("asyncpg")

from agent import tools


@pytest.mark.asyncio
async def test_generate_embedding_cache_hit_skips_provider(monkeypatch):
    async def fake_cache_get(_key):
        return [0.11, 0.22]

    async def fake_cache_set(_key, _value, _ttl):
        return True

    class FailingEmbeddings:
        async def create(self, **_kwargs):
            raise AssertionError("embedding provider should not be called on cache hit")

    monkeypatch.setattr(tools, "cache_get_json", fake_cache_get)
    monkeypatch.setattr(tools, "cache_set_json", fake_cache_set)
    monkeypatch.setattr(tools, "embedding_client", SimpleNamespace(embeddings=FailingEmbeddings()))
    monkeypatch.setenv("ENABLE_REDIS_CACHE", "true")

    embedding = await tools.generate_embedding("cache-hit-query")
    assert embedding == [0.11, 0.22]


@pytest.mark.asyncio
async def test_generate_embedding_cache_miss_calls_provider_and_sets_cache(monkeypatch):
    calls = {"create": 0, "set": 0}
    captured = {}

    async def fake_cache_get(_key):
        return None

    async def fake_cache_set(_key, value, _ttl):
        calls["set"] += 1
        assert value == [0.3, 0.4]
        return True

    class DummyEmbeddings:
        async def create(self, **_kwargs):
            calls["create"] += 1
            captured.update(_kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.3, 0.4])])

    monkeypatch.setattr(tools, "cache_get_json", fake_cache_get)
    monkeypatch.setattr(tools, "cache_set_json", fake_cache_set)
    monkeypatch.setattr(tools, "EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(tools, "embedding_client", SimpleNamespace(embeddings=DummyEmbeddings()))
    monkeypatch.setenv("ENABLE_REDIS_CACHE", "true")

    embedding = await tools.generate_embedding("cache-miss-query")
    assert embedding == [0.3, 0.4]
    assert calls["create"] == 1
    assert calls["set"] == 1
    assert captured["dimensions"] == 1024


@pytest.mark.asyncio
async def test_generate_embedding_omits_dimensions_for_bge_model(monkeypatch):
    captured = {}

    async def fake_cache_get(_key):
        return None

    async def fake_cache_set(_key, _value, _ttl):
        return True

    class DummyEmbeddings:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 0.6])])

    monkeypatch.setattr(tools, "cache_get_json", fake_cache_get)
    monkeypatch.setattr(tools, "cache_set_json", fake_cache_set)
    monkeypatch.setattr(tools, "EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setattr(tools, "embedding_client", SimpleNamespace(embeddings=DummyEmbeddings()))
    monkeypatch.setenv("ENABLE_REDIS_CACHE", "false")
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)

    embedding = await tools.generate_embedding("bge-query")

    assert embedding == [0.5, 0.6]
    assert "dimensions" not in captured
