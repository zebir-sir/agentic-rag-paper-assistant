from agent.providers import build_embedding_request_kwargs, get_embedding_dimensions


def test_get_embedding_dimensions_defaults_for_text_embedding_3_models(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)

    assert get_embedding_dimensions("text-embedding-3-small") == 1024
    assert get_embedding_dimensions("text-embedding-3-large") == 1024


def test_get_embedding_dimensions_defaults_for_qwen3_embedding_models(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)

    assert get_embedding_dimensions("Qwen/Qwen3-Embedding-4B") == 1024


def test_get_embedding_dimensions_omits_unsupported_models(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)

    assert get_embedding_dimensions("BAAI/bge-m3") is None


def test_get_embedding_dimensions_respects_explicit_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")

    assert get_embedding_dimensions("text-embedding-3-small") == 768


def test_get_embedding_dimensions_omits_unsupported_baai_model_even_with_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")

    assert get_embedding_dimensions("BAAI/bge-large-en-v1.5") is None


def test_build_embedding_request_kwargs_omits_dimensions_when_not_supported(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)

    payload = build_embedding_request_kwargs(
        model="BAAI/bge-m3",
        input_value="hello",
        encoding_format="float",
    )

    assert payload == {
        "model": "BAAI/bge-m3",
        "input": "hello",
        "encoding_format": "float",
    }


def test_build_embedding_request_kwargs_includes_dimensions_when_supported(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)

    payload = build_embedding_request_kwargs(
        model="text-embedding-3-small",
        input_value=["hello"],
        encoding_format="float",
    )

    assert payload == {
        "model": "text-embedding-3-small",
        "input": ["hello"],
        "encoding_format": "float",
        "dimensions": 1024,
    }
