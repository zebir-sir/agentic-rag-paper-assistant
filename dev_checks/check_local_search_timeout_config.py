from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "asyncpg" not in sys.modules:
    asyncpg_mod = types.ModuleType("asyncpg")
    asyncpg_pool_mod = types.ModuleType("asyncpg.pool")
    asyncpg_pool_mod.Pool = object
    asyncpg_mod.pool = asyncpg_pool_mod
    sys.modules["asyncpg"] = asyncpg_mod
    sys.modules["asyncpg.pool"] = asyncpg_pool_mod

if "pydantic_ai.providers.openai" not in sys.modules:
    pyd_mod = types.ModuleType("pydantic_ai")
    pyd_providers_mod = types.ModuleType("pydantic_ai.providers")
    pyd_provider_openai_mod = types.ModuleType("pydantic_ai.providers.openai")
    pyd_models_mod = types.ModuleType("pydantic_ai.models")
    pyd_model_openai_mod = types.ModuleType("pydantic_ai.models.openai")
    pyd_provider_openai_mod.OpenAIProvider = object
    pyd_model_openai_mod.OpenAIChatModel = object
    sys.modules["pydantic_ai"] = pyd_mod
    sys.modules["pydantic_ai.providers"] = pyd_providers_mod
    sys.modules["pydantic_ai.providers.openai"] = pyd_provider_openai_mod
    sys.modules["pydantic_ai.models"] = pyd_models_mod
    sys.modules["pydantic_ai.models.openai"] = pyd_model_openai_mod

if "openai" not in sys.modules:
    openai_mod = types.ModuleType("openai")
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.embeddings = types.SimpleNamespace(create=None)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=None))

    openai_mod.AsyncOpenAI = _DummyAsyncOpenAI
    sys.modules["openai"] = openai_mod

import agent.tools as tools  # noqa: E402


def main() -> None:
    assert hasattr(tools, "EMBEDDING_TIMEOUT_SECONDS"), "missing EMBEDDING_TIMEOUT_SECONDS"

    tools_text = Path("agent/tools.py").read_text(encoding="utf-8")
    routing_text = Path("agent/routing.py").read_text(encoding="utf-8")

    assert "asyncio.wait_for" in tools_text, "generate/search timeout not configured in tools.py"
    assert "asyncio.wait_for" in routing_text, "preflight timeout not configured in routing.py"
    print("PASS: local search timeout config present")


if __name__ == "__main__":
    main()
