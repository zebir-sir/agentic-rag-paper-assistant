import importlib
import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    runner_module = importlib.import_module("agent.agent_runner")
    get_agent_backend = getattr(runner_module, "get_agent_backend")

    original = os.environ.get("AGENT_BACKEND")
    try:
        os.environ.pop("AGENT_BACKEND", None)
        assert get_agent_backend() == "langchain", "default backend should be langchain"

        os.environ["AGENT_BACKEND"] = "pydantic_ai"
        assert get_agent_backend() == "pydantic_ai", "explicit pydantic_ai backend parse failed"

        os.environ["AGENT_BACKEND"] = "langchain"
        assert get_agent_backend() == "langchain", "explicit langchain backend parse failed"

        os.environ["AGENT_BACKEND"] = "bad_value"
        assert get_agent_backend() == "pydantic_ai", "bad backend should fallback to pydantic_ai"
    finally:
        if original is None:
            os.environ.pop("AGENT_BACKEND", None)
        else:
            os.environ["AGENT_BACKEND"] = original

    print("check_default_backend_langchain passed")


if __name__ == "__main__":
    main()
