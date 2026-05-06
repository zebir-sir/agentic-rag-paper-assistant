import importlib
import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    runner_module = importlib.import_module("agent.agent_runner")
    get_stream_backend = getattr(runner_module, "get_stream_backend")

    original = os.environ.get("STREAM_BACKEND")
    try:
        os.environ.pop("STREAM_BACKEND", None)
        assert get_stream_backend() == "langchain", "default stream backend should be langchain"

        os.environ["STREAM_BACKEND"] = "pydantic_ai"
        assert get_stream_backend() == "pydantic_ai", "explicit pydantic_ai stream backend parse failed"

        os.environ["STREAM_BACKEND"] = "langchain"
        assert get_stream_backend() == "langchain", "explicit langchain stream backend parse failed"

        os.environ["STREAM_BACKEND"] = "bad_value"
        assert get_stream_backend() == "pydantic_ai", "bad stream backend should fallback to pydantic_ai"
    finally:
        if original is None:
            os.environ.pop("STREAM_BACKEND", None)
        else:
            os.environ["STREAM_BACKEND"] = original

    print("check_stream_backend_switch passed")


if __name__ == "__main__":
    main()
