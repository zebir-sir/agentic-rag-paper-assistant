import asyncio
import importlib
import os
import sys
from pathlib import Path


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    db_utils_module = importlib.import_module("agent.db_utils")
    runtime_module = importlib.import_module("agent.agent_runtime")
    runner_module = importlib.import_module("agent.agent_runner")

    initialize_database = getattr(db_utils_module, "initialize_database")
    execute_init_sql = getattr(db_utils_module, "execute_init_sql")
    close_database = getattr(db_utils_module, "close_database")

    AgentDependencies = getattr(runtime_module, "AgentDependencies")
    get_agent_backend = getattr(runner_module, "get_agent_backend")
    run_agent = getattr(runner_module, "run_agent")

    old_backend = os.environ.pop("AGENT_BACKEND", None)
    try:
        assert get_agent_backend() == "langchain", "default backend should be langchain"

        os.environ["AGENT_BACKEND"] = "langchain"
        assert get_agent_backend() == "langchain", "langchain backend parse failed"

        os.environ["AGENT_BACKEND"] = "bad_value"
        assert get_agent_backend() == "pydantic_ai", "invalid backend should fallback to pydantic_ai"

        await initialize_database()
        try:
            await execute_init_sql("sql/schema.sql")

            os.environ["AGENT_BACKEND"] = "langchain"
            deps = AgentDependencies(
                session_id="dev_check_backend_switch",
                user_id="dev_check_user",
                use_web_search=False,
                search_preferences={
                    "default_search_type": "hybrid",
                    "default_limit": 3,
                },
            )
            result = await run_agent(
                full_prompt="请用一句中文介绍你是什么系统，不要联网。",
                deps=deps,
            )
            assert hasattr(result, "message"), "langchain result should have message"
            assert isinstance(result.message, str) and result.message.strip(), "langchain message should be non-empty"
            assert hasattr(result, "tools_used"), "langchain result should have tools_used"
            assert hasattr(result, "sources"), "langchain result should have sources"

            os.environ["AGENT_BACKEND"] = "pydantic_ai"
            deps2 = AgentDependencies(
                session_id="dev_check_backend_switch_pydantic",
                user_id="dev_check_user",
                use_web_search=False,
                search_preferences={
                    "default_search_type": "hybrid",
                    "default_limit": 3,
                },
            )
            result2 = await run_agent(
                full_prompt="请用一句中文介绍你是什么系统，不要联网。",
                deps=deps2,
            )
            assert hasattr(result2, "output"), "pydantic result should have output"
            assert str(result2.output).strip(), "pydantic output should be non-empty"
        finally:
            await close_database()
    finally:
        if old_backend is not None:
            os.environ["AGENT_BACKEND"] = old_backend
        else:
            os.environ.pop("AGENT_BACKEND", None)

    runner_text = Path("agent/agent_runner.py").read_text(encoding="utf-8")
    assert "run_langchain_agent" in runner_text, "runner should include langchain runner import"
    assert "async def iter_agent" in runner_text, "runner should include iter_agent"
    iter_section = runner_text.split("async def iter_agent", 1)[1]
    assert "iter_pydantic_agent" in iter_section, "iter_agent should remain on pydantic path in this phase"

    print("check_agent_backend_switch passed")


if __name__ == "__main__":
    asyncio.run(main())
