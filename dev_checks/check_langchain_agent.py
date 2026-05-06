import asyncio
import importlib
import sys
from pathlib import Path


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    db_utils_module = importlib.import_module("agent.db_utils")
    runtime_module = importlib.import_module("agent.agent_runtime")
    langchain_agent_module = importlib.import_module("agent.agent_langchain")

    initialize_database = getattr(db_utils_module, "initialize_database")
    execute_init_sql = getattr(db_utils_module, "execute_init_sql")
    close_database = getattr(db_utils_module, "close_database")

    AgentDependencies = getattr(runtime_module, "AgentDependencies")
    run_langchain_agent = getattr(langchain_agent_module, "run_langchain_agent")

    await initialize_database()
    try:
        await execute_init_sql("sql/schema.sql")

        deps = AgentDependencies(
            session_id="dev_check_langchain_agent",
            user_id="dev_check_user",
            use_web_search=False,
            search_preferences={
                "default_search_type": "hybrid",
                "default_limit": 3,
            },
        )
        result = await run_langchain_agent(
            full_prompt="请用一句中文介绍你是什么系统。可以在需要时查看知识库文档列表，但不要联网。",
            deps=deps,
        )
        assert isinstance(result.message, str), "result.message must be str"
        assert len(result.message.strip()) > 0, "result.message cannot be empty"
        assert result.raw_result is not None, "result.raw_result cannot be None"
        assert isinstance(result.tools_used, list), "result.tools_used must be list"
        assert isinstance(result.sources, list), "result.sources must be list"

        deps2 = AgentDependencies(
            session_id="dev_check_langchain_agent_2",
            user_id="dev_check_user",
            use_web_search=False,
            search_preferences={
                "default_search_type": "hybrid",
                "default_limit": 3,
            },
        )
        result2 = await run_langchain_agent(
            full_prompt=(
                "请先调用 list_documents 工具查看当前知识库有哪些文档，然后用一句中文总结你看到的结果。"
                "如果没有文档，也请说明当前知识库为空。"
            ),
            deps=deps2,
        )
        assert isinstance(result2.message, str), "result2.message must be str"
        assert len(result2.message.strip()) > 0, "result2.message cannot be empty"
        assert isinstance(result2.tools_used, list), "result2.tools_used must be list"
        if result2.tools_used:
            print("INFO tools_used:", [t.tool_name for t in result2.tools_used])

    finally:
        await close_database()

    api_text = Path("agent/api.py").read_text(encoding="utf-8")
    assert "build_langchain_agent" not in api_text, "api.py should not contain build_langchain_agent"
    assert "run_langchain_agent" not in api_text, "api.py should not contain run_langchain_agent"

    runner_text = Path("agent/agent_runner.py").read_text(encoding="utf-8")
    assert "run_pydantic_agent" in runner_text, "agent_runner should still include run_pydantic_agent"
    assert "iter_pydantic_agent" in runner_text, "agent_runner should still include iter_pydantic_agent"

    agent_text = Path("agent/agent.py").read_text(encoding="utf-8")
    assert "rag_agent = Agent" in agent_text, "agent.py must still contain rag_agent = Agent"

    print("check_langchain_agent passed")


if __name__ == "__main__":
    asyncio.run(main())
