import asyncio
import importlib
import sys
from pathlib import Path


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    try:
        importlib.import_module("langgraph.graph")
    except ImportError:
        print("FAIL: 请安装 langgraph，或确认依赖文件已包含 langgraph。")
        raise

    db_utils_module = importlib.import_module("agent.db_utils")
    runtime_module = importlib.import_module("agent.agent_runtime")
    langgraph_module = importlib.import_module("agent.agent_langgraph")

    initialize_database = getattr(db_utils_module, "initialize_database")
    execute_init_sql = getattr(db_utils_module, "execute_init_sql")
    close_database = getattr(db_utils_module, "close_database")

    AgentDependencies = getattr(runtime_module, "AgentDependencies")
    run_langgraph_analysis = getattr(langgraph_module, "run_langgraph_analysis")

    await initialize_database()
    try:
        await execute_init_sql("sql/schema.sql")

        deps = AgentDependencies(
            session_id="dev_check_langgraph_workflow",
            user_id="dev_check_user",
            use_web_search=False,
            search_preferences={
                "default_search_type": "hybrid",
                "default_limit": 3,
            },
        )
        result = await run_langgraph_analysis(
            question="请用中文说明你作为科研论文阅读助手能做什么，并说明如果知识库没有证据你会如何处理。",
            deps=deps,
        )
        assert isinstance(result.message, str) and result.message.strip(), "result.message should be non-empty str"
        assert isinstance(result.tools_used, list), "result.tools_used should be list"
        assert isinstance(result.sources, list), "result.sources should be list"
        assert isinstance(result.metadata, dict), "result.metadata should be dict"
        assert result.metadata.get("agent_backend") == "langgraph", "agent_backend should be langgraph"
        assert result.metadata.get("workflow") == "deep_analysis", "workflow should be deep_analysis"
        assert result.metadata.get("evidence_checked") is True, "evidence_checked should be True"
        tool_names = [t.tool_name for t in result.tools_used if hasattr(t, "tool_name")]
        assert "list_documents" in tool_names, "tools_used should include list_documents"
        warnings1 = list((result.raw_state or {}).get("warnings") or [])
        retrieval1 = list((result.raw_state or {}).get("retrieval_results") or [])
        if not retrieval1 and not warnings1:
            raise AssertionError("retrieval_results empty but warnings missing")

        deps2 = AgentDependencies(
            session_id="dev_check_langgraph_workflow_2",
            user_id="dev_check_user",
            use_web_search=False,
            search_preferences={
                "default_search_type": "hybrid",
                "default_limit": 3,
            },
        )
        result2 = await run_langgraph_analysis(
            question="请检索知识库中与 RRT 路径规划相关的内容，并基于检索结果给出简短总结。",
            deps=deps2,
        )
        assert isinstance(result2.message, str) and result2.message.strip(), "result2.message should be non-empty str"
        assert isinstance(result2.tools_used, list), "result2.tools_used should be list"
        assert isinstance(result2.sources, list), "result2.sources should be list"
        assert isinstance(result2.metadata, dict), "result2.metadata should be dict"
        assert result2.metadata.get("agent_backend") == "langgraph", "result2 agent_backend should be langgraph"
        assert result2.metadata.get("workflow") == "deep_analysis", "result2 workflow should be deep_analysis"

        if not result2.sources:
            message_or_warnings = (
                (result2.message or "")
                + " "
                + " ".join(list((result2.raw_state or {}).get("warnings") or []))
            ).lower()
            expected_phrases = [
                "证据不足",
                "未检索到",
                "no retrieval evidence found",
                "general guidance",
            ]
            if not any(p in message_or_warnings for p in expected_phrases):
                raise AssertionError("sources empty but no evidence-insufficient hint in message/warnings")
    finally:
        await close_database()

    print("check_langgraph_workflow passed")


if __name__ == "__main__":
    asyncio.run(main())
