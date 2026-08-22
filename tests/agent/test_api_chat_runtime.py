import pytest
import langchain.agents as langchain_agents
from unittest.mock import AsyncMock, patch

if not hasattr(langchain_agents, "create_agent"):
    langchain_agents.create_agent = lambda *args, **kwargs: None

from agent.api import execute_prepared_chat_runtime, openalex_status, prepare_chat_runtime
from agent.models import ChatRequest
from agent.query_rewrite_runtime import QueryRewriteResult


@pytest.mark.asyncio
async def test_openalex_status_reports_capability_state():
    with patch("agent.api._is_openalex_enabled", return_value=True):
        assert await openalex_status() == {"enabled": True}


@pytest.mark.asyncio
async def test_prepare_chat_runtime_includes_carryover_metadata_and_prompt_block():
    request = ChatRequest(
        message="这个方法和 RRT* 的差别呢？",
        session_id="session-1",
    )

    with patch("agent.api._prepare_agent_prompt", new_callable=AsyncMock) as mock_prepare, patch(
        "agent.api.rewrite_query_with_conversation",
        new_callable=AsyncMock,
        return_value=QueryRewriteResult(
            original_query="这个方法和 RRT* 的差别呢？",
            rewritten_query="AIT* 的方法和 RRT* 有什么差别？",
            model_used=True,
            reason="model_rewrite",
        ),
    ) as mock_rewrite:
        mock_prepare.return_value = {
            "full_prompt": "Previous conversation:\n...\n\nCurrent question: 这个方法和 RRT* 的差别呢？",
            "compression_used": False,
            "memory_updated": False,
            "memory_state": None,
            "history_messages": [
                {"role": "user", "content": "请总结 AIT* 这篇论文的方法核心。"},
                {"role": "assistant", "content": "AIT* 主要把启发式搜索和增量采样结合起来。"},
            ],
        }

        runtime = await prepare_chat_runtime(request)

    assert runtime.workflow_metadata["dialog_act"] == "follow_up_reference"
    assert runtime.workflow_metadata["history_resolution_used"] is True
    assert runtime.workflow_metadata["resolved_query"] == "AIT* 的方法和 RRT* 有什么差别？"
    assert runtime.retrieval_query == runtime.workflow_metadata["resolved_query"]
    assert runtime.workflow_metadata["query_rewrite_model_used"] is True
    assert mock_rewrite.await_count == 1
    assert runtime.workflow_metadata["simple_chat_candidate"] is False
    assert "[Conversation carry-over]" in runtime.full_prompt
    assert "resolved_query_for_retrieval" in runtime.full_prompt
    assert "[Conversation carry-over]" in runtime.langgraph_context_prompt


@pytest.mark.asyncio
async def test_prepare_chat_runtime_sends_self_contained_question_to_rewrite_model():
    request = ChatRequest(
        message="请解释 RRT* 和 Informed RRT* 的区别。",
        metadata={"selected_document_ids": ["legacy-document"]},
    )

    with patch("agent.api._prepare_agent_prompt", new_callable=AsyncMock) as mock_prepare, patch(
        "agent.api.rewrite_query_with_conversation",
        new_callable=AsyncMock,
        return_value=QueryRewriteResult(
            original_query="请解释 RRT* 和 Informed RRT* 的区别。",
            rewritten_query="请解释 RRT* 和 Informed RRT* 的区别。",
            model_used=True,
            reason="model_kept_original",
        ),
    ) as mock_rewrite:
        mock_prepare.return_value = {
            "full_prompt": "Current question: 请解释 RRT* 和 Informed RRT* 的区别。",
            "compression_used": False,
            "memory_updated": False,
            "memory_state": None,
            "history_messages": [
                {"role": "user", "content": "请总结 AIT* 这篇论文的方法核心。"},
                {"role": "assistant", "content": "AIT* 使用启发式搜索。"},
            ],
        }
        runtime = await prepare_chat_runtime(request)

    assert runtime.retrieval_query == request.message
    assert runtime.workflow_metadata["query_rewrite_model_used"] is True
    assert runtime.workflow_metadata["query_rewrite_reason"] == "model_kept_original"
    assert "selected_document_ids" not in runtime.deps.search_preferences
    assert mock_rewrite.await_count == 1


@pytest.mark.asyncio
async def test_deep_analysis_uses_resolved_query_for_first_workflow_input():
    request = ChatRequest(
        message="这个方法和 RRT* 的差别呢？",
        session_id="session-1",
        use_react=True,
    )

    with patch("agent.api._prepare_agent_prompt", new_callable=AsyncMock) as mock_prepare, patch(
        "agent.api.rewrite_query_with_conversation",
        new_callable=AsyncMock,
        return_value=QueryRewriteResult(
            original_query="这个方法和 RRT* 的差别呢？",
            rewritten_query="AIT* 的方法和 RRT* 有什么差别？",
            model_used=True,
            reason="model_rewrite",
        ),
    ):
        mock_prepare.return_value = {
            "full_prompt": "Current question: 这个方法和 RRT* 的差别呢？",
            "compression_used": False,
            "memory_updated": False,
            "memory_state": None,
            "history_messages": [
                {"role": "user", "content": "请总结 AIT* 这篇论文的方法核心。"},
                {"role": "assistant", "content": "AIT* 主要把启发式搜索和增量采样结合起来。"},
            ],
        }
        runtime = await prepare_chat_runtime(request)

    graph_result = type(
        "GraphResult",
        (),
        {"message": "回答", "tools_used": [], "sources": [], "metadata": {}},
    )()
    with patch("agent.api.run_langgraph_analysis", new_callable=AsyncMock, return_value=graph_result) as mock_graph:
        await execute_prepared_chat_runtime(
            request.message,
            runtime,
            save_conversation=False,
        )

    assert mock_graph.await_args.kwargs["question"] == runtime.retrieval_query
    assert mock_graph.await_args.kwargs["question"] != request.message
