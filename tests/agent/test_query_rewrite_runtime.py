import pytest

from agent.query_rewrite_runtime import build_query_rewrite_context, rewrite_query_with_conversation


class FakeModel:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return type("Response", (), {"content": self.content})()


def test_build_query_rewrite_context_prioritizes_memory_and_recent_turns(monkeypatch):
    monkeypatch.setattr("agent.query_rewrite_runtime.QUERY_REWRITE_RECENT_MESSAGE_COUNT", 10)
    context = build_query_rewrite_context(
        memory_summary={"goal": "比较采样规划论文", "constraints": ["中文回答"]},
        history_messages=[
            {"role": "user", "content": f"较早问题 {index}"}
            for index in range(12)
        ],
    )

    assert "长期会话记忆" in context
    recent_lines = [line for line in context.splitlines() if line.startswith("用户：较早问题")]
    assert recent_lines == [f"用户：较早问题 {index}" for index in range(2, 12)]
    assert "较早问题 2" in context


@pytest.mark.asyncio
async def test_rewrite_query_with_conversation_resolves_context_dependent_question():
    model = FakeModel('{"rewritten_query":"AIT* 的方法和 RRT* 有什么差别？"}')

    result = await rewrite_query_with_conversation(
        original_query="这个方法和 RRT* 的差别呢？",
        conversation_context="用户：请总结 AIT* 这篇论文的方法核心。",
        model=model,
    )

    assert result.rewritten_query == "AIT* 的方法和 RRT* 有什么差别？"
    assert result.model_used is True
    assert result.reason == "model_rewrite"
    assert len(model.calls) == 1
    assert "不得扩大问题" in model.calls[0][1]["content"]


@pytest.mark.asyncio
async def test_rewrite_query_with_conversation_keeps_self_contained_question():
    result = await rewrite_query_with_conversation(
        original_query="请解释 RRT* 和 Informed RRT* 的区别。",
        conversation_context="用户：请总结 AIT* 这篇论文的方法核心。",
        model=FakeModel('{"rewritten_query":"请解释 RRT* 和 Informed RRT* 的区别。"}'),
    )

    assert result.model_used is True
    assert result.reason == "model_kept_original"
    assert result.rewritten_query == "请解释 RRT* 和 Informed RRT* 的区别。"


@pytest.mark.asyncio
async def test_rewrite_query_with_conversation_falls_back_to_original_for_invalid_output():
    result = await rewrite_query_with_conversation(
        original_query="这个结果为什么更好？",
        conversation_context="用户：请比较 AIT* 和 BIT* 的实验结果。",
        model=FakeModel("这不是 JSON"),
    )

    assert result.model_used is False
    assert result.reason == "invalid_model_output_fallback"
    assert result.rewritten_query == "这个结果为什么更好？"
