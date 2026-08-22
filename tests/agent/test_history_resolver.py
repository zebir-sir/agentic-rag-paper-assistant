from agent.history_resolver import resolve_history_query


def test_resolve_history_query_attaches_previous_user_topic_for_follow_up():
    result = resolve_history_query(
        latest_query="这个方法和 RRT* 的差别呢？",
        history_messages=[
            {"role": "user", "content": "请总结 AIT* 这篇论文的方法核心。"},
            {"role": "assistant", "content": "AIT* 主要把启发式搜索和增量采样结合起来。"},
        ],
    )

    assert result.used_history is True
    assert result.topic_hint == "请总结 AIT* 这篇论文的方法核心。"
    assert result.resolved_query == "请总结 AIT* 这篇论文的方法核心。；当前追问：这个方法和 RRT* 的差别呢？"
    assert "用户：" in result.recent_history_summary


def test_resolve_history_query_keeps_self_contained_question_unchanged():
    result = resolve_history_query(
        latest_query="请解释 RRT* 和 Informed RRT* 的区别。",
        history_messages=[
            {"role": "user", "content": "先看 RRT 系列论文。"},
            {"role": "assistant", "content": "可以，我先帮你梳理基础脉络。"},
        ],
    )

    assert result.used_history is False
    assert result.resolved_query == "请解释 RRT* 和 Informed RRT* 的区别。"
    assert result.reason == "query_is_self_contained"
def test_resolve_history_query_falls_back_when_no_topic_found():
    result = resolve_history_query(
        latest_query="这个结果为什么更好？",
        history_messages=[
            {"role": "user", "content": "好"},
            {"role": "assistant", "content": "收到"},
        ],
    )

    assert result.used_history is False
    assert result.resolved_query == "这个结果为什么更好？"
    assert result.reason == "history_topic_not_found"
