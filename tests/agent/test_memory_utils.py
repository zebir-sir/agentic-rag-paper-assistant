from agent.memory_utils import (
    MemoryState,
    build_context_without_compaction,
    build_summary_update_prompt,
    get_messages_for_next_compaction,
    sanitize_history_messages,
    sanitize_message_for_context,
)


def test_sanitize_message_for_context_keeps_user_and_final_assistant_content():
    user_message = {
        "role": "user",
        "content": "根据上传论文总结方法贡献",
        "metadata": {"intent_plan": {"intent": "local_paper_qa"}},
    }
    assistant_message = {
        "role": "assistant",
        "content": "这篇论文的主要方法贡献包括三个方面。",
        "metadata": {"tools_executed": ["hybrid_search"]},
    }

    assert sanitize_message_for_context(user_message) == {
        "role": "user",
        "content": "根据上传论文总结方法贡献",
    }
    assert sanitize_message_for_context(assistant_message) == {
        "role": "assistant",
        "content": "这篇论文的主要方法贡献包括三个方面。",
    }


def test_sanitize_message_for_context_drops_debug_payload_content():
    polluted_message = {
        "role": "assistant",
        "content": (
            '{"intent_plan":{"intent":"direct_answer"},'
            '"tools_executed":["hybrid_search"],'
            '"raw_model_content_preview":"..."}'
        ),
        "metadata": {},
    }
    assert sanitize_message_for_context(polluted_message) is None


def test_build_context_without_compaction_excludes_planner_debug_fields():
    history_messages = [
        {"role": "user", "content": "你觉得我今晚吃什么比较好"},
        {
            "role": "assistant",
            "content": (
                '{"intent_plan":{"intent":"direct_answer"},'
                '"tools_planned":["hybrid_search"],'
                '"tools_executed":["hybrid_search"],'
                '"raw_model_content_preview":"..."}'
            ),
        },
        {"role": "assistant", "content": "今晚可以吃点清淡又有蛋白质的，比如鸡胸肉沙拉或盖饭。"},
    ]

    result = build_context_without_compaction(
        history_messages=history_messages,
        current_question="再给我一个偏热量低的版本",
        memory_state=MemoryState(),
    )

    assert "你觉得我今晚吃什么比较好" in result.full_prompt
    assert "今晚可以吃点清淡又有蛋白质的" in result.full_prompt
    assert "intent_plan" not in result.full_prompt
    assert "tools_planned" not in result.full_prompt
    assert "tools_executed" not in result.full_prompt
    assert "raw_model_content_preview" not in result.full_prompt
    assert "hybrid_search" not in result.full_prompt


def test_get_messages_for_next_compaction_filters_debug_only_messages():
    history_messages = [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": '{"tools_executed":["search_knowledge_base"]}'},
        {"role": "assistant", "content": "最终回答"},
        {"role": "user", "content": "继续追问"},
        {"role": "assistant", "content": "继续回答"},
        {"role": "user", "content": "第三问"},
        {"role": "assistant", "content": "第三答"},
        {"role": "user", "content": "第四问"},
        {"role": "assistant", "content": "第四答"},
    ]

    compact_messages = get_messages_for_next_compaction(
        history_messages=history_messages,
        compacted_message_count=0,
    )
    compact_texts = [msg["content"] for msg in compact_messages]
    assert '{"tools_executed":["search_knowledge_base"]}' not in compact_texts
    assert "用户问题" in compact_texts


def test_build_summary_update_prompt_requires_constraints_and_boundary_preservation():
    prompt = build_summary_update_prompt(
        old_summary="当前讨论对象：Hybrid-RRT。用户约束：只看 Abstract 与 Introduction。",
        messages_to_compact=[
            {
                "role": "user",
                "content": "只基于 Abstract 和 Introduction，不要扩展到 Results 或 Conclusion，也不要联网。",
            }
        ],
    )
    assert "用户约束" in prompt
    assert "章节范围" in prompt
    assert "禁止范围" in prompt
    assert "来源限制" in prompt
    assert "保留仍然有效的用户约束" in prompt
    assert "若新消息明确改变讨论对象或约束，以新消息为准" in prompt
    assert "简体中文" in prompt


def test_build_summary_update_prompt_forbids_debug_payload_and_tool_metadata():
    prompt = build_summary_update_prompt(
        old_summary="",
        messages_to_compact=[{"role": "assistant", "content": "继续分析方法流程"}],
    )
    assert "不要把任何 debug payload" in prompt
    assert "raw_model_content_preview" in prompt
    assert "tools_executed" in prompt
    assert "intent_plan" in prompt
