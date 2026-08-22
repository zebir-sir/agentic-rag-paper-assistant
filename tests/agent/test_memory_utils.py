from agent.memory_utils import (
    SessionMemoryState,
    build_context,
    build_memory_update_prompt,
    memory_eligible_messages,
    messages_for_memory_update,
    normalize_memory_summary,
    should_update_memory,
)


def test_memory_whitelist_excludes_partial_and_debug_messages():
    messages = [
        {"role": "user", "content": "比较两篇论文", "metadata": {"memory_eligible": True}},
        {"role": "assistant", "content": "最终回答", "metadata": {"memory_eligible": True}},
        {"role": "assistant", "content": "部分输出", "metadata": {"partial_response": True}},
        {"role": "assistant", "content": '{"tool_calls":["search"]}', "metadata": {}},
        {"role": "system", "content": "debug", "metadata": {"memory_eligible": True}},
    ]

    assert memory_eligible_messages(messages) == [
        {"role": "user", "content": "比较两篇论文"},
        {"role": "assistant", "content": "最终回答"},
    ]


def test_memory_discards_removed_document_scope_metadata():
    messages = [
        {
            "role": "user",
            "content": "比较选中的材料",
            "metadata": {
                "memory_eligible": True,
                "scope_mode": "selected_documents",
                "scope_document_ids": ["paper-a", "paper-b"],
                "sources": ["must-not-enter-memory"],
            },
        }
    ]

    assert memory_eligible_messages(messages) == [
        {
            "role": "user",
            "content": "比较选中的材料",
        }
    ]


def test_summary_normalization_removes_unknown_fields_and_bounds_values():
    summary = normalize_memory_summary(
        {
            "goal": "研究论文知识库" * 200,
            "constraints": ["只使用本地知识库"],
            "unknown": "debug payload",
            "open_questions": "not-a-list",
        }
    )

    assert summary["goal"]
    assert len(summary["goal"]) == 600
    assert summary["constraints"] == ["只使用本地知识库"]
    assert summary["open_questions"] == []
    assert "unknown" not in summary


def test_context_uses_snapshot_and_recent_eligible_turns():
    history = [
        {"role": "user", "content": f"问题 {index}"}
        for index in range(10)
    ]
    result = build_context(
        history_messages=history,
        current_question="当前问题",
        memory_state=SessionMemoryState(
            version=2,
            covered_message_count=8,
            summary={"goal": "完成方法比较", "constraints": ["中文回答"]},
        ),
    )

    assert result.compression_used is True
    assert "Structured conversation memory" in result.full_prompt
    assert "问题 1" not in result.full_prompt
    assert "问题 9" in result.full_prompt
    assert "当前问题" in result.full_prompt


def test_update_policy_uses_new_eligible_messages_and_trigger():
    history = [{"role": "user", "content": f"m{index}"} for index in range(8)]
    state = SessionMemoryState(covered_message_count=0)

    assert should_update_memory(history, state, "new question") is True
    assert messages_for_memory_update(history, 6) == history[6:]


def test_memory_update_prompt_forbids_retrieval_and_paper_facts():
    prompt = build_memory_update_prompt(
        {"goal": "比较方法"},
        [{"role": "assistant", "content": "最终回答"}],
    )

    assert "Return JSON only" in prompt
    assert "Never store paper claims" in prompt
    assert "tool calls" in prompt
