from agent.memory_runtime import build_session_memory_snapshot


def test_snapshot_reads_database_state_and_only_eligible_messages():
    snapshot = build_session_memory_snapshot(
        session_id="session-1",
        memory_snapshot={
            "version": 3,
            "covered_message_count": 4,
            "summary": {"goal": "论文对比", "open_questions": ["实验差异"]},
            "updated_at": "2026-08-12T10:00:00+00:00",
        },
        messages=[
            {"role": "user", "content": "第一问", "metadata": {"memory_eligible": True}},
            {"role": "assistant", "content": "最终回答", "metadata": {"memory_eligible": True}},
            {"role": "assistant", "content": "partial", "metadata": {"partial_response": True}},
        ],
    )

    assert snapshot["session_id"] == "session-1"
    assert snapshot["version"] == 3
    assert snapshot["covered_message_count"] == 4
    assert snapshot["summary"]["goal"] == "论文对比"
    assert snapshot["eligible_message_count"] == 2
    assert snapshot["recent_message_count"] == 2
    assert snapshot["using_structured_memory"] is True
    assert [item["content"] for item in snapshot["recent_messages_preview"]] == ["第一问", "最终回答"]
