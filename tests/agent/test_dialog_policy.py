from agent.dialog_policy import classify_dialog_turn


def test_classify_dialog_turn_marks_acknowledgement_as_brief_ack():
    decision = classify_dialog_turn(
        latest_query="好的",
        history_messages=[{"role": "assistant", "content": "我已经总结完方法部分。"}],
    )

    assert decision.dialog_act == "acknowledgement"
    assert decision.carry_context is False
    assert decision.response_style == "brief_ack"


def test_classify_dialog_turn_marks_constraint_update():
    decision = classify_dialog_turn(
        latest_query="只看实验部分，不要扩展到 related work。",
        history_messages=[{"role": "user", "content": "帮我分析这篇论文。"}],
    )

    assert decision.dialog_act == "constraint_update"
    assert decision.carry_context is True
    assert decision.response_style == "respect_constraints"


def test_classify_dialog_turn_marks_short_follow_up_reference():
    decision = classify_dialog_turn(
        latest_query="那这个表说明什么？",
        history_messages=[
            {"role": "user", "content": "请分析论文里的 Table 2。"},
            {"role": "assistant", "content": "Table 2 主要在比较不同算法的成功率和时间开销。"},
        ],
    )

    assert decision.dialog_act == "follow_up_reference"
    assert decision.carry_context is True
    assert decision.response_style == "normal"
