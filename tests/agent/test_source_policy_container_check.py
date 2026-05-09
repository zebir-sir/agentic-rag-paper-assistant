import json

from test_logs.source_policy_container_check import write_case_payload


def test_write_case_payload_keeps_chinese_question_readable(tmp_path):
    path = tmp_path / "planner_age.json"
    payload = {
        "mode": "planner_only",
        "question": "你多大了",
        "intent_plan": {
            "intent": "direct_answer",
            "needs_retrieval": False,
            "retrieval_steps": [],
        },
    }

    write_case_payload(path, payload)

    raw_text = path.read_text(encoding="utf-8")
    assert "你多大了" in raw_text
    assert "\\u4f60" not in raw_text
    assert json.loads(raw_text)["question"] == "你多大了"
