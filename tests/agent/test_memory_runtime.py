from agent.memory_runtime import (
    build_session_memory_snapshot,
    parse_memory_summary_sections,
)


def test_parse_memory_summary_sections_extracts_expected_fields():
    summary = (
        "1) 当前讨论对象：Hybrid-RRT 论文\n"
        "2) 用户约束：只看 Abstract 和 Introduction，不联网\n"
        "3) 已确认信息：用户用于秋招展示，强调工程感\n"
        "4) 用户关注点：项目结构、中间件、可观测性\n"
        "5) 待继续问题：是否补充记忆模块与配置能力\n"
        "6) 不确定或缺失信息：尚未验证容器中的新接口"
    )

    sections = parse_memory_summary_sections(summary)

    assert sections["current_topic"] == "Hybrid-RRT 论文"
    assert sections["user_constraints"] == "只看 Abstract 和 Introduction，不联网"
    assert sections["confirmed_facts"] == "用户用于秋招展示，强调工程感"
    assert sections["user_focus"] == "项目结构、中间件、可观测性"
    assert sections["pending_questions"] == "是否补充记忆模块与配置能力"
    assert sections["unknowns"] == "尚未验证容器中的新接口"


def test_build_session_memory_snapshot_summarizes_history_and_recent_preview():
    snapshot = build_session_memory_snapshot(
        session_id="session-1",
        memory_metadata={
            "latest_summary": (
                "1) 当前讨论对象：RAG 项目\n"
                "2) 用户约束：秋招导向\n"
                "3) 已确认信息：已接入健康检查\n"
                "4) 用户关注点：工程感\n"
                "5) 待继续问题：记忆可视化\n"
                "6) 不确定或缺失信息：暂无"
            ),
            "compression_count": 2,
            "compacted_message_count": 8,
            "summary_updated_at": "2026-07-08T23:10:00+00:00",
        },
        messages=[
            {"role": "user", "content": "先做工程化"},
            {"role": "assistant", "content": "已经补了健康检查"},
            {"role": "assistant", "content": '{"tools_executed":["hybrid_search"]}'},
            {"role": "user", "content": "再看看记忆模块"},
        ],
    )

    assert snapshot["session_id"] == "session-1"
    assert snapshot["compression_count"] == 2
    assert snapshot["compacted_message_count"] == 8
    assert snapshot["using_summary_context"] is True
    assert snapshot["history_message_count"] == 4
    assert snapshot["sanitized_history_count"] == 3
    assert snapshot["recent_message_count"] == 3
    assert snapshot["summary_sections"]["current_topic"] == "RAG 项目"
    assert snapshot["recent_messages_preview"][0]["role"] == "user"
    assert snapshot["summary_context_estimated_tokens"] > 0
