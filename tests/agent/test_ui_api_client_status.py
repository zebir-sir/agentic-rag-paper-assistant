from ui.api_client import (
    clean_assistant_display_text,
    map_status_text,
    should_display_status_event,
)


def test_status_internal_not_displayed():
    event = {"type": "status", "content": "内部调试", "phase": "internal", "user_visible": False}
    assert should_display_status_event(event) is False


def test_status_generation_displayed():
    event = {"type": "status", "content": "正在生成回答...", "phase": "generation", "user_visible": True}
    assert should_display_status_event(event) is True
    assert map_status_text(event["content"], event["phase"]) == "正在生成回答..."


def test_clean_assistant_display_text_rewrites_legacy_warning():
    legacy = "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    cleaned = clean_assistant_display_text(legacy)
    assert "以下内容" not in cleaned
    assert "本轮没有可核对的检索片段" in cleaned
