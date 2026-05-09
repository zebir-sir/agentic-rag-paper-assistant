from agent.warning_text import clean_legacy_warning_text


def test_clean_legacy_warning_text_default_call_is_backward_compatible():
    text = "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    cleaned = clean_legacy_warning_text(text)
    assert "一般性分析参考" not in cleaned
    assert "本轮没有可核对的检索片段" in cleaned


def test_clean_legacy_warning_text_drop_warning_true_removes_legacy_text():
    text = "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    cleaned = clean_legacy_warning_text(text, drop_warning=True)
    assert "一般性分析参考" not in cleaned
    assert cleaned == ""


def test_clean_legacy_warning_text_drop_warning_false_rewrites_to_neutral_text():
    text = "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    cleaned = clean_legacy_warning_text(text, drop_warning=False)
    assert "一般性分析参考" not in cleaned
    assert "本轮没有可核对的检索片段" in cleaned
