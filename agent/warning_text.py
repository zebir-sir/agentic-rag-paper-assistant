from __future__ import annotations


def clean_legacy_warning_text(text: str, drop_warning: bool = False) -> str:
    value = str(text or "")
    legacy_full = "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    legacy_tail = "以下内容更适合作为一般性分析参考。"
    neutral_full = "本轮没有可核对的检索片段；未被检索证据支持的结论请谨慎参考。"
    neutral_tail = "未被检索证据支持的结论请谨慎参考。"

    if drop_warning:
        return value.replace(legacy_full, "").replace(legacy_tail, "").strip()

    return value.replace(legacy_full, neutral_full).replace(legacy_tail, neutral_tail)
