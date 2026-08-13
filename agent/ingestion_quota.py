"""Classify provider quota failures so ingestion can stop safely."""

import re


_QUOTA_PATTERNS = (
    r"\b(?:http\s*)?429\b",
    r"insufficient[_\s-]?(?:quota|balance|credit|funds)",
    r"(?:quota|credit|balance|额度|余额).{0,40}(?:exceed|exhaust|insufficient|不足|耗尽|用完)",
    r"(?:exceed|exhaust).{0,40}(?:quota|credit|balance|额度|余额)",
    r"billing.{0,40}(?:limit|quota)",
)


def is_quota_exhausted_error(error: BaseException | str) -> bool:
    text = str(error or "")
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _QUOTA_PATTERNS)


def quota_pause_message(error: BaseException | str) -> str:
    detail = " ".join(str(error or "").split())[:360]
    suffix = f" 原始 PDF 已保留，可在额度恢复后重新入库。诊断：{detail}" if detail else " 原始 PDF 已保留，可在额度恢复后重新入库。"
    return "模型额度不足，已停止当前论文入库，不会自动重试或继续占用资源。" + suffix
