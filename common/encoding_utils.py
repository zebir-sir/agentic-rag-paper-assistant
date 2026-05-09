from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_COMMON_CHINESE_CHARS = set(
    "的一是不了人在有我他这中大来上个国到说们为子和你地出道也时年得就那要下以生会自"
    "着去之过家学对可里后小么心多天而能好都然没日于起还发成事只作当想看用无开手十方"
    "又如前所本见经头面公同三已老从动两长知因很给法间斯行理种将月分样现关些正话明问"
    "力它与把机实水加量都文点从业定其些然前外天政四日那社义平形相全表间样与关各重新"
    "你多大了联网上查一下资料总结知识库里这篇论文方法流程并说明依据来自哪些章节根据"
    "上传文档本地最新网页当前最近实验结果摘要方法章节小节附录图表算法论文知识库文档"
)


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def write_text_utf8(path: str | Path, text: str) -> None:
    target = _to_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(text), encoding="utf-8", newline="\n")


def read_text_utf8(path: str | Path) -> str:
    target = _to_path(path)
    raw = target.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def write_json_utf8(path: str | Path, payload: Any, indent: int = 2) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    write_text_utf8(path, text + "\n")


def _score_text_readability(text: str) -> tuple[int, int, int, int]:
    value = str(text or "")
    common_hits = sum(1 for ch in value if ch in _COMMON_CHINESE_CHARS)
    cjk_hits = sum(1 for ch in value if "\u4e00" <= ch <= "\u9fff")
    replacement_count = value.count("\ufffd")
    ignored_count = value.count("?")
    return (common_hits, cjk_hits, -replacement_count, -ignored_count)


def repair_mojibake_text(text: str) -> str:
    original = str(text or "")
    if not original:
        return original

    candidates = [original]
    for source_encoding in ("gbk", "cp936", "gb18030"):
        try:
            repaired = original.encode(source_encoding, errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            continue
        repaired = repaired.strip("\x00")
        if repaired:
            candidates.append(repaired)

    best = original
    best_score = _score_text_readability(original)
    for candidate in candidates[1:]:
        candidate_score = _score_text_readability(candidate)
        if candidate_score > best_score:
            best = candidate
            best_score = candidate_score

    return best or original


def read_json_robust(path: str | Path) -> Any:
    text = read_text_utf8(path).strip()
    try:
        return json.loads(text)
    except Exception:
        repaired = repair_mojibake_text(text)
        if repaired != text:
            try:
                return json.loads(repaired)
            except Exception:
                pass
        raise
