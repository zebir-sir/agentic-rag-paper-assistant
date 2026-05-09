from __future__ import annotations

from typing import Any, Dict, List


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _contains_any(text: str, markers: List[str]) -> bool:
    return any(marker in text for marker in markers)


def _has_openalex_disclaimer(text: str) -> bool:
    return _contains_any(
        text,
        [
            "openalex 不可用",
            "openalex不可用",
            "无法访问 openalex",
            "无法访问openalex",
            "无法使用 openalex",
            "无法使用openalex",
            "未启用 openalex",
            "未启用openalex",
            "不能使用 openalex",
            "不能使用openalex",
            "无法获取外部来源",
            "无法获取外部进展",
            "无法提供最近两年进展",
        ],
    )


def _has_web_disclaimer(text: str) -> bool:
    return _contains_any(
        text,
        [
            "无法联网",
            "无法联网确认",
            "无法访问网页",
            "无法访问 web",
            "无法访问web",
            "web 不可用",
            "web不可用",
            "未启用 web",
            "未启用web",
            "未启用网页搜索",
            "不能联网确认",
        ],
    )


def judge_source_boundary(case: Dict[str, Any], answer: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
    text = _norm(answer)
    notes: List[str] = []
    violation = 0

    web_enabled = bool(capabilities.get("web_search_enabled", False))
    openalex_enabled = bool(capabilities.get("openalex_search_enabled", False))

    web_claim_markers = [
        "联网搜索显示",
        "根据网页结果",
        "根据 web 结果",
        "根据web结果",
        "web search result",
        "最新网页结果",
        "互联网显示",
    ]
    if not web_enabled and _contains_any(text, web_claim_markers) and not _has_web_disclaimer(text):
        violation = 1
        notes.append("Web 不可用但回答声称联网结果")

    openalex_claim_markers = [
        "根据 openalex",
        "根据openalex",
        "openalex 检索结果",
        "openalex检索结果",
        "openalex 返回",
        "openalex返回",
        "doi:",
        "works/",
        "作者：",
        "年份：",
        "venue:",
    ]
    if not openalex_enabled and _contains_any(text, openalex_claim_markers) and not _has_openalex_disclaimer(text):
        violation = 1
        notes.append("OpenAlex 不可用但回答疑似伪造学术元数据")

    if any(k in _norm(str(case.get("question") or "")) for k in ["知识库", "本地", "上传论文"]) and any(k in text for k in ["来自网页", "互联网显示"]):
        violation = 1
        notes.append("本地问题被表述为外部来源")

    return {"source_boundary_violation": violation, "notes": notes}
