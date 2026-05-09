from __future__ import annotations

import re
from typing import Any, Dict, List


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return any(_norm(k) in t for k in keywords if str(k or "").strip())


def _document_coverage_score(source_blob: str, expected_docs: List[str]) -> int:
    docs = [doc for doc in expected_docs if str(doc or "").strip()]
    if not docs:
        return 2
    hit = sum(1 for doc in docs if _norm(doc) in _norm(source_blob))
    if hit == 0:
        return 0
    if hit < len(docs):
        return 1
    return 2


def _score_completeness(answer: str, must_include: List[str]) -> int:
    if not must_include:
        return 2
    hit = sum(1 for item in must_include if _norm(item) in _norm(answer))
    if hit == 0:
        return 0
    if hit < len(must_include):
        return 1
    return 2


def _score_clarity(answer: str) -> int:
    text = str(answer or "").strip()
    if len(text) < 40:
        return 0
    return 1


def _source_signals(sources: List[Dict[str, Any]]) -> set[str]:
    signals: set[str] = set()
    for source in sources:
        metadata = source.get("metadata") or {}
        metadata = metadata if isinstance(metadata, dict) else {}
        source_type = _norm(source.get("source_type") or metadata.get("source_type"))
        if source_type == "web":
            signals.add("general_web")
        elif source_type in {"openalex", "external_academic"}:
            signals.add("external_academic")
        else:
            signals.add("local_kb")
        if _norm(metadata.get("section_path_text") or metadata.get("section_title")):
            signals.add("local_section")
        if _norm(metadata.get("artifact_type")):
            signals.add("local_artifact")
    return signals


def _score_source_fit(case: Dict[str, Any], answer: str, sources: List[Dict[str, Any]]) -> int:
    expected = {str(item or "").strip() for item in list(case.get("expected_source_types") or []) if str(item or "").strip()}
    if not expected:
        return 2

    answer_text = _norm(answer)
    signals = _source_signals(sources)
    external_expected = expected & {"general_web", "external_academic"}
    local_expected = expected - {"general_web", "external_academic"}

    if external_expected and not (external_expected & signals):
        has_boundary_disclosure = any(
            marker in answer_text
            for marker in [
                "能力边界",
                "无法联网",
                "web 不可用",
                "web search unavailable",
                "无法使用 openalex",
                "openalex 不可用",
                "未启用 web",
                "未启用 openalex",
                "无法提供来源链接",
            ]
        )
        return 2 if has_boundary_disclosure else 0

    if not local_expected:
        return 2

    hit = sum(1 for item in local_expected if item in signals)
    if hit == 0:
        return 0
    if hit < len(local_expected):
        return 1
    return 2


def _score_forbidden_claims(answer: str, forbidden_claims: List[str]) -> int:
    claims = [claim for claim in forbidden_claims if str(claim or "").strip()]
    if not claims:
        return 2
    hit = sum(1 for claim in claims if _norm(claim) in _norm(answer))
    if hit == 0:
        return 2
    if hit < len(claims):
        return 1
    return 0


def _split_claim_units(answer: str) -> List[str]:
    text = str(answer or "").strip()
    if not text:
        return []
    units = re.split(r"[\n。！？!?]+", text)
    return [unit.strip(" -:*") for unit in units if unit.strip(" -:*")]


def _contains_phrase(text: str, phrases: List[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _is_disclaimer(unit_n: str) -> bool:
    return _contains_phrase(
        unit_n,
        [
            "无法访问 openalex",
            "无法访问openalex",
            "openalex 不可用",
            "openalex不可用",
            "无法联网",
            "不能联网",
            "无法获取外部来源",
            "不能获取外部来源",
            "当前环境不能获取外部来源",
            "当前环境无法获取外部来源",
            "不能确认最近两年进展",
            "无法确认最近两年进展",
            "无法提供最近两年进展",
            "无法提供最新研究进展",
            "环境限制",
            "能力边界",
        ],
    )


def _is_recommendation(unit_n: str) -> bool:
    return _contains_phrase(
        unit_n,
        [
            "建议",
            "手动检索",
            "可检索",
            "可以检索",
            "查阅",
            "可通过",
            "通过 ieee xplore",
            "通过 arxiv",
            "通过 google scholar",
            "关键词",
        ],
    ) and not _contains_phrase(
        unit_n,
        [
            "检索结果显示",
            "根据检索结果",
            "已检索到",
            "返回了",
            "结果表明",
        ],
    )


def _is_formatting_text(unit: str) -> bool:
    """
    判断是否为格式性文本，如标题、引用编号、检索关键词等。
    这类文本即使包含数字/年份，也不应计入 unsupported claim。
    """
    u = unit.strip()
    # 1. Markdown 标题 (e.g., # Title, ## Subtitle)
    if u.startswith("#"):
        return True
    # 2. 枚举标题 (e.g., "1. 核心优势：", "1. **算法基础**")
    # 匹配 数字+标点 或 列表符号 开头的行
    if re.match(r"^(\d+[\.\)\s]+|[\-\*]\s+).+$", u):
        # 如果长度较短，或含有粗体，或以冒号结尾，视为标题/格式行
        if len(u) < 25 or "**" in u or u.endswith(("：", ":")):
            return True
    # 3. 证据引用编号 (e.g., "证据[2]", "[1]")
    if re.match(r"^(证据)?\[\d+\]$", u):
        return True
    # 4. 章节/段落标题 (e.g., "关于最近两年（2023-2024）的外部进展：")
    if re.search(r"（20\d{2}-20\d{2}）", u) and u.endswith(("：", ":")):
        return True
    # 5. 检索建议中的关键词 (e.g., "关键词：Hybrid-RRT + 2023|2024")
    if u.startswith(("关键词", "Keywords")) and ("+" in u or "|" in u or "20" in u):
        return True
    # 6. 状态/边界说明的前缀符号 (e.g., "✅ 可完成：", "❌ 不可完成：")
    if any(u.startswith(icon) for icon in ["✅", "❌", "⚠️", "（注："]):
        return True
    return False


def _has_numeric_claim(unit: str) -> bool:
    return bool(re.search(r"\b20\d{2}\b|\b\d+(?:\.\d+)?%\b|\b\d+(?:\.\d+)?\b", unit))


def _has_external_fact_marker(unit_n: str) -> bool:
    return _contains_phrase(
        unit_n,
        [
            "openalex",
            "doi",
            "作者",
            "年份",
            "venue",
            "ieee xplore",
            "arxiv",
            "google scholar",
            "springer",
            "science direct",
            "icra",
            "iros",
        ],
    )


def _has_mechanism_marker(unit_n: str) -> bool:
    return _contains_phrase(
        unit_n,
        [
            "自适应机制",
            "自适应权重",
            "目标偏置",
            "goal-biased",
            "obstacle-aware",
            "障碍物感知",
            "rewiring",
            "混合采样",
            "adaptive strategy",
            "adaptive sampling",
        ],
    )


def _classify_claim_type(unit: str, source_blob_n: str) -> str | None:
    if _is_formatting_text(unit):
        return None
    unit_n = _norm(unit)
    if not unit_n:
        return None
    if _is_disclaimer(unit_n):
        return "disclaimer"
    if _is_recommendation(unit_n):
        return "recommendation"
    if _has_external_fact_marker(unit_n) and unit_n not in source_blob_n:
        return "unsupported_external_fact"
    if _has_numeric_claim(unit) and unit_n not in source_blob_n:
        return "unsupported_numeric_claim"
    if _has_mechanism_marker(unit_n) and unit_n not in source_blob_n:
        return "unsupported_mechanism_claim"
    if unit_n not in source_blob_n and _contains_phrase(unit_n, ["是", "表明", "说明", "证明", "提出", "采用", "面向", "支持"]):
        return "assertion"
    return None


def _claim_reason(claim_type: str) -> str:
    return {
        "disclaimer": "这是能力边界或来源不可用说明，不应视为未支撑事实断言",
        "recommendation": "这是后续检索建议，不应视为系统已检索到的事实结果",
        "assertion": "这是结论性表述，但当前 sources 中缺少对应直接支撑片段",
        "unsupported_external_fact": "提到具体外部来源、论文元数据或检索结果，但当前 sources 中没有对应支撑",
        "unsupported_numeric_claim": "提到具体年份、数字或实验数值，但当前 sources 中没有对应证据",
        "unsupported_mechanism_claim": "提到具体机制或实现细节，但当前 sources 中没有对应证据",
    }.get(claim_type, "当前 sources 中缺少对应支撑")


def _collect_unsupported_claim_notes(answer: str, sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answer_text = _norm(answer)
    source_blob = "\n".join(
        [
            str(source.get("snippet") or source.get("content") or "")
            for source in sources
        ]
    )
    source_blob_n = _norm(source_blob)
    notes: List[Dict[str, str]] = []
    seen: set[str] = set()
    claim_units = _split_claim_units(answer)
    for unit in claim_units:
        claim_type = _classify_claim_type(unit, source_blob_n)
        if not claim_type:
            continue
        trigger_matches = re.findall(r"\b20\d{2}\b|\b\d+(?:\.\d+)?%\b|openalex|doi|ieee xplore|arxiv|google scholar", _norm(unit))
        trigger = ", ".join(trigger_matches[:3]) if trigger_matches else claim_type
        key = f"{claim_type}|{trigger}|{unit}"
        if key in seen:
            continue
        seen.add(key)
        notes.append(
            {
                "claim_type": claim_type,
                "trigger": trigger,
                "text": unit,
                "reason": _claim_reason(claim_type),
            }
        )
    return notes


def judge_answer_with_rubric(case: Dict[str, Any], answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected_docs = list(case.get("expected_document_keywords") or [])
    expected_sections = list(case.get("expected_section_keywords") or [])
    must_include = list(case.get("must_include") or [])
    forbidden_claims = list(case.get("forbidden_claims") or [])

    source_blob = "\n".join(
        f"{s.get('document_title','')} {((s.get('metadata') or {}).get('section_path_text') or (s.get('metadata') or {}).get('section_title') or '')}"
        for s in sources
    )
    answer_text = str(answer or "")

    correct_document = _document_coverage_score(source_blob, expected_docs)
    source_accuracy = 1 if ("联网" not in answer_text and "web" not in _norm(answer_text)) else 0
    section_grounding = 2 if (not expected_sections or _contains_any(source_blob, expected_sections) or _contains_any(answer_text, expected_sections)) else 0
    source_boundary_violation = 1 if any(x in _norm(answer_text) for x in ["according to web", "联网搜索显示", "openalex full text"]) else 0
    source_fit = _score_source_fit(case, answer_text, sources)
    forbidden_claim_score = _score_forbidden_claims(answer_text, forbidden_claims)

    unsupported_claim_notes = _collect_unsupported_claim_notes(answer_text, sources)
    unsupported_claim_risk = 0
    risky_claim_types = {note["claim_type"] for note in unsupported_claim_notes if note.get("claim_type")}
    if risky_claim_types & {"unsupported_external_fact", "unsupported_numeric_claim", "unsupported_mechanism_claim"}:
        unsupported_claim_risk = 2
    elif "assertion" in risky_claim_types:
        unsupported_claim_risk = 1

    completeness_score = _score_completeness(answer_text, must_include)
    clarity_score = _score_clarity(answer_text)

    total_score = (
        correct_document
        + source_accuracy
        + section_grounding
        + source_fit
        + forbidden_claim_score
        + (0 if source_boundary_violation else 1)
        + max(0, 2 - unsupported_claim_risk)
        + completeness_score
        + clarity_score
    )

    notes: List[str] = []
    if not correct_document:
        notes.append("未显著命中预期论文")
    if source_boundary_violation:
        notes.append("存在来源边界违规表达")
    if unsupported_claim_risk > 0:
        notes.append("存在未支撑结论风险")

    return {
        "correct_document": correct_document,
        "source_accuracy": source_accuracy,
        "section_grounding": section_grounding,
        "source_fit": source_fit,
        "forbidden_claim_score": forbidden_claim_score,
        "source_boundary_violation": source_boundary_violation,
        "unsupported_claim_risk": unsupported_claim_risk,
        "unsupported_claim_notes": unsupported_claim_notes,
        "completeness_score": completeness_score,
        "clarity_score": clarity_score,
        "total_score": total_score,
        "notes": notes,
    }
