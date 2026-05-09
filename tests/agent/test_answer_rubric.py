from evals.judges.answer_rubric import judge_answer_with_rubric


def _local_sources() -> list[dict]:
    return [
        {
            "document_title": "Hybrid-RRT",
            "snippet": (
                "Hybrid-RRT improves convergence and balances exploration and exploitation. "
                "The abstract reports 40.83% improvement compared with Informed RRT*."
            ),
            "metadata": {
                "section_title": "Abstract",
                "section_path_text": "Hybrid-RRT > Abstract",
                "source_type": "local",
            },
        }
    ]


def test_openalex_disclaimer_does_not_increase_risk() -> None:
    result = judge_answer_with_rubric(
        case={"question": "补充最近两年的外部进展"},
        answer="当前无法访问 OpenAlex，因此不能确认最近两年的外部进展。",
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 0
    assert any(note["claim_type"] == "disclaimer" for note in result["unsupported_claim_notes"])


def test_manual_search_recommendation_does_not_increase_risk() -> None:
    result = judge_answer_with_rubric(
        case={"question": "补充最近两年的外部进展"},
        answer='建议手动检索 IEEE Xplore 或 arXiv，关键词可用 "Hybrid-RRT 2023 2024"。',
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 0
    assert any(note["claim_type"] == "recommendation" for note in result["unsupported_claim_notes"])


def test_unsupported_external_fact_increases_risk() -> None:
    result = judge_answer_with_rubric(
        case={"question": "补充最近两年的外部进展"},
        answer="2024 年某论文提出 X，DOI 为 10.1234/test，作者为 Zhang。",
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 2
    assert any(note["claim_type"] == "unsupported_external_fact" for note in result["unsupported_claim_notes"])


def test_unsupported_numeric_claim_increases_risk() -> None:
    result = judge_answer_with_rubric(
        case={"question": "总结实验结果"},
        answer="实验表明该方法在 2024 年数据集上提升了 17.3% 的成功率。",
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 2
    assert any(note["claim_type"] == "unsupported_numeric_claim" for note in result["unsupported_claim_notes"])


def test_formatting_text_does_not_increase_risk() -> None:
    # 1. 枚举标题
    result = judge_answer_with_rubric(
        case={"question": "总结优势"},
        answer="1. 核心优势：\n该方法显著提升了效率。",
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 0
    assert len(result["unsupported_claim_notes"]) == 0

    # 2. 证据引用编号
    result = judge_answer_with_rubric(
        case={"question": "总结优势"},
        answer="该方法效率高。\n证据[2]",
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 0
    assert len(result["unsupported_claim_notes"]) == 0

    # 3. Markdown 标题
    result = judge_answer_with_rubric(
        case={"question": "总结优势"},
        answer="### 实验结论\n效率提升明显。",
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 0
    assert len(result["unsupported_claim_notes"]) == 0

    # 4. 包含年份的章节标题
    result = judge_answer_with_rubric(
        case={"question": "外部进展"},
        answer="关于最近两年（2023-2024）的外部进展：\n目前暂无数据。",
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 0
    assert len(result["unsupported_claim_notes"]) == 0

    # 5. 检索建议中的关键词
    result = judge_answer_with_rubric(
        case={"question": "检索建议"},
        answer="关键词：Hybrid-RRT + 2023|2024",
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 0
    # 被 _is_formatting_text 过滤后不再进入 notes
    assert len(result["unsupported_claim_notes"]) == 0


def test_unsupported_mechanism_claim_increases_risk() -> None:
    result = judge_answer_with_rubric(
        case={"question": "机制细节"},
        answer="在开阔区域使用目标偏置采样，在狭窄通道切换混合采样策略。",
        sources=_local_sources(),
    )
    assert result["unsupported_claim_risk"] == 2
    assert any(note["claim_type"] == "unsupported_mechanism_claim" for note in result["unsupported_claim_notes"])
