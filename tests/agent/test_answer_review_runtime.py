from agent.answer_review_runtime import review_generated_answer


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


def test_review_generated_answer_records_risk_without_appending_a_chat_caveat():
    result = review_generated_answer(
        answer="实验表明该方法在 2024 年数据集上提升了 17.3% 的成功率。",
        sources=_local_sources(),
        is_local_question=True,
    )

    assert result.reviewed is True
    assert result.review_action == "retain_with_metadata"
    assert result.unsupported_claim_risk == 2
    assert result.revised_answer == "实验表明该方法在 2024 年数据集上提升了 17.3% 的成功率。"


def test_review_generated_answer_keeps_existing_chat_content_unchanged():
    result = review_generated_answer(
        answer=(
            "实验表明该方法在 2024 年数据集上提升了 17.3% 的成功率。\n\n"
            "注：以上判断基于当前检索片段，仍需回到原文进一步确认。"
        ),
        sources=_local_sources(),
        is_local_question=True,
    )

    assert result.reviewed is True
    assert result.review_action == "retain_with_metadata"
    assert result.revised_answer.count("仍需回到原文进一步确认") == 1


def test_review_generated_answer_skips_non_local_questions():
    result = review_generated_answer(
        answer="RRT* 是一种渐近最优的采样式路径规划算法。",
        sources=[],
        is_local_question=False,
    )

    assert result.reviewed is False
    assert result.review_action == "skip_non_local_question"
    assert result.revised_answer == "RRT* 是一种渐近最优的采样式路径规划算法。"
