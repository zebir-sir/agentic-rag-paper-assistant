from evals.judges.source_boundary_judge import judge_source_boundary


def test_openalex_unavailable_disclaimer_is_not_violation() -> None:
    result = judge_source_boundary(
        case={"question": "补充最近两年的外部进展"},
        answer="当前无法访问 OpenAlex，因此不能提供最近两年进展。",
        capabilities={"openalex_search_enabled": False, "web_search_enabled": False},
    )
    assert result["source_boundary_violation"] == 0


def test_openalex_claim_with_metadata_is_violation_when_unavailable() -> None:
    result = judge_source_boundary(
        case={"question": "补充最近两年的外部进展"},
        answer="根据 OpenAlex 检索结果，论文 X，DOI: 10.1234/test，作者：Zhang，年份：2024。",
        capabilities={"openalex_search_enabled": False, "web_search_enabled": False},
    )
    assert result["source_boundary_violation"] == 1


def test_web_unavailable_disclaimer_is_not_violation() -> None:
    result = judge_source_boundary(
        case={"question": "联网查一下最新资料"},
        answer="当前无法联网确认最新资料，因此这里只能基于本地证据回答。",
        capabilities={"openalex_search_enabled": False, "web_search_enabled": False},
    )
    assert result["source_boundary_violation"] == 0
