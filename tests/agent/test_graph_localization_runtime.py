from agent.graph_localization_schema import graph_localization_hash, normalize_graph_card, validate_graph_card


def test_graph_localization_hash_changes_with_profile():
    assert graph_localization_hash("Paper", "one") != graph_localization_hash("Paper", "two")


def test_normalize_graph_card_bounds_fields_and_keywords():
    card = normalize_graph_card({"title_zh": "中文题名", "problem_zh": "问题", "method_zh": "方法", "innovation_zh": "创新", "keywords_zh": ["采样", "路径规划", "RRT", "额外", "更多", "忽略"]}, "Original")
    assert card["title_zh"] == "中文题名"
    assert len(card["keywords_zh"]) == 5


def test_graph_card_validation_preserves_title_acronyms():
    valid = validate_graph_card({"title_zh": "HA-RRT 方法", "problem_zh": "问题", "method_zh": "HA-RRT", "innovation_zh": "创新"}, "Hybrid-Aware RRT")
    assert valid["valid"] is True
