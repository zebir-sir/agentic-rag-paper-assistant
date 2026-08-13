from agent.embedding_runtime import detect_embedding_language, get_embedding_route


def test_embedding_route_uses_chinese_model_for_chinese_text(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_ZH", "zh-model")
    assert detect_embedding_language("路径规划算法的实验结果") == "zh"
    assert get_embedding_route("路径规划算法的实验结果").model == "zh-model"


def test_embedding_route_uses_english_model_for_english_text(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_EN", "en-model")
    assert detect_embedding_language("Ablation results for path planning") == "en"
    assert get_embedding_route("Ablation results for path planning").model == "en-model"
