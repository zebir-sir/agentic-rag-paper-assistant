from pathlib import Path
from types import SimpleNamespace

import pytest

from ingestion.vision_client import ArkVisionClient, figure_context


def test_figure_context_uses_caption_neighbors():
    before, after = figure_context("Method\nWe sample paths.\nFigure 1: Results\nThe curve improves.", "Figure 1: Results")
    assert "We sample paths" in before
    assert "curve improves" in after


@pytest.mark.asyncio
async def test_ark_vision_client_includes_caption_and_context(tmp_path, monkeypatch):
    image_path = Path(tmp_path) / "figure.png"
    image_path.write_bytes(b"fake-image")
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"summary":"A trend","figure_type":"line chart","research_purpose":"compare methods","experimental_task":"planning","axes_and_units":["x: steps","y: success rate"],"series_or_methods":["HA-RRT"],"visible_text":["A"],"observations":["up"],"quantitative_findings":["HA-RRT is highest"],"comparative_claims":["HA-RRT improves"],"visual_tags":["visual:line_chart","role:experimental_result","evidence:comparison","topic:rrt","not-allowed"],"limitations":[],"evidence_confidence":"medium"}'
                        )
                    )
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setenv("VISION_LLM_CHOICE", "vision-test")
    result = await ArkVisionClient(client=client).analyze_figure(
        str(image_path), caption="Figure 1: Results", context_before="Method", context_after="Conclusion"
    )

    assert result.summary == "A trend"
    assert result.figure_type == "line chart"
    assert result.axes_and_units == ["x: steps", "y: success rate"]
    assert result.quantitative_findings == ["HA-RRT is highest"]
    assert result.visual_tags == ["visual:line_chart", "role:experimental_result", "evidence:comparison"]
    prompt = captured["messages"][0]["content"][0]["text"]
    assert "Figure 1: Results" in prompt
    assert "Method" in prompt
    assert "Conclusion" in prompt
    assert captured["model"] == "vision-test"
