"""火山方舟视觉模型适配器。

该模块只负责图片理解与结构化结果解析，不参与 LangGraph 主回答链路。
"""

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


_ALLOWED_VISUAL_TAGS = {
    "visual:line_chart", "visual:bar_chart", "visual:scatter_plot", "visual:box_plot",
    "visual:heatmap", "visual:table_image", "visual:method_diagram", "visual:architecture_diagram",
    "visual:algorithm_flow", "visual:trajectory_path", "visual:environment_map",
    "role:experimental_result", "role:method_framework", "role:method_workflow", "role:system_architecture",
    "role:qualitative_visualization", "role:comparison", "role:ablation", "role:benchmark",
    "role:dataset_overview", "role:case_study", "role:theory_analysis", "role:implementation_detail",
    "evidence:quantitative", "evidence:qualitative", "evidence:comparison", "evidence:ablation",
    "evidence:efficiency", "evidence:robustness", "evidence:generalization", "evidence:error_analysis",
}

_FIGURE_TYPE_TAGS = {
    "line": "visual:line_chart", "bar": "visual:bar_chart", "scatter": "visual:scatter_plot",
    "box": "visual:box_plot", "heatmap": "visual:heatmap", "table": "visual:table_image",
    "architecture": "visual:architecture_diagram", "flow": "visual:algorithm_flow",
    "trajectory": "visual:trajectory_path", "path": "visual:trajectory_path", "environment": "visual:environment_map",
}


def figure_context(markdown: str, caption: str, window: int = 3) -> tuple[str, str]:
    """从导出的 markdown 中提取图注附近的有限上下文。"""
    lines = str(markdown or "").splitlines()
    target = str(caption or "").strip().lower()
    if not target:
        return "", ""
    for index, line in enumerate(lines):
        if target in line.strip().lower():
            before = " ".join(v.strip() for v in lines[max(0, index - window):index] if v.strip())
            after = " ".join(v.strip() for v in lines[index + 1:index + 1 + window] if v.strip())
            return before[:600], after[:600]
    return "", ""


@dataclass
class VisionFigureAnalysis:
    """一张论文图的研究阅读卡，而不是通用图片描述。"""

    summary: str
    figure_type: str
    visible_text: List[str]
    observations: List[str]
    limitations: List[str]
    research_purpose: str
    experimental_task: str
    axes_and_units: List[str]
    series_or_methods: List[str]
    quantitative_findings: List[str]
    comparative_claims: List[str]
    visual_tags: List[str]
    evidence_confidence: str
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "figure_type": self.figure_type,
            "visible_text": self.visible_text,
            "observations": self.observations,
            "limitations": self.limitations,
            "research_purpose": self.research_purpose,
            "experimental_task": self.experimental_task,
            "axes_and_units": self.axes_and_units,
            "series_or_methods": self.series_or_methods,
            "quantitative_findings": self.quantitative_findings,
            "comparative_claims": self.comparative_claims,
            "visual_tags": self.visual_tags,
            "evidence_confidence": self.evidence_confidence,
            "model": self.model,
        }


def _extract_json(text: str) -> Dict[str, Any]:
    """兼容模型偶尔包裹 markdown fence 的 JSON 输出。"""
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            raise ValueError("视觉模型未返回可解析的 JSON")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("视觉模型 JSON 顶层必须是对象")
    return parsed


def _as_text_list(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_visual_tags(value: Any, figure_type: str) -> List[str]:
    tags: List[str] = []
    for tag in _as_text_list(value):
        normalized = tag.lower().replace("-", "_").replace(" ", "_")
        if normalized in _ALLOWED_VISUAL_TAGS and normalized not in tags:
            tags.append(normalized)
    for keyword, tag in _FIGURE_TYPE_TAGS.items():
        if keyword in str(figure_type or "").lower() and tag not in tags:
            tags.append(tag)
            break
    return tags


class ArkVisionClient:
    """使用 OpenAI 兼容协议调用火山方舟视觉模型。"""

    def __init__(self, client: Optional[AsyncOpenAI] = None):
        self.model = os.getenv("VISION_LLM_CHOICE", "doubao-seed-2.0-lite")
        self.client = client or AsyncOpenAI(
            api_key=os.getenv("VISION_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("VISION_OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3"),
        )

    async def analyze_figure(
        self,
        image_path: str,
        caption: str = "",
        context_before: str = "",
        context_after: str = "",
    ) -> VisionFigureAnalysis:
        path = Path(image_path)
        image_data = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lower().lstrip(".") or "png"
        if suffix == "jpg":
            suffix = "jpeg"
        prompt = (
            "你是论文图表阅读助手。请把图片转换为研究者可检索、可核验的实验/方法证据卡，"
            "而不是泛化地描述图片外观。必须结合图片、图注、图前后文；对曲线、柱状图和消融图，"
            "优先识别任务、指标、坐标轴、方法/系列、变化趋势、相对比较与可辨识数值。"
            "不得从模糊图像猜测精确数值；无法确认时明确写入 limitations。只返回 JSON，不要 markdown。\n"
            "JSON schema: {summary:string, figure_type:string, research_purpose:string, experimental_task:string, "
            "axes_and_units:string[], series_or_methods:string[], visible_text:string[], observations:string[], "
            "quantitative_findings:string[], comparative_claims:string[], visual_tags:string[], limitations:string[], evidence_confidence:string}. "
            "visual_tags may only use: visual:line_chart, visual:bar_chart, visual:scatter_plot, visual:box_plot, "
            "visual:heatmap, visual:table_image, visual:method_diagram, visual:architecture_diagram, visual:algorithm_flow, "
            "visual:trajectory_path, visual:environment_map, role:experimental_result, role:method_framework, role:method_workflow, "
            "role:system_architecture, role:qualitative_visualization, role:comparison, role:ablation, role:benchmark, "
            "role:dataset_overview, role:case_study, role:theory_analysis, role:implementation_detail, evidence:quantitative, "
            "evidence:qualitative, evidence:comparison, evidence:ablation, evidence:efficiency, evidence:robustness, "
            "evidence:generalization, evidence:error_analysis. Select at most one role:* tag, at most two visual:* tags, "
            "and only evidence:* tags directly supported by visible evidence. Do not infer a domain-specific topic.\n"
            f"图注：{caption or '[无图注]'}\n"
            f"图前文：{context_before or '[无]'}\n"
            f"图后文：{context_after or '[无]'}"
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=1200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/{suffix};base64,{image_data}"}},
                    ],
                }
            ],
        )
        payload = _extract_json(response.choices[0].message.content)
        figure_type = str(payload.get("figure_type", "unknown")).strip() or "unknown"
        return VisionFigureAnalysis(
            summary=str(payload.get("summary", "")).strip(),
            figure_type=figure_type,
            visible_text=_as_text_list(payload.get("visible_text")),
            observations=_as_text_list(payload.get("observations")),
            limitations=_as_text_list(payload.get("limitations")),
            research_purpose=str(payload.get("research_purpose", "")).strip(),
            experimental_task=str(payload.get("experimental_task", "")).strip(),
            axes_and_units=_as_text_list(payload.get("axes_and_units")),
            series_or_methods=_as_text_list(payload.get("series_or_methods")),
            quantitative_findings=_as_text_list(payload.get("quantitative_findings")),
            comparative_claims=_as_text_list(payload.get("comparative_claims")),
            visual_tags=_normalize_visual_tags(payload.get("visual_tags"), figure_type),
            evidence_confidence=str(payload.get("evidence_confidence", "unknown")).strip() or "unknown",
            model=self.model,
        )
