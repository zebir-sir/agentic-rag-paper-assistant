from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


_ACK_MESSAGES = {
    "好",
    "好的",
    "收到",
    "明白了",
    "知道了",
    "行",
    "可以",
    "ok",
    "okay",
    "thanks",
    "thank you",
}

_QUESTION_CUES = (
    "?",
    "？",
    "为什么",
    "怎么",
    "如何",
    "哪个",
    "哪里",
    "多少",
    "是否",
    "能不能",
    "区别",
    "差别",
)

_FOLLOW_UP_CUES = (
    "这个",
    "那个",
    "它",
    "其",
    "其中",
    "那",
    "再",
    "继续",
    "展开",
    "细说",
    "对比",
    "表",
    "图",
    "算法",
    "实验",
    "结果",
    "方法",
)

_CONSTRAINT_CUES = (
    "只看",
    "只基于",
    "不要",
    "别联网",
    "别扩展",
    "别看",
    "仅",
    "限定",
    "优先看",
    "忽略",
    "不联网",
    "只用",
    "不要用",
)


@dataclass
class DialogPolicyDecision:
    dialog_act: str
    carry_context: bool
    response_style: str
    reason: str


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def classify_dialog_turn(
    *,
    latest_query: str,
    history_messages: List[Dict[str, str]],
) -> DialogPolicyDecision:
    query = _normalize_text(latest_query)
    has_history = bool(history_messages)
    lowered = query.lower()

    if not query:
        return DialogPolicyDecision(
            dialog_act="empty_turn",
            carry_context=False,
            response_style="brief_ack",
            reason="当前输入为空。",
        )

    if lowered in _ACK_MESSAGES:
        return DialogPolicyDecision(
            dialog_act="acknowledgement",
            carry_context=False,
            response_style="brief_ack",
            reason="当前输入更像是简短确认或收尾。",
        )

    if any(cue in query for cue in _CONSTRAINT_CUES):
        return DialogPolicyDecision(
            dialog_act="constraint_update",
            carry_context=has_history,
            response_style="respect_constraints",
            reason="当前输入主要在补充回答范围、来源或展示约束。",
        )

    if has_history and any(cue in query for cue in _FOLLOW_UP_CUES):
        return DialogPolicyDecision(
            dialog_act="follow_up_reference",
            carry_context=True,
            response_style="normal",
            reason="当前输入依赖上文对象，属于多轮追问或指代承接。",
        )

    if any(cue in query for cue in _QUESTION_CUES):
        return DialogPolicyDecision(
            dialog_act="question_seek_answer",
            carry_context=has_history and len(query) <= 18,
            response_style="normal",
            reason="当前输入是明确提问。",
        )

    if has_history and len(query) <= 16:
        return DialogPolicyDecision(
            dialog_act="contextual_follow_up",
            carry_context=True,
            response_style="normal",
            reason="当前输入较短，默认按依赖上文的继续追问处理。",
        )

    return DialogPolicyDecision(
        dialog_act="standalone_statement",
        carry_context=False,
        response_style="normal",
        reason="当前输入可以独立理解。",
    )
