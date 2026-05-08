import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from agent.memory_utils import MemoryState, build_context_without_compaction

    history_messages = [
        {"role": "user", "content": "根据上传论文总结方法贡献"},
        {
            "role": "assistant",
            "content": (
                '{"intent_plan":{"intent":"local_paper_qa"},'
                '"tools_executed":["hybrid_search"],'
                '"raw_model_content_preview":"planner raw"}'
            ),
        },
        {"role": "assistant", "content": "这篇论文的方法贡献主要体现在检索策略与证据整合上。"},
    ]

    result = build_context_without_compaction(
        history_messages=history_messages,
        current_question="继续总结局限性",
        memory_state=MemoryState(),
    )
    prompt = result.full_prompt

    forbidden = [
        "intent_plan",
        "tools_executed",
        "raw_model_content_preview",
        "hybrid_search",
    ]
    for key in forbidden:
        if key in prompt:
            raise AssertionError(f"context prompt should not contain {key}")

    if "根据上传论文总结方法贡献" not in prompt:
        raise AssertionError("user content should be preserved")
    if "这篇论文的方法贡献主要体现在检索策略与证据整合上。" not in prompt:
        raise AssertionError("final assistant answer should be preserved")

    print("check_history_message_clean passed")


if __name__ == "__main__":
    main()
