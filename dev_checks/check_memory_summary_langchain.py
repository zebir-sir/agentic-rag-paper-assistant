import asyncio
import sys
from pathlib import Path


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from agent.api import _summarize_for_memory  # noqa: WPS433

    messages_to_compact = [
        {"role": "user", "content": "我正在比较 RRT、RRT* 和 Informed RRT* 在二维随机障碍环境中的表现。"},
        {"role": "assistant", "content": "我们讨论了路径长度、成功率、运行时间和节点数量等指标。"},
        {"role": "user", "content": "我希望后续重点关注无人艇路径规划场景，以及动态障碍物下的改进。"},
    ]

    summary = await _summarize_for_memory(
        session_id="dev_check_memory_summary_langchain",
        user_id="dev_check_user",
        old_summary="",
        messages_to_compact=messages_to_compact,
    )

    if not isinstance(summary, str) or not summary.strip():
        raise AssertionError("summary must be a non-empty string")
    if len(summary.strip()) <= 20:
        raise AssertionError("summary should be longer than 20 characters")

    keyword_hits = [
        kw for kw in ["RRT", "路径规划", "无人艇", "动态障碍"] if kw in summary
    ]
    if not keyword_hits:
        raise AssertionError("summary should contain at least one expected keyword")

    print("check_memory_summary_langchain passed")


if __name__ == "__main__":
    asyncio.run(main())
