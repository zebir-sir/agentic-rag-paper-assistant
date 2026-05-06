import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.prompt_templates import (
    build_multi_experiment_compare_prompt,
    build_multi_method_compare_prompt,
    build_multi_problem_compare_prompt,
    build_multi_value_compare_prompt,
    build_single_experiment_prompt,
    build_single_innovation_prompt,
    build_single_inspiration_prompt,
    build_single_limitation_prompt,
    build_single_method_prompt,
    build_single_summary_prompt,
)


def main() -> None:
    title = "Sample Paper"
    single_prompts = [
        build_single_summary_prompt(title),
        build_single_innovation_prompt(title),
        build_single_method_prompt(title),
        build_single_experiment_prompt(title),
        build_single_limitation_prompt(title),
        build_single_inspiration_prompt(title),
    ]
    for prompt in single_prompts:
        assert title in prompt, "single-paper prompt must include the selected title"
        assert "{title}" not in prompt, "single-paper prompt leaked a title placeholder"

    titles = ["Paper A", "Paper B"]
    multi_prompts = [
        build_multi_problem_compare_prompt(titles),
        build_multi_method_compare_prompt(titles),
        build_multi_experiment_compare_prompt(titles),
        build_multi_value_compare_prompt(titles),
    ]
    for prompt in multi_prompts:
        assert all(item in prompt for item in titles), "multi-paper prompt must include all selected titles"
        assert "{title}" not in prompt and "{titles}" not in prompt, "multi-paper prompt leaked a placeholder"

    print("PASS: UI prompt templates include selected titles")


if __name__ == "__main__":
    main()
