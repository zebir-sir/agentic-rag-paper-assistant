from agent.prompt_registry import (
    BASE_SYSTEM_PROMPT_ID,
    SOURCE_SELECTION_POLICY_ID,
    SYSTEM_PROMPT_ID,
    build_system_prompt,
    get_prompt,
)
from agent.prompts import SOURCE_SELECTION_POLICY, SYSTEM_PROMPT


def test_prompt_registry_exposes_base_and_full_prompts():
    base = get_prompt(BASE_SYSTEM_PROMPT_ID)
    full = get_prompt(SYSTEM_PROMPT_ID)

    assert "科研论文阅读与分析助手" in base.template
    assert "Source selection policy" in full.template
    assert "OpenAlex" in full.template


def test_build_system_prompt_matches_compatibility_exports():
    built = build_system_prompt()

    assert built == SYSTEM_PROMPT
    assert get_prompt(SOURCE_SELECTION_POLICY_ID).template == SOURCE_SELECTION_POLICY


def test_unknown_prompt_id_raises_key_error():
    try:
        get_prompt("missing.prompt")
    except KeyError as exc:
        assert "Unknown prompt_id" in str(exc)
    else:
        raise AssertionError("expected KeyError for unknown prompt id")
