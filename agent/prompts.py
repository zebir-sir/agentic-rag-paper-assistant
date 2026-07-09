from .prompt_registry import (
    SOURCE_SELECTION_POLICY_ID,
    SYSTEM_PROMPT_ID,
    get_prompt,
)


SOURCE_SELECTION_POLICY = get_prompt(SOURCE_SELECTION_POLICY_ID).template
SYSTEM_PROMPT = get_prompt(SYSTEM_PROMPT_ID).template
