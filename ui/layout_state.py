from __future__ import annotations

from typing import Any, Sequence


def should_show_welcome_guide_from_state(
    messages: Sequence[Any] | None,
    pending_prompt: Any,
    is_streaming: bool,
    streaming_response: str | None,
) -> bool:
    return (
        not list(messages or [])
        and not pending_prompt
        and not bool(is_streaming)
        and not str(streaming_response or "").strip()
    )
