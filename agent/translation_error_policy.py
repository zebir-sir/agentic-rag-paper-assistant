"""HTTP classification policy for translation provider failures."""

from __future__ import annotations

from openai import APIConnectionError, APIStatusError, APITimeoutError


def is_translation_service_unavailable(error: Exception) -> bool:
    """Return whether retrying later may succeed without changing the request."""
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIStatusError):
        return int(error.status_code or 0) >= 500
    return False
