"""OpenAlex capability checks shared by planning and runtime diagnostics."""

import os


def _is_openalex_enabled() -> bool:
    return bool(os.getenv("OPENALEX_API_KEY", "").strip())
