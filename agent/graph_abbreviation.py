"""Stable display abbreviations for paper graph nodes."""
import re


def build_title_abbreviation(title: str) -> str:
    """Build a readable, deterministic abbreviation from an original title."""
    normalized = re.sub(r"\s+", " ", str(title or "").strip())
    explicit = re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b", normalized)
    if explicit:
        return explicit[-1]
    words = re.findall(r"[A-Za-z0-9]+", normalized)
    ignored = {"a", "an", "and", "the", "of", "for", "to", "with", "via", "on", "in", "from"}
    terminal_acronym = next((word for word in reversed(words) if re.fullmatch(r"[A-Z0-9]{2,8}", word)), "")
    if terminal_acronym:
        prefix = [word[0].upper() for word in words[:words.index(terminal_acronym)] if word.lower() not in ignored]
        if prefix:
            return "".join(prefix[:4]) + ("-" if len(prefix) > 1 else "") + terminal_acronym
    initials = [word[0].upper() for word in words if word.lower() not in ignored]
    return "".join(initials[:6]) or "PAPER"
