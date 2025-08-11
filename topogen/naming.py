"""Naming utilities for stable identifiers.

Provides a single source of truth for generating metro slugs used in
configuration keys and path-like identifiers.
"""

from __future__ import annotations

import re


def metro_slug(name: str) -> str:
    """Return a stable slug for a metro display name.

    The slug is used for configuration keys (e.g., build_overrides) and other
    path-like identifiers where a concise, normalized identifier is needed.

    Rules:
    - Remove everything after the first comma (drops state suffixes).
    - Lowercase the string.
    - Replace whitespace and hyphen sequences with a single hyphen.
    - Remove any remaining characters except ``a-z``, ``0-9``, and ``-``.
    - Collapse duplicate hyphens and strip leading/trailing hyphens.
    - Truncate to 30 characters for practicality.

    Args:
        name: Human-readable metro name (possibly with punctuation/state suffix).

    Returns:
        Normalized slug string (e.g., "Salt Lake City" -> "salt-lake-city").
    """

    if not isinstance(name, str):
        name = str(name)

    # Remove state suffix after comma
    base = name.split(",")[0]

    # Lowercase and normalize separators
    lowered = base.lower().strip()
    sep_norm = re.sub(r"[\s\-]+", "-", lowered)

    # Remove disallowed characters (keep a-z, 0-9, hyphen)
    cleaned = re.sub(r"[^a-z0-9-]", "", sep_norm)

    # Collapse duplicates and trim
    collapsed = re.sub(r"-+", "-", cleaned).strip("-")

    # Practical length limit
    return collapsed[:30]
