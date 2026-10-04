"""Metro configuration names and stable site-adjacency identifiers."""

from __future__ import annotations

import re


def metro_slug(name: str) -> str:
    """Normalize a metro name to a lowercase slug of at most 30 characters.

    Remove the comma-separated state suffix and punctuation; collapse whitespace
    and hyphens. For example, ``Salt Lake City, UT`` becomes ``salt-lake-city``.
    """

    if not isinstance(name, str):
        name = str(name)

    base = name.split(",")[0]

    lowered = base.lower().strip()
    sep_norm = re.sub(r"[\s\-]+", "-", lowered)

    cleaned = re.sub(r"[^a-z0-9-]", "", sep_norm)

    collapsed = re.sub(r"-+", "-", cleaned).strip("-")

    return collapsed[:30]


def site_edge_id(source: str, target: str, key: str) -> str:
    """Identify one site adjacency in the graph's edge iteration order."""
    return f"{source}|{target}|{key}"
