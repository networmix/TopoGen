"""Utility helpers for scenario assembly.

Provides small, focused helpers that are shared by scenario-building modules.
"""

from __future__ import annotations

from typing import Any


def _count_nodes_with_role(
    blueprints: dict[str, dict[str, Any]], blueprint_name: str, role: str
) -> int:
    """Return the number of nodes with a given role in a blueprint.

    Args:
        blueprints: Mapping of blueprint names to definitions.
        blueprint_name: Name of the blueprint to inspect.
        role: Role attribute to match (e.g., "core", "dc").

    Returns:
        Number of nodes across all groups whose ``attrs.role`` equals ``role``.
        Returns 0 when the blueprint or role is not found.
    """
    try:
        bp = blueprints.get(blueprint_name, {})
        groups = bp.get("groups", {}) or {}
        total = 0
        for _gname, gdef in groups.items():
            attrs = gdef.get("attrs", {}) or {}
            if attrs.get("role") == role:
                try:
                    total += int(gdef.get("node_count", 0))
                except Exception:
                    continue
        return int(total)
    except Exception:
        return 0
