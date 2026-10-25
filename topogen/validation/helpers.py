"""Helper utilities for scenario validation."""

from __future__ import annotations

from collections.abc import Mapping


def _node_hw_from_attrs(node_attrs: Mapping[str, object]) -> tuple[str | None, float]:
    """Extract node hardware component name and count from node attrs.

    Returns:
        Tuple of (component name or None, count as float). When hardware is not
        assigned, returns (None, 0.0).
    """
    hw = node_attrs.get("hardware")
    if isinstance(hw, Mapping):
        comp_name = str(hw.get("component", "")).strip()
        if comp_name:
            count = float(hw.get("count", 1.0))
            return comp_name, count
    return None, 0.0
