"""Helper utilities for scenario validation."""

from __future__ import annotations

from typing import Any


def _build_ig_coord_map(ig_json: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Map metro names to projected (x, y) coordinates from integrated graph JSON."""
    return {
        node["name"]: (float(node["x"]), float(node["y"]))
        for node in ig_json["nodes"]
        if node["node_type"] == "metro"
    }


def _float_or_nan(value: Any) -> float:
    """Convert value to float or return NaN if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _node_hw_from_attrs(node_attrs: dict[str, object]) -> tuple[str | None, float]:
    """Extract node hardware component name and count from node attrs.

    Returns:
        Tuple of (component name or None, count as float). When hardware is not
        assigned, returns (None, 0.0).
    """
    hw = node_attrs.get("hardware")
    if isinstance(hw, dict):
        comp_name = str(hw.get("component", "")).strip()
        if comp_name:
            count = float(hw.get("count", 1.0))
            return comp_name, count
    return None, 0.0
