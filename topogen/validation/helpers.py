"""Helper utilities for scenario validation."""

from __future__ import annotations

from typing import Any


def _build_ig_coord_map(ig_json: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Return metro name to (x, y) coordinate map from integrated graph JSON.

    Args:
        ig_json: Parsed JSON object for the integrated graph.

    Returns:
        Mapping from metro display name to coordinate tuple in the target CRS.
    """
    mapping: dict[str, tuple[float, float]] = {}
    for node in ig_json.get("nodes", []) or []:
        try:
            node_type = node.get("node_type")
        except AttributeError:
            continue
        if node_type not in ("metro", "metro+highway"):
            continue
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        try:
            x = float(node.get("x", 0.0))
            y = float(node.get("y", 0.0))
        except Exception:
            continue
        mapping[name] = (x, y)
    return mapping


def _float_or_nan(value: Any) -> float:
    """Convert value to float or return NaN if conversion fails."""
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
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
            try:
                count = float(hw.get("count", 1.0))
            except Exception:
                count = 1.0
            return comp_name, count
    return None, 0.0
