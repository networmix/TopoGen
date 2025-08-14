"""Hardware capacity vs total attached capacity feasibility check."""

from __future__ import annotations

from typing import Any, Dict

from ..helpers import _node_hw_from_attrs


def check_node_hw_capacity(net: Any, comp_lib: Dict[str, Any]) -> list[str]:
    """For each node with explicit hardware, ensure attached capacity <= HW capacity.

    Unknown components are skipped here (already reported by node_hw_presence).
    """
    issues: list[str] = []

    # Pre-compute total attached capacity per node (sum of incident link capacities)
    attached: dict[str, float] = {}
    for link in net.links.values():
        cap = float(getattr(link, "capacity", 0.0) or 0.0)
        src = str(getattr(link, "source", ""))
        dst = str(getattr(link, "target", ""))
        if src:
            attached[src] = attached.get(src, 0.0) + cap
        if dst:
            attached[dst] = attached.get(dst, 0.0) + cap

    for node in net.nodes.values():
        node_name = str(getattr(node, "name", ""))
        if not node_name:
            continue
        comp_name, count = _node_hw_from_attrs(getattr(node, "attrs", {}) or {})
        if not comp_name or count <= 0.0:
            continue
        comp = comp_lib.get(comp_name)
        if comp is None:
            # Avoid duplicate reporting; node_hw_presence handles unknown comps.
            continue
        hw_cap = float(comp.get("capacity", 0.0)) * float(count)
        total_attached = float(attached.get(node_name, 0.0))
        if total_attached > hw_cap + 1e-9:
            issues.append(
                (
                    "hardware capacity: node '\n"
                    f"{node_name}\n' total attached capacity {total_attached:,.0f} "
                    f"exceeds hardware capacity {hw_cap:,.0f} from component "
                    f"'{comp_name}' (hw_count={count:g})."
                ).replace("\n", "")
            )

    return issues
