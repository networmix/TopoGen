"""Checks for node hardware assignment completeness and basic validity."""

from __future__ import annotations

from typing import Any, Dict


def check_node_hw_presence(
    net: Any,
    comps_section: Dict[str, Any],
    comp_lib: Dict[str, Any],
) -> list[str]:
    """If role is mapped in components.hw_component, enforce that node has hardware.

    Also flags unknown components referenced by node hardware and capacity calc errors.
    """
    issues: list[str] = []

    if not isinstance(comps_section, dict) or not comps_section:
        return issues

    for node in net.nodes.values():
        nname = str(getattr(node, "name", ""))
        attrs = getattr(node, "attrs", {}) or {}
        role = str(attrs.get("role", "")).strip()
        if not role or role not in comps_section:
            continue
        hw = attrs.get("hardware")
        if not isinstance(hw, dict) or not str(hw.get("component", "")).strip():
            issues.append(
                f"node hardware: role '{role}' mapped in components.hw_component but node '{nname}' has no hardware assignment"
            )
        else:
            try:
                comp_name = str(hw.get("component", "")).strip()
                count = float(hw.get("count", 1.0))
                comp = comp_lib.get(comp_name)
                if comp is None:
                    issues.append(
                        f"node hardware: node '{nname}' references unknown component '{comp_name}'"
                    )
                else:
                    # Compute capacity once to surface calculation issues (parity with original).
                    _ = float(comp.get("capacity", 0.0)) * float(count)
            except Exception as e2:  # pragma: no cover
                issues.append(
                    f"node hardware: capacity calculation error for node '{nname}': {e2}"
                )

    return issues
