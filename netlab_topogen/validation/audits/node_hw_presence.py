"""Validate declared platform references and optional role assignment coverage."""

from __future__ import annotations

import math
from typing import Any

from ngraph import Network

from ..helpers import _node_hw_from_attrs


def check_node_hw_presence(
    net: Network, assignments: dict[str, str], comp_lib: dict[str, Any]
) -> list[str]:
    issues = []
    roles = {node.attrs.get("role", "") for node in net.nodes.values()}
    if assignments:
        for role in sorted(roles - assignments.keys() - {""}):
            issues.append(
                f"node hardware: components.hw_component missing mapping for role '{role}'"
            )
    for node in net.nodes.values():
        role = node.attrs.get("role", "")
        component, count = _node_hw_from_attrs(node.attrs)
        if component is None:
            if assignments.get(role):
                issues.append(
                    f"node hardware: role '{role}' mapped in components.hw_component but node '{node.name}' has no hardware assignment"
                )
            continue
        if component not in comp_lib:
            issues.append(
                f"node hardware: node '{node.name}' references unknown component '{component}'"
            )
        if not math.isfinite(count) or count <= 0:
            issues.append(
                f"node hardware: node '{node.name}' has invalid hardware count {count}"
            )
    return issues
