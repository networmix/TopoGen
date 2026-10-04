"""Compare total attached link capacity with each declared platform budget."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ngraph import Network

from ..helpers import _node_hw_from_attrs


def check_node_hw_capacity(net: Network, comp_lib: dict[str, Any]) -> list[str]:
    attached: dict[str, float] = defaultdict(float)
    for link in net.links.values():
        attached[link.source] += link.capacity
        attached[link.target] += link.capacity
    issues = []
    for node in net.nodes.values():
        component, count = _node_hw_from_attrs(node.attrs)
        if component is None or component not in comp_lib:
            continue  # Presence audit reports unknown references.
        capacity = float(comp_lib[component]["capacity"]) * count
        if not math.isfinite(capacity) or capacity < 0:
            issues.append(
                f"hardware capacity: node '{node.name}' has invalid capacity {capacity} from '{component}'"
            )
        elif attached[node.name] > capacity + 1e-9:
            issues.append(
                f"hardware capacity: node '{node.name}' total attached capacity {attached[node.name]:,.0f} exceeds hardware capacity {capacity:,.0f} from component '{component}' (hw_count={count:g})."
            )
    return issues
