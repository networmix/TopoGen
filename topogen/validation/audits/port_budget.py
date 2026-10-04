"""Audit declared optic module counts against declared platform port budgets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ngraph import Network

from ..helpers import _node_hw_from_attrs


def audit_port_budget(net: Network, comp_lib: dict[str, Any]) -> list[str]:
    required: dict[str, float] = defaultdict(float)
    for link in net.links.values():
        hardware = link.attrs.get("hardware", {})
        for end, node in (("source", link.source), ("target", link.target)):
            optic = hardware.get(end)
            if optic is None or optic["component"] not in comp_lib:
                continue  # Optics audit reports incomplete/unknown assignments.
            component = comp_lib[optic["component"]]
            required[node] += float(optic.get("count", 1)) * component["ports"]
    issues = []
    for node, needed in sorted(required.items()):
        platform, count = _node_hw_from_attrs(net.nodes[node].attrs)
        if platform is None or platform not in comp_lib:
            continue  # Hardware presence audit reports this separately.
        available = comp_lib[platform]["ports"] * count
        if needed > available:
            issues.append(
                f"hardware ports: node '{node}' requires {needed:g} ports for link optics but only {available:g} ports are available on '{platform}' (count={count:g})."
            )
    return issues
