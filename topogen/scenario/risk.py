"""Risk definitions use the length of their owning corridor path."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import networkx as nx

from topogen.corridors import corridor_risk_name

if TYPE_CHECKING:
    from topogen.config import TopologyConfig


def _build_risk_groups_section(
    graph: nx.MultiGraph, config: "TopologyConfig"
) -> list[dict[str, Any]]:
    if not config.corridors.risk_groups.enabled:
        return []
    distances: dict[str, int] = {}
    referenced: set[str] = set()
    for source, target, key, data in graph.edges(keys=True, data=True):
        name = corridor_risk_name(
            config.corridors.risk_groups.group_prefix,
            graph.nodes[source]["name"],
            graph.nodes[target]["name"],
            key,
        )
        if name in distances:
            raise ValueError(f"Duplicate corridor risk identity: {name}")
        length = float(data["length_km"])
        if not math.isfinite(length) or length <= 0:
            raise ValueError(f"Corridor '{name}' must have positive finite length")
        distances[name] = math.ceil(length)
        referenced.update(data.get("risk_groups", []))
    unknown = referenced - distances.keys()
    if unknown:
        raise ValueError(f"Risk groups have no owning corridor path: {sorted(unknown)}")
    return [
        {
            "name": name,
            "attrs": {"type": "corridor_risk", "distance_km": distances[name]},
        }
        for name in sorted(referenced)
    ]
