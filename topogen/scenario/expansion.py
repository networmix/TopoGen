"""Resolve site budgets and hardware against one complete device network."""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any

import networkx as nx
from ngraph import Link, Network
from ngraph.dsl.blueprints.expand import expand_network_dsl

from topogen.naming import site_edge_id
from topogen.roles import role_optics


def resolve_network(
    graph: nx.MultiGraph,
    scenario: dict[str, Any],
    optics: dict[str, str],
) -> Network:
    """Expand once, then resolve capacities and optics for each site edge.

    The returned device network, emitted DSL, and site graph receive the same
    capacities and hardware. Internal blueprint links keep their own budgets.
    Node rules and actual scenario blueprints participate in this expansion.
    """
    network = expand_network_dsl(scenario)
    links_by_edge: dict[str, list[Link]] = defaultdict(list)
    for link in network.links.values():
        edge_id = link.attrs.get("site_edge")
        if edge_id is not None:
            links_by_edge[edge_id].append(link)
    rules = {rule["attrs"]["site_edge"]: rule for rule in scenario["network"]["links"]}
    optics_by_pair = role_optics(optics)

    for source, target, key, data in graph.edges(keys=True, data=True):
        edge_id = site_edge_id(source, target, key)
        links = links_by_edge[edge_id]
        if not links:
            raise ValueError(
                f"Adjacency expansion produced zero links for {source}↔{target}"
            )
        base = float(data["base_capacity"])
        if not math.isfinite(base) or base <= 0:
            raise ValueError(f"Edge {source}-{target} has invalid base_capacity={base}")
        capacity = base / len(links)
        data["capacity"] = rules[edge_id]["capacity"] = capacity
        for link in links:
            link.capacity = capacity

        if not optics:
            continue
        hardware = {}
        for endpoint, local, remote in (
            ("source", "source", "target"),
            ("target", "target", "source"),
        ):
            pairs = {
                (
                    network.nodes[getattr(link, local)].attrs["role"],
                    network.nodes[getattr(link, remote)].attrs["role"],
                )
                for link in links
            }
            missing = pairs - optics_by_pair.keys()
            if missing:
                raise ValueError(
                    f"Adjacency '{edge_id}' has no {endpoint}-end optic mapping for {sorted(missing)}"
                )
            components = {optics_by_pair[pair] for pair in pairs}
            if len(components) != 1:
                raise ValueError(
                    f"Adjacency '{edge_id}' requires different optics on its {endpoint} ends; select one role pair per adjacency"
                )
            component = next(iter(components))
            spec = scenario["components"][component]
            optic_capacity = float(spec["capacity"])
            if not math.isfinite(optic_capacity) or optic_capacity <= 0:
                raise ValueError(f"Optic '{component}' must have positive capacity")
            hardware[endpoint] = {
                "component": component,
                "count": float(math.ceil(capacity / optic_capacity)),
            }
        data["hardware"] = deepcopy(hardware)
        rules[edge_id]["attrs"]["hardware"] = deepcopy(hardware)
        for link in links:
            link.attrs["hardware"] = deepcopy(hardware)
    return network
