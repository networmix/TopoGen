"""Exact local equal splitting and site-link budgets on the metro graph."""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

import networkx as nx

from topogen.config import TopologyConfig
from topogen.naming import site_edge_id
from topogen.numeric import decimal

from .records import Sizing, TrafficMatrix


def _split_loads(
    graph: nx.MultiDiGraph, demands: tuple[tuple[str, str, Fraction], ...]
) -> dict[tuple[str, str, str], Fraction]:
    """Split each inflow over individual outgoing shortest-route links."""
    loads = {edge: Fraction(0) for edge in graph.edges(keys=True)}
    distances = {}
    for source, target, volume in demands:
        if volume == 0 or source == target:
            continue
        if target not in distances:
            distances[target] = nx.single_source_dijkstra_path_length(
                graph.reverse(copy=False), target, weight="cost"
            )
        distance = distances[target]
        if source not in distance:
            raise ValueError(
                f"TM sizing: no corridor route for metro {source}->{target}"
            )
        inflow: dict[str, Fraction] = defaultdict(Fraction)
        inflow[source] = volume
        for node in sorted(distance, key=distance.__getitem__, reverse=True):
            if node == target or not inflow[node]:
                continue
            outgoing = [
                (node, neighbor, key)
                for _, neighbor, key, data in graph.out_edges(
                    node, keys=True, data=True
                )
                if neighbor in distance
                and data["cost"] + distance[neighbor] == distance[node]
            ]
            share = inflow[node] / len(outgoing)
            for edge in outgoing:
                loads[edge] += share
                inflow[edge[1]] += share
    return loads


def size_links(
    plan: nx.MultiGraph, traffic: tuple[TrafficMatrix, ...], config: TopologyConfig
) -> Sizing:
    """Return new exact budgets without changing the plan or traffic."""
    capacity = {
        site_edge_id(u, v, key): decimal(data["base_capacity"])
        for u, v, key, data in plan.edges(keys=True, data=True)
    }
    cfg = config.build.tm_sizing
    if not cfg.enabled:
        return Sizing({}, capacity)
    if not config.traffic.enabled:
        raise ValueError(
            "TM sizing is enabled but traffic generation is disabled in configuration"
        )
    if not any(data["site_kind"] == "dc" for _, data in plan.nodes(data=True)):
        raise ValueError("TM sizing is enabled but no DC regions are configured")
    name = cfg.matrix_name or config.traffic.matrix_name
    matrices = {matrix.name: matrix for matrix in traffic}
    if name not in matrices:
        raise ValueError(
            f"TM sizing: unknown traffic matrix '{name}'; available: {sorted(matrices)}"
        )
    if not matrices[name].demands:
        raise ValueError(
            f"TM sizing: traffic matrix '{name}' is empty despite enabled traffic"
        )
    metro_names = {data["metro_name"] for _, data in plan.nodes(data=True)}
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(metro_names)
    corridors = []
    for u, v, key, data in plan.edges(keys=True, data=True):
        if data["link_type"] != "inter_metro_corridor":
            continue
        source, target = data.get("source_metro"), data.get("target_metro")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(
                "TM sizing: inter-metro edge missing source_metro/target_metro attributes"
            )
        if source not in metro_names or target not in metro_names:
            raise ValueError(
                f"TM sizing: unknown metro name(s) on inter-metro edge: {source!r}, {target!r}"
            )
        cost = data["cost"]
        if isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0:
            raise ValueError(
                f"TM sizing: corridor {u}->{v} requires positive integer cost"
            )
        edge = site_edge_id(u, v, key)
        graph.add_edge(source, target, key=edge, cost=cost)
        graph.add_edge(target, source, key=edge, cost=cost)
        corridors.append((u, v, edge, source, target))
    if not corridors:
        raise ValueError(
            "TM sizing: no inter-metro corridor edges present in site graph"
        )
    demands = []
    for demand in matrices[name].demands:
        try:
            volume = decimal(demand.volume)
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                "TM sizing: demand volume must be finite non-negative"
            ) from exc
        if volume < 0:
            raise ValueError("TM sizing: demand volume must be finite non-negative")
        if demand.source not in plan or demand.target not in plan:
            raise ValueError(
                f"TM sizing: unsupported demand endpoint ({demand.source!r} -> {demand.target!r})"
            )
        demands.append(
            (
                plan.nodes[demand.source]["metro_name"],
                plan.nodes[demand.target]["metro_name"],
                volume,
            )
        )
    directed = _split_loads(graph, tuple(demands))
    quantum, headroom = decimal(cfg.quantum_gbps), decimal(cfg.headroom)

    def sized(edge: str, target: Fraction) -> Fraction:
        result = quantum * math.ceil(target / quantum) if quantum else target
        return max(capacity[edge], result) if cfg.respect_min_base_capacity else result

    peak = {}
    egress: dict[str, Fraction] = defaultdict(Fraction)
    for u, v, edge, source, target in corridors:
        peak[edge] = max(directed[source, target, edge], directed[target, source, edge])
        if peak[edge]:
            capacity[edge] = sized(edge, headroom * peak[edge])
        egress[u] += capacity[edge]
        egress[v] += capacity[edge]
    for u, v, key, data in plan.edges(keys=True, data=True):
        edge = site_edge_id(u, v, key)
        if data["link_type"] == "dc_to_pop":
            pop = v if plan.nodes[v]["site_kind"] == "pop" else u
            if plan.nodes[pop]["site_kind"] != "pop":
                raise ValueError(
                    f"TM sizing: DC->PoP edge does not connect to a PoP endpoint: {u}<->{v}"
                )
            capacity[edge] = sized(edge, decimal(cfg.alpha_dc_to_pop) * egress[pop])
        elif data["link_type"] == "intra_metro":
            capacity[edge] = sized(
                edge, decimal(cfg.beta_intra_pop) * min(egress[u], egress[v])
            )
    return Sizing(peak, capacity)
