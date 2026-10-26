"""Exact local equal splitting and site-link budgets on the metro graph."""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

import networkx as nx

from topogen.config import TopologyConfig
from topogen.numeric import decimal

from .records import SitePlan, Sizing, TrafficMatrix


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
    plan: SitePlan, traffic: tuple[TrafficMatrix, ...], config: TopologyConfig
) -> Sizing:
    """Return exact budgets without changing any stage's records."""
    capacity = {link.id: decimal(link.base_capacity) for link in plan.links}
    cfg = config.build.tm_sizing
    if not cfg.enabled:
        return Sizing({}, capacity)
    if not config.traffic.enabled:
        raise ValueError(
            "TM sizing is enabled but traffic generation is disabled in configuration"
        )
    if not any(site.kind == "dc" for site in plan.sites):
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
    sites = {site.id: site for site in plan.sites}
    graph = nx.MultiDiGraph()
    graph.add_nodes_from({site.metro.name for site in plan.sites})
    corridors = []
    for link in plan.links:
        if link.kind != "inter_metro_corridor":
            continue
        if link.source not in sites or link.target not in sites:
            raise ValueError(
                f"TM sizing: unsupported site-link endpoint ({link.source!r} -> {link.target!r})"
            )
        source, target = sites[link.source].metro.name, sites[link.target].metro.name
        if (
            isinstance(link.cost, bool)
            or not isinstance(link.cost, int)
            or link.cost <= 0
        ):
            raise ValueError(
                f"TM sizing: corridor {link.source}->{link.target} requires positive integer cost"
            )
        graph.add_edge(source, target, key=link.id, cost=link.cost)
        graph.add_edge(target, source, key=link.id, cost=link.cost)
        corridors.append((link, source, target))
    if not corridors:
        raise ValueError(
            "TM sizing: no inter-metro corridor links present in site plan"
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
        if demand.source not in sites or demand.target not in sites:
            raise ValueError(
                f"TM sizing: unsupported demand endpoint ({demand.source!r} -> {demand.target!r})"
            )
        demands.append(
            (sites[demand.source].metro.name, sites[demand.target].metro.name, volume)
        )
    directed = _split_loads(graph, tuple(demands))
    quantum, headroom = decimal(cfg.quantum_gbps), decimal(cfg.headroom)

    def sized(edge: str, target: Fraction) -> Fraction:
        result = quantum * math.ceil(target / quantum) if quantum else target
        return max(capacity[edge], result) if cfg.respect_min_base_capacity else result

    peak = {}
    egress: dict[str, Fraction] = defaultdict(Fraction)
    for link, source, target in corridors:
        peak[link.id] = max(
            directed[source, target, link.id], directed[target, source, link.id]
        )
        if peak[link.id]:
            capacity[link.id] = sized(link.id, headroom * peak[link.id])
        egress[link.source] += capacity[link.id]
        egress[link.target] += capacity[link.id]
    for link in plan.links:
        if link.kind == "dc_to_pop":
            pop = link.target if sites[link.target].kind == "pop" else link.source
            if sites[pop].kind != "pop":
                raise ValueError(
                    f"TM sizing: DC->PoP edge does not connect to a PoP endpoint: {link.source}<->{link.target}"
                )
            capacity[link.id] = sized(
                link.id, decimal(cfg.alpha_dc_to_pop) * egress[pop]
            )
        elif link.kind == "intra_metro":
            capacity[link.id] = sized(
                link.id,
                decimal(cfg.beta_intra_pop)
                * min(egress[link.source], egress[link.target]),
            )
    return Sizing(peak, capacity)
