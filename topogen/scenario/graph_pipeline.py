"""Build a site-level MultiGraph with explicit adjacency and stripe contracts."""

from __future__ import annotations

import itertools
import math
from typing import Any

import networkx as nx

from topogen.log_config import get_logger
from topogen.roles import role_pair

from .striping import StripePlanner

logger = get_logger(__name__)


def _metro_index_maps(
    metros: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[Any, dict[str, Any]]]:
    return (
        {metro["name"]: idx for idx, metro in enumerate(metros, 1)},
        {metro["node_key"]: metro for metro in metros},
    )


def _site_node_id(metro_idx: int, kind: str, ordinal: int) -> str:
    return f"metro{metro_idx}/{kind}{ordinal}"


def _assign_site_positions(
    G: nx.MultiGraph, metros: list[dict[str, Any]], metro_idx_map: dict[str, int]
) -> None:
    """Position PoPs/DCs on separate rings in one pass over the site inventory."""
    sites: dict[tuple[int, str], list[str]] = {}
    for node, attrs in G.nodes(data=True):
        sites.setdefault((attrs["metro_idx"], attrs["site_kind"]), []).append(node)
    for metro in metros:
        idx = metro_idx_map[metro["name"]]
        radius = 1000.0 * metro["radius_km"]
        for kind, fraction, phase in (("pop", 0.7, 0.0), ("dc", 0.4, math.pi / 4)):
            nodes = sorted(
                sites.get((idx, kind), []),
                key=lambda name: G.nodes[name]["site_ordinal"],
            )
            for ordinal, name in enumerate(nodes):
                theta = phase + math.tau * ordinal / len(nodes)
                G.nodes[name].update(
                    pos_x=metro["x"] + fraction * radius * math.cos(theta),
                    pos_y=metro["y"] + fraction * radius * math.sin(theta),
                    center_x=metro["x"],
                    center_y=metro["y"],
                    radius_m=radius,
                )


def _role_match(link: dict[str, Any]) -> dict[str, Any]:
    roles = sorted(
        {role for pair in link.get("role_pairs", []) for role in role_pair(pair)}
    )
    return (
        {
            "conditions": [
                {"attr": "role", "op": "==", "value": role} for role in roles
            ],
            "logic": "or",
        }
        if roles
        else link.get("match", {})
    )


def _ring_cost(count: int, source: int, target: int, radius: float) -> int:
    delta = abs(source - target) % count
    return math.ceil(min(delta, count - delta) * math.tau * radius / count)


def _add_site_edge(
    G: nx.MultiGraph,
    source: str,
    target: str,
    key: str,
    link: dict[str, Any],
    kind: str,
    cost: int,
    adjacency_id: str,
    source_metro: str,
    target_metro: str,
    match: dict[str, Any],
    **metadata: Any,
) -> None:
    capacity = link["capacity"]
    G.add_edge(
        source,
        target,
        key=key,
        attrs=link.get("attrs", {}),
        link_type=kind,
        base_capacity=capacity,
        target_capacity=capacity,
        cost=max(cost, link["cost"]),
        adjacency_id=adjacency_id,
        distance_km=cost,
        source_metro=source_metro,
        target_metro=target_metro,
        match=match,
        role_pairs=link.get("role_pairs", []),
        **metadata,
    )


def _stripe_match(attribute: str, label: str) -> dict[str, Any]:
    return {"conditions": [{"attr": attribute, "op": "==", "value": label}]}


def _add_intra_metro_edges(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    metro_idx_map: dict[str, int],
) -> None:
    for metro in metros:
        name = metro["name"]
        idx = metro_idx_map[name]
        settings = metro_settings[name]
        link = settings["intra_metro_link"]
        count = settings["pop_per_metro"]
        match = _role_match(link)
        adjacency = f"intra_metro:{name}"
        for a, b in itertools.combinations(range(1, count + 1), 2):
            _add_site_edge(
                G,
                _site_node_id(idx, "pop", a),
                _site_node_id(idx, "pop", b),
                f"{adjacency}:{a}-{b}",
                link,
                "intra_metro",
                _ring_cost(count, a, b, metro["radius_km"]),
                adjacency,
                name,
                name,
                match,
            )


def _add_dc_to_pop_edges(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    metro_idx_map: dict[str, int],
    stripes: StripePlanner,
) -> None:
    for metro in metros:
        name = metro["name"]
        idx = metro_idx_map[name]
        settings = metro_settings[name]
        pops, dcs = settings["pop_per_metro"], settings["dc_regions_per_metro"]
        if not dcs:
            continue
        link = settings["dc_to_pop_link"]
        match = _role_match(link)
        adjacency = f"dc_to_pop:{name}"
        if striping := link.get("striping"):
            pop_groups = stripes.groups(settings["site_blueprint"], striping, match)
            dc_groups = stripes.groups(settings["dc_region_blueprint"], striping, match)
            asymmetric = (
                striping.get("mode", "width") == "width" and striping["width"] == 1
            )
            if not asymmetric and pop_groups.keys() != dc_groups.keys():
                raise ValueError(
                    f"DC-to-PoP striping mismatch in {name}: pop={list(pop_groups)}, dc={list(dc_groups)}"
                )
            attribute = f"stripe_dc_{idx}"
            for kind, count, groups in (
                ("pop", pops, pop_groups),
                ("dc", dcs, dc_groups),
            ):
                for ordinal in range(1, count + 1):
                    stripes.attach(_site_node_id(idx, kind, ordinal), attribute, groups)
            match = _stripe_match(attribute, next(iter(pop_groups)))
        for dc in range(1, dcs + 1):
            fraction = (dc - 1) * pops / dcs
            candidates = {
                math.floor(fraction) % pops + 1,
                math.ceil(fraction) % pops + 1,
            }
            for pop in range(1, pops + 1):
                cost = max(
                    1,
                    min(
                        _ring_cost(pops, pop, candidate, metro["radius_km"])
                        for candidate in candidates
                    ),
                )
                _add_site_edge(
                    G,
                    _site_node_id(idx, "dc", dc),
                    _site_node_id(idx, "pop", pop),
                    f"{adjacency}:{dc}-{pop}",
                    link,
                    "dc_to_pop",
                    cost,
                    adjacency,
                    name,
                    name,
                    match,
                )


def _add_inter_metro_edges(
    G: nx.MultiGraph,
    metro_settings: dict[str, dict[str, Any]],
    graph: nx.MultiGraph,
    metro_idx_map: dict[str, int],
    metro_by_node: dict[Any, dict[str, Any]],
    stripes: StripePlanner,
) -> None:
    for u, v, path_index, edge in graph.edges(keys=True, data=True):
        source, target = metro_by_node[u], metro_by_node[v]
        a, b = source["name"], target["name"]
        ai, bi = metro_idx_map[a], metro_idx_map[b]
        ac, bc = metro_settings[a], metro_settings[b]
        # The first metro in the graph's node order supplies the corridor policy.
        link = ac["inter_metro_link"]
        match = _role_match(link)
        ap, bp = ac["pop_per_metro"], bc["pop_per_metro"]
        mode = link.get("mode", "mesh")
        if mode == "one_to_one":
            pairs = [(p, p) for p in range(1, min(ap, bp) + 1)]
        elif mode == "mesh":
            pairs = list(itertools.product(range(1, ap + 1), range(1, bp + 1)))
        else:
            raise ValueError(f"Unknown inter-metro mode: {mode}")
        cost = math.ceil(edge["length_km"])
        adjacency = f"inter_metro:{min(ai, bi)}-{max(ai, bi)}:path{path_index}"
        labels: list[str] = []
        attribute = f"stripe_im_{min(ai, bi)}_{max(ai, bi)}"
        if striping := link.get("striping"):
            left = stripes.groups(ac["site_blueprint"], striping, match)
            right = stripes.groups(bc["site_blueprint"], striping, match)
            if left.keys() != right.keys():
                raise ValueError(
                    f"Inter-metro striping mismatch: {a}={list(left)}, {b}={list(right)}"
                )
            labels = list(left)
            for idx, count, groups in ((ai, ap, left), (bi, bp, right)):
                for ordinal in range(1, count + 1):
                    stripes.attach(
                        _site_node_id(idx, "pop", ordinal), attribute, groups
                    )
        for ordinal, (p, q) in enumerate(pairs):
            label = labels[ordinal % len(labels)] if labels else None
            _add_site_edge(
                G,
                _site_node_id(ai, "pop", p),
                _site_node_id(bi, "pop", q),
                f"{adjacency}:{p}-{q}",
                link,
                "inter_metro_corridor",
                cost,
                f"{adjacency}:{label}" if label is not None else adjacency,
                a,
                b,
                _stripe_match(attribute, label) if label is not None else match,
                risk_groups=list(edge.get("risk_groups", [])),
                euclidean_km=edge.get("euclidean_km"),
                detour_ratio=edge.get("detour_ratio"),
            )


def build_site_graph(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    integrated_graph: nx.MultiGraph,
    blueprints: dict[str, Any],
) -> nx.MultiGraph:
    """Construct a site-level MultiGraph with nodes and edges per adjacency.

    Nodes are created for sites: ``metro{n}/pop{i}`` and ``metro{n}/dc{j}`` with
    their blueprints attached as node attributes. Edges are added for three
    adjacency families:

    - intra_metro: full mesh of PoPs within a metro, with shortest ring-arc costs.
    - dc_to_pop: DC region↔PoP within a metro, optionally striped.
    - inter_metro_corridor: PoP↔PoP across metros along discovered corridors,
      optionally striped.

    Each edge carries ``link_type``, ``cost``, ``base_capacity`` (total intended
    budget prior to per-link split), ``target_capacity`` (kept equal to base),
    and metadata for downstream sizing and visualization.
    """
    logger.info("Building site-level MultiGraph")
    G = nx.MultiGraph()
    metro_idx_map, metro_by_node = _metro_index_maps(metros)

    for metro in metros:
        name = metro["name"]
        s = metro_settings[name]["pop_per_metro"]
        d = metro_settings[name]["dc_regions_per_metro"]
        if (s > 1 or d > 0) and float(metro["radius_km"]) <= 0.0:
            logger.error(
                "Metro %s requires positive radius_km for ring-based adjacency (pop_per_metro=%d, dc_regions=%d)",
                name,
                s,
                d,
            )
            raise ValueError(
                f"Metro '{name}' has radius_km={metro.get('radius_km', 0.0)}; expected > 0 for ring-based adjacency"
            )

    for metro in metros:
        name = metro["name"]
        idx = metro_idx_map[name]
        s = metro_settings[name]["pop_per_metro"]
        d = metro_settings[name]["dc_regions_per_metro"]
        pop_blueprint = metro_settings[name]["site_blueprint"]
        dc_blueprint = metro_settings[name]["dc_region_blueprint"]
        for p in range(1, s + 1):
            node_id = _site_node_id(idx, "pop", p)
            G.add_node(
                node_id,
                metro_idx=idx,
                metro_name=name,
                site_kind="pop",
                site_ordinal=p,
                site_blueprint=pop_blueprint,
            )
        for j in range(1, d + 1):
            node_id = _site_node_id(idx, "dc", j)
            G.add_node(
                node_id,
                metro_idx=idx,
                metro_name=name,
                site_kind="dc",
                site_ordinal=j,
                site_blueprint=dc_blueprint,
            )

    _assign_site_positions(G, metros, metro_idx_map)

    stripes = StripePlanner(blueprints)
    _add_intra_metro_edges(G, metros, metro_settings, metro_idx_map)
    _add_dc_to_pop_edges(G, metros, metro_settings, metro_idx_map, stripes)
    _add_inter_metro_edges(
        G,
        metro_settings,
        integrated_graph,
        metro_idx_map,
        metro_by_node,
        stripes,
    )
    if rules := stripes.node_rules():
        G.graph["node_overrides"] = rules

    logger.info(
        "MultiGraph built: %d nodes, %d edges (multi-edges counted)",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G
