"""Graph-based scenario generation pipeline.

Builds a site-level MultiGraph, assigns capacities, and serializes to YAML.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
from ngraph.algorithms.base import EdgeSelect as _NgEdgeSelect
from ngraph.algorithms.base import FlowPlacement as _NgFlowPlacement
from ngraph.algorithms.flow_init import init_flow_graph as _ng_init_flow_graph
from ngraph.algorithms.placement import (
    place_flow_on_graph as _ng_place_flow,
)
from ngraph.algorithms.placement import (
    remove_flow_from_graph as _ng_remove_flow,
)
from ngraph.algorithms.spf import spf as _ng_spf
from ngraph.dsl.blueprints.expand import expand_network_dsl as _ng_expand
from ngraph.graph.strict_multidigraph import StrictMultiDiGraph as _NgSMDG

from topogen.blueprints_lib import get_builtin_blueprints as _get_builtins
from topogen.corridors import (
    extract_corridor_edges_for_metros_graph as _extract_corridor_edges,
)
from topogen.log_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


# Per-node component assignments removed from the pipeline.


def _metro_index_maps(
    metros: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[Any, dict[str, Any]]]:
    """Build helper maps for metro indices and lookup by node key."""
    metro_idx_map = {metro["name"]: idx for idx, metro in enumerate(metros, 1)}
    metro_by_node = {metro["node_key"]: metro for metro in metros}
    return metro_idx_map, metro_by_node


def _site_node_id(metro_idx: int, kind: str, ordinal: int) -> str:
    """Return a canonical site node id like 'metro1/pop2' or 'metro3/dc1'."""
    return f"metro{metro_idx}/{kind}{ordinal}"


def _add_intra_metro_edges(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
    metro_idx_map: dict[str, int],
) -> None:
    """Add PoP-to-PoP edges within each metro with ring-arc cost model."""
    logger.info("Forming intra-metro PoP mesh edges for %d metros", len(metros))
    for metro in metros:
        metro_name = metro["name"]
        idx = metro_idx_map[metro_name]
        # Resolve per-metro link config
        link_cfg = metro_settings[metro_name]["intra_metro_link"]
        base_capacity = (
            int(link_cfg["capacity"])
            if isinstance(link_cfg, dict)
            else int(getattr(link_cfg, "capacity", 0))
        )
        base_cost = (
            int(link_cfg["cost"])
            if isinstance(link_cfg, dict)
            else int(getattr(link_cfg, "cost", 0))
        )
        s = int(metro_settings[metro_name]["pop_per_metro"])
        if s <= 1:
            continue
        radius_km = float(metro.get("radius_km", 0.0))
        ring_radius_km = max(0.0, radius_km)

        def arc_ceil(
            n: int,
            i: int,
            j: int,
            *,
            _radius_km: float = ring_radius_km,
            _base_cost: int = base_cost,
        ) -> int:
            if n <= 1 or _radius_km <= 0.0:
                return _base_cost
            delta = abs(i - j) % n
            steps = min(delta, n - delta)
            arc = steps * ((2.0 * math.pi * _radius_km) / float(n))
            return int(math.ceil(arc))

        pairs = list(itertools.combinations(range(1, s + 1), 2))
        adj_id = f"intra_metro:{metro_name}"
        for i, j in pairs:
            u = _site_node_id(idx, "pop", i)
            v = _site_node_id(idx, "pop", j)
            cost = arc_ceil(s, i, j)
            # Build symmetric match if role_pairs provided
            rp = (
                link_cfg.get("role_pairs", [])
                if isinstance(link_cfg, dict)
                else getattr(link_cfg, "role_pairs", [])
            )
            if rp:
                roles: set[str] = set()
                for item in rp:
                    if isinstance(item, str):
                        parts = [p.strip() for p in item.split("|") if p.strip()]
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        parts = [str(item[0]).strip(), str(item[1]).strip()]
                    else:
                        parts = []
                    for r in parts:
                        if r:
                            roles.add(r)
                try:
                    logger.debug(
                        "intra_metro: metro=%s role_pairs=%s -> match.roles=%s",
                        metro_name,
                        rp,
                        sorted(list(roles)),
                    )
                except Exception:
                    pass
                match_obj = {
                    "conditions": [
                        {"attr": "role", "operator": "==", "value": r}
                        for r in sorted(roles)
                    ]
                }

            else:
                match_obj = (
                    link_cfg.get("match", {})
                    if isinstance(link_cfg, dict)
                    else getattr(link_cfg, "match", {})
                ) or {}

            G.add_edge(
                u,
                v,
                key=f"{adj_id}:{i}-{j}",
                link_type="intra_metro",
                base_capacity=base_capacity,
                target_capacity=base_capacity,
                cost=cost,
                adjacency_id=adj_id,
                distance_km=cost,
                source_metro=metro_name,
                target_metro=metro_name,
                match=match_obj,
                role_pairs=rp,
            )


def _add_dc_to_pop_edges(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
    metro_idx_map: dict[str, int],
) -> None:
    """Add DC-to-PoP edges within each metro."""
    logger.info("Forming DC-to-PoP edges where DC regions exist")
    for metro in metros:
        metro_name = metro["name"]
        idx = metro_idx_map[metro_name]
        link_cfg = metro_settings[metro_name]["dc_to_pop_link"]
        base_capacity = (
            int(link_cfg["capacity"])
            if isinstance(link_cfg, dict)
            else int(getattr(link_cfg, "capacity", 0))
        )
        base_cost = (
            int(link_cfg["cost"])
            if isinstance(link_cfg, dict)
            else int(getattr(link_cfg, "cost", 0))
        )
        s = int(metro_settings[metro_name]["pop_per_metro"])
        d = int(metro_settings[metro_name]["dc_regions_per_metro"])  # may be 0
        if s <= 0 or d <= 0:
            continue
        adj_id = f"dc_to_pop:{metro_name}"
        radius_km = float(metro.get("radius_km", 0.0))
        ring_radius_km = max(0.0, radius_km)

        def arc_ceil(
            n: int,
            i: int,
            j: int,
            *,
            _radius_km: float = ring_radius_km,
            _base_cost: int = base_cost,
        ) -> int:
            if n <= 1 or _radius_km <= 0.0:
                return _base_cost
            delta = abs(i - j) % n
            steps = min(delta, n - delta)
            arc = steps * ((2.0 * math.pi * _radius_km) / float(n))
            return int(math.ceil(arc))

        # Build symmetric match if role_pairs provided
        rp = (
            link_cfg.get("role_pairs", [])
            if isinstance(link_cfg, dict)
            else getattr(link_cfg, "role_pairs", [])
        )
        if rp:
            roles: set[str] = set()
            for item in rp:
                if isinstance(item, str):
                    parts = [p.strip() for p in item.split("|") if p.strip()]
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    parts = [str(item[0]).strip(), str(item[1]).strip()]
                else:
                    parts = []
                for r in parts:
                    if r:
                        roles.add(r)
            try:
                logger.debug(
                    "dc_to_pop: metro=%s role_pairs=%s -> match.roles=%s",
                    metro_name,
                    rp,
                    sorted(list(roles)),
                )
            except Exception:
                pass
            match_obj = {
                "conditions": [
                    {"attr": "role", "operator": "==", "value": r}
                    for r in sorted(roles)
                ],
                "logic": "or",
            }

        else:
            match_obj = (
                link_cfg.get("match", {})
                if isinstance(link_cfg, dict)
                else getattr(link_cfg, "match", {})
            ) or {}

        for dc in range(1, d + 1):
            for p in range(1, s + 1):
                u = _site_node_id(idx, "dc", dc)
                v = _site_node_id(idx, "pop", p)
                # Place DCs evenly around the ring and compute nearest POP-based arc distance
                n = s
                frac_index = (dc - 1) * (n / max(d, 1))
                cand_idxs = {
                    int(math.floor(frac_index)) % n + 1,
                    int(math.ceil(frac_index)) % n + 1,
                }
                raw_arc = (
                    min(arc_ceil(n, p, c) for c in cand_idxs)
                    if ring_radius_km > 0.0
                    else base_cost
                )
                # Enforce strictly positive cost to avoid zero-length links
                cost_arc = max(1, int(raw_arc))

                G.add_edge(
                    u,
                    v,
                    key=f"{adj_id}:{dc}-{p}",
                    link_type="dc_to_pop",
                    base_capacity=base_capacity,
                    target_capacity=base_capacity,
                    cost=cost_arc,
                    adjacency_id=adj_id,
                    distance_km=cost_arc,
                    source_metro=metro_name,
                    target_metro=metro_name,
                    match=match_obj,
                    role_pairs=rp,
                )


def _add_inter_metro_edges(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    graph: nx.Graph,
    config: TopologyConfig,
    metro_idx_map: dict[str, int],
    metro_by_node: dict[Any, dict[str, Any]],
) -> None:
    """Add PoP-to-PoP edges across metro corridors."""
    logger.info("Forming inter-metro corridor edges from integrated graph")
    corridor_edges = _extract_corridor_edges(graph)
    for edge in corridor_edges:
        src = metro_by_node.get(edge["source"])  # type: ignore[index]
        tgt = metro_by_node.get(edge["target"])  # type: ignore[index]
        if not src or not tgt:
            logger.warning("Skipping corridor with unknown metros: %s", edge)
            continue
        s_name = src["name"]
        t_name = tgt["name"]
        s_idx = metro_idx_map[s_name]
        t_idx = metro_idx_map[t_name]
        s_sites = int(metro_settings[s_name]["pop_per_metro"])  # type: ignore[index]
        t_sites = int(metro_settings[t_name]["pop_per_metro"])  # type: ignore[index]
        # Inter-metro precedence: use source metro's link config
        src_cfg = metro_settings[s_name]["inter_metro_link"]
        base_capacity = (
            int(src_cfg["capacity"])
            if isinstance(src_cfg, dict)
            else int(getattr(src_cfg, "capacity", 0))
        )
        # Corridor length km preferred; fallback to source metro's configured cost
        cost = int(
            math.ceil(
                float(
                    edge.get(
                        "length_km",
                        (
                            src_cfg.get("cost", 1)
                            if isinstance(src_cfg, dict)
                            else getattr(src_cfg, "cost", 1)
                        ),
                    )
                )
            )
        )
        risk_groups = list(edge.get("risk_groups", []))
        euclid_km = edge.get("euclidean_km")
        detour_ratio = edge.get("detour_ratio")
        adj_id = f"inter_metro:{min(s_idx, t_idx)}-{max(s_idx, t_idx)}"
        for p in range(1, s_sites + 1):
            for q in range(1, t_sites + 1):
                u = _site_node_id(s_idx, "pop", p)
                v = _site_node_id(t_idx, "pop", q)
                G.add_edge(
                    u,
                    v,
                    key=f"{adj_id}:{p}-{q}",
                    link_type="inter_metro_corridor",
                    base_capacity=base_capacity,
                    target_capacity=base_capacity,
                    cost=cost,
                    adjacency_id=adj_id,
                    distance_km=cost,
                    source_metro=s_name,
                    target_metro=t_name,
                    risk_groups=risk_groups,
                    euclidean_km=euclid_km,
                    detour_ratio=detour_ratio,
                    # Symmetric match applied to both endpoints; build from role_pairs if present
                    match=(
                        {
                            "conditions": [
                                {
                                    "attr": "role",
                                    "operator": "==",
                                    "value": r,
                                }
                                for r in sorted(
                                    {
                                        part.strip()
                                        for item in (
                                            src_cfg.get("role_pairs", [])
                                            if isinstance(src_cfg, dict)
                                            else getattr(src_cfg, "role_pairs", [])
                                        )
                                        for part in (
                                            item.split("|")
                                            if isinstance(item, str)
                                            else [
                                                str(item[0]) if len(item) > 0 else "",
                                                str(item[1]) if len(item) > 1 else "",
                                            ]
                                            if isinstance(item, (list, tuple))
                                            else []
                                        )
                                        if part.strip()
                                    }
                                )
                            ],
                            "logic": "or",
                        }
                        if (
                            src_cfg.get("role_pairs", [])
                            if isinstance(src_cfg, dict)
                            else getattr(src_cfg, "role_pairs", [])
                        )
                        else (
                            src_cfg.get("match", {})
                            if isinstance(src_cfg, dict)
                            else getattr(src_cfg, "match", {})
                        )
                    ),
                    role_pairs=(
                        src_cfg.get("role_pairs", [])
                        if isinstance(src_cfg, dict)
                        else getattr(src_cfg, "role_pairs", [])
                    ),
                )


def build_site_graph(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    integrated_graph: nx.Graph,
    config: TopologyConfig,
) -> nx.MultiGraph:
    """Construct a site-level MultiGraph with nodes and edges per adjacency.

    Nodes: 'metro{n}/pop{i}' and 'metro{n}/dc{j}'.
    Edges carry 'link_type', 'cost', 'base_capacity', and metadata.
    """
    logger.info("Building site-level MultiGraph")
    G = nx.MultiGraph()
    metro_idx_map, metro_by_node = _metro_index_maps(metros)

    # Early validation: ring radius must be positive when ring-based adjacencies are needed
    for metro in metros:
        name = metro["name"]
        s = int(metro_settings[name]["pop_per_metro"])  # type: ignore[index]
        d = int(metro_settings[name]["dc_regions_per_metro"])  # type: ignore[index]
        if (s > 1 or d > 0) and float(metro.get("radius_km", 0.0)) <= 0.0:
            logger.error(
                "Metro %s requires positive radius_km for ring-based adjacency (pop_per_metro=%d, dc_regions=%d)",
                name,
                s,
                d,
            )
            raise ValueError(
                f"Metro '{name}' has radius_km={metro.get('radius_km', 0.0)}; expected > 0 for ring-based adjacency"
            )

    # Nodes with per-site blueprint and component assignments
    for metro in metros:
        name = metro["name"]
        idx = metro_idx_map[name]
        s = int(metro_settings[name]["pop_per_metro"])  # type: ignore[index]
        d = int(metro_settings[name]["dc_regions_per_metro"])  # type: ignore[index]
        pop_blueprint = str(metro_settings[name]["site_blueprint"])  # type: ignore[index]
        dc_blueprint = str(metro_settings[name]["dc_region_blueprint"])  # type: ignore[index]
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

    # Edges by category
    _add_intra_metro_edges(G, metros, metro_settings, config, metro_idx_map)
    _add_dc_to_pop_edges(G, metros, metro_settings, config, metro_idx_map)
    _add_inter_metro_edges(
        G,
        metros,
        metro_settings,
        integrated_graph,
        config,
        metro_idx_map,
        metro_by_node,
    )

    logger.info(
        "MultiGraph built: %d nodes, %d edges (multi-edges counted)",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def assign_per_link_capacity(G: nx.MultiGraph, config: TopologyConfig) -> None:
    """Assign per-link capacity based on final DSL expansion link count.

    The configured ``base_capacity`` on an edge is interpreted as the total
    capacity budget for that site-to-site adjacency. We compute how many
    concrete links the NetGraph DSL will produce for this adjacency (given the
    endpoint site blueprints, match filters, and the ``one_to_one`` pattern),
    then set per-link capacity to ``base_capacity / num_links``.

    This logic is adjacency-agnostic and uses the same expansion rules as
    NetGraph by calling its DSL expander on a minimal network containing only
    the two endpoints and the single adjacency under evaluation.

    Args:
        G: Site-level MultiGraph with edges carrying ``base_capacity`` and node
            attributes ``site_blueprint``.
        config: Topology configuration (unused; present for interface stability).

    Raises:
        ValueError: If an edge lacks ``base_capacity`` or the expansion yields
            zero links.
    """
    logger.info("Assigning per-link capacities by splitting base over expansion size")

    builtins = _get_builtins()

    def _estimate_link_count(u_id: str, v_id: str, edge_data: dict[str, Any]) -> int:
        bp_u = str(G.nodes[u_id].get("site_blueprint", ""))
        bp_v = str(G.nodes[v_id].get("site_blueprint", ""))
        if not bp_u or not bp_v:
            raise ValueError(f"Missing site_blueprint for nodes {u_id} or {v_id}")

        # Build a minimal DSL using only the two endpoints and one adjacency.
        # Tag links from our adjacency so we can count them precisely.
        # Ensure match is a plain dict; tests may inject mocks
        raw_match = edge_data.get("match", {})
        match_dict: dict[str, Any] = raw_match if isinstance(raw_match, dict) else {}

        dsl = {
            "blueprints": builtins,
            "network": {
                "groups": {
                    u_id: {"use_blueprint": bp_u},
                    v_id: {"use_blueprint": bp_v},
                },
                "adjacency": [
                    {
                        "source": {"path": u_id, "match": match_dict},
                        "target": {"path": v_id, "match": match_dict},
                        "pattern": "one_to_one",
                        "link_params": {
                            "capacity": 1.0,
                            "cost": 1.0,
                            "attrs": {"_tg_tmp": "count_me"},
                        },
                    }
                ],
            },
        }
        net = _ng_expand(dsl)
        # Count only links created by our adjacency (tagged with _tg_tmp)
        count = sum(
            1
            for _lid, link in net.links.items()
            if link.attrs.get("_tg_tmp") == "count_me"
        )
        return int(count)

    for u, v, k, data in G.edges(keys=True, data=True):
        base_val = data.get("base_capacity")
        if base_val is None:
            raise ValueError(f"Edge {u}-{v} missing base_capacity")
        try:
            base_capacity = float(base_val)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Edge {u}-{v} has non-numeric base_capacity: {base_val!r}"
            ) from exc
        if base_capacity <= 0:
            raise ValueError(
                f"Edge {u}-{v} (type={data.get('link_type')}) has invalid base_capacity={base_capacity}"
            )

        num_links = _estimate_link_count(u, v, data)
        if num_links <= 0:
            raise ValueError(
                f"Adjacency expansion produced zero links for {u}↔{v} (pattern=one_to_one, match={data.get('match')})"
            )

        per_link = base_capacity / float(num_links)
        G.edges[u, v, k]["capacity"] = per_link


def resolve_and_assign_link_hardware(G: nx.MultiGraph, config: TopologyConfig) -> None:
    """No-op: link hardware is assigned during YAML emission."""
    return None


def _parse_tm_endpoint_to_metro_idx(endpoint: str) -> int | None:
    """Extract metro index from a traffic matrix endpoint regex/path.

    Accepts strings like '^metro3/dc2/.*' or 'metro3/dc2'. Returns 3.
    """
    try:
        s = endpoint.lstrip("^")
        if not s.startswith("metro"):
            return None
        rest = s[5:]
        digits: list[str] = []
        for ch in rest:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        return int("".join(digits)) if digits else None
    except Exception:
        return None


def _resolve_ngraph_enums(
    flow_placement: str, edge_select: str
) -> tuple[_NgFlowPlacement, _NgEdgeSelect]:
    """Map config strings to ngraph enums with safe defaults."""
    fp = (
        _NgFlowPlacement.EQUAL_BALANCED
        if str(flow_placement).upper() == "EQUAL_BALANCED"
        else _NgFlowPlacement.PROPORTIONAL
    )
    es = _NgEdgeSelect.ALL_MIN_COST
    es_in = str(edge_select).upper()
    if es_in == "ALL_MIN_COST":
        es = _NgEdgeSelect.ALL_MIN_COST
    elif es_in == "ALL_MIN_COST_WITH_CAP_REMAINING":
        es = _NgEdgeSelect.ALL_MIN_COST_WITH_CAP_REMAINING
    return fp, es


def tm_based_size_capacities(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
) -> None:
    """Adjust base capacities using an early TM and ECMP on collapsed graph.

    Pipeline:
    - Generate TM using traffic_matrix.generate_traffic_matrix (in-memory).
    - Build a metro-level StrictMultiDiGraph with parallel edges per corridor
      (one per PoP-to-PoP corridor edge in G), cost from G, unit capacity.
    - For each directed TM demand in the matrix, compute shortest-path ECMP
      fractions and accumulate load on inter-metro corridor edges only.
    - Quantize inter-metro base capacities with headroom.
    - Derive DC->PoP and intra-metro PoP<->PoP base capacities from metro/PoP
      egress with configurable multipliers and quantization.
    """
    sizing_cfg = getattr(getattr(config, "build", None), "tm_sizing", None)
    if getattr(sizing_cfg, "enabled", False) is not True:
        return

    from topogen.traffic_matrix import generate_traffic_matrix

    logger.info("TM sizing: generating early traffic matrix")

    # Pre-flight checks
    traffic_cfg = getattr(config, "traffic", None)
    if not getattr(traffic_cfg, "enabled", False):
        raise ValueError(
            "TM sizing is enabled but traffic generation is disabled in configuration"
        )

    # Generate TM (in-memory)
    tm_map = generate_traffic_matrix(metros, metro_settings, config)
    matrix_name = getattr(sizing_cfg, "matrix_name", None) or getattr(
        getattr(config, "traffic", object()), "matrix_name", "default"
    )
    demands = list(tm_map.get(matrix_name, []))
    # Ensure DC inventory exists when TM sizing is requested
    total_dc = 0
    try:
        for settings in metro_settings.values():
            total_dc += int(settings.get("dc_regions_per_metro", 0))
    except Exception as exc:
        raise ValueError("TM sizing: failed to determine DC region count") from exc
    if total_dc <= 0:
        raise ValueError(
            "TM sizing is enabled but no DC regions are configured (dc_regions_per_metro == 0)"
        )
    if not demands:
        raise ValueError(
            f"TM sizing: traffic matrix '{matrix_name}' is empty despite enabled traffic"
        )

    # Build metro-level graph H and reference mapping to G edges
    metro_idx_map = {m["name"]: idx for idx, m in enumerate(metros, 1)}
    # Allow reverse map from 'metro{idx}' string
    for idx, _ in enumerate(metros, 1):
        metro_idx_map.setdefault(f"metro{idx}", idx)

    H = _NgSMDG()
    for idx in set(metro_idx_map.values()):
        if idx not in H:
            # Pre-initialize node flow attributes expected by placement
            H.add_node(idx, name=f"metro{idx}", flow_tm=0.0, flows_tm={})

    # Map H edge id -> (g_u, g_v, g_k) for back-mapping loads to G
    h_edge_to_g_edge: dict[str, tuple[str, str, str]] = {}

    # Add directed corridor edges for each inter-metro edge in G
    for u_g, v_g, k_g, data in G.edges(keys=True, data=True):
        if str(data.get("link_type")) != "inter_metro_corridor":
            continue
        src_name = data.get("source_metro")
        tgt_name = data.get("target_metro")
        if not isinstance(src_name, str) or not isinstance(tgt_name, str):
            raise ValueError(
                "TM sizing: inter-metro edge missing source_metro/target_metro attributes"
            )

        s_idx = metro_idx_map.get(src_name)
        t_idx = metro_idx_map.get(tgt_name)
        if s_idx is None or t_idx is None:
            raise ValueError(
                f"TM sizing: unknown metro name(s) on inter-metro edge: {src_name!r}, {tgt_name!r}"
            )

        cost = int(data.get("cost", 1))
        # Forward
        ekey_fwd = H.add_edge(
            s_idx,
            t_idx,
            key=None,
            cost=cost,
            capacity_tm=1e18,
            flow_tm=0.0,
            flows_tm={},
        )
        h_edge_to_g_edge[str(ekey_fwd)] = (str(u_g), str(v_g), str(k_g))
        # Reverse
        ekey_rev = H.add_edge(
            t_idx,
            s_idx,
            key=None,
            cost=cost,
            capacity_tm=1e18,
            flow_tm=0.0,
            flows_tm={},
        )
        h_edge_to_g_edge[str(ekey_rev)] = (str(v_g), str(u_g), str(k_g))

    if H.number_of_edges() == 0:
        raise ValueError(
            "TM sizing: no inter-metro corridor edges present in site graph"
        )

    # Ensure flow attributes exist on nodes and edges for custom attr names
    _ng_init_flow_graph(
        H, flow_attr="flow_tm", flows_attr="flows_tm", reset_flow_graph=True
    )

    fp_enum, es_enum = _resolve_ngraph_enums(
        getattr(sizing_cfg, "flow_placement", "EQUAL_BALANCED"),
        getattr(sizing_cfg, "edge_select", "ALL_MIN_COST"),
    )

    # Accumulate loads per G edge key (u, v, k) in Gbps
    edge_loads: dict[tuple[str, str, str], float] = {}

    # Iterate demands
    for d in demands:
        src = str(d.get("source_path", ""))
        dst = str(d.get("sink_path", ""))
        demand_val = float(d.get("demand", 0.0))
        if demand_val <= 0.0:
            continue
        s_idx = _parse_tm_endpoint_to_metro_idx(src)
        t_idx = _parse_tm_endpoint_to_metro_idx(dst)
        if s_idx is None or t_idx is None or s_idx == t_idx:
            # Skip intra-metro or unparsable entries for corridor sizing
            continue
        if s_idx not in H or t_idx not in H:
            raise ValueError(
                f"TM sizing: metro index not in graph (src={s_idx}, dst={t_idx})"
            )

        # SPF with early-exit toward destination
        try:
            costs, pred = _ng_spf(
                H, s_idx, edge_select=es_enum, multipath=True, dst_node=t_idx
            )
        except KeyError as exc:
            raise ValueError(
                f"TM sizing: SPF failed for metro {s_idx}->{t_idx}: {exc}"
            ) from exc
        if t_idx not in pred:
            raise ValueError(f"TM sizing: no path between metros {s_idx} and {t_idx}")

        # Place actual flow onto edges (equal split across parallel edges per adjacency)
        _ng_place_flow(
            H,
            s_idx,
            t_idx,
            pred,
            flow=demand_val,
            flow_index=None,
            flow_placement=fp_enum,
            capacity_attr="capacity_tm",
            flow_attr="flow_tm",
            flows_attr="flows_tm",
        )
        try:
            min_cost = float(costs.get(t_idx, 0.0))
            logger.debug(
                "TM sizing: placed %s Gbps from metro%d->metro%d (min_cost=%s)",
                f"{demand_val:,.1f}",
                s_idx,
                t_idx,
                f"{int(min_cost):,}",
            )
        except Exception:
            pass
        # Accumulate load per underlying G edge
        for e_id, (_u, _v, _key, attrs) in H.get_edges().items():
            f = float(attrs.get("flow_tm", 0.0))
            if f <= 0.0:
                continue
            ref = h_edge_to_g_edge.get(str(e_id))
            if ref is None:
                continue
            edge_loads[ref] = edge_loads.get(ref, 0.0) + f
        # Clear placed flow to avoid capacity coupling across demands
        _ng_remove_flow(H, flow_attr="flow_tm", flows_attr="flows_tm")

    Q = float(getattr(sizing_cfg, "quantum_gbps", 3200.0))
    h_factor = float(getattr(sizing_cfg, "headroom", 1.3))
    respect_min = bool(getattr(sizing_cfg, "respect_min_base_capacity", True))

    # Snapshot pre-adjustment aggregated capacities per directed metro corridor
    prev_corridor_caps: dict[tuple[str, str], float] = {}
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        if str(data.get("link_type")) != "inter_metro_corridor":
            continue
        src_name = data.get("source_metro")
        tgt_name = data.get("target_metro")
        if not isinstance(src_name, str) or not isinstance(tgt_name, str):
            continue
        prev_cap = float(data.get("base_capacity", data.get("capacity", 0.0)))
        key = (src_name, tgt_name)
        prev_corridor_caps[key] = prev_corridor_caps.get(key, 0.0) + prev_cap

    # Apply sized capacities to inter-metro edges in G
    for (u_g, v_g, k_g), L in edge_loads.items():
        try:
            data = G.edges[(u_g, v_g, k_g)]
        except Exception:
            try:
                data = G.get_edge_data(u_g, v_g, k_g) or {}
            except Exception:
                continue
        if str(data.get("link_type")) != "inter_metro_corridor":
            continue
        # c_e = Q * ceil(h * L / Q)
        import math as _m

        sized = Q * int(_m.ceil((h_factor * L) / Q)) if Q > 0 else h_factor * L
        prev = float(data.get("base_capacity", 0.0))
        new_base = max(prev, sized) if respect_min else sized
        G.edges[u_g, v_g, k_g]["base_capacity"] = new_base
        # Keep target_capacity aligned with total intended capacity for the adjacency
        G.edges[u_g, v_g, k_g]["target_capacity"] = new_base

    # Aggregate post-adjustment capacities per directed metro corridor
    post_corridor_caps: dict[tuple[str, str], float] = {}
    for _u2, _v2, _k2, data in G.edges(keys=True, data=True):
        if str(data.get("link_type")) != "inter_metro_corridor":
            continue
        src_name = data.get("source_metro")
        tgt_name = data.get("target_metro")
        if not isinstance(src_name, str) or not isinstance(tgt_name, str):
            continue
        cap = float(data.get("base_capacity", data.get("capacity", 0.0)))
        key = (src_name, tgt_name)
        post_corridor_caps[key] = post_corridor_caps.get(key, 0.0) + cap

    # Log corridor capacity deltas (per directed pair)
    if post_corridor_caps:
        try:
            total_before = sum(
                prev_corridor_caps.get(k, 0.0) for k in post_corridor_caps
            )
            total_after = sum(post_corridor_caps.values())
            logger.info(
                "TM sizing: corridor capacity totals (Gbps) before=%s after=%s delta=%s",
                f"{total_before:,.1f}",
                f"{total_after:,.1f}",
                f"{(total_after - total_before):,.1f}",
            )
            # Only emit lines for pairs that changed
            for pair in sorted(post_corridor_caps):
                before = float(prev_corridor_caps.get(pair, 0.0))
                after = float(post_corridor_caps[pair])
                if abs(after - before) <= 0.0:
                    continue
                factor = (after / before) if before > 0.0 else float("inf")
                src, dst = pair
                logger.info(
                    "TM sizing: %s -> %s corridor capacity %s -> %s (delta=%s, x%s)",
                    src,
                    dst,
                    f"{before:,.1f}",
                    f"{after:,.1f}",
                    f"{(after - before):,.1f}",
                    "inf" if not (factor < float("inf")) else f"{factor:.2f}",
                )
        except Exception:
            # Logging must not affect sizing; swallow any formatting errors
            pass

    # Compute PoP egress based on sized corridor capacities
    pop_egress: dict[str, float] = {}
    for u_g, v_g, data in G.edges(data=True):
        if str(data.get("link_type")) != "inter_metro_corridor":
            continue
        cap = float(data.get("base_capacity", data.get("capacity", 0.0)))
        pop_egress[u_g] = pop_egress.get(u_g, 0.0) + cap
        pop_egress[v_g] = pop_egress.get(v_g, 0.0) + cap

    alpha = float(getattr(sizing_cfg, "alpha_dc_to_pop", 1.2))
    beta = float(getattr(sizing_cfg, "beta_intra_pop", 0.8))

    # Derive DC->PoP base capacities
    for u_g, v_g, k_g, data in G.edges(keys=True, data=True):
        if str(data.get("link_type")) != "dc_to_pop":
            continue
        # Endpoint PoP is either u or v depending on orientation
        try:
            u_kind = str(G.nodes[u_g].get("site_kind", ""))
            v_kind = str(G.nodes[v_g].get("site_kind", ""))
        except Exception as exc:
            raise ValueError(
                f"TM sizing: missing site_kind on DC->PoP edge endpoints: {u_g}, {v_g}"
            ) from exc
        if v_kind == "pop":
            pop_node = v_g
        elif u_kind == "pop":
            pop_node = u_g
        else:
            raise ValueError(
                f"TM sizing: DC->PoP edge does not connect to a PoP endpoint: {u_g}<->{v_g}"
            )
        egress = float(pop_egress.get(pop_node, 0.0))
        target = alpha * egress
        import math as _m

        sized = Q * int(_m.ceil(target / Q)) if Q > 0 else target
        prev = float(data.get("base_capacity", 0.0))
        new_base = max(prev, sized) if respect_min else sized
        G.edges[u_g, v_g, k_g]["base_capacity"] = new_base
        G.edges[u_g, v_g, k_g]["target_capacity"] = new_base

    # Derive PoP<->PoP (intra-metro) base capacities
    for u_g, v_g, k_g, data in G.edges(keys=True, data=True):
        if str(data.get("link_type")) != "intra_metro":
            continue
        eg_u = float(pop_egress.get(u_g, 0.0))
        eg_v = float(pop_egress.get(v_g, 0.0))
        target = beta * min(eg_u, eg_v)
        import math as _m

        sized = Q * int(_m.ceil(target / Q)) if Q > 0 else target
        prev = float(data.get("base_capacity", 0.0))
        new_base = max(prev, sized) if respect_min else sized
        G.edges[u_g, v_g, k_g]["base_capacity"] = new_base
        G.edges[u_g, v_g, k_g]["target_capacity"] = new_base

    logger.info(
        "TM sizing: applied capacities (Q=%s Gb/s, h=%s, alpha=%s, beta=%s)",
        f"{Q:.0f}",
        f"{h_factor:.3f}",
        f"{alpha:.3f}",
        f"{beta:.3f}",
    )


def to_network_sections(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Serialize MultiGraph to NetGraph 'groups' and 'adjacency' sections."""
    logger.info("Serializing MultiGraph to scenario network sections")
    # Groups
    groups: dict[str, Any] = {}
    # Determine max counts
    metro_idx_map = {m["name"]: idx for idx, m in enumerate(metros, 1)}
    for metro in metros:
        name = metro["name"]
        idx = metro_idx_map[name]
        s = int(metro_settings[name]["pop_per_metro"])  # type: ignore[index]
        d = int(metro_settings[name]["dc_regions_per_metro"])  # type: ignore[index]
        pop_group_path = f"metro{idx}/pop[1-{s}]"
        groups[pop_group_path] = {
            "use_blueprint": metro_settings[name]["site_blueprint"],
            "attrs": {
                "metro_name": name,
                "metro_name_orig": metro.get("name_orig", name),
                "metro_id": metro.get("metro_id", ""),
                "location_x": metro.get("x", 0.0),
                "location_y": metro.get("y", 0.0),
                "radius_km": metro.get("radius_km", 0.0),
                "node_type": "pop",
            },
        }
        try:
            logger.debug(
                "group: path=%s use_blueprint=%s node_type=pop metro=%s count=%d",
                pop_group_path,
                metro_settings[name]["site_blueprint"],
                name,
                s,
            )
        except Exception:
            pass
        if d > 0:
            dc_group_path = f"metro{idx}/dc[1-{d}]"
            groups[dc_group_path] = {
                "use_blueprint": metro_settings[name]["dc_region_blueprint"],
                "attrs": {
                    "metro_name": name,
                    "metro_name_orig": metro.get("name_orig", name),
                    "metro_id": metro.get("metro_id", ""),
                    "location_x": metro.get("x", 0.0),
                    "location_y": metro.get("y", 0.0),
                    "radius_km": metro.get("radius_km", 0.0),
                    "node_type": "dc_region",
                    # Required by validation and traffic
                    "mw_per_dc_region": float(
                        getattr(
                            getattr(config, "traffic", None), "mw_per_dc_region", 0.0
                        )
                    ),
                    "gbps_per_mw": float(
                        getattr(getattr(config, "traffic", None), "gbps_per_mw", 0.0)
                    ),
                },
            }
            try:
                logger.debug(
                    "group: path=%s use_blueprint=%s node_type=dc_region metro=%s count=%d",
                    dc_group_path,
                    metro_settings[name]["dc_region_blueprint"],
                    name,
                    d,
                )
            except Exception:
                pass

    # Adjacency as explicit per-pair entries from graph edges
    adjacency: list[dict[str, Any]] = []
    for u, v, k, data in G.edges(keys=True, data=True):
        # Only serialize if capacity assigned
        cap = data.get("capacity", data.get("base_capacity", 1))
        cost = int(data.get("cost", 1))
        attrs = {
            "link_type": data.get("link_type", "unknown"),
            "source_metro": data.get("source_metro"),
            "target_metro": data.get("target_metro"),
        }
        # Emit total expected capacity for the adjacency (pre per-link split)
        try:
            tcap = float(data.get("target_capacity", data.get("base_capacity", 0.0)))
        except Exception:
            tcap = float(data.get("base_capacity", 0.0))
        attrs["target_capacity"] = tcap
        # Tag this adjacency with a stable site-edge key and optional adjacency id
        attrs["site_edge"] = f"{u}|{v}|{k}"
        if "adjacency_id" in data:
            attrs["adjacency_id"] = data.get("adjacency_id")
        if "distance_km" in data:
            attrs["distance_km"] = int(
                data["distance_km"]
            )  # ensure int for YAML stability
        if "euclidean_km" in data and data["euclidean_km"] is not None:
            attrs["euclidean_km"] = float(
                data["euclidean_km"]
            )  # corridor straight-line distance
        if "detour_ratio" in data and data["detour_ratio"] is not None:
            attrs["detour_ratio"] = float(
                data["detour_ratio"]
            )  # corridor length / euclidean
        link_params = {"capacity": cap, "cost": cost, "attrs": attrs}
        # Include computed per-end hardware, if present on this edge
        hw = data.get("hardware")
        if isinstance(hw, dict) and hw:
            link_params["attrs"]["hardware"] = hw
        # Risk groups belong at link_params level to align with validation/tests
        if "risk_groups" in data and data["risk_groups"]:
            link_params["risk_groups"] = data["risk_groups"]

        adjacency.append(
            {
                # Preserve role-based match filters on endpoints
                # so DSL expansion restricts nodes within each site.
                "source": (
                    {"path": u, "match": data.get("match")}
                    if isinstance(data.get("match"), dict) and data.get("match")
                    else u
                ),
                "target": (
                    {"path": v, "match": data.get("match")}
                    if isinstance(data.get("match"), dict) and data.get("match")
                    else v
                ),
                "pattern": "one_to_one",
                "link_params": link_params,
            }
        )

    logger.info(
        "Serialized network: %d groups, %d adjacency entries",
        len(groups),
        len(adjacency),
    )
    return groups, adjacency


def save_site_graph_json(G: nx.MultiGraph, path: Path, *, json_indent: int = 2) -> None:
    """Save site-level MultiGraph to JSON using string node ids.

    The format mirrors the integrated graph JSON structure but uses string
    identifiers for nodes and includes the MultiGraph edge key to
    disambiguate parallel edges.

    Args:
        G: Site-level MultiGraph.
        path: Output JSON file path.
        json_indent: Indentation for JSON pretty-printing.
    """
    logger.info(f"Saving site-level network graph to JSON: {path}")

    def _to_jsonable(obj: Any) -> Any:
        if isinstance(obj, (set, tuple)):
            try:
                return list(sorted(obj)) if isinstance(obj, set) else list(obj)
            except Exception:
                return list(obj)
        try:
            import numpy as np  # type: ignore

            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
        except Exception:
            pass
        return obj

    out: dict[str, Any] = {"graph_type": "site_network", "nodes": [], "edges": []}

    for node_id, data in G.nodes(data=True):
        out["nodes"].append(
            {"id": str(node_id), **{k: _to_jsonable(v) for k, v in data.items()}}
        )

    for u, v, k, data in G.edges(keys=True, data=True):
        out["edges"].append(
            {
                "source": str(u),
                "target": str(v),
                "key": str(k),
                **{k2: _to_jsonable(v2) for k2, v2 in data.items()},
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(out, f, indent=int(json_indent))
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved site network graph JSON ({size_mb:.2f} MB) → {path}")
