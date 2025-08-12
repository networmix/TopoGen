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
from ngraph.dsl.blueprints.expand import expand_network_dsl as _ng_expand

from topogen.blueprints_lib import get_builtin_blueprints as _get_builtins
from topogen.corridors import (
    extract_corridor_edges_for_metros_graph as _extract_corridor_edges,
)
from topogen.log_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def _component_assignments_for_site(
    config: "TopologyConfig", site_kind: str
) -> dict[str, dict[str, Any]]:
    """Build per-site component assignments from global config.

    Args:
        config: Top-level topology configuration.
        site_kind: Either "pop" or "dc" indicating site type.

    Returns:
        Mapping from role name to assignment dictionary (keys like
        "hw_component" and optionally "optics"). Roles included are
        {"core", "leaf", "spine"} for PoP sites and {"dc"} for DC sites.
    """
    roles_for_kind = ["core", "leaf", "spine"] if site_kind == "pop" else ["dc"]
    result: dict[str, dict[str, Any]] = {}
    components_cfg = getattr(config, "components", None)
    assignments = getattr(components_cfg, "assignments", None)

    for role in roles_for_kind:
        entry = getattr(assignments, role, None) if assignments is not None else None
        if entry is None:
            continue
        role_map: dict[str, Any] = {}
        # Only include keys that are set to non-empty values to keep JSON concise
        hw = getattr(entry, "hw_component", "")
        if hw:
            role_map["hw_component"] = hw
        optics = getattr(entry, "optics", "")
        if optics:
            role_map["optics"] = optics
        if role_map:
            result[role] = role_map
    return result


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
            G.add_edge(
                u,
                v,
                key=f"{adj_id}:{i}-{j}",
                link_type="intra_metro",
                base_capacity=base_capacity,
                cost=cost,
                adjacency_id=adj_id,
                distance_km=cost,
                source_metro=metro_name,
                target_metro=metro_name,
                match=(
                    link_cfg.get("match", {})
                    if isinstance(link_cfg, dict)
                    else getattr(link_cfg, "match", {})
                )
                or {},
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
                    cost=cost_arc,
                    adjacency_id=adj_id,
                    distance_km=cost_arc,
                    source_metro=metro_name,
                    target_metro=metro_name,
                    match=(
                        link_cfg.get("match", {})
                        if isinstance(link_cfg, dict)
                        else getattr(link_cfg, "match", {})
                    )
                    or {},
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
                    cost=cost,
                    adjacency_id=adj_id,
                    distance_km=cost,
                    source_metro=s_name,
                    target_metro=t_name,
                    risk_groups=risk_groups,
                    euclidean_km=euclid_km,
                    detour_ratio=detour_ratio,
                    # Symmetric match applied to both endpoints
                    match=(
                        src_cfg.get("match", {})
                        if isinstance(src_cfg, dict)
                        else getattr(src_cfg, "match", {})
                    )
                    or {},
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
        pop_assign = _component_assignments_for_site(config, "pop")
        dc_assign = _component_assignments_for_site(config, "dc")
        for p in range(1, s + 1):
            node_id = _site_node_id(idx, "pop", p)
            G.add_node(
                node_id,
                metro_idx=idx,
                metro_name=name,
                site_kind="pop",
                site_ordinal=p,
                site_blueprint=pop_blueprint,
                components_assignments=pop_assign,
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
                components_assignments=dc_assign,
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
        groups[f"metro{idx}/pop[1-{s}]"] = {
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
        if d > 0:
            groups[f"metro{idx}/dc[1-{d}]"] = {
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

    # Adjacency as explicit per-pair entries from graph edges
    adjacency: list[dict[str, Any]] = []
    for u, v, data in G.edges(data=True):
        # Only serialize if capacity assigned
        cap = data.get("capacity", data.get("base_capacity", 1))
        cost = int(data.get("cost", 1))
        attrs = {
            "link_type": data.get("link_type", "unknown"),
            "source_metro": data.get("source_metro"),
            "target_metro": data.get("target_metro"),
        }
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
