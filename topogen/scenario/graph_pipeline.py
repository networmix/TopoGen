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

from topogen.corridors import (
    extract_corridor_edges_for_metros_graph as _extract_corridor_edges,
)
from topogen.log_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


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
                cost_arc = (
                    min(arc_ceil(n, p, c) for c in cand_idxs)
                    if ring_radius_km > 0.0
                    else base_cost
                )
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

    # Nodes
    for metro in metros:
        name = metro["name"]
        idx = metro_idx_map[name]
        s = int(metro_settings[name]["pop_per_metro"])  # type: ignore[index]
        d = int(metro_settings[name]["dc_regions_per_metro"])  # type: ignore[index]
        for p in range(1, s + 1):
            node_id = _site_node_id(idx, "pop", p)
            G.add_node(
                node_id,
                metro_idx=idx,
                metro_name=name,
                site_kind="pop",
                site_ordinal=p,
            )
        for j in range(1, d + 1):
            node_id = _site_node_id(idx, "dc", j)
            G.add_node(
                node_id,
                metro_idx=idx,
                metro_name=name,
                site_kind="dc",
                site_ordinal=j,
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
    """Assign per-link capacity by splitting config default over link count.

    For each adjacency group (by 'adjacency_id'), take the config default
    capacity for that link_type and split evenly across realized links.
    """
    logger.info("Assigning per-link capacities from defaults")
    # Group edge keys by adjacency_id
    by_adj: dict[str, list[tuple[str, str, str]]] = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        adj_id = str(data.get("adjacency_id", ""))
        if not adj_id:
            raise ValueError(f"Edge {u}-{v} missing mandatory adjacency_id")
        by_adj.setdefault(adj_id, []).append((u, v, key))

    for adj_id, ek_list in by_adj.items():
        # Determine base from stored base_capacity on any edge in this group
        any_u, any_v, any_k = ek_list[0]
        data = G.get_edge_data(any_u, any_v, any_k)
        link_type = data.get("link_type") if isinstance(data, dict) else None
        # Safely coerce base_capacity to int with explicit None handling to satisfy type checker
        try:
            if isinstance(data, dict):
                base_val = data.get("base_capacity")
                base = int(0 if base_val is None else base_val)
            else:
                base = 0
        except Exception as exc:
            # Provide clear error with the original value for debugging
            bad_val = data.get("base_capacity") if isinstance(data, dict) else None
            raise ValueError(
                f"Adjacency {adj_id} has non-integer base_capacity: {bad_val}"
            ) from exc
        if base <= 0:
            raise ValueError(
                f"Adjacency {adj_id} (type={link_type}) has invalid base_capacity={base}"
            )
        count = max(1, len(ek_list))
        per = base // count
        if per <= 0:
            raise ValueError(
                f"Adjacency {adj_id} (type={link_type}) base_capacity={base} insufficient for {count} links"
            )
        logger.info(
            "Adjacency %s: type=%s base=%s links=%d -> per-link=%s",
            adj_id,
            link_type,
            base,
            count,
            per,
        )
        # Assign capacity to each edge in group
        for u, v, k in ek_list:
            G.edges[u, v, k]["capacity"] = int(per)


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
        cap = int(data.get("capacity", data.get("base_capacity", 1)))
        cost = int(data.get("cost", 1))
        attrs = {
            "link_type": data.get("link_type", "unknown"),
            "source_metro": data.get("source_metro"),
            "target_metro": data.get("target_metro"),
        }
        if "distance_km" in data:
            attrs["distance_km"] = data["distance_km"]
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
