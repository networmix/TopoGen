"""Graph-based scenario generation pipeline.

Builds a site-level MultiGraph, assigns capacities, and serializes to YAML.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import netgraph_core
import networkx as nx
from ngraph.dsl.blueprints.expand import expand_network_dsl as _ng_expand
from ngraph.lib.nx import from_networkx as _from_networkx
from ngraph.types.base import FlowPlacement

from topogen.blueprints_lib import get_builtin_blueprints as _get_builtins
from topogen.corridors import (
    extract_corridor_edges_for_metros_graph as _extract_corridor_edges,
)
from topogen.log_config import get_logger

from .striping import (
    build_node_overrides_for_site as _stripe_build_overrides,
)
from .striping import (
    eligible_device_names_from_blueprint as _stripe_names,
)
from .striping import (
    group_by_attr as _stripe_group_by_attr,
)
from .striping import (
    group_by_width as _stripe_group_by_width,
)
from .striping import (
    make_stripe_attr_name as _stripe_attr_name,
)

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


def _assign_site_positions(
    G: nx.MultiGraph, metros: list[dict[str, Any]], metro_idx_map: dict[str, int]
) -> None:
    """Assign deterministic 2D positions for site nodes within each metro radius.

    Positions are stored on nodes as ``pos_x`` and ``pos_y`` in the target CRS
    coordinate units (meters for EPSG:5070). Sites are placed on rings centered
    at the metro centroid, strictly within the metro radius to avoid overlap
    with the outline.

    Args:
        G: Site-level MultiGraph with nodes carrying ``metro_idx``,
            ``site_kind`` and ``site_ordinal`` attributes.
        metros: List of metro dicts as produced by ``_extract_metros_from_graph``.
        metro_idx_map: Mapping from metro name to metro index used in node ids.
    """
    # Build reverse map: idx -> metro record
    idx_to_metro: dict[int, dict[str, Any]] = {}
    for name, idx in metro_idx_map.items():
        # ``metros`` is enumerated 1-based in _metro_index_maps; reconstruct by name
        for m in metros:
            if m.get("name") == name:
                idx_to_metro[idx] = m
                break

    # Group nodes by metro index and site kind
    by_idx_pop: dict[int, list[tuple[str, int]]] = {}
    by_idx_dc: dict[int, list[tuple[str, int]]] = {}
    for node_id, data in G.nodes(data=True):
        try:
            idx = int(data.get("metro_idx", 0))
            kind = str(data.get("site_kind", ""))
            ordn = int(data.get("site_ordinal", 0))
        except Exception:
            # Skip nodes without expected metadata
            continue
        if kind == "pop":
            by_idx_pop.setdefault(idx, []).append((str(node_id), ordn))
        elif kind == "dc":
            by_idx_dc.setdefault(idx, []).append((str(node_id), ordn))

    # Assign positions per metro
    import math as _m

    for idx, metro in idx_to_metro.items():
        cx = float(metro.get("x", 0.0))
        cy = float(metro.get("y", 0.0))
        r_km = float(metro.get("radius_km", 0.0))
        r_m = max(0.0, 1000.0 * r_km)

        # POP ring radius and DC ring radius as fractions of metro radius
        # Keep strictly inside the circle to avoid overlap with outline
        r_pop = 0.70 * r_m if r_m > 0.0 else 0.0
        r_dc = 0.40 * r_m if r_m > 0.0 else 0.0

        # Deterministic angular placement
        pops = sorted(by_idx_pop.get(idx, []), key=lambda t: t[1])
        dcs = sorted(by_idx_dc.get(idx, []), key=lambda t: t[1])

        # Helper to place nodes on a ring
        def place_ring(
            nodes: list[tuple[str, int]],
            radius: float,
            phase: float,
            *,
            center_x: float = cx,
            center_y: float = cy,
        ) -> None:
            n = len(nodes)
            if n == 0:
                return
            if n == 1:
                theta = phase
                x = center_x + radius * _m.cos(theta)
                y = center_y + radius * _m.sin(theta)
                G.nodes[nodes[0][0]]["pos_x"] = float(x)
                G.nodes[nodes[0][0]]["pos_y"] = float(y)
                return
            for i, (node_name, _ord) in enumerate(nodes):
                theta = phase + (2.0 * _m.pi * i) / float(n)
                x = center_x + radius * _m.cos(theta)
                y = center_y + radius * _m.sin(theta)
                G.nodes[node_name]["pos_x"] = float(x)
                G.nodes[node_name]["pos_y"] = float(y)

        # Place POPs and DCs
        place_ring(pops, r_pop, phase=0.0)
        # Phase DC ring by 45 degrees to reduce overlap with POPs when counts match
        place_ring(dcs, r_dc, phase=_m.pi / 4.0)

        # Attach metro center and radius metadata to nodes (useful for visualization)
        for node_name, _ in pops + dcs:
            G.nodes[node_name]["center_x"] = float(cx)
            G.nodes[node_name]["center_y"] = float(cy)
            G.nodes[node_name]["radius_m"] = float(r_m)


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
    """Add DC-to-PoP edges within each metro with optional striping."""
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

        # Build symmetric match if role_pairs provided (fallback when no striping)
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

        # Optional striping resolution
        striping = (
            link_cfg.get("striping", {})
            if isinstance(link_cfg, dict)
            else getattr(link_cfg, "striping", {})
        ) or {}

        # Precompute stripe groups and attach node_overrides when configured
        stripe_attr: str | None = None
        stripe_labels: list[str] = []
        if striping:
            # Resolve eligible roles
            role_set: set[str] | None = set()
            for item in rp:
                if isinstance(item, str):
                    parts = [p.strip() for p in item.split("|") if p.strip()]
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    parts = [str(item[0]).strip(), str(item[1]).strip()]
                else:
                    parts = []
                for r in parts:
                    if r:
                        role_set.add(r)
            if not role_set:
                role_set = None

            bps = _get_builtins()
            pop_bp = bps.get(metro_settings[metro_name]["site_blueprint"])
            dc_bp = bps.get(metro_settings[metro_name]["dc_region_blueprint"])
            if not pop_bp or not dc_bp:
                raise ValueError(f"Unknown blueprint for metro {metro_name}")
            mode = str(striping.get("mode", "width")).strip()
            if mode == "width":
                width_val = int(striping.get("width", 0))
                pop_names = _stripe_names(pop_bp, role_set)
                dc_names = _stripe_names(dc_bp, role_set)
                pop_groups = _stripe_group_by_width(pop_names, width_val)
                dc_groups = _stripe_group_by_width(dc_names, width_val)
                # For width==1 allow asymmetric group counts (one-vs-many is valid).
                # For width>1 require equal number of groups to maintain pairing semantics.
                if width_val > 1 and (len(pop_groups) != len(dc_groups)):
                    raise ValueError(
                        f"DC-to-PoP striping mismatch in {metro_name}: groups pop={len(pop_groups)} dc={len(dc_groups)}"
                    )
                stripe_attr = _stripe_attr_name(f"dc_{idx}")
                label_to_names_pop = {
                    f"g{i + 1}": grp for i, grp in enumerate(pop_groups)
                }
                label_to_names_dc = {
                    f"g{i + 1}": grp for i, grp in enumerate(dc_groups)
                }
            elif mode == "by_attr":
                attr = str(striping.get("attribute", "")).strip()
                if not attr:
                    raise ValueError("striping.by_attr requires 'attribute'")
                pop_map = _stripe_group_by_attr(pop_bp, attr=attr, roles=role_set)
                dc_map = _stripe_group_by_attr(dc_bp, attr=attr, roles=role_set)
                if set(pop_map.keys()) != set(dc_map.keys()):
                    raise ValueError(
                        f"DC-to-PoP striping by_attr mismatch in {metro_name}: labels pop={sorted(pop_map)} dc={sorted(dc_map)}"
                    )
                stripe_attr = _stripe_attr_name(f"dc_{idx}")
                # Stable order
                label_to_names_pop = {lab: pop_map[lab] for lab in sorted(pop_map)}
                label_to_names_dc = {lab: dc_map[lab] for lab in sorted(dc_map)}
            else:
                raise ValueError(f"Unknown striping.mode: {mode}")

            # Build node_overrides for this metro's DC regions and POPs
            G.graph.setdefault("node_overrides", [])
            for p_ord in range(1, s + 1):
                site = _site_node_id(idx, "pop", p_ord)
                G.graph["node_overrides"].extend(
                    _stripe_build_overrides(site, stripe_attr, label_to_names_pop)
                )
            for d_ord in range(1, d + 1):
                site = _site_node_id(idx, "dc", d_ord)
                G.graph["node_overrides"].extend(
                    _stripe_build_overrides(site, stripe_attr, label_to_names_dc)
                )
            stripe_labels = list(label_to_names_pop.keys())

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

                # Build match: either role-based (no striping) or stripe-only
                if not striping:
                    match_payload = match_obj
                else:
                    # pick first stripe by default
                    label = stripe_labels[0] if stripe_labels else "g1"
                    match_payload = {
                        "conditions": [
                            {"attr": stripe_attr, "operator": "==", "value": label}
                        ]
                    }

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
                    match=match_payload,
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
    """Add PoP-to-PoP edges across metro corridors with optional striping."""
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
        # Corridor length km preferred; otherwise use source metro's configured cost
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
        # Optional striping
        striping = (
            src_cfg.get("striping", {})
            if isinstance(src_cfg, dict)
            else getattr(src_cfg, "striping", {})
        ) or {}

        match_base = (
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
        )

        if not striping:
            # Determine adjacency mode: mesh (default) or one_to_one
            mode_val = (
                src_cfg.get("mode", "mesh")
                if isinstance(src_cfg, dict)
                else getattr(src_cfg, "mode", "mesh")
            )
            mode = str(mode_val).strip().lower()
            if mode == "one_to_one":
                limit = min(int(s_sites), int(t_sites))
                for p in range(1, limit + 1):
                    u = _site_node_id(s_idx, "pop", p)
                    v = _site_node_id(t_idx, "pop", p)
                    G.add_edge(
                        u,
                        v,
                        key=f"{adj_id}:{p}-{p}",
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
                        match=match_base,
                        role_pairs=(
                            src_cfg.get("role_pairs", [])
                            if isinstance(src_cfg, dict)
                            else getattr(src_cfg, "role_pairs", [])
                        ),
                    )
            else:
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
                            match=match_base,
                            role_pairs=(
                                src_cfg.get("role_pairs", [])
                                if isinstance(src_cfg, dict)
                                else getattr(src_cfg, "role_pairs", [])
                            ),
                        )
            continue

        # With striping configured, compute stripe labels and node_overrides per site
        bps = _get_builtins()
        roles: set[str] | None = set()
        for item in (
            src_cfg.get("role_pairs", [])
            if isinstance(src_cfg, dict)
            else getattr(src_cfg, "role_pairs", [])
        ):
            if isinstance(item, str):
                parts = [p.strip() for p in item.split("|") if p.strip()]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                parts = [str(item[0]).strip(), str(item[1]).strip()]
            else:
                parts = []
            for r in parts:
                if r:
                    roles.add(r)
        if not roles:
            roles = None

        bp_A = bps.get(metro_settings[s_name]["site_blueprint"])  # type: ignore[index]
        bp_B = bps.get(metro_settings[t_name]["site_blueprint"])  # type: ignore[index]
        if not bp_A or not bp_B:
            raise ValueError(f"Unknown blueprint for metros {s_name} or {t_name}")
        mode = str(striping.get("mode", "width")).strip()

        if mode == "width":
            width_val = int(striping.get("width", 0))
            names_A = _stripe_names(bp_A, roles)
            names_B = _stripe_names(bp_B, roles)
            groups_A = _stripe_group_by_width(names_A, width_val)
            groups_B = _stripe_group_by_width(names_B, width_val)
            if len(groups_A) != len(groups_B):
                raise ValueError(
                    f"Inter-metro striping mismatch: groups {s_name}={len(groups_A)} {t_name}={len(groups_B)}"
                )
            label_to_names_A = {f"g{i + 1}": grp for i, grp in enumerate(groups_A)}
            label_to_names_B = {f"g{i + 1}": grp for i, grp in enumerate(groups_B)}
        elif mode == "by_attr":
            attr = str(striping.get("attribute", "")).strip()
            if not attr:
                raise ValueError("striping.by_attr requires 'attribute'")
            map_A = _stripe_group_by_attr(bp_A, attr=attr, roles=roles)
            map_B = _stripe_group_by_attr(bp_B, attr=attr, roles=roles)
            if set(map_A.keys()) != set(map_B.keys()):
                raise ValueError(
                    f"Inter-metro striping by_attr mismatch: {s_name} labels={sorted(map_A)} {t_name} labels={sorted(map_B)}"
                )
            label_to_names_A = {lab: map_A[lab] for lab in sorted(map_A)}
            label_to_names_B = {lab: map_B[lab] for lab in sorted(map_B)}
        else:
            raise ValueError(f"Unknown striping.mode: {mode}")

        stripe_attr = _stripe_attr_name(f"im_{min(s_idx, t_idx)}_{max(s_idx, t_idx)}")
        G.graph.setdefault("node_overrides", [])
        # Emit overrides for each site in both metros
        for p_ord in range(1, s_sites + 1):
            site = _site_node_id(s_idx, "pop", p_ord)
            G.graph["node_overrides"].extend(
                _stripe_build_overrides(site, stripe_attr, label_to_names_A)
            )
        for q_ord in range(1, t_sites + 1):
            site = _site_node_id(t_idx, "pop", q_ord)
            G.graph["node_overrides"].extend(
                _stripe_build_overrides(site, stripe_attr, label_to_names_B)
            )

        labels = list(label_to_names_A.keys())
        # Deterministic assignment: cycle sites then groups across corridor edges
        link_index = 0
        for p in range(1, s_sites + 1):
            for q in range(1, t_sites + 1):
                label = labels[(link_index) % len(labels)]
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
                    adjacency_id=f"{adj_id}:{label}",
                    distance_km=cost,
                    source_metro=s_name,
                    target_metro=t_name,
                    risk_groups=risk_groups,
                    euclidean_km=euclid_km,
                    detour_ratio=detour_ratio,
                    match={
                        "conditions": [
                            {
                                "attr": stripe_attr,
                                "operator": "==",
                                "value": label,
                            }
                        ]
                    },
                    role_pairs=(
                        src_cfg.get("role_pairs", [])
                        if isinstance(src_cfg, dict)
                        else getattr(src_cfg, "role_pairs", [])
                    ),
                )
                link_index += 1


def build_site_graph(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    integrated_graph: nx.Graph,
    config: TopologyConfig,
) -> nx.MultiGraph:
    """Construct a site-level MultiGraph with nodes and edges per adjacency.

    Nodes are created for sites: ``metro{n}/pop{i}`` and ``metro{n}/dc{j}`` with
    their blueprints attached as node attributes. Edges are added for three
    adjacency families with clear semantics:

    - intra_metro: PoP↔PoP within a metro on a ring with arc-length-based cost.
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

    # Assign deterministic positions for visualization/export consumers
    _assign_site_positions(G, metros, metro_idx_map)

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
        # Inject node_overrides (e.g., striping attributes) if present on graph
        try:
            overrides = G.graph.get("node_overrides", [])
            if isinstance(overrides, list) and overrides:
                dsl["network"]["node_overrides"] = overrides
        except Exception:
            pass
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


def _resolve_flow_placement(flow_placement: str) -> FlowPlacement:
    """Map config string to FlowPlacement enum."""
    fp_str = str(flow_placement).upper()
    if fp_str == "EQUAL_BALANCED":
        return FlowPlacement.EQUAL_BALANCED
    return FlowPlacement.PROPORTIONAL


def tm_based_size_capacities(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
) -> None:
    """Adjust base capacities using an early TM and ECMP on collapsed graph.

    Pipeline:
    - Generate TM using traffic_matrix.generate_traffic_matrix (in-memory).
    - Build a metro-level NetworkX MultiDiGraph with explicit forward and reverse
      edges for each inter-metro corridor in G. This allows traffic to flow in
      both directions while keeping per-direction loads separate.
    - Convert to netgraph_core StrictMultiDiGraph via from_networkx().
    - For each directed TM demand in the matrix, compute shortest-path ECMP
      fractions and accumulate load on inter-metro corridor edges.
    - Track forward and reverse flows separately using different edge refs.
      Since G is undirected, both refs point to the same G edge, but respect_min
      ensures final capacity = max(sized_forward, sized_reverse).
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

    # Build metro-level NetworkX graph from G inter-metro corridors
    metro_idx_map = {m["name"]: idx for idx, m in enumerate(metros, 1)}
    for idx, _ in enumerate(metros, 1):
        metro_idx_map.setdefault(f"metro{idx}", idx)

    # Build temporary NetworkX MultiDiGraph with metro nodes and inter-metro corridors.
    # MultiDiGraph is required to preserve parallel edges between the same metro pair
    # (e.g., striped corridors with multiple links per corridor).
    H: nx.MultiDiGraph = nx.MultiDiGraph()
    for idx in set(metro_idx_map.values()):
        H.add_node(idx)

    # Map to track correspondence between H edges and G edges.
    # Key is (src_metro_idx, dst_metro_idx, h_edge_key) to handle parallel edges.
    # For each undirected G edge, we create two directed H edges (forward and reverse)
    # with DIFFERENT refs so that flows in each direction are tracked separately.
    g_edge_refs: dict[tuple[int, int, Any], tuple[str, str, str]] = {}

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
        # Add forward edge: source_metro -> target_metro
        h_key_fwd = H.add_edge(s_idx, t_idx, key=f"{k_g}:fwd", capacity=1e15, cost=cost)
        g_edge_refs[(s_idx, t_idx, h_key_fwd)] = (str(u_g), str(v_g), str(k_g))

        # Add reverse edge: target_metro -> source_metro
        # Use DIFFERENT ref (v_g, u_g, k_g) so reverse flows are tracked separately.
        # When applied to undirected G, both refs point to the same edge but are
        # processed separately, resulting in max(sized_forward, sized_reverse).
        h_key_rev = H.add_edge(t_idx, s_idx, key=f"{k_g}:rev", capacity=1e15, cost=cost)
        g_edge_refs[(t_idx, s_idx, h_key_rev)] = (str(v_g), str(u_g), str(k_g))

    if H.number_of_edges() == 0:
        raise ValueError(
            "TM sizing: no inter-metro corridor edges present in site graph"
        )

    # Convert NetworkX graph to netgraph_core format.
    # bidirectional=False because we already added explicit reverse edges above.
    multidigraph, node_map, edge_map = _from_networkx(H, bidirectional=False)
    num_nodes = multidigraph.num_nodes()

    # Build Core graph handle
    backend = netgraph_core.Backend.cpu()
    algorithms = netgraph_core.Algorithms(backend)
    handle = algorithms.build_graph(multidigraph)

    # Create FlowState for accumulating flows
    flow_state = netgraph_core.FlowState(multidigraph)

    fp_enum = _resolve_flow_placement(
        getattr(sizing_cfg, "flow_placement", "EQUAL_BALANCED")
    )
    core_fp = (
        netgraph_core.FlowPlacement.EQUAL_BALANCED
        if fp_enum == FlowPlacement.EQUAL_BALANCED
        else netgraph_core.FlowPlacement.PROPORTIONAL
    )

    # Build edge selection
    edge_selection = netgraph_core.EdgeSelection(
        multi_edge=True,
        require_capacity=False,  # IP-style routing based on cost only
        tie_break=netgraph_core.EdgeTieBreak.DETERMINISTIC,
    )

    # Iterate demands and accumulate flows
    for d in demands:
        src = str(d.get("source_path", ""))
        dst = str(d.get("sink_path", ""))
        demand_val = float(d.get("demand", 0.0))
        if demand_val <= 0.0:
            continue
        s_metro = _parse_tm_endpoint_to_metro_idx(src)
        t_metro = _parse_tm_endpoint_to_metro_idx(dst)
        if s_metro is None or t_metro is None or s_metro == t_metro:
            continue
        # Convert 1-based metro indices to 0-based netgraph node indices
        s_idx = node_map.to_index.get(s_metro)
        t_idx = node_map.to_index.get(t_metro)
        if s_idx is None or t_idx is None:
            raise ValueError(
                f"TM sizing: metro index out of range (src={s_metro}, dst={t_metro}, num_nodes={num_nodes})"
            )

        # Compute SPF
        try:
            dists, pred_dag = algorithms.spf(
                handle,
                src=s_idx,
                dst=t_idx,
                selection=edge_selection,
                multipath=True,
            )
        except Exception as exc:
            raise ValueError(
                f"TM sizing: SPF failed for metro {s_metro}->{t_metro}: {exc}"
            ) from exc

        # Place flow on DAG
        placed = flow_state.place_on_dag(
            src=s_idx,
            dst=t_idx,
            dag=pred_dag,
            requested_flow=demand_val,
            flow_placement=core_fp,
        )

        try:
            cost = int(dists[t_idx])
            logger.debug(
                "TM sizing: placed %s Gbps from metro%d->metro%d (cost=%s)",
                f"{placed:,.1f}",
                s_metro,
                t_metro,
                f"{cost:,}",
            )
        except Exception:
            pass

    # Extract edge flows and map back to G edges
    edge_flows_arr = flow_state.edge_flow_view()
    ext_edge_ids_view = multidigraph.ext_edge_ids_view()
    edge_loads: dict[tuple[str, str, str], float] = {}
    for edge_idx in range(len(edge_flows_arr)):
        flow_val = float(edge_flows_arr[edge_idx])
        if flow_val <= 0.0:
            continue
        ext_id = int(ext_edge_ids_view[edge_idx])
        # Map ext_id -> H edge reference (src_idx, dst_idx, key)
        h_edge_ref = edge_map.to_ref.get(ext_id)
        if h_edge_ref:
            src_idx, dst_idx, h_key = h_edge_ref
            # H nodes are int (metro indices), so cast is safe
            if isinstance(src_idx, int) and isinstance(dst_idx, int):
                # Look up G edge using full (src, dst, key) tuple.
                # Forward and reverse H edges have different refs:
                # - Forward: (u_g, v_g, k_g)
                # - Reverse: (v_g, u_g, k_g)
                # This keeps flows in each direction separate. Since G is undirected,
                # both refs point to the same physical edge, but respect_min logic
                # below ensures capacity = max(sized_forward, sized_reverse).
                g_edge_ref = g_edge_refs.get((src_idx, dst_idx, h_key))
                if g_edge_ref:
                    edge_loads[g_edge_ref] = edge_loads.get(g_edge_ref, 0.0) + flow_val

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

    # Deduplicate and store node_overrides on graph for assembly to embed
    try:
        node_overrides = G.graph.get("node_overrides", [])
        if isinstance(node_overrides, list) and node_overrides:
            seen: set[tuple[str, str, str]] = set()
            dedup: list[dict[str, Any]] = []
            for ov in node_overrides:
                path = str(ov.get("path", ""))
                attrs = (
                    ov.get("attrs", {}) if isinstance(ov.get("attrs", {}), dict) else {}
                )
                for k, v in attrs.items():
                    key = (path, str(k), str(v))
                    if key in seen:
                        continue
                    seen.add(key)
                    dedup.append({"path": path, "attrs": {k: v}})
            G.graph["__emitted_node_overrides__"] = dedup
    except Exception:
        pass

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
