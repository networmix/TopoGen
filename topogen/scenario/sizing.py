"""Size site links from traffic routed across the metro corridor graph."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Hashable, Iterator
from typing import Any

import netgraph_core
import networkx as nx
from ngraph.lib.nx import from_networkx as _from_networkx

from topogen.config import TopologyConfig
from topogen.log_config import get_logger

logger = get_logger(__name__)


def _parse_tm_endpoint_to_metro_idx(endpoint: str) -> int | None:
    """Extract metro index from a traffic matrix endpoint regex/path.

    Accepts strings like '^metro3/dc2/.*' or 'metro3/dc2'. Returns 3.
    """
    match = re.fullmatch(r"\^?metro([1-9][0-9]*)/dc[1-9][0-9]*(?:/\.\*)?", endpoint)
    return int(match[1]) if match else None


def _resolve_flow_placement(flow_placement: str) -> netgraph_core.FlowPlacement:
    """Resolve the configured split directly to the Core routing enum."""
    try:
        return {
            "EQUAL_BALANCED": netgraph_core.FlowPlacement.EQUAL_BALANCED,
            "PROPORTIONAL": netgraph_core.FlowPlacement.PROPORTIONAL,
        }[flow_placement]
    except KeyError as exc:
        raise ValueError(f"Unknown flow_placement: {flow_placement}") from exc


def _tm_metro_demands(
    demands: list[dict[str, Any]], dc_counts: dict[int, int]
) -> Iterator[tuple[int, int, float]]:
    """Collapse the generator's concrete and uniform demands to metro pairs."""
    uniform_selector = "(metro[0-9]+/dc[0-9]+)"
    for demand in demands:
        volume = float(demand.get("demand", 0.0))
        if not math.isfinite(volume) or volume < 0:
            raise ValueError("TM sizing: demand volume must be finite non-negative")
        if volume == 0:
            continue
        source = str(demand.get("source_path", ""))
        target = str(demand.get("sink_path", ""))
        if (
            source == target == uniform_selector
            and demand.get("mode") == "pairwise"
            and demand.get("group_mode") == "group_pairwise"
        ):
            total_dc = sum(dc_counts.values())
            if total_dc < 2:
                raise ValueError(
                    "TM sizing: pairwise traffic requires at least two DCs"
                )
            # NetGraph splits volume over all ordered, distinct DC site pairs.
            # Same-metro pairs remain in the denominator but use no corridors.
            per_pair = volume / (total_dc * (total_dc - 1))
            for src, src_count in dc_counts.items():
                for dst, dst_count in dc_counts.items():
                    if src != dst and src_count > 0 and dst_count > 0:
                        yield src, dst, per_pair * src_count * dst_count
            continue
        src = _parse_tm_endpoint_to_metro_idx(source)
        dst = _parse_tm_endpoint_to_metro_idx(target)
        if src is None or dst is None:
            raise ValueError(
                f"TM sizing: unsupported demand endpoint ({source!r} -> {target!r})"
            )
        if src != dst:
            yield src, dst, volume


def tm_based_size_capacities(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
    traffic_matrices: dict[str, list[dict[str, Any]]],
) -> None:
    """Size links from generated traffic routed on a collapsed metro graph.

    Preserve parallel corridors and route each demand with the configured flow
    split. Size each corridor for its peak directional load, then apply headroom
    and round up to capacity increments. Derive DC-to-PoP and intra-metro
    capacities from PoP egress, respecting configured minimums when enabled.
    """
    sizing_cfg = config.build.tm_sizing
    if sizing_cfg.enabled is not True:
        return

    traffic_cfg = config.traffic
    if not traffic_cfg.enabled:
        raise ValueError(
            "TM sizing is enabled but traffic generation is disabled in configuration"
        )

    matrix_name = sizing_cfg.matrix_name or config.traffic.matrix_name
    demands = traffic_matrices.get(matrix_name, [])
    dc_counts = {
        idx: metro_settings[metro["name"]]["dc_regions_per_metro"]
        for idx, metro in enumerate(metros, 1)
    }
    if sum(dc_counts.values()) <= 0:
        raise ValueError(
            "TM sizing is enabled but no DC regions are configured (dc_regions_per_metro == 0)"
        )
    if not demands:
        raise ValueError(
            f"TM sizing: traffic matrix '{matrix_name}' is empty despite enabled traffic"
        )

    metro_idx_map = {m["name"]: idx for idx, m in enumerate(metros, 1)}
    # Build temporary NetworkX MultiDiGraph with metro nodes and inter-metro corridors.
    # MultiDiGraph is required to preserve parallel edges between the same metro pair
    # (e.g., striped corridors with multiple links per corridor).
    H: nx.MultiDiGraph = nx.MultiDiGraph()
    H.add_nodes_from(metro_idx_map.values())

    # Map to track correspondence between H edges and G edges.
    # Key is (src_metro_idx, dst_metro_idx, h_edge_key) to handle parallel edges.
    # Both directions reference the same physical edge, sized once for peak load.
    g_edge_refs: dict[tuple[Hashable, Hashable, Any], tuple[Any, Any, Any]] = {}

    corridor_edges = [
        (u, v, key, data)
        for u, v, key, data in G.edges(keys=True, data=True)
        if data["link_type"] == "inter_metro_corridor"
    ]
    for u_g, v_g, k_g, data in corridor_edges:
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
        # Let H allocate keys: G keys can repeat across different PoP pairs
        # that collapse to the same metro pair.
        h_key_fwd = H.add_edge(s_idx, t_idx, capacity=1e15, cost=cost)
        g_edge_refs[(s_idx, t_idx, h_key_fwd)] = (u_g, v_g, k_g)

        h_key_rev = H.add_edge(t_idx, s_idx, capacity=1e15, cost=cost)
        g_edge_refs[(t_idx, s_idx, h_key_rev)] = (u_g, v_g, k_g)

    if H.number_of_edges() == 0:
        raise ValueError(
            "TM sizing: no inter-metro corridor edges present in site graph"
        )

    # Convert NetworkX graph to netgraph_core format.
    # bidirectional=False because we already added explicit reverse edges above.
    multidigraph, node_map, edge_map = _from_networkx(H, bidirectional=False)
    num_nodes = multidigraph.num_nodes()

    backend = netgraph_core.Backend.cpu()
    algorithms = netgraph_core.Algorithms(backend)
    handle = algorithms.build_graph(multidigraph)

    flow_state = netgraph_core.FlowState(multidigraph)

    core_fp = _resolve_flow_placement(sizing_cfg.flow_placement)

    edge_selection = netgraph_core.EdgeSelection(
        multi_edge=True,
        require_capacity=False,  # IP-style routing based on cost only
        tie_break=netgraph_core.EdgeTieBreak.DETERMINISTIC,
    )

    by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    for source, target, volume in _tm_metro_demands(demands, dc_counts):
        by_pair[source, target].append(volume)
    for (s_metro, t_metro), volumes in by_pair.items():
        demand_val = math.fsum(volumes)
        s_idx = node_map.to_index.get(s_metro)
        t_idx = node_map.to_index.get(t_metro)
        if s_idx is None or t_idx is None:
            raise ValueError(
                f"TM sizing: metro index out of range (src={s_metro}, dst={t_metro}, num_nodes={num_nodes})"
            )

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

        placed = flow_state.place_on_dag(
            src=s_idx,
            dst=t_idx,
            dag=pred_dag,
            requested_flow=demand_val,
            flow_placement=core_fp,
        )

        if not math.isclose(placed, demand_val, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"TM sizing: could not route demand metro{s_metro}->metro{t_metro}: {placed}/{demand_val}"
            )
        cost = int(dists[t_idx])
        logger.debug(
            "TM sizing: placed %s Gbps from metro%d->metro%d (cost=%s)",
            f"{placed:,.1f}",
            s_metro,
            t_metro,
            f"{cost:,}",
        )

    edge_flows_arr = flow_state.edge_flow_view()
    ext_edge_ids_view = multidigraph.ext_edge_ids_view()
    edge_loads: dict[tuple[Any, Any, Any], float] = {}
    for edge_idx in range(len(edge_flows_arr)):
        flow_val = float(edge_flows_arr[edge_idx])
        if flow_val <= 0.0:
            continue
        ext_id = int(ext_edge_ids_view[edge_idx])
        src_idx, dst_idx, h_key = edge_map.to_ref[ext_id]
        g_edge_ref = g_edge_refs[src_idx, dst_idx, h_key]
        edge_loads[g_edge_ref] = max(edge_loads.get(g_edge_ref, 0.0), flow_val)

    Q = float(sizing_cfg.quantum_gbps)
    h_factor = float(sizing_cfg.headroom)
    respect_min = bool(sizing_cfg.respect_min_base_capacity)

    def apply_capacity(data: dict[str, Any], target: float) -> None:
        sized = Q * math.ceil(target / Q) if Q > 0 else target
        capacity = max(float(data["base_capacity"]), sized) if respect_min else sized
        data["base_capacity"] = data["target_capacity"] = capacity

    # Each undirected site edge is sized once for its peak directional load.
    # Accumulate metro-pair totals and PoP egress in the same pass.
    previous: dict[tuple[str, str], float] = defaultdict(float)
    current: dict[tuple[str, str], float] = defaultdict(float)
    pop_egress: dict[str, float] = defaultdict(float)
    for source, target, key, data in corridor_edges:
        pair = (data["source_metro"], data["target_metro"])
        previous[pair] += float(data["base_capacity"])
        if (source, target, key) in edge_loads:
            apply_capacity(data, h_factor * edge_loads[source, target, key])
        capacity = float(data["base_capacity"])
        current[pair] += capacity
        pop_egress[source] += capacity
        pop_egress[target] += capacity

    total_before, total_after = sum(previous.values()), sum(current.values())
    logger.info(
        "TM sizing: corridor capacity totals (Gbps) before=%.1f after=%.1f delta=%.1f",
        total_before,
        total_after,
        total_after - total_before,
    )
    for pair in sorted(current):
        before, after = previous[pair], current[pair]
        if after != before:
            logger.info(
                "TM sizing: %s <-> %s corridor capacity %.1f -> %.1f (delta=%.1f)",
                *pair,
                before,
                after,
                after - before,
            )

    alpha = float(sizing_cfg.alpha_dc_to_pop)
    beta = float(sizing_cfg.beta_intra_pop)

    for u_g, v_g, data in G.edges(data=True):
        if str(data.get("link_type")) != "dc_to_pop":
            continue
        # An undirected edge may expose either endpoint first.
        u_kind = G.nodes[u_g]["site_kind"]
        v_kind = G.nodes[v_g]["site_kind"]
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
        apply_capacity(data, target)

    for u_g, v_g, data in G.edges(data=True):
        if str(data.get("link_type")) != "intra_metro":
            continue
        eg_u = float(pop_egress.get(u_g, 0.0))
        eg_v = float(pop_egress.get(v_g, 0.0))
        target = beta * min(eg_u, eg_v)
        apply_capacity(data, target)

    logger.info(
        "TM sizing: applied capacities (Q=%s Gb/s, h=%s, alpha=%s, beta=%s)",
        f"{Q:.0f}",
        f"{h_factor:.3f}",
        f"{alpha:.3f}",
        f"{beta:.3f}",
    )
