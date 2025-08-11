"""Traffic and risk sections for scenario building."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import networkx as nx

from topogen.log_config import get_logger
from topogen.traffic_matrix import generate_traffic_matrix

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def _build_traffic_matrix_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: "TopologyConfig",
) -> dict[str, list[dict[str, Any]]]:
    """Build the ``traffic_matrix_set`` section if enabled."""
    return generate_traffic_matrix(metros, metro_settings, config)


def _build_risk_groups_section(
    graph: nx.Graph, config: "TopologyConfig"
) -> list[dict[str, Any]]:
    """Build the ``risk_groups`` section from corridor-tagged edges.

    This reads risk-group tags attached to corridor edges and creates scenario
    risk group entries with a canonical per-pair distance in kilometers.
    """
    if not config.corridors.risk_groups.enabled:
        return []

    prefix = getattr(config.corridors.risk_groups, "group_prefix", "corridor_risk")
    pair_distance_km: dict[str, int] = {}
    for source, target, data in graph.edges(data=True):
        src = graph.nodes[source]
        tgt = graph.nodes[target]
        if src.get("node_type") in ["metro", "metro+highway"] and tgt.get(
            "node_type"
        ) in [
            "metro",
            "metro+highway",
        ]:
            src_name = str(src.get("name", "")).strip()
            tgt_name = str(tgt.get("name", "")).strip()
            a_name, b_name = (
                (src_name, tgt_name) if src_name < tgt_name else (tgt_name, src_name)
            )
            pair_rg = f"{prefix}_{a_name}_{b_name}"
            dist_km = int(math.ceil(float(data.get("length_km", 0.0))))
            prev = pair_distance_km.get(pair_rg)
            pair_distance_km[pair_rg] = dist_km if prev is None else max(prev, dist_km)

    risk_group_distance_km: dict[str, int] = {}
    for _source, _target, data in graph.edges(data=True):
        src = graph.nodes[_source]
        tgt = graph.nodes[_target]
        if src.get("node_type") in ["metro", "metro+highway"] and tgt.get(
            "node_type"
        ) in [
            "metro",
            "metro+highway",
        ]:
            edge_dist_km = int(math.ceil(float(data.get("length_km", 0.0))))
            edge_risk_groups = data.get("risk_groups", []) or []
            for rg in edge_risk_groups:
                name = str(rg)
                base = name
                idx = base.rfind("_path")
                if idx != -1 and idx + 5 <= len(base):
                    tail = base[idx + 5 :]
                    if tail.isdigit():
                        base = base[:idx]
                canonical = pair_distance_km.get(base)
                chosen = canonical if canonical is not None else edge_dist_km
                prev = risk_group_distance_km.get(name)
                risk_group_distance_km[name] = (
                    chosen if prev is None else max(prev, chosen)
                )

    if not risk_group_distance_km:
        logger.info("No risk groups found in corridor edges")
        return []

    risk_groups: list[dict[str, Any]] = []
    for rg_name in sorted(risk_group_distance_km.keys()):
        risk_groups.append(
            {
                "name": rg_name,
                "attrs": {
                    "type": "corridor_risk",
                    "distance_km": int(risk_group_distance_km.get(rg_name, 0)),
                },
            }
        )
    logger.info(f"Generated {len(risk_groups)} risk group definitions")
    return risk_groups
