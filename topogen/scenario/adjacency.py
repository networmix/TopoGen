"""Adjacency formation helpers for scenario building.

Provides functions that construct adjacency rules using NetGraph's DSL.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import networkx as nx

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.corridors import (
    extract_corridor_edges_for_metros_graph as _extract_corridor_edges_impl,
)
from topogen.log_config import get_logger

from .utils import _count_nodes_with_role

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def _extract_corridor_edges(graph: nx.Graph) -> list[dict[str, Any]]:
    """Backwards wrapper to extract corridor edges from metro-level graph."""
    return _extract_corridor_edges_impl(graph)


def _form_inter_metro_adjacency(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    graph: nx.Graph,
    config: "TopologyConfig",
) -> list[dict[str, Any]]:
    """Build cartesian inter-metro adjacency rules using NetGraph DSL.

    Emits one DSL rule per corridor pair with cartesian expansion across POPs.
    """
    adjacency: list[dict[str, Any]] = []
    blueprint_defs = get_builtin_blueprints()
    metro_by_node = {metro["node_key"]: metro for metro in metros}
    metro_idx_map = {metro["name"]: idx for idx, metro in enumerate(metros, 1)}

    corridor_edges = _extract_corridor_edges(graph)
    logger.info(f"Found {len(corridor_edges)} corridor connections")
    for edge in corridor_edges:
        source_metro = metro_by_node.get(edge["source"])
        target_metro = metro_by_node.get(edge["target"])
        if not source_metro or not target_metro:
            logger.warning(f"Skipping corridor edge with unknown metro: {edge}")
            continue
        source_idx = metro_idx_map[source_metro["name"]]
        target_idx = metro_idx_map[target_metro["name"]]
        source_sites = metro_settings[source_metro["name"]]["pop_per_metro"]
        target_sites = metro_settings[target_metro["name"]]["pop_per_metro"]
        source_settings = metro_settings[source_metro["name"]]
        default_capacity = source_settings["inter_metro_link"]["capacity"]
        base_cost = source_settings["inter_metro_link"]["cost"]
        src_bp = source_settings["site_blueprint"]
        tgt_bp = metro_settings[target_metro["name"]]["site_blueprint"]
        src_core = max(1, _count_nodes_with_role(blueprint_defs, src_bp, "core"))
        tgt_core = max(1, _count_nodes_with_role(blueprint_defs, tgt_bp, "core"))
        inter_divisor = max(1, min(src_core, tgt_core))
        link_params = {
            "capacity": int(
                max(
                    1, int(edge.get("capacity", default_capacity)) // int(inter_divisor)
                )
            ),
            "cost": math.ceil(edge.get("length_km", base_cost)),
            "attrs": {
                **source_settings["inter_metro_link"]["attrs"],
                "distance_km": math.ceil(edge.get("length_km", 0.0)),
                "source_metro": source_metro["name"],
                "source_metro_orig": source_metro.get(
                    "name_orig", source_metro["name"]
                ),
                "target_metro": target_metro["name"],
                "target_metro_orig": target_metro.get(
                    "name_orig", target_metro["name"]
                ),
            },
        }
        edge_risk_groups = edge.get("risk_groups", [])
        if edge_risk_groups:
            link_params["risk_groups"] = edge_risk_groups
        adjacency.append(
            {
                "source": {
                    "path": f"metro{source_idx}/pop{{src_idx}}",
                    "match": {
                        "conditions": [
                            {"attr": "role", "operator": "==", "value": "core"}
                        ]
                    },
                },
                "target": {
                    "path": f"metro{target_idx}/pop{{tgt_idx}}",
                    "match": {
                        "conditions": [
                            {"attr": "role", "operator": "==", "value": "core"}
                        ]
                    },
                },
                "expand_vars": {
                    "src_idx": list(range(1, source_sites + 1)),
                    "tgt_idx": list(range(1, target_sites + 1)),
                },
                "expansion_mode": "cartesian",
                "pattern": "one_to_one",
                "link_params": link_params,
            }
        )

    return adjacency
