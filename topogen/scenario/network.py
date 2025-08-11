"""Network topology section builders for scenarios.

Builds groups and adjacency sections and computes link overrides.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import networkx as nx

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.log_config import get_logger

from .utils import _count_nodes_with_role

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def _extract_metros_from_graph(graph: nx.Graph) -> list[dict[str, Any]]:
    """Extract metro node information from the integrated graph.

    Args:
        graph: Integrated graph containing metro and highway nodes.

    Returns:
        List of metro node dictionaries with required attributes.

    Raises:
        ValueError: If metro nodes are missing required attributes.
    """
    metros: list[dict[str, Any]] = []
    for node, data in graph.nodes(data=True):
        if data.get("node_type") in ["metro", "metro+highway"]:
            required_attrs = ["name", "metro_id", "radius_km"]
            for attr in required_attrs:
                if attr not in data:
                    raise ValueError(
                        f"Metro node {node} missing required attribute '{attr}'"
                    )
            metros.append(
                {
                    "node_key": node,
                    "name": data["name"],
                    "name_orig": data.get("name_orig", data["name"]),
                    "metro_id": data["metro_id"],
                    "x": data.get("x", 0.0),
                    "y": data.get("y", 0.0),
                    "radius_km": data["radius_km"],
                }
            )
    return metros


def _build_network_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    max_sites: int,
    max_dc_regions: int,
    graph: nx.Graph,
    config: "TopologyConfig",
) -> dict[str, Any]:
    """Build the network section of the NetGraph scenario."""
    network: dict[str, Any] = {}
    network["groups"] = _build_groups_section(
        metros, metro_settings, max_sites, max_dc_regions, config
    )
    network["adjacency"] = _build_adjacency_section(
        metros, metro_settings, graph, config
    )
    overrides = _build_intra_metro_link_overrides(metros, metro_settings, config)
    if overrides:
        network["link_overrides"] = overrides
    return network


def _build_groups_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    max_sites: int,
    max_dc_regions: int,
    config: "TopologyConfig",
) -> dict[str, Any]:
    """Build the groups section defining site hierarchies."""
    groups: dict[str, Any] = {}
    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        blueprint_name = settings["site_blueprint"]
        group_name = f"metro{idx}/pop[1-{max_sites}]"
        groups[group_name] = {
            "use_blueprint": blueprint_name,
            "attrs": {
                "metro_name": metro_name,
                "metro_name_orig": metro.get("name_orig", metro_name),
                "metro_id": metro["metro_id"],
                "location_x": metro["x"],
                "location_y": metro["y"],
                "radius_km": metro["radius_km"],
                "node_type": "pop",
            },
        }
        dc_regions_for_metro = int(settings.get("dc_regions_per_metro", 0))
        if dc_regions_for_metro > 0:
            dc_blueprint_name = settings["dc_region_blueprint"]
            dc_group_name = f"metro{idx}/dc[1-{dc_regions_for_metro}]"
            groups[dc_group_name] = {
                "use_blueprint": dc_blueprint_name,
                "attrs": {
                    "metro_name": metro_name,
                    "metro_name_orig": metro.get("name_orig", metro_name),
                    "metro_id": metro["metro_id"],
                    "location_x": metro["x"],
                    "location_y": metro["y"],
                    "radius_km": metro["radius_km"],
                    "node_type": "dc_region",
                    "mw_per_dc_region": float(
                        getattr(
                            getattr(config, "traffic", object()),
                            "mw_per_dc_region",
                            0.0,
                        )
                    ),
                    "gbps_per_mw": float(
                        getattr(
                            getattr(config, "traffic", object()), "gbps_per_mw", 0.0
                        )
                    ),
                },
            }
    return groups


def _build_adjacency_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    graph: nx.Graph,
    config: "TopologyConfig",
) -> list[dict[str, Any]]:
    """Build the adjacency section defining inter-metro connectivity."""
    adjacency: list[dict[str, Any]] = []
    blueprint_defs = get_builtin_blueprints()
    metro_by_node = {metro["node_key"]: metro for metro in metros}
    metro_idx_map = {metro["name"]: idx for idx, metro in enumerate(metros, 1)}

    # Intra-metro PoP mesh
    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        sites_count = settings["pop_per_metro"]
        if sites_count > 1:
            radius_km = float(metro.get("radius_km", 0.0))
            circle_frac = 1.0
            ring_radius_km = max(0.0, radius_km * circle_frac)
            site_bp = settings["site_blueprint"]
            core_per_site = max(
                1, _count_nodes_with_role(blueprint_defs, site_bp, "core")
            )
            adjacency.append(
                {
                    "source": {
                        "path": f"/metro{idx}/pop[1-{sites_count}]",
                        "match": {
                            "conditions": [
                                {"attr": "role", "operator": "==", "value": "core"}
                            ]
                        },
                    },
                    "target": {
                        "path": f"/metro{idx}/pop[1-{sites_count}]",
                        "match": {
                            "conditions": [
                                {"attr": "role", "operator": "==", "value": "core"}
                            ]
                        },
                    },
                    "pattern": "one_to_one",
                    "link_params": {
                        "capacity": int(
                            max(
                                1,
                                int(settings["intra_metro_link"]["capacity"])
                                // int(core_per_site),
                            )
                        ),
                        "cost": int(
                            math.ceil(
                                ((2.0 * math.pi * ring_radius_km) / float(sites_count))
                                if ring_radius_km > 0.0 and sites_count > 0
                                else settings["intra_metro_link"]["cost"]
                            )
                        ),
                        "attrs": {
                            **settings["intra_metro_link"]["attrs"],
                            "metro_name": metro_name,
                            "metro_name_orig": metro.get("name_orig", metro_name),
                            "distance_model": "metro_circle_arc",
                            "circle_radius_km": ring_radius_km,
                            "pop_count": int(sites_count),
                        },
                    },
                }
            )

    # DC-to-PoP connectivity within metro
    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        sites_count = settings["pop_per_metro"]
        dc_regions_count = settings["dc_regions_per_metro"]
        if dc_regions_count > 0 and sites_count > 0:
            site_bp = settings["site_blueprint"]
            dc_bp = settings["dc_region_blueprint"]
            core_per_site = max(
                1, _count_nodes_with_role(blueprint_defs, site_bp, "core")
            )
            dc_nodes = max(1, _count_nodes_with_role(blueprint_defs, dc_bp, "dc"))
            dp_divisor = max(1, core_per_site * dc_nodes)
            adjacency.append(
                {
                    "source": {
                        "path": f"/metro{idx}/dc[1-{dc_regions_count}]",
                        "match": {
                            "conditions": [
                                {"attr": "role", "operator": "==", "value": "dc"}
                            ]
                        },
                    },
                    "target": {
                        "path": f"/metro{idx}/pop[1-{sites_count}]",
                        "match": {
                            "conditions": [
                                {"attr": "role", "operator": "==", "value": "core"}
                            ]
                        },
                    },
                    "pattern": "mesh",
                    "link_params": {
                        "capacity": int(
                            max(
                                1,
                                int(settings["dc_to_pop_link"]["capacity"])
                                // int(dp_divisor),
                            )
                        ),
                        "cost": settings["dc_to_pop_link"]["cost"],
                        "attrs": {
                            **settings["dc_to_pop_link"]["attrs"],
                            "metro_name": metro_name,
                            "metro_name_orig": metro.get("name_orig", metro_name),
                        },
                    },
                }
            )

    # Inter-metro corridor connectivity
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


def _build_intra_metro_link_overrides(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: "TopologyConfig",
) -> list[dict[str, Any]]:
    """Create link_overrides assigning circle-arc distances as per-pair costs."""
    overrides: list[dict[str, Any]] = []
    circle_frac = 1.0
    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        sites_count = int(settings["pop_per_metro"])
        dc_count = int(settings["dc_regions_per_metro"])
        if sites_count <= 0:
            continue
        ring_radius_km = max(0.0, float(metro.get("radius_km", 0.0)) * circle_frac)
        if ring_radius_km <= 0.0:
            continue

        def arc_ceil(
            n: int, i: int, j: int, *, _radius_km: float = ring_radius_km
        ) -> int:
            if n <= 1:
                return 0
            delta = abs(i - j) % n
            steps = min(delta, n - delta)
            arc = steps * ((2.0 * math.pi * _radius_km) / float(n))
            return int(math.ceil(arc))

        if sites_count > 1:
            for i in range(1, sites_count + 1):
                for j in range(i + 1, sites_count + 1):
                    cost_ij = arc_ceil(sites_count, i, j)
                    if cost_ij <= 0:
                        continue
                    overrides.append(
                        {
                            "source": f"metro{idx}/pop{i}",
                            "target": f"metro{idx}/pop{j}",
                            "any_direction": True,
                            "link_params": {
                                "cost": cost_ij,
                                "attrs": {
                                    "distance_km": cost_ij,
                                    "distance_model": "metro_circle_arc",
                                    "circle_radius_km": ring_radius_km,
                                },
                            },
                        }
                    )

        if dc_count > 0 and sites_count > 0:
            for d in range(1, dc_count + 1):
                for p in range(1, sites_count + 1):
                    n = sites_count
                    frac_index = (d - 1) * (n / max(dc_count, 1))
                    cand_idxs = {
                        int(math.floor(frac_index)) % n + 1,
                        int(math.ceil(frac_index)) % n + 1,
                    }
                    cost_dp = min(arc_ceil(n, p, c) for c in cand_idxs)
                    if cost_dp <= 0:
                        continue
                    overrides.append(
                        {
                            "source": f"metro{idx}/dc{d}",
                            "target": f"metro{idx}/pop{p}",
                            "any_direction": True,
                            "link_params": {
                                "cost": cost_dp,
                                "attrs": {
                                    "distance_km": cost_dp,
                                    "distance_model": "metro_circle_arc",
                                    "circle_radius_km": ring_radius_km,
                                },
                            },
                        }
                    )
    return overrides


def _extract_corridor_edges(graph: nx.Graph) -> list[dict[str, Any]]:
    """Extract corridor edges between metro nodes from the integrated graph."""
    corridor_edges: list[dict[str, Any]] = []
    for source, target, data in graph.edges(data=True):
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]
        if source_data.get("node_type") in [
            "metro",
            "metro+highway",
        ] and target_data.get("node_type") in ["metro", "metro+highway"]:
            edge_entry: dict[str, Any] = {
                "source": source,
                "target": target,
                "length_km": data.get("length_km", 0.0),
                "edge_type": data.get("edge_type", "corridor"),
                "risk_groups": data.get("risk_groups", []),
            }
            if "capacity" in data:
                edge_entry["capacity"] = data["capacity"]
            corridor_edges.append(edge_entry)
    return corridor_edges
