"""Legacy network topology builders (deprecated).

Graph-based pipeline supersedes this. Functions remain for API compatibility.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import networkx as nx

from topogen.log_config import get_logger

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
    # Deprecated in new pipeline; kept only for module exports compatibility
    network["groups"] = {}
    network["adjacency"] = []
    return network


def _build_groups_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    max_sites: int,
    max_dc_regions: int,
    config: "TopologyConfig",
) -> dict[str, Any]:
    """Deprecated: groups are built via graph_pipeline serialization."""
    return {}


def _build_adjacency_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    graph: nx.Graph,
    config: "TopologyConfig",
) -> list[dict[str, Any]]:
    """Deprecated: adjacency is built via graph_pipeline serialization."""
    return []


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


## Note: corridor extraction moved to topogen.scenario.adjacency
