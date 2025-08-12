"""Network helpers retained for internal use.

This module exposes only `_extract_metros_from_graph`. All scenario building is
handled by `topogen.scenario.graph_pipeline`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

from topogen.log_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    pass

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


__all__ = ["_extract_metros_from_graph"]


## Note: corridor extraction logic is implemented in graph_pipeline
