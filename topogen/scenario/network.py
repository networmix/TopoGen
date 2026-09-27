"""Extract metro attributes for scenario assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

from topogen.log_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    pass

logger = get_logger(__name__)


def _extract_metros_from_graph(graph: nx.Graph) -> list[dict[str, Any]]:
    """Extract metro nodes and their identity, coordinates, and radius.

    Raise ValueError if a metro lacks ``name``, ``metro_id``, or ``radius_km``.
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
