"""Extract metro descriptors and serialize site graphs to NetGraph DSL and JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from topogen.config import TopologyConfig
from topogen.log_config import get_logger
from topogen.naming import site_edge_id

logger = get_logger(__name__)


def _extract_metros_from_graph(graph: nx.MultiGraph) -> list[dict[str, Any]]:
    if not graph.is_multigraph() or graph.is_directed():
        raise TypeError("Expected an undirected corridor MultiGraph")
    required = ("name", "name_orig", "metro_id", "x", "y", "radius_km")
    metros = []
    names: set[str] = set()
    ids: set[str] = set()
    for node, data in graph.nodes(data=True):
        if data.get("node_type") != "metro":
            raise ValueError(f"Expected metro node at {node}")
        missing = set(required) - data.keys()
        if missing:
            raise ValueError(
                f"Metro node {node} missing required attributes: {sorted(missing)}"
            )
        for key, seen in (("name", names), ("metro_id", ids)):
            value = data[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Metro {node} requires nonempty string '{key}'")
            if value in seen:
                raise ValueError(f"Duplicate metro {key}: {value}")
            seen.add(value)
        for key in ("x", "y", "radius_km"):
            value = data[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Metro {node} requires finite numeric '{key}'")
        if data["radius_km"] < 0:
            raise ValueError(f"Metro {node} has negative radius_km")
        metros.append({"node_key": node, **{key: data[key] for key in required}})
    return metros


def to_network_sections(
    G: nx.MultiGraph,
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Serialize MultiGraph to NetGraph 'nodes' and 'links' sections."""
    logger.info("Serializing MultiGraph to scenario network sections")
    groups: dict[str, Any] = {}
    for idx, metro in enumerate(metros, 1):
        name = metro["name"]
        settings = metro_settings[name]
        attrs = {
            "metro_name": name,
            "metro_name_orig": metro["name_orig"],
            "metro_id": metro["metro_id"],
            "location_x": metro["x"],
            "location_y": metro["y"],
            "radius_km": metro["radius_km"],
        }
        for kind, count, blueprint in (
            ("pop", settings["pop_per_metro"], settings["site_blueprint"]),
            ("dc", settings["dc_regions_per_metro"], settings["dc_region_blueprint"]),
        ):
            if count == 0:
                continue
            site_attrs = {**attrs, "node_type": "pop" if kind == "pop" else "dc_region"}
            if kind == "dc":
                site_attrs.update(
                    mw_per_dc_region=float(config.traffic.mw_per_dc_region),
                    gbps_per_mw=float(config.traffic.gbps_per_mw),
                )
            groups[f"metro{idx}/{kind}[1-{count}]"] = {
                "blueprint": blueprint,
                "attrs": site_attrs,
            }

    adjacency: list[dict[str, Any]] = []
    for u, v, k, data in G.edges(keys=True, data=True):
        cap = data["base_capacity"]
        cost = int(data.get("cost", 1))
        attrs = {
            **data.get("attrs", {}),
            "link_type": data.get("link_type", "unknown"),
            "source_metro": data.get("source_metro"),
            "target_metro": data.get("target_metro"),
        }
        # Emit total expected capacity for the adjacency (pre per-link split)
        tcap = float(data.get("target_capacity", data.get("base_capacity", 0.0)))
        attrs["target_capacity"] = tcap
        # Tag this adjacency with a stable site-edge key and optional adjacency id
        attrs["site_edge"] = site_edge_id(u, v, k)
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
        hw = data.get("hardware")
        if isinstance(hw, dict) and hw:
            attrs["hardware"] = hw

        link_entry: dict[str, Any] = {
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
            "capacity": cap,
            "cost": cost,
            "attrs": attrs,
        }
        if config.corridors.risk_groups.enabled and data.get("risk_groups"):
            link_entry["risk_groups"] = data["risk_groups"]

        adjacency.append(link_entry)

    logger.info(
        "Serialized network: %d groups, %d adjacency entries",
        len(groups),
        len(adjacency),
    )
    return groups, adjacency


def save_site_graph_json(G: nx.MultiGraph, path: Path, *, json_indent: int = 2) -> None:
    """Save site nodes and edges as JSON.

    String node IDs and edge keys preserve parallel links.
    """
    logger.info(f"Saving site-level network graph to JSON: {path}")

    def _to_jsonable(obj: Any) -> Any:
        if isinstance(obj, set):
            return sorted(obj)

        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(f"Cannot serialize {type(obj).__name__}")

    out: dict[str, Any] = {"graph_type": "site_network", "nodes": [], "edges": []}

    for node_id, data in G.nodes(data=True):
        out["nodes"].append({"id": str(node_id), **data})

    for u, v, k, data in G.edges(keys=True, data=True):
        out["edges"].append(
            {
                "source": str(u),
                "target": str(v),
                "key": str(k),
                **data,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(out, f, indent=json_indent, default=_to_jsonable, allow_nan=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved site network graph JSON ({size_mb:.2f} MB) → {path}")
