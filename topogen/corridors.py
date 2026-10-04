"""Discover highway paths between metros, assign risk groups, and extract corridors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy.spatial import KDTree  # pyright: ignore[reportMissingTypeStubs]
from shapely.geometry import Point

from topogen.log_config import get_logger
from topogen.metro_clusters import MetroCluster

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import CorridorsConfig, ValidationConfig

logger = get_logger(__name__)


PathId = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class CorridorPath:
    """Concrete corridor path details connecting two metros.

    Attributes:
        edges: Ordered endpoint pairs along the path.
        segment_ids: Ordered contracted highway segment IDs (excludes anchor edges).
        length_km: Total length of path in kilometers.
        geometry: Ordered polyline coordinates (concatenated edge geometries).
    """

    edges: list[tuple[tuple[float, float], tuple[float, float]]]
    segment_ids: list[str]
    length_km: float
    geometry: list[tuple[float, float]]


def add_corridors(
    graph: nx.Graph,
    metros: list[MetroCluster],
    corridors_config: CorridorsConfig,
) -> None:
    """Register shortest simple paths for the union of metro k-nearest pairs.

    Distances use projected meters; equal-distance neighbors are ordered by
    coordinates and ID. Path geometry must match its road edge endpoints.
    """
    from itertools import islice

    if len(metros) < 2:
        raise ValueError("At least two metros are required for corridor discovery")
    coords = np.array([m.node_key for m in metros])
    tree = KDTree(coords)
    pairs = set()
    by_id = {m.metro_id: m for m in metros}
    if len(by_id) != len(metros) or len({m.node_key for m in metros}) != len(metros):
        raise ValueError("Metro IDs and coordinates must be unique")
    k = min(corridors_config.k_nearest, len(metros) - 1)
    for index, metro in enumerate(metros):
        distances, _ = tree.query(coords[index], k=k + 1)
        candidates = tree.query_ball_point(
            coords[index], np.nextafter(distances[-1], np.inf)
        )
        candidates = sorted(
            (j for j in candidates if j != index),
            key=lambda j: (
                float(np.linalg.norm(coords[index] - coords[j])),
                metros[j].node_key,
                metros[j].metro_id,
            ),
        )
        for neighbor in candidates[:k]:
            if (
                np.linalg.norm(coords[index] - coords[neighbor]) / 1000
                <= corridors_config.max_edge_km
            ):
                pairs.add(tuple(sorted((metro.metro_id, metros[neighbor].metro_id))))
    if not pairs:
        raise ValueError("No adjacent metro pairs found for corridor discovery")

    registry: dict[PathId, CorridorPath] = {}
    graph.graph["corridor_paths"] = registry
    for _, _, data in graph.edges(data=True):
        data.pop("corridor", None)
        data.pop("risk_groups", None)
    for a, b in sorted(pairs):
        try:
            paths = nx.shortest_simple_paths(
                graph, by_id[a].node_key, by_id[b].node_key, weight="length_km"
            )
            for path_index, nodes in enumerate(islice(paths, corridors_config.k_paths)):
                edges = list(zip(nodes, nodes[1:], strict=False))
                length = sum(float(graph[u][v]["length_km"]) for u, v in edges)
                if length > corridors_config.max_corridor_distance_km:
                    break  # Subsequent paths are at least as long.
                geometry = []
                segments = []
                pid = (a, b, path_index)
                for u, v in edges:
                    data = graph[u][v]
                    points = [tuple(point) for point in data["geometry"]]
                    if points[0] == v and points[-1] == u:
                        points.reverse()
                    if points[0] != u or points[-1] != v:
                        raise ValueError(
                            f"Road geometry does not match endpoints {u}, {v}"
                        )
                    geometry.extend(points if not geometry else points[1:])
                    if "segment_id" in data:
                        segments.append(data["segment_id"])
                    data.setdefault("corridor", []).append(
                        {
                            "metro_a": a,
                            "metro_b": b,
                            "path_index": path_index,
                            "distance_km": length,
                        }
                    )
                registry[pid] = CorridorPath(edges, segments, length, geometry)
        except nx.NetworkXNoPath:
            logger.info("No road path between metros %s and %s", a, b)
    if not registry:
        raise ValueError("No corridors found - corridor discovery failed")
    logger.info(
        "Discovered %d paths for %d adjacent metro pairs", len(registry), len(pairs)
    )


def corridor_risk_name(prefix: str, source: str, target: str, path_index: int) -> str:
    """Name one corridor path consistently in geography and scenario output."""
    first, second = sorted((source, target))
    name = f"{prefix}_{first}_{second}"
    return name if path_index == 0 else f"{name}_path{path_index}"


def assign_risk_groups(
    graph: nx.Graph, metros: list[MetroCluster], corridors_config: CorridorsConfig
) -> None:
    """Tag edges with a risk group for each corridor path they carry.

    When metro-radius exclusion is enabled, skip edges with either endpoint
    inside any metro circle.
    """
    if not corridors_config.risk_groups.enabled:
        logger.info("Risk group assignment disabled - skipping")
        return

    logger.info("Assigning risk groups to corridor edges")

    metro_points = {}
    metro_id_to_name = {}
    for metro in metros:
        metro_points[metro.metro_id] = {
            "center": Point(metro.centroid_x, metro.centroid_y),
            "radius_m": metro.radius_km * 1000.0,
        }
        metro_id_to_name[metro.metro_id] = metro.name

    corridor_counter = 0
    excluded_counter = 0
    assigned_counter = 0

    for u, v, edge_data in graph.edges(data=True):
        if "corridor" not in edge_data or not edge_data["corridor"]:
            continue

        corridor_counter += 1

        # Exclude edges with either endpoint in a metro circle.
        if corridors_config.risk_groups.exclude_metro_radius_shared:
            u_point = Point(u)
            v_point = Point(v)
            is_shared = False
            for _metro_id, metro_info in metro_points.items():
                center = metro_info["center"]
                radius_m = metro_info["radius_m"]
                if (
                    u_point.distance(center) <= radius_m
                    or v_point.distance(center) <= radius_m
                ):
                    is_shared = True
                    break
            if is_shared:
                excluded_counter += 1
                logger.debug(f"Excluding corridor edge {u}-{v} - within metro radius")
                continue

        for corridor_info in edge_data["corridor"]:
            metro_a_id = corridor_info["metro_a"]
            metro_b_id = corridor_info["metro_b"]
            path_index = corridor_info["path_index"]

            metro_a_name = metro_id_to_name[metro_a_id]
            metro_b_name = metro_id_to_name[metro_b_id]

            risk_group_name = corridor_risk_name(
                corridors_config.risk_groups.group_prefix,
                metro_a_name,
                metro_b_name,
                path_index,
            )

            if "risk_groups" not in edge_data:
                edge_data["risk_groups"] = []
            if risk_group_name not in edge_data["risk_groups"]:
                edge_data["risk_groups"].append(risk_group_name)
                assigned_counter += 1

    logger.info(
        f"Risk group assignment complete: Processed {corridor_counter} corridor edges, "
        f"Excluded {excluded_counter} within metro radius, Assigned {assigned_counter} risk group tags"
    )


def extract_corridor_graph(
    full_graph: nx.Graph, metros: list[MetroCluster]
) -> nx.MultiGraph:
    """Preserve every registered path and aggregate risks along its own edges."""
    registry: dict[PathId, CorridorPath] = full_graph.graph["corridor_paths"]
    if not registry:
        raise ValueError("Corridor path registry is empty")
    graph = nx.MultiGraph()
    for metro in metros:
        graph.add_node(
            metro.node_key,
            node_type="metro",
            metro_id=metro.metro_id,
            name=metro.name,
            name_orig=metro.name_orig,
            x=metro.centroid_x,
            y=metro.centroid_y,
            radius_km=metro.radius_km,
            uac_code=metro.uac_code,
            land_area_km2=metro.land_area_km2,
        )
    by_id = {metro.metro_id: metro.node_key for metro in metros}
    for (a, b, index), path in sorted(registry.items()):
        u, v = by_id[a], by_id[b]
        euclidean = Point(u).distance(Point(v)) / 1000
        risks = {
            risk
            for x, y in path.edges
            for risk in full_graph[x][y].get("risk_groups", [])
        }
        graph.add_edge(
            u,
            v,
            key=index,
            path_index=index,
            edge_type="corridor",
            length_km=path.length_km,
            metro_a=a,
            metro_b=b,
            euclidean_km=euclidean,
            detour_ratio=path.length_km / euclidean,
            geometry=path.geometry,
            contracted_segments=path.segment_ids,
            risk_groups=sorted(risks),
        )
    return graph


def validate_corridor_graph(
    corridor_graph: nx.Graph,
    metros: list[MetroCluster],
    validation_config: ValidationConfig,
) -> None:
    """Validate corridor-level graph connectivity and structure.

    Raises ValueError if required properties are not met.
    """
    logger.info("Validating corridor-level graph")

    if len(corridor_graph.nodes) != len(metros):
        raise ValueError(
            f"Corridor graph node count mismatch: {len(corridor_graph.nodes)} nodes vs {len(metros)} metros"
        )

    if len(corridor_graph.edges) == 0:
        raise ValueError(
            "Corridor graph has no edges - network is completely disconnected"
        )

    if not nx.is_connected(corridor_graph):
        components = list(nx.connected_components(corridor_graph))
        largest_component_size = max(len(c) for c in components)
        largest_component_fraction = largest_component_size / len(corridor_graph.nodes)
        logger.warning(
            f"Corridor graph is disconnected: {len(components)} components, "
            f"largest has {largest_component_size}/{len(corridor_graph.nodes)} metros "
            f"({largest_component_fraction:.1%})"
        )
        if validation_config.require_connected:
            raise ValueError(
                f"Corridor graph connectivity validation failed: Graph has {len(components)} disconnected components, "
                f"but require_connected=True. Largest component: {largest_component_size}/{len(corridor_graph.nodes)}"
            )
        if (
            largest_component_fraction
            < validation_config.min_largest_component_fraction
        ):
            raise ValueError(
                "Corridor graph connectivity validation failed: "
                f"Largest component fraction {largest_component_fraction:.1%} < "
                f"required minimum {validation_config.min_largest_component_fraction:.1%}"
            )
    else:
        logger.info("Corridor graph is connected")

    corridor_edges = 0
    total_distance = 0.0
    for _u, _v, data in corridor_graph.edges(data=True):
        if data.get("edge_type") == "corridor":
            corridor_edges += 1
            total_distance += data.get("length_km", 0.0)

    if corridor_edges != len(corridor_graph.edges):
        raise ValueError(
            f"Edge type validation failed: {corridor_edges} corridor edges vs {len(corridor_graph.edges)} total edges"
        )

    avg_distance = total_distance / corridor_edges if corridor_edges > 0 else 0.0
    logger.info(
        f"Corridor graph validation successful: {len(corridor_graph.nodes):,} metros, {corridor_edges:,} corridors, avg distance {avg_distance:.1f}km"
    )
