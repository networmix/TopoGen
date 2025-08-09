"""Corridor discovery, extraction, and validation.

Provides functions to:
- Discover corridor paths between metros and tag highway edges
- Assign risk groups to corridor-tagged edges
- Extract a metro-to-metro corridor graph
- Validate the resulting corridor-level graph

Intended to be used with a full integrated graph containing highway and metro
nodes. See ``topogen.integrated_graph`` for building the integrated graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy.spatial import KDTree  # type: ignore[import-untyped]
from shapely.geometry import Point

from topogen.log_config import get_logger
from topogen.metro_clusters import MetroCluster

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import CorridorsConfig, ValidationConfig

logger = get_logger(__name__)


def add_corridors(
    graph: nx.Graph,
    metros: list[MetroCluster],
    corridors_config: CorridorsConfig,
) -> None:
    """Add corridor tags to edges along metro-to-metro k-shortest paths.

    Computes k-shortest paths between adjacent metro pairs using the integrated
    graph that already contains metro nodes connected to their highway anchors.
    Paths therefore take the form: ``metro -> anchor -> ...highway... -> anchor -> metro``.
    The path length is the sum of ``length_km`` over all edges in the path. For
    ``metro_anchor`` edges, ``length_km`` is the Euclidean distance in kilometers.

    Args:
        graph: Integrated graph containing highway and metro nodes and edges.
        metros: List of metro clusters.
        corridors_config: Corridor discovery configuration.

    Raises:
        ValueError: If no adjacent metro pairs or no corridors are found.

    Notes:
        This function annotates edges in the full integrated graph with a
        ``corridor`` list of metadata entries. It does not create metro-to-metro
        edges. Use ``extract_corridor_graph`` to produce the corridor-level graph
        with metro nodes and corridor edges for downstream processing.
    """
    logger.info(f"Starting corridor discovery for {len(metros)} metros")
    logger.info(
        f"Corridor configuration: k_paths={corridors_config.k_paths}, "
        f"k_nearest={corridors_config.k_nearest}, max_edge_km={corridors_config.max_edge_km}km"
    )

    # Build adjacency using k-nearest neighbors
    if len(metros) < 2:
        raise ValueError("At least two metros are required for corridor discovery")

    metro_coords = np.array([[m.centroid_x, m.centroid_y] for m in metros])
    tree = KDTree(metro_coords)

    adjacent_pairs: list[tuple[str, str, float]] = []
    for metro in metros:
        # Find k nearest neighbors
        k_query = min(corridors_config.k_nearest + 1, len(metros))
        distances, indices = tree.query([metro.centroid_x, metro.centroid_y], k=k_query)

        # Handle numpy array results properly
        indices_list = (
            indices.tolist() if isinstance(indices, np.ndarray) else [indices]
        )
        distances_list = (
            distances.tolist() if isinstance(distances, np.ndarray) else [distances]
        )

        for j in range(1, min(len(indices_list), corridors_config.k_nearest + 1)):
            neighbor_idx = indices_list[j]
            distance_km = distances_list[j] / 1000.0

            if distance_km <= corridors_config.max_edge_km:
                # Add pair in sorted order to avoid duplicates
                metro_a_id = metro.metro_id
                metro_b_id = metros[neighbor_idx].metro_id
                if metro_a_id < metro_b_id:
                    adjacent_pairs.append((metro_a_id, metro_b_id, distance_km))

    logger.info(
        f"Found {len(adjacent_pairs)} adjacent metro pairs for corridor discovery"
    )

    if not adjacent_pairs:
        raise ValueError("No adjacent metro pairs found for corridor discovery")

    # Process corridors with progress reporting
    corridor_count = 0
    processed_pairs = 0
    skipped_pairs = 0
    successful_pairs = 0

    logger.info(f"Processing {len(adjacent_pairs)} metro pairs for corridor discovery")

    # Precompute metro_id -> node_key for path endpoints
    metro_id_to_node: dict[str, tuple[float, float]] = {
        m.metro_id: m.node_key for m in metros
    }

    for metro_a_id, metro_b_id, pair_distance in adjacent_pairs:
        processed_pairs += 1

        # Log progress every 10 pairs
        if processed_pairs % 10 == 0 or processed_pairs == len(adjacent_pairs):
            logger.info(
                f"Progress: {processed_pairs:,}/{len(adjacent_pairs):,} pairs processed"
            )

        if pair_distance > corridors_config.max_corridor_distance_km:
            skipped_pairs += 1
            logger.debug(
                f"Skipping {metro_a_id}-{metro_b_id}: {pair_distance:.1f}km > {corridors_config.max_corridor_distance_km}km"
            )
            continue

        # Endpoints are the metro nodes themselves
        if metro_a_id not in metro_id_to_node or metro_b_id not in metro_id_to_node:
            logger.warning(
                f"Skipping {metro_a_id}-{metro_b_id}: metro node(s) missing in mapping"
            )
            continue
        node_a = metro_id_to_node[metro_a_id]
        node_b = metro_id_to_node[metro_b_id]

        # Validate endpoints exist in the graph
        if not graph.has_node(node_a) or not graph.has_node(node_b):
            logger.warning(
                f"Skipping {metro_a_id}-{metro_b_id}: metro node(s) not present in graph"
            )
            continue

        logger.debug(
            f"Finding paths between {metro_a_id} and {metro_b_id} ({pair_distance:.1f}km)"
        )

        # Find k-shortest paths between anchors
        try:
            import itertools
            import time

            start_time = time.time()
            paths_generator = nx.shortest_simple_paths(
                graph, node_a, node_b, weight="length_km"
            )
            paths = list(itertools.islice(paths_generator, corridors_config.k_paths))
            elapsed = time.time() - start_time

            if not paths:
                logger.warning(f"No paths found between {metro_a_id} and {metro_b_id}")
                continue

            logger.debug(
                f"Found {len(paths)} paths between {metro_a_id}-{metro_b_id} in {elapsed:.2f}s"
            )

        except nx.NetworkXNoPath:
            logger.warning(
                f"No path found between metros {metro_a_id} and {metro_b_id}"
            )
            continue
        except Exception as e:  # pragma: no cover - defensive
            logger.error(
                f"Error finding paths between {metro_a_id} and {metro_b_id}: {e}"
            )
            continue

        # Tag edges in all paths using actual path distance (sum of edge lengths)
        successful_pairs += 1
        for path_idx, path in enumerate(paths):
            # Compute path length
            path_length_km = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if not graph.has_edge(u, v):
                    continue
                path_length_km += float(graph[u][v].get("length_km", 0.0))

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if graph.has_edge(u, v):
                    edge_data = graph[u][v]
                    if "corridor" not in edge_data:
                        edge_data["corridor"] = []
                    edge_data["corridor"].append(
                        {
                            "metro_a": metro_a_id,
                            "metro_b": metro_b_id,
                            "path_index": path_idx,
                            "distance_km": path_length_km,
                        }
                    )
                    corridor_count += 1

    # Final summary
    failed_pairs = processed_pairs - skipped_pairs - successful_pairs
    logger.info(
        "Corridor discovery complete: "
        f"Processed {processed_pairs:,}/{len(adjacent_pairs):,} pairs, "
        f"Successful {successful_pairs:,}, "
        f"Failed {failed_pairs:,}, "
        f"Skipped {skipped_pairs:,} (too far), "
        f"Tagged {corridor_count:,} highway edges with corridor labels"
    )

    if corridor_count == 0:
        raise ValueError("No corridors found - corridor discovery failed")


def assign_risk_groups(
    graph: nx.Graph, metros: list[MetroCluster], corridors_config: CorridorsConfig
) -> None:
    """Assign risk groups to corridor edges, avoiding shared metro radius segments.

    Creates unique risk groups for each corridor while excluding highway segments
    within a metro radius to avoid shared risk.
    """
    if not corridors_config.risk_groups.enabled:
        logger.info("Risk group assignment disabled - skipping")
        return

    logger.info("Assigning risk groups to corridor edges")

    # Build spatial index for metro areas and ID-to-name mapping
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

        # Exclude edges within any metro radius if configured
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

        # Assign risk groups to this corridor edge
        for corridor_info in edge_data["corridor"]:
            metro_a_id = corridor_info["metro_a"]
            metro_b_id = corridor_info["metro_b"]
            path_index = corridor_info["path_index"]

            metro_a_name = metro_id_to_name[metro_a_id]
            metro_b_name = metro_id_to_name[metro_b_id]

            if metro_a_name < metro_b_name:
                risk_group_name = f"{corridors_config.risk_groups.group_prefix}_{metro_a_name}_{metro_b_name}"
            else:
                risk_group_name = f"{corridors_config.risk_groups.group_prefix}_{metro_b_name}_{metro_a_name}"

            if path_index > 0:
                risk_group_name += f"_path{path_index}"

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
) -> nx.Graph:
    """Extract corridor-level graph from integrated highway graph.

    Creates a graph where:
    - Nodes are metro clusters
    - Edges represent corridor connections between metros
    - Edge weights are shortest path lengths through the highway network
      (derived from corridor metadata on highway edges).

    Edge attributes include:
    - ``length_km``: shortest path length between metro anchors
    - ``euclidean_km``: straight-line centroid distance (km)
    - ``detour_ratio``: ratio of length_km / euclidean_km
    - ``risk_groups``: list of risk group labels aggregated from highway edges
    """
    logger.info("Extracting corridor-level graph from integrated network")

    corridor_graph = nx.Graph()

    # Add metro nodes
    for metro in metros:
        corridor_graph.add_node(
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

    metro_id_to_coords = {metro.metro_id: metro.node_key for metro in metros}
    # Aggregate both min distance and risk-groups in a single pass over edges
    min_distance_by_pair: dict[tuple[str, str], float] = {}
    risks_by_pair: dict[tuple[str, str], set[str]] = {}

    for _u, _v, edge_data in full_graph.edges(data=True):
        corridors = edge_data.get("corridor")
        if not corridors:
            continue
        # Edge-level risk groups (if assigned)
        edge_risks = set(edge_data.get("risk_groups", []))
        for c in corridors:
            a_id = c["metro_a"]
            b_id = c["metro_b"]
            if a_id == b_id:
                continue
            key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            dist = float(c["distance_km"])
            prev = min_distance_by_pair.get(key)
            if prev is None or dist < prev:
                min_distance_by_pair[key] = dist
            if edge_risks:
                if key not in risks_by_pair:
                    risks_by_pair[key] = set()
                risks_by_pair[key].update(edge_risks)

    # Add corridor edges to graph
    edges_added = 0
    for (metro_a_id, metro_b_id), distance_km in min_distance_by_pair.items():
        if metro_a_id not in metro_id_to_coords or metro_b_id not in metro_id_to_coords:
            continue
        node_a = metro_id_to_coords[metro_a_id]
        node_b = metro_id_to_coords[metro_b_id]

        # Compute Euclidean distance and detour ratio
        euclidean_km = Point(node_a).distance(Point(node_b)) / 1000.0
        detour_ratio = (distance_km / euclidean_km) if euclidean_km > 0 else None

        corridor_graph.add_edge(
            node_a,
            node_b,
            edge_type="corridor",
            length_km=distance_km,
            metro_a=metro_a_id,
            metro_b=metro_b_id,
            euclidean_km=euclidean_km,
            detour_ratio=detour_ratio,
            risk_groups=sorted(risks_by_pair.get((metro_a_id, metro_b_id), set())),
        )
        edges_added += 1

    logger.info(
        f"Extracted corridor graph: {len(corridor_graph.nodes)} metro nodes, {edges_added} corridor edges"
    )

    if edges_added == 0:
        raise ValueError("Corridor graph extraction failed: no corridor edges found")

    return corridor_graph


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
