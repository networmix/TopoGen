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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from scipy.spatial import KDTree  # type: ignore[import-untyped]
from shapely.geometry import Point

from topogen.log_config import get_logger
from topogen.metro_clusters import MetroCluster

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import CorridorsConfig, ValidationConfig

logger = get_logger(__name__)


# Public typed registry structures
PathId = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class CorridorPath:
    """Concrete corridor path details connecting two metros.

    Attributes:
        metros: Sorted pair of metro IDs (metro_a, metro_b).
        path_index: K-shortest path index.
        nodes: Ordered node coordinates along the path (including metros and anchors).
        edges: Ordered endpoint pairs along the path.
        segment_ids: Ordered contracted highway segment IDs (excludes anchor edges).
        length_km: Total length of path in kilometers.
        geometry: Ordered polyline coordinates (concatenated edge geometries).
    """

    metros: tuple[str, str]
    path_index: int
    nodes: list[tuple[float, float]]
    edges: list[tuple[tuple[float, float], tuple[float, float]]]
    segment_ids: list[str]
    length_km: float
    geometry: list[tuple[float, float]]


def add_corridors(
    graph: nx.Graph,
    metros: list[MetroCluster],
    corridors_config: CorridorsConfig,
) -> None:
    """Add corridor tags to edges along metro-to-metro k-shortest paths.

    Computes k-shortest paths between adjacent metro pairs using the integrated
    graph that already contains metro nodes connected to their highway anchors.
    Paths take the form ``metro → anchor → ...highway... → anchor → metro``.
    The path length is the sum of ``length_km`` over all edges in the path.
    ``metro_anchor`` edges use Euclidean length in kilometers.

    Args:
        graph: Integrated graph containing highway and metro nodes and edges.
        metros: List of metro clusters.
        corridors_config: Corridor discovery configuration.

    Raises:
        ValueError: If no adjacent metro pairs or no corridors are found.

    Notes:
        - Adjacency is built via k-nearest neighbors on metro centroids and
          filtered by ``max_edge_km`` (Euclidean km).
        - ``max_corridor_distance_km`` is enforced on the actual path length over
          the graph (sum of per-edge ``length_km``), not on Euclidean separation.
        - Each path edge is tagged with a ``corridor`` entry and a membership set
          ``corridor_path_ids``. Use ``extract_corridor_graph`` to build a
          metro-to-metro corridor graph for downstream use.
    """
    logger.info(f"Starting corridor discovery for {len(metros)} metros")
    logger.info(
        f"Corridor configuration: k_paths={corridors_config.k_paths}, "
        f"k_nearest={corridors_config.k_nearest}, "
        f"max_edge_km={corridors_config.max_edge_km}km, "
        f"max_corridor_distance_km={corridors_config.max_corridor_distance_km}km"
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

    # Initialize path registry on the graph
    if "corridor_paths" not in graph.graph:
        graph.graph["corridor_paths"] = {}

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

        # Do not apply max_corridor_distance_km to Euclidean separation.
        # The threshold is enforced below using the actual path length along the graph.

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

        # Find k-shortest paths between metros (via anchors/highways)
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
                f"Found {len(paths)} candidate paths between {metro_a_id}-{metro_b_id} in {elapsed:.2f}s"
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

        # Compute and tag only paths that satisfy the path-length threshold
        added_any_path_for_pair = False
        too_long_paths = 0
        for path_idx, path in enumerate(paths):
            # Compute path length
            path_length_km = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if not graph.has_edge(u, v):
                    continue
                path_length_km += float(graph[u][v].get("length_km", 0.0))

            # Enforce max_corridor_distance_km using actual path length over multi-edge fiber
            if path_length_km > corridors_config.max_corridor_distance_km:
                too_long_paths += 1
                logger.debug(
                    f"Skipping path {path_idx} for {metro_a_id}-{metro_b_id}: "
                    f"{path_length_km:.1f}km > {corridors_config.max_corridor_distance_km}km"
                )
                continue

            # Build ordered edge list
            ordered_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
            for i in range(len(path) - 1):
                ordered_edges.append((path[i], path[i + 1]))

            # Collect segment IDs from contracted highway edges only
            segment_ids: list[str] = []
            for u, v in ordered_edges:
                if not graph.has_edge(u, v):
                    continue
                seg = graph[u][v].get("segment_id")
                if seg:
                    segment_ids.append(str(seg))

            # Build merged geometry by concatenating per-edge geometry
            merged_geometry: list[tuple[float, float]] = []

            for u, v in ordered_edges:
                if not graph.has_edge(u, v):
                    continue
                ed = graph[u][v]
                geom = ed.get("geometry")
                if isinstance(geom, list) and geom:
                    # Normalize direction to match u->v when possible
                    g0 = tuple(geom[0])
                    g1 = tuple(geom[-1])
                    if g0 == u and g1 == v:
                        coords: list[tuple[float, float]] = [
                            (float(p[0]), float(p[1])) for p in geom
                        ]
                    elif g0 == v and g1 == u:
                        coords = [(float(p[0]), float(p[1])) for p in reversed(geom)]
                    else:
                        # Fallback to endpoints if geometry endpoints don't align
                        coords = [
                            (float(u[0]), float(u[1])),
                            (float(v[0]), float(v[1])),
                        ]
                else:
                    coords = [
                        (float(u[0]), float(u[1])),
                        (float(v[0]), float(v[1])),
                    ]

                if not merged_geometry:
                    merged_geometry.extend(coords)
                else:
                    # Drop duplicate shared vertex when appending
                    if merged_geometry[-1] == coords[0]:
                        merged_geometry.extend(coords[1:])
                    else:
                        merged_geometry.extend(coords)

            # Create registry entry
            m_sorted = (
                (metro_a_id, metro_b_id)
                if metro_a_id < metro_b_id
                else (
                    metro_b_id,
                    metro_a_id,
                )
            )
            path_id: PathId = (m_sorted[0], m_sorted[1], path_idx)

            nodes_seq: list[tuple[float, float]] = [
                (float(n[0]), float(n[1])) for n in path
            ]
            edges_seq: list[tuple[tuple[float, float], tuple[float, float]]] = [
                ((float(u[0]), float(u[1])), (float(v[0]), float(v[1])))
                for (u, v) in ordered_edges
            ]

            cp = CorridorPath(
                metros=m_sorted,
                path_index=path_idx,
                nodes=nodes_seq,
                edges=edges_seq,
                segment_ids=segment_ids,
                length_km=path_length_km,
                geometry=merged_geometry,
            )
            graph.graph["corridor_paths"][path_id] = cp

            # Tag each path edge with edge-scoped metadata for migration + path ids
            for u, v in ordered_edges:
                if not graph.has_edge(u, v):
                    continue
                edge_data = graph[u][v]
                # Tag retained for migration visibility
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
                # New explicit path membership set for risk aggregation
                if "corridor_path_ids" not in edge_data:
                    edge_data["corridor_path_ids"] = set()
                try:
                    edge_data["corridor_path_ids"].add(path_id)
                except Exception:
                    s = edge_data.get("corridor_path_ids")
                    edge_data["corridor_path_ids"] = (
                        set(s) if isinstance(s, list) else set()
                    )
                    edge_data["corridor_path_ids"].add(path_id)
                corridor_count += 1
                added_any_path_for_pair = True

        if added_any_path_for_pair:
            successful_pairs += 1
        else:
            skipped_pairs += 1
            if too_long_paths > 0:
                logger.debug(
                    f"Skipping {metro_a_id}-{metro_b_id}: all {too_long_paths} candidate paths exceed max_corridor_distance_km"
                )

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
    """Extract corridor-level graph from registry of concrete paths.

    Uses ``full_graph.graph["corridor_paths"]`` as the single source of truth.
    """
    logger.info("Extracting corridor-level graph from corridor path registry")

    registry = full_graph.graph.get("corridor_paths")
    if not registry:
        raise ValueError(
            "Corridor path registry missing: graph.graph['corridor_paths'] not found"
        )

    # Normalize registry iteration
    items = registry.items() if isinstance(registry, dict) else []

    # Choose shortest path per metro pair
    best: dict[tuple[str, str], tuple[PathId, CorridorPath]] = {}
    for pid, cp in items:
        metros_sorted = tuple(sorted(cp.metros))
        key = (metros_sorted[0], metros_sorted[1])
        prev = best.get(key)
        if (
            prev is None
            or (cp.length_km < prev[1].length_km)
            or (cp.length_km == prev[1].length_km and pid[2] < prev[0][2])
        ):
            best[key] = (pid, cp)  # type: ignore[assignment]

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

    # Aggregate risk groups only from edges that include the chosen path id
    risks_by_pair: dict[tuple[str, str], set[str]] = {}
    for (metro_a_id, metro_b_id), (pid, _cp) in best.items():
        key = (metro_a_id, metro_b_id)
        risks: set[str] = set()
        for _u, _v, ed in full_graph.edges(data=True):
            pids = ed.get("corridor_path_ids")
            if not pids:
                continue
            # Normalize set/list
            if isinstance(pids, set):
                has = pid in pids
            else:
                try:
                    has = pid in set(pids)
                except Exception:
                    has = False
            if has:
                for rg in ed.get("risk_groups", []) or []:
                    risks.add(str(rg))
        risks_by_pair[key] = risks

    # Emit edges
    edges_added = 0
    for (metro_a_id, metro_b_id), (_pid, cp) in best.items():
        if metro_a_id not in metro_id_to_coords or metro_b_id not in metro_id_to_coords:
            continue
        node_a = metro_id_to_coords[metro_a_id]
        node_b = metro_id_to_coords[metro_b_id]

        euclidean_km = Point(node_a).distance(Point(node_b)) / 1000.0
        detour_ratio = (cp.length_km / euclidean_km) if euclidean_km > 0 else None

        corridor_graph.add_edge(
            node_a,
            node_b,
            edge_type="corridor",
            length_km=cp.length_km,
            metro_a=metro_a_id,
            metro_b=metro_b_id,
            euclidean_km=euclidean_km,
            detour_ratio=detour_ratio,
            geometry=cp.geometry,
            contracted_segments=cp.segment_ids,
            risk_groups=sorted(risks_by_pair.get((metro_a_id, metro_b_id), set())),
        )
        edges_added += 1

    logger.info(
        f"Extracted corridor graph: {len(corridor_graph.nodes):,} metro nodes, {edges_added} corridor edges"
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


def extract_corridor_edges_for_metros_graph(graph: nx.Graph) -> list[dict[str, Any]]:
    """Extract simplified metro-to-metro corridor edges for downstream pipelines.

    Returns list entries with:
    - source: metro node key
    - target: metro node key
    - length_km: corridor path length in km
    - edge_type: 'corridor'
    - risk_groups: list[str]
    - capacity: optional capacity if already present
    """
    edges: list[dict[str, Any]] = []
    for u, v, data in graph.edges(data=True):
        src = graph.nodes[u]
        tgt = graph.nodes[v]
        if src.get("node_type") in {"metro", "metro+highway"} and tgt.get(
            "node_type"
        ) in {"metro", "metro+highway"}:
            entry: dict[str, Any] = {
                "source": u,
                "target": v,
                "length_km": data.get("length_km", 0.0),
                "edge_type": data.get("edge_type", "corridor"),
                "risk_groups": data.get("risk_groups", []),
            }
            if "capacity" in data:
                entry["capacity"] = data["capacity"]
            edges.append(entry)
    return edges
