"""Integrated metro and highway graph construction and management.

This module assembles a highway network and metropolitan clusters into two
graphs used by downstream steps:

1. A full integrated graph (internal) containing the highway backbone, metro
   nodes, and metro anchor edges. Highway edges along k-shortest inter-metro
   paths are tagged with corridor metadata.
2. A corridor-level graph (returned) where nodes are metros and edges are
   metro-to-metro corridors. Edge attributes include `length_km` (sum of
   highway edge lengths along the chosen path), `euclidean_km` (straight-line
   centroid separation), and `detour_ratio` (length_km / euclidean_km).

Note:
- The public ``build_integrated_graph`` function returns the corridor-level
  graph by design because the scenario builder consumes metro-to-metro
  corridors directly. If callers need the full integrated graph, they should
  modify the pipeline to capture it before corridor extraction or call
  ``extract_corridor_graph`` themselves.

Saves to JSON with tuple encoding/decoding via ``save_to_json`` / ``load_from_json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from scipy.spatial import KDTree  # type: ignore[import-untyped]
from shapely.geometry import Point

from topogen.corridors import (
    CorridorPath,  # for JSON (type hints only)
    add_corridors,
    assign_risk_groups,
    extract_corridor_graph,
    validate_corridor_graph,
)
from topogen.highway_graph import build_highway_graph
from topogen.log_config import get_logger
from topogen.metro_clusters import MetroCluster, load_metro_clusters

if TYPE_CHECKING:
    from topogen.config import (
        FormattingConfig,
        TopologyConfig,
        ValidationConfig,
    )

logger = get_logger(__name__)


def _contract_degree2_chains(
    G: nx.Graph, protected_nodes: set[tuple[float, float]] | None = None
) -> nx.Graph:
    """Contract maximal degree-2 chains while preserving protected nodes.

    Keeps every junction (deg != 2), any end-points of dangling chains,
    and all protected nodes (such as metro anchors).
    Aggregates edge length into 'length_km' and stores the traversed
    coordinate list in 'geometry'.

    Args:
        G: Input graph with intersection-level detail.
        protected_nodes: Set of node coordinates that must never be removed.

    Returns:
        Contracted graph with only junctions and protected nodes as vertices.
    """
    logger.info("Contracting degree-2 chains")

    if protected_nodes is None:
        protected_nodes = set()

    if protected_nodes:
        logger.info(
            f"Protecting {len(protected_nodes)} nodes from contraction (metro anchors)"
        )

    # Enhanced logging: analyze input graph structure
    components_before = list(nx.connected_components(G))
    logger.debug(
        f"Input graph: {len(G.nodes)} nodes, {len(G.edges)} edges, {len(components_before)} components"
    )
    if len(components_before) > 1:
        component_sizes = sorted([len(c) for c in components_before], reverse=True)
        logger.debug(f"Component sizes before contraction: {component_sizes[:10]}")

    # Degree analysis
    degree_counts = {}
    for node in G.nodes():
        deg = len(list(G.neighbors(node)))
        degree_counts[deg] = degree_counts.get(deg, 0) + 1
    logger.debug(f"Degree distribution: {dict(sorted(degree_counts.items()))}")

    contracted = nx.Graph()
    visited = set()
    processed_nodes = set()  # Track all nodes that were part of contracted paths

    def _segment_id(u: tuple[float, float], v: tuple[float, float]) -> str:
        """Return deterministic segment id based on sorted integerized endpoints.

        Endpoints are snapped to a grid in upstream processing, so integerization
        via round() is stable.
        """
        (ux, uy) = (int(round(u[0])), int(round(u[1])))
        (vx, vy) = (int(round(v[0])), int(round(v[1])))
        a = (ux, uy)
        b = (vx, vy)
        a, b = (a, b) if a <= b else (b, a)
        return f"{a[0]}_{a[1]}__{b[0]}_{b[1]}"

    def is_contractible(node):
        """Check if a node can be contracted (degree-2 and not protected)."""
        return len(list(G.neighbors(node))) == 2 and node not in protected_nodes

    chains_contracted = 0
    for node in G.nodes():
        if not is_contractible(node):  # a junction, dead-end, or protected node
            for nbr in G.neighbors(node):
                key = tuple(sorted((node, nbr)))
                if key in visited:
                    continue

                path = [node]  # Start with just the starting junction/protected node
                length = G.edges[node, nbr]["length_km"]
                visited.add(key)

                prev, curr = node, nbr
                # Add intermediate nodes including nbr
                path.append(curr)

                while is_contractible(curr):
                    nxt = next(n for n in G.neighbors(curr) if n != prev)
                    length += G.edges[curr, nxt]["length_km"]
                    visited.add(tuple(sorted((curr, nxt))))
                    path.append(nxt)
                    prev, curr = curr, nxt

                # path now contains [start_junction, intermediate_nodes..., end_junction] without duplication
                contracted.add_edge(
                    path[0],
                    path[-1],
                    length_km=length,
                    geometry=path,
                    segment_id=_segment_id(path[0], path[-1]),
                )
                # Track all nodes in this path as processed
                processed_nodes.update(path)
                chains_contracted += 1

                # Log very long chains that might span critical connections
                if len(path) > 100:
                    logger.debug(
                        f"Long chain contracted: {len(path)} nodes, {length:.1f}km from {path[0]} to {path[-1]}"
                    )

    logger.debug(
        f"Phase 1: Contracted {chains_contracted} chains connected to junctions"
    )

    # Second pass: handle isolated degree-2 cycles (rings)
    # Find all nodes that weren't included in any contracted path
    remaining_nodes = set(G.nodes()) - processed_nodes

    processed_cycles = set()
    for node in remaining_nodes:
        if node in processed_cycles or len(list(G.neighbors(node))) != 2:
            continue

        # This node is part of an isolated degree-2 cycle
        # Walk around the entire cycle to collect all nodes and total length
        cycle_nodes = []
        cycle_length = 0.0
        current = node
        prev = None

        while True:
            cycle_nodes.append(current)
            processed_cycles.add(current)

            # Find next node in the cycle
            neighbors = list(G.neighbors(current))
            if len(neighbors) != 2:
                # Node no longer has exactly 2 neighbors, break to avoid infinite loop
                break
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]

            # Add edge length
            cycle_length += G.edges[current, next_node]["length_km"]

            # Move to next node
            prev = current
            current = next_node

            # Stop when we complete the cycle
            if current == node:
                break

        # Contract the entire cycle into a single edge
        if len(cycle_nodes) >= 3:  # Only contract cycles with 3+ nodes
            # Create a single edge representing the cycle
            # Use first and "middle" node as endpoints to avoid self-loops
            start_node = cycle_nodes[0]
            mid_node = cycle_nodes[len(cycle_nodes) // 2]

            # Avoid duplicate edge insertion when start_node equals mid_node
            if start_node != mid_node:
                contracted.add_edge(
                    start_node,
                    mid_node,
                    length_km=cycle_length,
                    geometry=cycle_nodes + [cycle_nodes[0]],
                    segment_id=_segment_id(start_node, mid_node),
                )  # Close the loop

    if contracted.number_of_edges() == 0:
        raise ValueError("Graph contraction produced empty result")

    # Enhanced logging: analyze output graph structure
    components_after = list(nx.connected_components(contracted))
    logger.debug(
        f"Output graph: {len(contracted.nodes)} nodes, {len(contracted.edges)} edges, {len(components_after)} components"
    )

    if len(components_after) > 1:
        component_sizes_after = sorted([len(c) for c in components_after], reverse=True)
        logger.debug(f"Component sizes after contraction: {component_sizes_after[:10]}")

        # Check if number of components changed
        if len(components_after) != len(components_before):
            logger.warning(
                f"Component count changed during contraction: {len(components_before)} → {len(components_after)}"
            )

    # Summary
    nodes_removed = G.number_of_nodes() - contracted.number_of_nodes()
    original_nodes_count = G.number_of_nodes()
    if original_nodes_count > 0:
        percentage = nodes_removed / original_nodes_count * 100
        logger.debug(
            f"Contraction summary: removed {nodes_removed} degree-2 nodes ({percentage:.1f}%)"
        )
    else:
        logger.debug("Contraction summary: no nodes to process")

    logger.info(
        f"Contracted graph: {contracted.number_of_nodes():,} nodes, {contracted.number_of_edges():,} edges"
    )
    return contracted


def _remove_slivers(
    G: nx.Graph, min_length_km: float, validation_config: ValidationConfig
) -> nx.Graph:
    """Remove edges shorter than minimum length threshold.

    Args:
        G: Input graph.
        min_length_km: Minimum edge length to keep.
        validation_config: Validation parameters for fragmentation checks.

    Returns:
        Graph with short edges removed.

    Raises:
        ValueError: If sliver removal fragments the network excessively.
    """
    logger.info(
        f"Removing edges shorter than {min_length_km}km (sliver removal threshold)"
    )

    # Check initial connectivity
    initial_connected = nx.is_connected(G)
    initial_nodes = len(G.nodes)
    initial_edges = len(G.edges)

    logger.info(
        f"Pre-sliver removal: {initial_nodes:,} nodes, {initial_edges:,} edges, "
        f"{'connected' if initial_connected else 'disconnected'}"
    )

    edges_to_remove = [
        (u, v) for u, v, data in G.edges(data=True) if data["length_km"] < min_length_km
    ]

    G_clean = G.copy()
    G_clean.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    isolated_nodes = [
        node for node in G_clean.nodes() if len(list(G_clean.neighbors(node))) == 0
    ]
    G_clean.remove_nodes_from(isolated_nodes)

    nodes_remaining = len(G_clean.nodes)
    edges_remaining = len(G_clean.edges)

    logger.info(
        f"Removed {len(edges_to_remove):,} short edges and {len(isolated_nodes):,} isolated nodes"
    )
    logger.info(
        f"Post-sliver removal: {nodes_remaining:,} nodes ({nodes_remaining / initial_nodes:.1%} kept), "
        f"{edges_remaining:,} edges ({edges_remaining / initial_edges:.1%} kept)"
    )

    if G_clean.number_of_nodes() == 0:
        raise ValueError("No nodes remain after sliver removal")

    # Critical validation: Check for network fragmentation
    components = list(nx.connected_components(G_clean))
    num_components = len(components)

    if num_components > 1:
        component_sizes = sorted([len(c) for c in components], reverse=True)
        largest_component_size = component_sizes[0]
        largest_component_fraction = largest_component_size / initial_nodes

        # Only blame sliver removal for fragmentation if edges were actually removed
        if len(edges_to_remove) > 0:
            nodes_lost_to_fragmentation = initial_nodes - largest_component_size
            logger.warning(
                f"🚨 SLIVER REMOVAL CAUSED FRAGMENTATION: {num_components} components created! "
                f"Largest: {largest_component_size:,}/{initial_nodes:,} nodes ({largest_component_fraction:.1%}). "
                f"Lost {nodes_lost_to_fragmentation:,} nodes to disconnected fragments."
            )

            # Log component sizes for visibility
            if len(component_sizes) > 1:
                other_components = component_sizes[
                    1:6
                ]  # Show up to 5 smaller components
                logger.warning(f"Other component sizes: {other_components}")
        else:
            logger.info(
                f"Network has {num_components} pre-existing components (no edges removed). "
                f"Largest: {largest_component_size:,} nodes ({largest_component_fraction:.1%} of original)"
            )

        # Error if we lose more than configured fraction of the network to fragmentation
        if (
            largest_component_fraction
            < validation_config.min_largest_component_fraction
        ):
            raise ValueError(
                f"Sliver removal threshold ({min_length_km}km) too aggressive: "
                f"largest component only {largest_component_fraction:.1%} of original network. "
                f"Consider reducing min_edge_length_km or using 0.0 to disable sliver removal."
            )

    else:
        # Log success case for connected networks
        if initial_connected:
            logger.info("Sliver removal preserved network connectivity")
        else:
            logger.info(
                f"Sliver removal complete - network remains connected ({nodes_remaining:,} nodes)"
            )

    return G_clean


def _keep_largest_component(
    G: nx.Graph, validation_config: ValidationConfig
) -> nx.Graph:
    """Keep only the largest connected component.

    Args:
        G: Input graph potentially with multiple components.
        validation_config: Validation parameters for component size checks.

    Returns:
        Subgraph containing only the largest connected component.

    Raises:
        ValueError: If graph has no connected components or largest component is too small.
    """
    if nx.is_connected(G):
        logger.info("Graph is already connected")
        return G

    components = list(nx.connected_components(G))
    if not components:
        raise ValueError("Graph has no connected components")

    component_sizes = sorted([len(c) for c in components], reverse=True)
    total_nodes = len(G.nodes)
    largest_size = component_sizes[0]
    largest_fraction = largest_size / total_nodes

    logger.debug(
        f"Graph has {len(components)} components, sizes: {component_sizes[:5]}"
    )

    # Warn about significant data loss
    nodes_lost = total_nodes - largest_size
    if nodes_lost > 0:
        logger.warning(
            f"Keeping largest component will discard {nodes_lost:,} nodes "
            f"({(nodes_lost / total_nodes):.1%} of network)"
        )

    # Log warning if largest component is suspiciously small
    if largest_fraction < 0.1:  # Fixed threshold - if largest component < 10%
        logger.warning(
            f"Largest component only {largest_fraction:.1%} of network "
            f"({largest_size:,}/{total_nodes:,} nodes). "
            f"This suggests severe network fragmentation."
        )

    largest_component = max(components, key=len)
    G_main = G.subgraph(largest_component).copy()

    logger.info(f"Kept largest component: {len(G_main.nodes):,} nodes")
    return G_main


def anchor_metros(
    metros: list[MetroCluster],
    highway_graph: nx.Graph,
    validation_config: ValidationConfig,
) -> dict[str, tuple[float, float]]:
    """Find nearest highway node for each metro cluster.

    Args:
        metros: List of metro clusters to anchor.
        highway_graph: Highway graph with (x, y) tuple node keys.
        validation_config: Validation configuration containing max distance.

    Returns:
        Dict mapping metro_id to nearest highway node (x, y) tuple.

    Raises:
        ValueError: If any metro is farther than max allowed distance from highway network.
    """
    # Build KDTree from highway node coordinates
    highway_coords = np.array([list(node) for node in highway_graph.nodes()])
    highway_nodes = list(highway_graph.nodes())
    tree = KDTree(highway_coords)

    # Log spatial debugging info
    highway_x_range = (highway_coords[:, 0].min(), highway_coords[:, 0].max())
    highway_y_range = (highway_coords[:, 1].min(), highway_coords[:, 1].max())
    logger.info(
        f"Highway network spatial coverage: "
        f"X: [{highway_x_range[0]:.0f}, {highway_x_range[1]:.0f}], "
        f"Y: [{highway_y_range[0]:.0f}, {highway_y_range[1]:.0f}]"
    )

    anchors = {}
    metro_distances = []  # Track all distances for analysis

    for metro in metros:
        metro_coords = np.array([metro.centroid_x, metro.centroid_y])

        # Find nearest highway node
        distance, idx = tree.query(metro_coords)
        distance_km = distance / 1000.0  # Convert meters to km
        metro_distances.append((metro.name, distance_km))

        if distance_km > validation_config.max_metro_highway_distance_km:
            nearest_node = highway_nodes[idx]
            logger.error(
                f"Metro anchoring failed for {metro.name} (ID {metro.metro_id}): "
                f"Metro coords: ({metro.centroid_x:.0f}, {metro.centroid_y:.0f}), "
                f"Nearest highway node: ({nearest_node[0]:.0f}, {nearest_node[1]:.0f}), "
                f"Distance: {distance_km:.1f}km (max: {validation_config.max_metro_highway_distance_km}km)"
            )

            # Log distance statistics before failing
            sorted_distances = sorted(metro_distances, key=lambda x: x[1])
            logger.info("Metro-highway distances so far:")
            for name, dist in sorted_distances[:10]:  # Show first 10
                logger.info(f"  {name}: {dist:.1f}km")
            if len(sorted_distances) > 10:
                logger.info(f"  ... and {len(sorted_distances) - 10} more metros")

            raise ValueError(
                f"Metro {metro.name} (ID {metro.metro_id}) is {distance_km:.1f}km "
                f"from nearest highway node (max: {validation_config.max_metro_highway_distance_km}km)"
            )

        nearest_node = highway_nodes[idx]
        anchors[metro.metro_id] = nearest_node

        logger.debug(
            f"Anchored metro {metro.name} ({metro.metro_id}) to highway node {nearest_node} "
            f"at {distance_km:.2f}km"
        )

    # Log final distance summary
    sorted_distances = sorted(metro_distances, key=lambda x: x[1])
    avg_distance = sum(dist for _, dist in sorted_distances) / len(sorted_distances)
    max_distance = max(dist for _, dist in sorted_distances)

    logger.info(f"Successfully anchored {len(anchors)} metros to highway network")
    logger.info(
        f"Anchoring distances - Avg: {avg_distance:.1f}km, Max: {max_distance:.1f}km"
    )

    # Detailed distances at DEBUG level
    logger.debug("Metro-highway anchoring distances:")
    for name, dist in sorted_distances:
        logger.debug(f"  {name}: {dist:.1f}km")
    return anchors


def build_integrated_graph(config: TopologyConfig) -> nx.Graph:
    """Build integrated metro and highway graph and return corridor graph.

    Args:
        config: Complete topology configuration object.

    Returns:
        Corridor-level NetworkX graph whose nodes are metros and whose edges
        represent metro-to-metro corridors. This is the graph expected by the
        scenario builder.

    Raises:
        ValueError: If integration fails at any step.

    Notes:
        The full integrated highway + metro graph is an intermediate product.
        If you need that graph, capture it prior to calling
        ``extract_corridor_graph`` or adapt this function to return both.
    """
    logger.info("Building integrated metro and highway graph")

    # Step 1: Load metro clusters
    metros = load_metro_clusters(
        uac_path=config.data_sources.uac_polygons,
        k=config.clustering.metro_clusters,
        target_crs=config.projection.target_crs,
        clustering_config=config.clustering,
        formatting_config=config.output.formatting,
        conus_boundary_path=config.data_sources.conus_boundary,
    )
    logger.info(f"Loaded {len(metros)} metro clusters")

    # Step 2: Build highway graph
    highway_graph = build_highway_graph(
        tiger_zip=config.data_sources.tiger_roads,
        target_crs=config.projection.target_crs,
        highway_config=config.highway_processing,
        validation_config=config.validation,
    )
    logger.info(
        f"Built highway graph: {len(highway_graph.nodes):,} nodes, {len(highway_graph.edges):,} edges"
    )

    # Step 3: Anchor metros to highway network
    anchors = anchor_metros(metros, highway_graph, config.validation)

    # Step 4: Contract highway graph (now that metros are anchored)
    logger.info("Contracting highway graph")
    # Protect metro anchor nodes from being removed during contraction
    anchor_nodes = set(anchors.values())
    highway_contracted = _contract_degree2_chains(
        highway_graph, protected_nodes=anchor_nodes
    )

    # Step 5: Remove slivers (after contraction for better accuracy)
    highway_clean = _remove_slivers(
        highway_contracted,
        config.highway_processing.min_edge_length_km,
        config.validation,
    )

    # Step 5.5: Optional component filtering
    if config.highway_processing.filter_largest_component:
        highway_final = _keep_largest_component(highway_clean, config.validation)
    else:
        highway_final = highway_clean
        logger.info("Component filtering disabled - keeping all highway components")

    # Step 6: Verify metro anchors still exist in contracted graph
    logger.info("Verifying metro anchors after contraction")

    # With protected nodes, all anchors should be preserved
    missing_anchors = []
    for metro in metros:
        anchor = anchors[metro.metro_id]
        if not highway_final.has_node(anchor):
            missing_anchors.append((metro.name, anchor))

    if missing_anchors:
        anchor_list = ", ".join(
            [f"{name} ({anchor})" for name, anchor in missing_anchors]
        )
        raise RuntimeError(
            f"BUG: Protected anchor nodes were removed during contraction: {anchor_list}. "
            f"This should never happen with anchor protection enabled."
        )

    logger.info(f"All {len(metros)} metro anchors preserved during contraction")

    # Step 7: Build integrated graph
    logger.info("Building integrated graph")
    G = highway_final.copy()

    # Add metro nodes and anchor edges
    for metro in metros:
        key = metro.node_key  # (x, y) tuple
        anchor = anchors[metro.metro_id]

        # Add metro node (merge if collision)
        if G.has_node(key):
            # Merge full metro attributes into existing highway node
            G.nodes[key]["node_type"] = "metro+highway"
            G.nodes[key]["name"] = metro.name
            G.nodes[key]["name_orig"] = metro.name_orig
            G.nodes[key]["radius_km"] = metro.radius_km
            G.nodes[key]["metro_id"] = metro.metro_id
            G.nodes[key]["x"] = metro.centroid_x
            G.nodes[key]["y"] = metro.centroid_y
            G.nodes[key]["uac_code"] = metro.uac_code
            G.nodes[key]["land_area_km2"] = metro.land_area_km2
        else:
            G.add_node(
                key,
                node_type="metro",
                x=metro.centroid_x,
                y=metro.centroid_y,
                name=metro.name,
                name_orig=metro.name_orig,
                radius_km=metro.radius_km,
                metro_id=metro.metro_id,
                uac_code=metro.uac_code,
                land_area_km2=metro.land_area_km2,
            )

        # Add anchor edge
        dist_km = Point(metro.coordinates).distance(Point(anchor)) / 1000.0
        G.add_edge(
            key,
            anchor,
            edge_type="metro_anchor",
            length_km=dist_km,
            geometry=[key, anchor],  # straight line from metro centroid to anchor
        )

    logger.info(f"Added {len(metros)} metro nodes with anchor connections")

    # Step 8: Add corridor tags (paths from metro to metro via anchors and highways)
    logger.info("Discovering corridors")
    add_corridors(G, metros, config.corridors)

    # Step 8.5: Assign risk groups to corridor edges
    assign_risk_groups(G, metros, config.corridors)

    # Step 9: Validate integration
    validate_integrated_graph(G, metros, config.validation)

    # Step 10: Extract corridor-level graph
    corridor_graph = extract_corridor_graph(G, metros)

    # Step 11: Validate corridor connectivity
    validate_corridor_graph(corridor_graph, metros, config.validation)

    # Step 12: Export visualization if requested
    if config.clustering.export_integrated_graph:
        logger.info("Exporting corridor graph visualization")
        try:
            from topogen.visualization import export_integrated_graph_map

            # Use config-derived prefix and CWD for artefacts
            output_dir = Path.cwd()
            prefix = getattr(config, "_source_path", None)
            stem = Path(prefix).stem if isinstance(prefix, Path) else "scenario"
            visualization_path = output_dir / f"{stem}_integrated_graph.jpg"

            # Visualize the corridor graph instead of the full highway graph
            export_integrated_graph_map(
                metros=metros,
                graph=corridor_graph,  # Use corridor graph for cleaner visualization
                output_path=visualization_path,
                conus_boundary_path=config.data_sources.conus_boundary,
                target_crs=config.projection.target_crs,
                use_real_geometry=bool(
                    getattr(config, "_use_real_corridor_geometry", False)
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to export corridor graph visualization: {e}"
            ) from e

    logger.info(
        f"Successfully built integrated graph with corridor extraction: "
        f"Full graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges; "
        f"Corridor graph: {len(corridor_graph.nodes):,} metros, {len(corridor_graph.edges):,} corridors"
    )

    # Return the corridor graph as the final result
    return corridor_graph


def validate_integrated_graph(
    graph: nx.Graph, metros: list[MetroCluster], validation_config: ValidationConfig
) -> None:
    """Validate integrated graph structure.

    Args:
        graph: Integrated graph to validate.
        metros: List of metro clusters that should be in graph.
        validation_config: Validation configuration with thresholds.

    Raises:
        ValueError: If validation fails.
    """
    logger.info("Validating integrated graph")

    # Report connectivity for informational purposes (but don't fail on disconnected highway components)
    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        component_sizes = sorted([len(c) for c in components], reverse=True)
        logger.info(
            f"Integrated graph has {len(components)} disconnected components. "
            f"Largest: {component_sizes[0]:,} nodes ({component_sizes[0] / len(graph.nodes):.1%})"
        )
    else:
        logger.info("Integrated graph is fully connected")

    # Check metro anchor connections
    metro_anchor_count = 0
    for metro in metros:
        key = metro.node_key

        # Count anchor edges for this metro
        anchor_edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if (u == key or v == key) and d.get("edge_type") == "metro_anchor"
        ]

        if len(anchor_edges) != 1:
            raise ValueError(
                f"Metro {metro.name} ({metro.metro_id}) has {len(anchor_edges)} anchor edges (expected 1)"
            )
        metro_anchor_count += 1

    # Check node degree sanity
    node_count = graph.number_of_nodes()
    if node_count > 0:
        node_degrees = list(graph.degree())  # type: ignore[arg-type]
        degrees = [d for _, d in node_degrees]
        max_degree = max(degrees) if degrees else 0
        if max_degree > validation_config.max_degree_threshold:
            raise ValueError(
                f"Maximum node degree {max_degree} exceeds threshold {validation_config.max_degree_threshold}"
            )

        high_degree_nodes = [
            n for n, d in node_degrees if d > validation_config.high_degree_warning
        ]
    else:
        max_degree = 0
        high_degree_nodes = []
    if high_degree_nodes:
        logger.warning(
            f"Found {len(high_degree_nodes)} nodes with degree > {validation_config.high_degree_warning}"
        )

    # Count corridor tags
    corridor_edges = sum(
        1 for _, _, data in graph.edges(data=True) if "corridor" in data
    )

    if corridor_edges == 0:
        raise ValueError("No corridor tags found - corridor discovery failed")

    logger.info(
        f"Validation successful: {metro_anchor_count} metro anchors, {corridor_edges} corridor edges"
    )


def save_to_json(
    graph: nx.Graph,
    path: Path,
    crs: str,
    formatting_config: FormattingConfig,
) -> None:
    """Save integrated graph to JSON format with tuple encoding.

    Args:
        graph: NetworkX graph to save.
        path: Output path for JSON file.
        crs: Coordinate reference system string.
        formatting_config: Formatting configuration for JSON output.
    """
    logger.info(f"Saving integrated graph to JSON: {path}")

    def _to_python(obj: Any):
        """Convert common non-JSON types to JSON-friendly Python types."""
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, set):
            # Sort for determinism when possible
            try:
                return sorted(list(obj))
            except Exception:
                return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    out: dict[str, Any] = {"target_crs": crs, "nodes": [], "edges": []}

    # Serialize nodes
    for node, data in graph.nodes(data=True):
        x, y = node  # Unpack tuple
        node_data = {
            "id": [float(x), float(y)],
            **{k: _to_python(v) for k, v in data.items()},
        }
        out["nodes"].append(node_data)

    # Serialize edges
    for u, v, data in graph.edges(data=True):
        edge_data = {
            "source": [float(u[0]), float(u[1])],
            "target": [float(v[0]), float(v[1])],
            **{k: _to_python(v) for k, v in data.items()},
        }
        out["edges"].append(edge_data)

    # Serialize corridor path registry if present
    registry = graph.graph.get("corridor_paths")
    if registry:
        serialized_paths: list[dict[str, Any]] = []
        # registry may be a dict keyed by PathId or already a list; normalize to dict iteration
        if isinstance(registry, dict):
            items = registry.items()
        else:  # pragma: no cover - defensive
            try:
                items = list(enumerate(registry))  # type: ignore[assignment]
            except Exception:
                items = []
        for _key, cp in items:  # type: ignore[misc]
            # cp may be a CorridorPath dataclass or a plain dict
            if hasattr(cp, "metros") and hasattr(cp, "geometry"):
                # Likely a CorridorPath
                metros_tuple = cp.metros
                path_index = int(cp.path_index)
                length_km = float(cp.length_km)
                segment_ids = list(cp.segment_ids)
                geometry = [(float(p[0]), float(p[1])) for p in cp.geometry]
                serialized_paths.append(
                    {
                        "metro_a": metros_tuple[0],
                        "metro_b": metros_tuple[1],
                        "path_index": path_index,
                        "length_km": length_km,
                        "segment_ids": segment_ids,
                        "geometry": [list(p) for p in geometry],
                    }
                )
            elif isinstance(cp, dict):
                # Assume already in serializable form
                serialized_paths.append(cp)
        out["corridor_paths"] = serialized_paths

    # Save to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(out, f, indent=formatting_config.json_indent)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved integrated graph: {file_size_mb:.1f} MB")


def load_from_json(path: Path) -> tuple[nx.Graph, str]:
    """Load integrated graph from JSON format with tuple decoding.

    Args:
        path: Input path for JSON file.

    Returns:
        Tuple of (NetworkX graph, CRS string).
    """
    logger.info(f"Loading integrated graph from JSON: {path}")

    with path.open("r") as f:
        data = json.load(f)

    graph = nx.Graph()

    # Rebuild nodes
    for node_data in data["nodes"]:
        key = tuple(node_data.pop("id"))  # Convert list back to tuple
        graph.add_node(key, **node_data)

    # Rebuild edges
    for edge_data in data["edges"]:
        u = tuple(edge_data.pop("source"))
        v = tuple(edge_data.pop("target"))
        graph.add_edge(u, v, **edge_data)

    crs = data["target_crs"]

    # Rebuild corridor path registry if present
    try:
        raw_paths = data.get("corridor_paths")
    except Exception:  # pragma: no cover - defensive
        raw_paths = None
    if raw_paths:
        registry: dict[tuple[str, str, int], CorridorPath] = {}
        for entry in raw_paths:
            try:
                metro_a = str(entry["metro_a"])
                metro_b = str(entry["metro_b"])
                path_index = int(entry["path_index"])
                length_km = float(entry.get("length_km", 0.0))
                segment_ids = list(entry.get("segment_ids", []))
                geometry = [
                    (float(p[0]), float(p[1]))
                    for p in entry.get("geometry", [])
                    if isinstance(p, (list, tuple)) and len(p) >= 2
                ]
                metros = tuple(sorted((metro_a, metro_b)))  # type: ignore[assignment]
                cp = CorridorPath(
                    metros=metros,  # type: ignore[arg-type]
                    path_index=path_index,
                    nodes=[],
                    edges=[],
                    segment_ids=segment_ids,
                    length_km=length_km,
                    geometry=geometry,
                )
                registry[(metros[0], metros[1], path_index)] = cp
            except Exception:  # pragma: no cover - robust to partial data
                continue
        if registry:
            graph.graph["corridor_paths"] = registry
    logger.info(
        f"Loaded integrated graph: {len(graph.nodes):,} nodes, {len(graph.edges):,} edges"
    )

    return graph, crs
