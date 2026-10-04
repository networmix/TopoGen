"""Build metro-to-metro corridor graphs from urban areas and highways.

The intermediate graph contains highway segments, metros, and anchor edges.
The returned graph has metro nodes and corridor edges with path length,
Euclidean separation, and detour ratio. JSON helpers preserve coordinate keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy.spatial import KDTree  # pyright: ignore[reportMissingTypeStubs]
from shapely.geometry import Point

from topogen.context import RunContext
from topogen.corridors import (
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
    """Contract chains without erasing alternative routes or distorting cycles.

    Keep one interior vertex for parallel chains so routing remains a simple
    graph. Preserve rings and loops at junctions as their original segments.
    """
    from collections import defaultdict
    from hashlib import blake2b

    protected = set() if protected_nodes is None else protected_nodes
    terminals = {node for node in G if G.degree[node] != 2 or node in protected}
    chains = defaultdict(list)
    visited = set()
    for start in sorted(terminals):
        for neighbor in sorted(G[start]):
            if frozenset((start, neighbor)) in visited:
                continue
            path = [start, neighbor]
            visited.add(frozenset((start, neighbor)))
            while path[-1] not in terminals:
                previous, current = path[-2:]
                following = next(node for node in G[current] if node != previous)
                visited.add(frozenset((current, following)))
                path.append(following)
            chains[tuple(sorted((path[0], path[-1])))].append(path)

    result = nx.Graph()
    result.add_nodes_from((node, dict(G.nodes[node])) for node in sorted(terminals))

    def add_path(path):
        canonical = min(path, list(reversed(path)))
        segment_id = blake2b(repr(canonical).encode(), digest_size=12).hexdigest()
        result.add_edge(
            path[0],
            path[-1],
            length_km=sum(
                G[u][v]["length_km"] for u, v in zip(path, path[1:], strict=False)
            ),
            geometry=path,
            segment_id=segment_id,
        )

    for (start, end), paths in sorted(chains.items()):
        for path in paths:
            if start == end:
                for u, v in zip(path, path[1:], strict=False):
                    add_path([u, v])
            elif len(paths) > 1 and len(path) > 2:
                add_path(path[:2])
                add_path(path[1:])
            else:
                add_path(path)
    # Unvisited edges belong to isolated rings; preserving them preserves both arcs.
    for u, v in sorted(G.edges()):
        if frozenset((u, v)) not in visited:
            add_path([u, v])
    if not result.edges:
        raise ValueError("Graph contraction produced empty result")
    logger.info("Contracted graph: %d nodes, %d edges", len(result), len(result.edges))
    return result


def _remove_slivers(
    G: nx.Graph, min_length_km: float, validation_config: ValidationConfig
) -> nx.Graph:
    """Copy the graph without edges below ``min_length_km`` or isolated nodes.

    Reject an empty result or a disconnected result whose largest component
    falls below the configured fraction of the original node count.
    """
    if min_length_km == 0:
        return G
    logger.info(
        f"Removing edges shorter than {min_length_km}km (sliver removal threshold)"
    )

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

    isolated_nodes = list(nx.isolates(G_clean))
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

            if len(component_sizes) > 1:
                other_components = component_sizes[1:6]
                logger.warning(f"Other component sizes: {other_components}")
        else:
            logger.info(
                f"Network has {num_components} pre-existing components (no edges removed). "
                f"Largest: {largest_component_size:,} nodes ({largest_component_fraction:.1%} of original)"
            )

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
        if initial_connected:
            logger.info("Sliver removal preserved network connectivity")
        else:
            logger.info(
                f"Sliver removal complete - network remains connected ({nodes_remaining:,} nodes)"
            )

    return G_clean


def _keep_largest_component(G: nx.Graph) -> nx.Graph:
    """Return G if connected, otherwise a copy of its largest component.

    Log discarded nodes; component size is not a rejection criterion here.
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

    nodes_lost = total_nodes - largest_size
    if nodes_lost > 0:
        logger.warning(
            f"Keeping largest component will discard {nodes_lost:,} nodes "
            f"({(nodes_lost / total_nodes):.1%} of network)"
        )

    if largest_fraction < 0.1:
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
    highway_coords = np.array([list(node) for node in highway_graph.nodes()])
    highway_nodes = list(highway_graph.nodes())
    tree = KDTree(highway_coords)

    highway_x_range = (highway_coords[:, 0].min(), highway_coords[:, 0].max())
    highway_y_range = (highway_coords[:, 1].min(), highway_coords[:, 1].max())
    logger.info(
        f"Highway network spatial coverage: "
        f"X: [{highway_x_range[0]:.0f}, {highway_x_range[1]:.0f}], "
        f"Y: [{highway_y_range[0]:.0f}, {highway_y_range[1]:.0f}]"
    )

    anchors = {}
    metro_distances = []

    for metro in metros:
        metro_coords = np.array([metro.centroid_x, metro.centroid_y])

        distance, idx = tree.query(metro_coords)
        distance_km = distance / 1000.0
        metro_distances.append((metro.name, distance_km))

        if distance_km > validation_config.max_metro_highway_distance_km:
            nearest_node = highway_nodes[idx]
            logger.error(
                f"Metro anchoring failed for {metro.name} (ID {metro.metro_id}): "
                f"Metro coords: ({metro.centroid_x:.0f}, {metro.centroid_y:.0f}), "
                f"Nearest highway node: ({nearest_node[0]:.0f}, {nearest_node[1]:.0f}), "
                f"Distance: {distance_km:.1f}km (max: {validation_config.max_metro_highway_distance_km}km)"
            )

            sorted_distances = sorted(metro_distances, key=lambda x: x[1])
            logger.info("Metro-highway distances so far:")
            for name, dist in sorted_distances[:10]:
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

    sorted_distances = sorted(metro_distances, key=lambda x: x[1])
    avg_distance = sum(dist for _, dist in sorted_distances) / len(sorted_distances)
    max_distance = max(dist for _, dist in sorted_distances)

    logger.info(f"Successfully anchored {len(anchors)} metros to highway network")
    logger.info(
        f"Anchoring distances - Avg: {avg_distance:.1f}km, Max: {max_distance:.1f}km"
    )

    logger.debug("Metro-highway anchoring distances:")
    for name, dist in sorted_distances:
        logger.debug(f"  {name}: {dist:.1f}km")
    return anchors


def build_integrated_graph(
    config: TopologyConfig, *, context: RunContext | None = None
) -> nx.MultiGraph:
    """Build and return the metro-to-metro corridor graph for scenario assembly.

    The highway graph, metro anchors, and tagged corridor paths are intermediate
    products. Raises ValueError when graph validation fails.
    """
    logger.info("Building integrated metro and highway graph")

    metros = load_metro_clusters(
        uac_path=config.data_sources.uac_polygons,
        k=config.clustering.metro_clusters,
        target_crs=config.projection.target_crs,
        clustering_config=config.clustering,
        conus_boundary_path=config.data_sources.conus_boundary,
        context=context,
    )
    logger.info(f"Loaded {len(metros)} metro clusters")

    highway_graph = build_highway_graph(
        tiger_zip=config.data_sources.tiger_roads,
        target_crs=config.projection.target_crs,
        highway_config=config.highway_processing,
        validation_config=config.validation,
    )
    logger.info(
        f"Built highway graph: {len(highway_graph.nodes):,} nodes, {len(highway_graph.edges):,} edges"
    )

    anchors = anchor_metros(metros, highway_graph, config.validation)

    logger.info("Contracting highway graph")
    # Protect metro anchor nodes from being removed during contraction
    anchor_nodes = set(anchors.values())
    highway_contracted = _contract_degree2_chains(
        highway_graph, protected_nodes=anchor_nodes
    )

    del highway_graph
    highway_clean = _remove_slivers(
        highway_contracted,
        config.highway_processing.min_edge_length_km,
        config.validation,
    )

    del highway_contracted
    if config.highway_processing.filter_largest_component:
        highway_final = _keep_largest_component(highway_clean)
    else:
        highway_final = highway_clean
        logger.info("Component filtering disabled - keeping all highway components")

    del highway_clean
    logger.info("Verifying metro anchors after highway filtering")

    missing_anchors = []
    for metro in metros:
        anchor = anchors[metro.metro_id]
        if not highway_final.has_node(anchor):
            missing_anchors.append((metro.name, anchor))

    if missing_anchors:
        anchor_list = ", ".join(
            [f"{name} ({anchor})" for name, anchor in missing_anchors]
        )
        raise ValueError(
            f"Highway filtering removed metro anchors: {anchor_list}. "
            "Check min_edge_length_km, filter_largest_component, and selected metros."
        )

    logger.info(f"All {len(metros)} metro anchors survived highway filtering")

    logger.info("Building integrated graph")
    G = highway_final

    for metro in metros:
        key = metro.node_key  # (x, y) tuple
        anchor = anchors[metro.metro_id]

        G.add_node(
            key,
            node_type="metro+highway" if G.has_node(key) else "metro",
            x=metro.centroid_x,
            y=metro.centroid_y,
            name=metro.name,
            name_orig=metro.name_orig,
            radius_km=metro.radius_km,
            metro_id=metro.metro_id,
            uac_code=metro.uac_code,
            land_area_km2=metro.land_area_km2,
        )

        dist_km = Point(metro.coordinates).distance(Point(anchor)) / 1000.0
        if key == anchor:
            continue
        G.add_edge(
            key,
            anchor,
            edge_type="metro_anchor",
            length_km=dist_km,
            geometry=[key, anchor],  # Straight line from representative point to road.
        )

    logger.info(f"Added {len(metros)} metro nodes with anchor connections")

    logger.info("Discovering corridors")
    add_corridors(G, metros, config.corridors)

    assign_risk_groups(G, metros, config.corridors)

    validate_integrated_graph(G, metros, config.validation)

    corridor_graph = extract_corridor_graph(G, metros)

    validate_corridor_graph(corridor_graph, metros, config.validation)

    if context is not None and config.clustering.export_integrated_graph:
        from topogen.visualization import export_integrated_graph_map

        context.output_dir.mkdir(parents=True, exist_ok=True)
        export_integrated_graph_map(
            metros=metros,
            graph=corridor_graph,
            output_path=context.path("integrated_graph.jpg"),
            conus_boundary_path=config.data_sources.conus_boundary,
            target_crs=config.projection.target_crs,
            use_real_geometry=config.visualization.use_real_corridor_geometry,
            dpi=config.visualization.dpi,
        )

    logger.info(
        f"Successfully built integrated graph with corridor extraction: "
        f"Full graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges; "
        f"Corridor graph: {len(corridor_graph.nodes):,} metros, {len(corridor_graph.edges):,} corridors"
    )

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

    metro_anchor_count = 0
    for metro in metros:
        key = metro.node_key

        anchor_edges = [
            (u, v)
            for u, v, d in graph.edges(key, data=True)
            if d.get("edge_type") == "metro_anchor"
        ]

        expected = 0 if graph.nodes[key]["node_type"] == "metro+highway" else 1
        if len(anchor_edges) != expected:
            raise ValueError(
                f"Metro {metro.name} ({metro.metro_id}) has {len(anchor_edges)} anchor edges (expected {expected})"
            )
        metro_anchor_count += 1

    node_count = graph.number_of_nodes()
    if node_count > 0:
        node_degrees = [(node, graph.degree[node]) for node in graph]
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

    corridor_edges = sum(
        1 for _, _, data in graph.edges(data=True) if "corridor" in data
    )

    if corridor_edges == 0:
        raise ValueError("No corridor tags found - corridor discovery failed")

    logger.info(
        f"Validation successful: {metro_anchor_count} metro anchors, {corridor_edges} corridor edges"
    )


def save_to_json(
    graph: nx.MultiGraph, path: Path, crs: str, formatting_config: FormattingConfig
) -> None:
    """Save the corridor multigraph with explicit edge keys and projected CRS."""
    if not graph.is_multigraph() or graph.is_directed():
        raise TypeError("Expected a corridor MultiGraph")
    out = {
        "graph_type": "corridors",
        "target_crs": crs,
        "nodes": [
            {"id": list(node), **attrs} for node, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {"source": list(u), "target": list(v), "key": key, **attrs}
            for u, v, key, attrs in graph.edges(keys=True, data=True)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, indent=formatting_config.json_indent, allow_nan=False) + "\n"
    )


def load_from_json(path: Path) -> tuple[nx.MultiGraph, str]:
    """Read the corridor contract; reject obsolete and duplicate records."""
    data = json.loads(path.read_text())
    if data.get("graph_type") != "corridors":
        raise ValueError(
            "Expected graph_type='corridors'; regenerate the integrated graph"
        )
    graph = nx.MultiGraph()
    for record in data["nodes"]:
        attrs = dict(record)
        node = tuple(attrs.pop("id"))
        if len(node) != 2 or graph.has_node(node):
            raise ValueError(f"Invalid or duplicate metro coordinates: {node}")
        graph.add_node(node, **attrs)
    for record in data["edges"]:
        attrs = dict(record)
        u, v = tuple(attrs.pop("source")), tuple(attrs.pop("target"))
        key = attrs.pop("key")
        if u not in graph or v not in graph or graph.has_edge(u, v, key):
            raise ValueError(
                f"Invalid or duplicate corridor endpoints/key: {u}, {v}, {key}"
            )
        attrs["geometry"] = [tuple(point) for point in attrs["geometry"]]
        graph.add_edge(u, v, key=key, **attrs)
    return graph, data["target_crs"]
