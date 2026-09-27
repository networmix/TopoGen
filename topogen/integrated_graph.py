"""Build metro-to-metro corridor graphs from urban areas and highways.

The intermediate graph contains highway segments, metros, and anchor edges.
The returned graph has metro nodes and corridor edges with path length,
Euclidean separation, and detour ratio. JSON helpers preserve coordinate keys.
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
    CorridorPath,
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
    """Collapse degree-2 chains, preserving junctions, dead ends, and protected nodes.

    Sum lengths into ``length_km`` and retain traversed coordinates in ``geometry``.
    Isolated cycles become one edge between two of their nodes.
    """
    logger.info("Contracting degree-2 chains")

    if protected_nodes is None:
        protected_nodes = set()

    if protected_nodes:
        logger.info(
            f"Protecting {len(protected_nodes)} nodes from contraction (metro anchors)"
        )

    components_before = list(nx.connected_components(G))
    logger.debug(
        f"Input graph: {len(G.nodes)} nodes, {len(G.edges)} edges, {len(components_before)} components"
    )
    if len(components_before) > 1:
        component_sizes = sorted([len(c) for c in components_before], reverse=True)
        logger.debug(f"Component sizes before contraction: {component_sizes[:10]}")

    degree_counts = {}
    for node in G.nodes():
        deg = len(list(G.neighbors(node)))
        degree_counts[deg] = degree_counts.get(deg, 0) + 1
    logger.debug(f"Degree distribution: {dict(sorted(degree_counts.items()))}")

    contracted = nx.Graph()
    visited = set()
    processed_nodes = set()

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

                path = [node]
                length = G.edges[node, nbr]["length_km"]
                visited.add(key)

                prev, curr = node, nbr
                path.append(curr)

                while is_contractible(curr):
                    nxt = next(n for n in G.neighbors(curr) if n != prev)
                    length += G.edges[curr, nxt]["length_km"]
                    visited.add(tuple(sorted((curr, nxt))))
                    path.append(nxt)
                    prev, curr = curr, nxt

                contracted.add_edge(
                    path[0],
                    path[-1],
                    length_km=length,
                    geometry=path,
                    segment_id=_segment_id(path[0], path[-1]),
                )
                processed_nodes.update(path)
                chains_contracted += 1

                if len(path) > 100:
                    logger.debug(
                        f"Long chain contracted: {len(path)} nodes, {length:.1f}km from {path[0]} to {path[-1]}"
                    )

    logger.debug(
        f"Phase 1: Contracted {chains_contracted} chains connected to junctions"
    )

    # Second pass: handle isolated degree-2 cycles (rings)
    remaining_nodes = set(G.nodes()) - processed_nodes

    processed_cycles = set()
    for node in remaining_nodes:
        if node in processed_cycles or len(list(G.neighbors(node))) != 2:
            continue

        cycle_nodes = []
        cycle_length = 0.0
        current = node
        prev = None

        while True:
            cycle_nodes.append(current)
            processed_cycles.add(current)

            neighbors = list(G.neighbors(current))
            if len(neighbors) != 2:
                # Node no longer has exactly 2 neighbors, break to avoid infinite loop
                break
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]

            cycle_length += G.edges[current, next_node]["length_km"]

            prev = current
            current = next_node

            if current == node:
                break

        if len(cycle_nodes) >= 3:  # Only contract cycles with 3+ nodes
            # Use first and "middle" node as endpoints to avoid self-loops
            start_node = cycle_nodes[0]
            mid_node = cycle_nodes[len(cycle_nodes) // 2]

            if start_node != mid_node:
                contracted.add_edge(
                    start_node,
                    mid_node,
                    length_km=cycle_length,
                    geometry=cycle_nodes + [cycle_nodes[0]],
                    segment_id=_segment_id(start_node, mid_node),
                )

    if contracted.number_of_edges() == 0:
        raise ValueError("Graph contraction produced empty result")

    components_after = list(nx.connected_components(contracted))
    logger.debug(
        f"Output graph: {len(contracted.nodes)} nodes, {len(contracted.edges)} edges, {len(components_after)} components"
    )

    if len(components_after) > 1:
        component_sizes_after = sorted([len(c) for c in components_after], reverse=True)
        logger.debug(f"Component sizes after contraction: {component_sizes_after[:10]}")

        if len(components_after) != len(components_before):
            logger.warning(
                f"Component count changed during contraction: {len(components_before)} → {len(components_after)}"
            )

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
    """Copy the graph without edges below ``min_length_km`` or isolated nodes.

    Reject an empty result or a disconnected result whose largest component
    falls below the configured fraction of the original node count.
    """
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


def _keep_largest_component(
    G: nx.Graph, validation_config: ValidationConfig
) -> nx.Graph:
    """Return G if connected, otherwise a copy of its largest component.

    Log discarded nodes. ``validation_config`` is unused; component size is
    not a rejection criterion here.
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


def build_integrated_graph(config: TopologyConfig) -> nx.Graph:
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
        formatting_config=config.output.formatting,
        conus_boundary_path=config.data_sources.conus_boundary,
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

    highway_clean = _remove_slivers(
        highway_contracted,
        config.highway_processing.min_edge_length_km,
        config.validation,
    )

    if config.highway_processing.filter_largest_component:
        highway_final = _keep_largest_component(highway_clean, config.validation)
    else:
        highway_final = highway_clean
        logger.info("Component filtering disabled - keeping all highway components")

    logger.info("Verifying metro anchors after contraction")

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

    logger.info("Building integrated graph")
    G = highway_final.copy()

    for metro in metros:
        key = metro.node_key  # (x, y) tuple
        anchor = anchors[metro.metro_id]

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

        dist_km = Point(metro.coordinates).distance(Point(anchor)) / 1000.0
        G.add_edge(
            key,
            anchor,
            edge_type="metro_anchor",
            length_km=dist_km,
            geometry=[key, anchor],  # straight line from metro centroid to anchor
        )

    logger.info(f"Added {len(metros)} metro nodes with anchor connections")

    logger.info("Discovering corridors")
    add_corridors(G, metros, config.corridors)

    assign_risk_groups(G, metros, config.corridors)

    validate_integrated_graph(G, metros, config.validation)

    corridor_graph = extract_corridor_graph(G, metros)

    validate_corridor_graph(corridor_graph, metros, config.validation)

    if config.clustering.export_integrated_graph:
        cfg_out = getattr(config, "_output_dir", None)
        if cfg_out is not None and isinstance(cfg_out, (str, Path)):
            logger.info("Exporting corridor graph visualization")
            try:
                from topogen.visualization import export_integrated_graph_map

                output_dir = Path(cfg_out)
                prefix = getattr(config, "_source_path", None)
                stem = Path(prefix).stem if isinstance(prefix, Path) else "scenario"
                visualization_path = output_dir / f"{stem}_integrated_graph.jpg"

                export_integrated_graph_map(
                    metros=metros,
                    graph=corridor_graph,
                    output_path=visualization_path,
                    conus_boundary_path=config.data_sources.conus_boundary,
                    target_crs=config.projection.target_crs,
                    use_real_geometry=bool(
                        getattr(config, "_use_real_corridor_geometry", False)
                    ),
                    dpi=int(getattr(config, "_visualization_dpi", 300)),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to export corridor graph visualization: {e}"
                ) from e
        else:
            logger.debug(
                "Skipping corridor graph visualization (no output directory configured)"
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
            for u, v, d in graph.edges(data=True)
            if (u == key or v == key) and d.get("edge_type") == "metro_anchor"
        ]

        if len(anchor_edges) != 1:
            raise ValueError(
                f"Metro {metro.name} ({metro.metro_id}) has {len(anchor_edges)} anchor edges (expected 1)"
            )
        metro_anchor_count += 1

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

    for node, data in graph.nodes(data=True):
        x, y = node
        node_data = {
            "id": [float(x), float(y)],
            **{k: _to_python(v) for k, v in data.items()},
        }
        out["nodes"].append(node_data)

    for u, v, data in graph.edges(data=True):
        edge_data = {
            "source": [float(u[0]), float(u[1])],
            "target": [float(v[0]), float(v[1])],
            **{k: _to_python(v) for k, v in data.items()},
        }
        out["edges"].append(edge_data)

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

    for node_data in data["nodes"]:
        key = tuple(node_data.pop("id"))
        graph.add_node(key, **node_data)

    for edge_data in data["edges"]:
        u = tuple(edge_data.pop("source"))
        v = tuple(edge_data.pop("target"))
        graph.add_edge(u, v, **edge_data)

    crs = data["target_crs"]

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
