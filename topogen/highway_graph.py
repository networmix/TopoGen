"""Highway graph construction for backbone topology generation.

Take raw TIGER/Line Primary Roads and return a small, connected, weighted nx.Graph
whose vertices are only real highway intersections; every edge has an accurate
length_km; geometry detail below intersection-to-intersection is removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import numpy as np

from topogen.log_config import get_logger

if TYPE_CHECKING:
    from topogen.config import HighwayProcessingConfig, ValidationConfig

logger = get_logger(__name__)


def _load_and_validate_tiger_data(tiger_zip: Path, target_crs: str) -> gpd.GeoDataFrame:
    """Load TIGER data with schema guard and immediate validation.

    Args:
        tiger_zip: Path to TIGER ZIP file.
        target_crs: Target coordinate reference system.

    Returns:
        GeoDataFrame with MTFCC and geometry columns in target CRS.

    Raises:
        ValueError: If data is missing, empty, or invalid.
        OSError: If file cannot be read.
    """
    logger.info(f"Loading TIGER data from {tiger_zip}")

    # Load with column filtering for efficiency
    try:
        gdf = gpd.read_file(f"zip://{tiger_zip}", columns=["MTFCC", "geometry"])
    except Exception as e:
        raise OSError(f"Cannot read TIGER ZIP file: {e}") from e

    if gdf.empty:
        raise ValueError("TIGER data is empty")

    # Validate CRS - accept both WGS84 (4326) and NAD83 (4269) for CONUS data
    if gdf.crs is None:
        raise ValueError("TIGER data has no CRS information")

    epsg_code = gdf.crs.to_epsg()
    if epsg_code not in (4326, 4269):
        raise ValueError(
            f"Expected EPSG:4326 (WGS84) or EPSG:4269 (NAD83), got {gdf.crs}"
        )

    logger.info(f"Source CRS: {gdf.crs} (EPSG:{epsg_code})")

    # Reproject to target CRS
    try:
        gdf = gdf.to_crs(target_crs)
    except Exception as e:
        raise ValueError(f"Cannot reproject to {target_crs}: {e}") from e

    logger.info(f"Loaded {len(gdf):,} road segments in {target_crs}")
    return gdf


def _filter_highway_classes(
    gdf: gpd.GeoDataFrame, highway_classes: list[str]
) -> gpd.GeoDataFrame:
    """Keep only classes appropriate for long-haul routing.

    Args:
        gdf: Input GeoDataFrame with MTFCC column.
        highway_classes: List of TIGER highway classes to keep.

    Returns:
        Filtered GeoDataFrame with specified highway classes.

    Raises:
        ValueError: If no highway segments remain after filtering.
    """
    original_count = len(gdf)
    logger.info(
        f"Filtering to backbone highway classes: {highway_classes} (from {original_count:,} total segments)"
    )

    # Log available MTFCC codes for debugging
    available_codes = gdf["MTFCC"].value_counts()
    logger.info(f"Available MTFCC codes in data: {dict(available_codes.head(10))}")

    # Keep only specified highway classes
    mask = gdf.MTFCC.isin(highway_classes)
    filtered_gdf = gdf[mask].copy()
    assert isinstance(filtered_gdf, gpd.GeoDataFrame)
    gdf = filtered_gdf

    if gdf.empty:
        raise ValueError(
            f"No backbone highway segments found for classes: {highway_classes}"
        )

    filtered_count = len(gdf)
    excluded_count = original_count - filtered_count
    logger.info(
        f"Retained {filtered_count:,} backbone highway segments (excluded {excluded_count:,})"
    )
    return gdf


def _fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix or drop bad geometries.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame with valid geometries only.

    Raises:
        ValueError: If no valid geometries remain.
    """
    initial_count = len(gdf)
    logger.info("Fixing geometries")

    # Drop empty and invalid geometries
    mask = ~gdf.geometry.is_empty & gdf.geometry.is_valid
    filtered_gdf = gdf[mask].copy()
    assert isinstance(filtered_gdf, gpd.GeoDataFrame)
    gdf = filtered_gdf

    if gdf.empty:
        raise ValueError("No valid geometries remain after geometry filter")

    # Explode multipart geometries to individual lines
    exploded = gdf.explode(index_parts=False, ignore_index=True)
    # Ensure result is a GeoDataFrame
    gdf = exploded if isinstance(exploded, gpd.GeoDataFrame) else gdf

    final_count = len(gdf)
    excluded_count = initial_count - final_count
    if excluded_count > 0:
        logger.info(
            f"Geometry cleanup: {initial_count:,} -> {final_count:,} segments (excluded {excluded_count:,} invalid)"
        )
    else:
        logger.info(
            f"Geometry cleanup: {final_count:,} segments (no invalid geometries)"
        )
    return gdf


def _iter_snapped_edges(lines: list, snap_m: float):
    """Iterate through geometries and yield snapped edges with lengths.

    Linear-time replacement for O(n²) overlay operation. Snaps coordinates
    to a grid, automatically creating shared nodes at intersections.

    Args:
        lines: List of LineString geometries.
        snap_m: Grid snap precision in meters.

    Yields:
        Tuples of (start_point, end_point, length_km) for each edge.
    """
    for geom in lines:
        if geom is None or geom.is_empty:
            continue

        # Extract and snap coordinates to grid
        coords = np.array(geom.coords)
        xs = np.round(coords[:, 0] / snap_m) * snap_m
        ys = np.round(coords[:, 1] / snap_m) * snap_m

        # Create edges from consecutive snapped points
        for i in range(len(xs) - 1):
            p = (xs[i], ys[i])
            q = (xs[i + 1], ys[i + 1])

            if p != q:  # Skip degenerate edges
                # Calculate edge length using numpy hypot for speed
                length_km = np.hypot(p[0] - q[0], p[1] - q[1]) / 1000
                yield p, q, length_km


def _build_intersection_graph(lines: list, snap_precision_m: float) -> nx.Graph:
    """Build graph from geometries using grid-snap approach.

    Args:
        lines: List of LineString geometries.
        snap_precision_m: Grid snap precision in meters.

    Returns:
        NetworkX Graph with snapped coordinate tuples as node IDs.

    Raises:
        ValueError: If no valid edges can be created.
    """
    logger.info(
        f"Building intersection graph with grid-snap approach (snap precision: {snap_precision_m}m)"
    )

    G = nx.Graph()

    # Use the linear-time grid-snap approach
    for p, q, length_km in _iter_snapped_edges(lines, snap_precision_m):
        if length_km <= 0:
            continue

        # Add edge, automatically merging parallel edges to keep shorter one
        if G.has_edge(p, q):
            existing_length = G.edges[p, q]["length_km"]
            if length_km < existing_length:
                G.edges[p, q]["length_km"] = length_km
        else:
            G.add_edge(p, q, length_km=length_km)

    if len(G.edges) == 0:
        raise ValueError("No valid edges created from geometries")

    logger.info(f"Built graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    return G


def _validate_final_graph(G: nx.Graph, validation_config: ValidationConfig) -> None:
    """Validate uncontracted highway graph before returning.

    Note: This validates the raw intersection-level graph before chain contraction.
    Connectivity validation is deferred to the integrated graph pipeline.

    Args:
        G: Final highway graph to validate.
        validation_config: Validation parameters for graph quality checks.

    Raises:
        ValueError: If graph fails validation checks.
    """
    logger.info("Validating uncontracted highway graph")

    if len(G.nodes) == 0:
        raise ValueError("Highway graph has no nodes")

    if len(G.edges) == 0:
        raise ValueError("Highway graph has no edges")

    # Report connectivity for informational purposes (but don't fail)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        component_sizes = sorted([len(c) for c in components], reverse=True)
        logger.debug(
            f"Highway graph has {len(components)} disconnected components. "
            f"Largest: {component_sizes[0]:,} nodes. "
            f"Components will be filtered during contraction in integrated pipeline."
        )

    # Validate all edges have positive length_km
    for u, v, data in G.edges(data=True):
        if "length_km" not in data:
            raise ValueError(f"Edge {u}-{v} missing length_km attribute")

        length = data["length_km"]
        if not isinstance(length, (int, float)) or length <= 0 or np.isnan(length):
            raise ValueError(f"Edge {u}-{v} has invalid length_km: {length}")

    # Check for excessive node degrees (indicates snapping bugs)
    degrees = [len(list(G.neighbors(node))) for node in G.nodes()]
    max_degree = max(degrees) if degrees else 0
    if max_degree > validation_config.max_degree_threshold:
        raise ValueError(
            f"Node with degree {max_degree} found - indicates accidental mass snapping bug"
        )

    high_degree_nodes = [
        node
        for node in G.nodes()
        if len(list(G.neighbors(node))) > validation_config.high_degree_warning
    ]
    if high_degree_nodes:
        logger.warning(
            f"Found {len(high_degree_nodes)} nodes with degree > {validation_config.high_degree_warning}"
        )

    logger.info(f"Graph validation passed - max degree: {max_degree}")


def build_highway_graph(
    tiger_zip: Path,
    target_crs: str,
    highway_config: HighwayProcessingConfig,
    validation_config: ValidationConfig,
) -> nx.Graph:
    """Build highway graph from TIGER Primary Roads data.

    Take raw TIGER/Line Primary Roads and return a detailed intersection-level
    nx.Graph whose vertices are highway intersections; every edge has an
    accurate length_km and preserves geometry detail.

    Uses linear-time grid-snap approach instead of O(n²) overlay for efficiency.
    Snaps coordinates to 10m grid to merge twin carriageways while preserving
    distinct interchanges.

    Args:
        tiger_zip: Path to TIGER/Line ZIP file.
        target_crs: Target coordinate reference system.
        highway_config: Highway data processing configuration.
        validation_config: Validation parameters for graph quality checks.

    Returns:
        NetworkX Graph with highway segments (uncontracted, full detail).

    Raises:
        ValueError: If data is invalid or processing fails.
        OSError: If files cannot be accessed.
    """
    logger.info(f"Building highway graph from {tiger_zip}")

    # 1. Load once, with schema guard
    gdf = _load_and_validate_tiger_data(tiger_zip, target_crs)

    # 2. Keep only classes appropriate for long-haul routing
    gdf = _filter_highway_classes(gdf, highway_config.highway_classes)

    # 3. Fix or drop bad geometries
    gdf = _fix_geometries(gdf)

    # 4. Build graph using linear grid-snap approach (replaces overlay)
    G = _build_intersection_graph(
        gdf.geometry.tolist(), highway_config.snap_precision_m
    )

    # 5. Validate before returning
    _validate_final_graph(G, validation_config)

    logger.info(f"Final highway graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    return G
