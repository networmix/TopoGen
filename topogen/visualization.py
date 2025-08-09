"""Visualization utilities for topology generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from topogen.log_config import get_logger

if TYPE_CHECKING:
    from topogen.metro_clusters import MetroCluster

logger = get_logger(__name__)


def export_cluster_map(
    centroids: np.ndarray, output_path: Path, conus_boundary_path: Path, target_crs: str
) -> None:
    """Export JPEG preview map of cluster centroids.

    Args:
        centroids: Cluster centroids (x, y coordinates).
        output_path: Path to save JPEG file.
        conus_boundary_path: Path to CONUS boundary shapefile for visual context.
        target_crs: Target coordinate reference system.

    Raises:
        RuntimeError: If JPEG export fails for any critical reason.
    """
    fig = None

    try:
        logger.info(f"Exporting cluster map to {output_path}")

        # Validate inputs
        if len(centroids) == 0:
            raise ValueError("Cannot create map: no centroids provided")

        if not target_crs or not target_crs.strip():
            raise ValueError("Cannot create map: invalid or empty target_crs")

        if not output_path:
            raise ValueError("Cannot create map: invalid output_path")

        if not conus_boundary_path or not conus_boundary_path.exists():
            raise ValueError(
                f"Cannot create map: CONUS boundary file not found at {conus_boundary_path}"
            )

        # Create GeoDataFrame of centroids
        try:
            from shapely.geometry import Point

            geometry = [Point(float(x), float(y)) for x, y in centroids]
            gdf = gpd.GeoDataFrame(
                {
                    "x": centroids[:, 0].astype(float),
                    "y": centroids[:, 1].astype(float),
                },
                geometry=geometry,
                crs=target_crs,
            )
            logger.debug(
                f"Created centroids GeoDataFrame: {len(gdf)} points in {target_crs}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create GeoDataFrame for centroids: {e}"
            ) from e

        # Load CONUS boundary for visual context (NOT for filtering - that's done in UAC processor)
        try:
            from topogen.geo_utils import create_conus_mask

            conus_poly = create_conus_mask(conus_boundary_path, target_crs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CONUS boundary for map context: {e}"
            ) from e

        # Create plot
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            logger.debug("Created matplotlib figure (12x8) for cluster map")
        except Exception as e:
            raise RuntimeError(f"Failed to create matplotlib figure: {e}") from e

        # Plot CONUS boundary for geographic context
        try:
            conus_poly.plot(ax=ax, edgecolor="k", facecolor="none", linewidth=0.4)
        except Exception as e:
            raise RuntimeError(f"Failed to plot CONUS boundary: {e}") from e

        # Plot centroids
        try:
            gdf.plot(ax=ax, markersize=12, color="red", alpha=0.7)
            logger.debug(f"Plotted {len(gdf)} cluster centroids (red circles)")
        except Exception as e:
            raise RuntimeError(f"Failed to plot cluster centroids: {e}") from e

        # Add basemap using contextily
        try:
            cx.add_basemap(ax, crs=gdf.crs, attribution=False, alpha=0.3)
            logger.debug("Added contextily basemap to cluster map")
        except Exception as e:
            raise RuntimeError(f"Failed to add basemap: {e}") from e

        # Style the plot
        try:
            ax.set_axis_off()
            ax.set_title(f"Metro Clusters (n={len(centroids)})", fontsize=14, pad=20)
            plt.tight_layout()
        except Exception as e:
            raise RuntimeError(f"Failed to style the plot: {e}") from e

        # Save JPEG - this is critical
        try:
            logger.debug(f"Saving cluster map to {output_path} (DPI=150, format=JPEG)")
            plt.savefig(output_path, dpi=150, format="jpeg", bbox_inches="tight")
        except Exception as e:
            raise RuntimeError(f"Failed to save JPEG file to {output_path}: {e}") from e
        finally:
            plt.close(fig)
            fig = None  # Mark as closed

        # Validate the JPEG was actually created and has reasonable size
        if not output_path.exists():
            raise RuntimeError(f"JPEG file was not created at {output_path}")

        file_size = output_path.stat().st_size
        if file_size < 1000:  # Less than 1KB suggests a problem
            raise RuntimeError(f"JPEG file appears corrupt (only {file_size} bytes)")

        logger.info(
            f"Saved centroid preview → {output_path} ({file_size / 1024:.1f} KB)"
        )

    except Exception as e:
        # Clean up the figure if it exists
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

        # Clean up partial file if it exists
        if output_path.exists():
            try:
                output_path.unlink()
                logger.debug(f"Cleaned up partial file: {output_path}")
            except Exception:
                pass

        # Re-raise with clear context
        if isinstance(e, (ValueError, RuntimeError)):
            raise  # Re-raise our custom errors as-is
        else:
            raise RuntimeError(
                f"Unexpected error during cluster map export: {e}"
            ) from e


def export_integrated_graph_map(
    metros: list[MetroCluster],
    graph: nx.Graph,
    output_path: Path,
    conus_boundary_path: Path,
    target_crs: str,
    *,
    use_real_geometry: bool = False,
) -> None:
    """Export JPEG preview map of integrated graph with metro clusters and corridors.

    Args:
        metros: List of metro clusters.
        graph: Integrated NetworkX graph with corridor tags.
        output_path: Path to save JPEG file.
        conus_boundary_path: Path to CONUS boundary shapefile for visual context.
        target_crs: Target coordinate reference system.

    Raises:
        RuntimeError: If JPEG export fails for any critical reason.
    """
    fig = None

    try:
        logger.info(f"Exporting integrated graph map to {output_path}")

        # Validate inputs
        if len(metros) == 0:
            raise ValueError("Cannot create map: no metros provided")

        if graph is None or len(graph.nodes) == 0:
            raise ValueError("Cannot create map: graph is empty or None")

        if not target_crs or not target_crs.strip():
            raise ValueError("Cannot create map: invalid or empty target_crs")

        if not output_path:
            raise ValueError("Cannot create map: invalid output_path")

        if not conus_boundary_path or not conus_boundary_path.exists():
            raise ValueError(
                f"Cannot create map: CONUS boundary file not found at {conus_boundary_path}"
            )

        # Extract metro centroids
        centroids = np.array([metro.coordinates for metro in metros])

        # Extract corridor connections from graph
        corridor_lines = []
        # Extract metro coordinates for plotting
        # metro_coords_dict = {metro.node_key: metro for metro in metros}  # Kept for potential future use
        metro_id_to_coords = {metro.metro_id: metro.coordinates for metro in metros}

        # Find unique metro pairs from corridor-tagged edges
        corridor_pairs = set()

        # Handle both full highway graphs (with corridor tags) and corridor graphs (direct edges)
        corridor_edges_found = 0

        # Check if this is a corridor graph (edges have edge_type="corridor")
        is_corridor_graph = any(
            data.get("edge_type") == "corridor" for u, v, data in graph.edges(data=True)
        )

        if is_corridor_graph:
            # Direct corridor graph - construct linestrings from edge geometry when available
            logger.debug("Processing corridor graph (direct metro-to-metro edges)")
            for _u, _v, data in graph.edges(data=True):
                if data.get("edge_type") != "corridor":
                    continue
                corridor_edges_found += 1

                if use_real_geometry:
                    geom = data.get("geometry")
                    if not (isinstance(geom, list) and len(geom) >= 2):
                        raise ValueError(
                            "Configured to use real geometry, but corridor edge missing geometry"
                        )
                    try:
                        coords = [(float(x), float(y)) for (x, y) in geom]
                    except Exception as e:
                        raise ValueError(
                            f"Invalid corridor geometry coordinates: {e}"
                        ) from e
                    corridor_lines.append(LineString(coords))
                else:
                    # Straight line required
                    metro_a_id = data.get("metro_a")
                    metro_b_id = data.get("metro_b")
                    if metro_a_id and metro_b_id:
                        if metro_a_id not in metro_id_to_coords:
                            raise ValueError(
                                f"Corridor references unknown metro ID: {metro_a_id}"
                            )
                        if metro_b_id not in metro_id_to_coords:
                            raise ValueError(
                                f"Corridor references unknown metro ID: {metro_b_id}"
                            )
                        a_coords = metro_id_to_coords[metro_a_id]
                        b_coords = metro_id_to_coords[metro_b_id]
                        corridor_lines.append(LineString([a_coords, b_coords]))
                    else:
                        raise ValueError(
                            "Corridor edge missing metro IDs for straight-line rendering"
                        )
        else:
            # Full highway graph - extract from corridor tags
            logger.debug(
                "Processing full highway graph (extracting from corridor tags)"
            )
            for _u, _v, data in graph.edges(data=True):
                if "corridor" in data and data["corridor"]:
                    corridor_edges_found += 1
                    # Extract metro pairs from corridor information
                    for corridor_info in data["corridor"]:
                        try:
                            metro_a_id = corridor_info["metro_a"]
                            metro_b_id = corridor_info["metro_b"]

                            # Validate metro IDs exist
                            if metro_a_id not in metro_id_to_coords:
                                raise ValueError(
                                    f"Corridor references unknown metro ID: {metro_a_id}"
                                )
                            if metro_b_id not in metro_id_to_coords:
                                raise ValueError(
                                    f"Corridor references unknown metro ID: {metro_b_id}"
                                )

                            # Create sorted tuple to avoid duplicates
                            pair = tuple(sorted([metro_a_id, metro_b_id]))
                            corridor_pairs.add(pair)

                        except KeyError as e:
                            raise ValueError(
                                f"Invalid corridor metadata: missing key {e}"
                            ) from e
                        except Exception as e:
                            raise RuntimeError(
                                f"Error processing corridor metadata: {e}"
                            ) from e

        # For full graph, create LineString for each unique metro pair (straight line)
        if not is_corridor_graph:
            for metro_a_id, metro_b_id in corridor_pairs:
                coords_a = metro_id_to_coords[metro_a_id]
                coords_b = metro_id_to_coords[metro_b_id]
                corridor_lines.append(LineString([coords_a, coords_b]))

        # Validate corridor discovery results
        if corridor_edges_found > 0 and len(corridor_lines) == 0:
            raise RuntimeError(
                f"Corridor extraction failed: found {corridor_edges_found} corridor-tagged edges "
                "but no valid metro pairs could be extracted"
            )

        logger.info(
            f"Found {len(corridor_lines)} corridor connections from {corridor_edges_found} corridor-tagged edges"
        )
        logger.debug(
            f"Extracted {len(corridor_pairs)} unique metro pairs from corridor metadata"
        )

        # Log informational message if no corridors (but don't fail - graph building would have failed first)
        if len(corridor_lines) == 0:
            logger.warning(
                "No corridor connections found for visualization. "
                "Map will show metro clusters only without corridor lines."
            )

        # Create GeoDataFrame of centroids
        try:
            geometry = [Point(float(x), float(y)) for x, y in centroids]
            metros_gdf = gpd.GeoDataFrame(
                {
                    "x": centroids[:, 0].astype(float),
                    "y": centroids[:, 1].astype(float),
                    "name": [metro.name for metro in metros],
                    "radius_km": [float(metro.radius_km) for metro in metros],
                },
                geometry=geometry,
                crs=target_crs,
            )
            logger.debug(
                f"Created metros GeoDataFrame: {len(metros_gdf)} points in {target_crs}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create GeoDataFrame for metros: {e}") from e

        # Create GeoDataFrame for corridors if any exist
        corridors_gdf = None
        if corridor_lines:
            try:
                corridors_gdf = gpd.GeoDataFrame(
                    {"corridor_id": list(range(len(corridor_lines)))},
                    geometry=corridor_lines,
                    crs=target_crs,
                )
                logger.debug(
                    f"Created corridors GeoDataFrame: {len(corridors_gdf)} lines in {target_crs}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create GeoDataFrame for corridors: {e}"
                ) from e
        else:
            logger.debug("No corridor lines to create GeoDataFrame")

        # Load CONUS boundary for visual context
        try:
            from topogen.geo_utils import create_conus_mask

            conus_poly = create_conus_mask(conus_boundary_path, target_crs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CONUS boundary for map context: {e}"
            ) from e

        # Create plot
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            logger.debug("Created matplotlib figure (14x10)")
        except Exception as e:
            raise RuntimeError(f"Failed to create matplotlib figure: {e}") from e

        # Plot CONUS boundary for geographic context
        try:
            conus_poly.plot(ax=ax, edgecolor="k", facecolor="none", linewidth=0.4)
        except Exception as e:
            raise RuntimeError(f"Failed to plot CONUS boundary: {e}") from e

        # Plot corridors first (so they appear behind metros)
        if corridors_gdf is not None:
            try:
                corridors_gdf.plot(ax=ax, color="blue", linewidth=1.5, alpha=0.6)
                logger.debug(f"Plotted {len(corridors_gdf)} corridor connections")
            except Exception as e:
                raise RuntimeError(f"Failed to plot corridor connections: {e}") from e

        # Plot metro centroids
        try:
            metros_gdf.plot(
                ax=ax,
                markersize=20,
                color="red",
                alpha=0.8,
                edgecolor="darkred",
                linewidth=0.5,
            )
            logger.debug(f"Plotted {len(metros_gdf)} metro centroids (red circles)")
        except Exception as e:
            raise RuntimeError(f"Failed to plot metro centroids: {e}") from e

        # Add basemap using contextily
        try:
            cx.add_basemap(ax, crs=metros_gdf.crs, attribution=False, alpha=0.3)
            logger.debug("Added contextily basemap to integrated graph map")
        except Exception as e:
            raise RuntimeError(f"Failed to add basemap: {e}") from e

        # Style the plot
        try:
            ax.set_axis_off()
            corridor_count = len(corridor_lines) if corridor_lines else 0
            ax.set_title(
                f"Integrated Graph: {len(metros)} Metro Clusters, {corridor_count} Corridors",
                fontsize=14,
                pad=20,
            )
            plt.tight_layout()
        except Exception as e:
            raise RuntimeError(f"Failed to style the plot: {e}") from e

        # Save JPEG - this is critical
        try:
            logger.debug(
                f"Saving integrated graph visualization to {output_path} (DPI=150, format=JPEG)"
            )
            plt.savefig(output_path, dpi=150, format="jpeg", bbox_inches="tight")
        except Exception as e:
            raise RuntimeError(f"Failed to save JPEG file to {output_path}: {e}") from e
        finally:
            plt.close(fig)
            fig = None  # Mark as closed

        # Validate the JPEG was actually created and has reasonable size
        if not output_path.exists():
            raise RuntimeError(f"JPEG file was not created at {output_path}")

        file_size = output_path.stat().st_size
        if file_size < 1000:  # Less than 1KB suggests a problem
            raise RuntimeError(f"JPEG file appears corrupt (only {file_size} bytes)")

        logger.info(
            f"Saved integrated graph preview → {output_path} ({file_size / 1024:.1f} KB)"
        )

    except Exception as e:
        # Clean up the figure if it exists
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

        # Clean up partial file if it exists
        if output_path.exists():
            try:
                output_path.unlink()
                logger.debug(f"Cleaned up partial file: {output_path}")
            except Exception:
                pass

        # Re-raise with clear context
        if isinstance(e, (ValueError, RuntimeError)):
            raise  # Re-raise our custom errors as-is
        else:
            raise RuntimeError(
                f"Unexpected error during integrated graph map export: {e}"
            ) from e
