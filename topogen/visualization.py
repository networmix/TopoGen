"""Visualization utilities for topology generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import contextily as cx  # type: ignore
import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle
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
            )  # type: ignore
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
            gdf.plot(ax=ax, markersize=24, color="red", alpha=0.7)
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
    dpi: int = 150,
) -> None:
    """Export JPEG preview map of integrated graph with metro clusters and corridors.

    Args:
        metros: List of metro clusters.
        graph: Integrated NetworkX graph with corridor tags.
        output_path: Path to save JPEG file.
        conus_boundary_path: Path to CONUS boundary shapefile for visual context.
        target_crs: Target coordinate reference system.
        dpi: Output image dots-per-inch when saving.

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

        # Extract metro centroids and radii (meters in target CRS)
        centroids = np.array([metro.coordinates for metro in metros])
        metro_radii_m = [
            float(getattr(metro, "radius_km", 0.0)) * 1000.0 for metro in metros
        ]

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
            )  # type: ignore
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
                )  # type: ignore
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

        # Plot metro circles (accurate radius) behind points
        try:
            for (x0, y0), r_m in zip(centroids, metro_radii_m, strict=False):
                if r_m > 0.0:
                    circ = Circle(
                        (float(x0), float(y0)),
                        float(r_m),
                        fill=True,
                        facecolor="royalblue",
                        edgecolor="royalblue",
                        linewidth=0.8,
                        alpha=0.2,
                        zorder=1,
                    )
                    ax.add_patch(circ)
        except Exception as e:
            raise RuntimeError(f"Failed to draw metro radius circles: {e}") from e

        # Metro name labels outside the circle with leader lines (deterministic angles)
        label_extent_x: list[float] = []
        label_extent_y: list[float] = []
        try:
            import math as _m

            metro_names_list = [str(m.name) for m in metros]
            for (x0, y0), r_m, label in zip(
                centroids, metro_radii_m, metro_names_list, strict=False
            ):
                try:
                    if float(r_m) <= 0.0:
                        continue
                    h = abs(hash(label))
                    phi = (h % 360) * _m.pi / 180.0
                    offset = max(0.08 * float(r_m), 8000.0)
                    bx = float(x0) + float(r_m) * _m.cos(phi)
                    by = float(y0) + float(r_m) * _m.sin(phi)
                    lx = float(x0) + (float(r_m) + offset) * _m.cos(phi)
                    ly = float(y0) + (float(r_m) + offset) * _m.sin(phi)
                    ax.plot(
                        [bx, lx],
                        [by, ly],
                        color="gray",
                        linewidth=0.5,
                        alpha=0.5,
                        zorder=2,
                    )
                    ha = "left" if _m.cos(phi) >= 0.0 else "right"
                    ax.text(
                        lx,
                        ly,
                        label,
                        fontsize=8,
                        color="black",
                        ha=ha,
                        va="center",
                        zorder=6,
                        bbox=dict(
                            boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.5
                        ),
                    )
                    label_extent_x.append(lx)
                    label_extent_y.append(ly)
                except Exception:
                    continue
        except Exception:
            pass

        # Plot metro centroids
        try:
            metros_gdf.plot(
                ax=ax,
                markersize=60,
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
            # Include metro circle extents (and labels) in view
            try:
                circ_xs: list[float] = []
                circ_ys: list[float] = []
                for (x0, y0), r_m in zip(centroids, metro_radii_m, strict=False):
                    circ_xs.extend([float(x0) - float(r_m), float(x0) + float(r_m)])
                    circ_ys.extend([float(y0) - float(r_m), float(y0) + float(r_m)])
                all_xs = circ_xs + label_extent_x
                all_ys = circ_ys + label_extent_y
                if len(all_xs) >= 2 and len(all_ys) >= 2:
                    pad = 0.05
                    min_x, max_x = min(all_xs), max(all_xs)
                    min_y, max_y = min(all_ys), max(all_ys)
                    dx = max_x - min_x
                    dy = max_y - min_y
                    if dx > 0 and dy > 0:
                        ax.set_xlim(min_x - pad * dx, max_x + pad * dx)
                        ax.set_ylim(min_y - pad * dy, max_y + pad * dy)
            except Exception:
                pass
            plt.tight_layout()
        except Exception as e:
            raise RuntimeError(f"Failed to style the plot: {e}") from e

        # Save JPEG - this is critical
        try:
            logger.debug(
                f"Saving integrated graph visualization to {output_path} (DPI={int(dpi)}, format=JPEG)"
            )
            plt.savefig(output_path, dpi=int(dpi), format="jpeg", bbox_inches="tight")
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


def export_site_graph_map(
    G: nx.MultiGraph,
    output_path: Path,
    *,
    figure_size: tuple[int, int] = (14, 10),
    metro_scale: float = 1.0,
    dpi: int = 300,
    target_crs: str | None = None,
) -> None:
    """Export a JPEG visualization of the site-level network graph.

    Draws each metro's radius as a thin circle, places sites at their assigned
    ``pos_x``/``pos_y`` coordinates, and renders edges as straight segments
    between sites. Each edge is labeled with its adjacency capacity
    (``target_capacity`` if present; else ``base_capacity``; else ``capacity``).

    Required node attributes:
        - ``pos_x`` and ``pos_y``: site coordinates in target CRS units.
        - ``center_x`` and ``center_y``: metro center coordinates.
        - ``radius_m``: metro radius in meters.

    Args:
        G: Site-level MultiGraph with required node metadata.
        output_path: Path to save the JPEG.
        figure_size: Matplotlib figure size in inches (width, height).
        metro_scale: Multiplier applied to visualized metro circle radius.
        dpi: Output image dots-per-inch when saving.
        target_crs: Target coordinate reference system string for basemap.

    Raises:
        ValueError: If required attributes are missing.
        RuntimeError: If rendering or file output fails.
    """
    fig = None
    try:
        logger.info(f"Exporting site graph visualization to {output_path}")

        if G is None or G.number_of_nodes() == 0:
            raise ValueError("Cannot create site graph map: graph is empty or None")

        # Validate node attributes and collect per-metro circle parameters
        metros: dict[int, tuple[float, float, float]] = {}
        metro_names: dict[int, str] = {}
        xs: list[float] = []
        ys: list[float] = []
        for node_id, data in G.nodes(data=True):
            try:
                x = float(data["pos_x"])  # type: ignore[index]
                y = float(data["pos_y"])  # type: ignore[index]
                cx = float(data["center_x"])  # type: ignore[index]
                cy = float(data["center_y"])  # type: ignore[index]
                r = float(data["radius_m"])  # type: ignore[index]
                idx = int(data.get("metro_idx", 0))
            except Exception as exc:
                raise ValueError(
                    f"Node '{node_id}' missing required visualization attributes"
                ) from exc
            xs.append(x)
            ys.append(y)
            if idx not in metros:
                metros[idx] = (cx, cy, r)
            # Capture metro canonical name if present
            try:
                name_val = str(data.get("metro_name", "")).strip()
                if name_val:
                    metro_names[idx] = name_val
            except Exception:
                pass

        # Create figure/axes
        try:
            fig, ax = plt.subplots(figsize=figure_size)
        except Exception as e:
            raise RuntimeError(f"Failed to create matplotlib figure: {e}") from e

        # Draw metro circles (light outline) and labels (outside the circle)
        label_extent_x: list[float] = []
        label_extent_y: list[float] = []
        for _idx, (cx, cy, r) in metros.items():
            rr = float(r) * float(metro_scale)
            if rr > 0.0:
                try:
                    circ = Circle(
                        (cx, cy),
                        rr,
                        fill=False,
                        edgecolor="gray",
                        linewidth=0.8,
                        alpha=0.6,
                    )
                    ax.add_patch(circ)
                except Exception as e:
                    raise RuntimeError(f"Failed to draw metro circle: {e}") from e
            # Metro name label outside the circle at deterministic angle
            try:
                label = metro_names.get(_idx, f"metro{_idx}")
                import math as _m

                h = abs(hash(label))
                phi = (h % 360) * _m.pi / 180.0
                # Offset beyond the circle radius
                offset = max(0.08 * rr, 8000.0)
                lx = float(cx) + (rr + offset) * _m.cos(phi)
                ly = float(cy) + (rr + offset) * _m.sin(phi)
                # Keep extents inclusive of labels for autoscale
                label_extent_x.append(lx)
                label_extent_y.append(ly)
                # Leader line from circle boundary to label
                bx = float(cx) + rr * _m.cos(phi)
                by = float(cy) + rr * _m.sin(phi)
                ax.plot(
                    [bx, lx], [by, ly], color="gray", linewidth=0.5, alpha=0.5, zorder=3
                )
                ha = "left" if _m.cos(phi) >= 0.0 else "right"
                ax.text(
                    lx,
                    ly,
                    label,
                    fontsize=8,
                    color="black",
                    ha=ha,
                    va="center",
                    zorder=4,
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.5
                    ),
                )
            except Exception:
                pass

        # Draw edges and labels
        for u, v, k, data in G.edges(keys=True, data=True):
            try:
                ux = float(G.nodes[u]["pos_x"])  # type: ignore[index]
                uy = float(G.nodes[u]["pos_y"])  # type: ignore[index]
                vx = float(G.nodes[v]["pos_x"])  # type: ignore[index]
                vy = float(G.nodes[v]["pos_y"])  # type: ignore[index]
            except Exception as e:
                raise ValueError(f"Edge {u}-{v} endpoints missing positions") from e

            try:
                ax.plot([ux, vx], [uy, vy], color="steelblue", linewidth=1.2, alpha=0.9)
            except Exception as e:
                raise RuntimeError(f"Failed to draw edge {u}-{v}") from e

            # Label with adjacency capacity (prefer total target capacity)
            val = data.get(
                "target_capacity", data.get("base_capacity", data.get("capacity", None))
            )
            # Skip inter-metro adjacency labels; corridor will be labeled once per metro pair
            if str(data.get("link_type", "")) == "inter_metro_corridor":
                continue
            if val is not None:
                try:
                    cap = float(val)
                    # Deterministic jitter to reduce overlap across edges
                    edge_key = f"{u}|{v}|{k}"
                    h = abs(hash(edge_key))
                    # Along-edge parameter in [0.35, 0.65]
                    t = 0.35 + 0.30 * ((h % 7) / 6.0)
                    mx = (1.0 - t) * ux + t * vx
                    my = (1.0 - t) * uy + t * vy
                    # Slight perpendicular offset for readability (vary magnitude/sign)
                    dx = vx - ux
                    dy = vy - uy
                    import math as _m

                    length = _m.hypot(dx, dy)
                    if length > 0.0:
                        step = (h >> 3) % 4  # 0..3
                        off_frac = 0.03 + 0.02 * float(step)  # 0.03..0.09
                        sign = -1.0 if ((h >> 9) % 2 == 0) else 1.0
                        off = sign * off_frac
                        ox = -dy * off
                        oy = dx * off
                    else:
                        ox = 0.0
                        oy = 0.0
                    ax.text(
                        mx + ox,
                        my + oy,
                        f"{cap:,.0f}",
                        fontsize=6,
                        color="black",
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6
                        ),
                        zorder=5,
                    )
                except Exception:
                    # Labeling is optional; continue on format errors
                    pass

        # Aggregate corridor capacities per directed metro pair and place labels along centerline
        corridor_caps: dict[tuple[str, str], float] = {}
        for _u2, _v2, _k2, d2 in G.edges(keys=True, data=True):
            if str(d2.get("link_type", "")) != "inter_metro_corridor":
                continue
            s_name = d2.get("source_metro")
            t_name = d2.get("target_metro")
            if not isinstance(s_name, str) or not isinstance(t_name, str):
                continue
            cap_val = d2.get("target_capacity", d2.get("base_capacity", None))
            try:
                cap_num = float(cap_val) if cap_val is not None else 0.0
            except Exception:
                cap_num = 0.0
            key = (s_name, t_name)
            corridor_caps[key] = corridor_caps.get(key, 0.0) + cap_num

        # Build name -> (cx, cy) lookup from collected metros and names
        name_to_center: dict[str, tuple[float, float]] = {}
        for idx, (cx, cy, _r) in metros.items():
            nm = metro_names.get(idx, f"metro{idx}")
            name_to_center[nm] = (float(cx), float(cy))

        # Place corridor labels
        corr_label_x: list[float] = []
        corr_label_y: list[float] = []
        for (s_name, t_name), cap_num in corridor_caps.items():
            a = name_to_center.get(str(s_name))
            b = name_to_center.get(str(t_name))
            if a is None or b is None:
                continue
            ax0, ay0 = a
            bx0, by0 = b
            dx = bx0 - ax0
            dy = by0 - ay0
            import math as _m

            length = _m.hypot(dx, dy)
            if length <= 0.0:
                continue
            # Along-line parameter and perpendicular offset based on pair hash
            h = abs(hash(f"{s_name}->{t_name}"))
            tpar = 0.45 + 0.10 * ((h % 5) / 4.0)  # 0.45..0.55
            mx = ax0 + tpar * dx
            my = ay0 + tpar * dy
            step = (h >> 3) % 4
            off_frac = 0.04 + 0.02 * float(step)  # 0.04..0.10
            sign = -1.0 if ((h >> 9) % 2 == 0) else 1.0
            ox = sign * (-dy) * off_frac
            oy = sign * (dx) * off_frac
            lx = mx + ox
            ly = my + oy
            corr_label_x.append(lx)
            corr_label_y.append(ly)
            try:
                ax.text(
                    lx,
                    ly,
                    f"{cap_num:,.0f}",
                    fontsize=7,
                    color="black",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6
                    ),
                    zorder=5,
                )
            except Exception:
                pass

        # Draw site nodes on top
        try:
            ax.scatter(
                xs,
                ys,
                s=20,
                c="crimson",
                alpha=0.9,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to draw site nodes: {e}") from e

        # Add basemap using contextily to match integrated graph background
        try:
            if isinstance(target_crs, str) and target_crs.strip():
                import contextily as _cx  # type: ignore

                _cx.add_basemap(ax, crs=target_crs, attribution=False, alpha=0.3)
        except Exception as e:
            raise RuntimeError(f"Failed to add basemap to site graph: {e}") from e

        # Style and bounds
        try:
            # Use 'box' so explicit limits below are respected without warnings
            ax.set_aspect("equal", adjustable="box")
            pad = 0.05
            circ_xs: list[float] = []
            circ_ys: list[float] = []
            for cx, cy, r in metros.values():
                rr = float(r) * float(metro_scale)
                circ_xs.extend([float(cx) - rr, float(cx) + rr])
                circ_ys.extend([float(cy) - rr, float(cy) + rr])
            all_xs = xs + circ_xs + label_extent_x + corr_label_x
            all_ys = ys + circ_ys + label_extent_y + corr_label_y
            if all_xs and all_ys:
                min_x, max_x = min(all_xs), max(all_xs)
                min_y, max_y = min(all_ys), max(all_ys)
                dx = max_x - min_x
                dy = max_y - min_y
                ax.set_xlim(min_x - pad * dx, max_x + pad * dx)
                ax.set_ylim(min_y - pad * dy, max_y + pad * dy)
            ax.set_axis_off()
            ax.set_title("Site-level Network Graph", fontsize=14, pad=16)
            plt.tight_layout()
        except Exception as e:
            raise RuntimeError(f"Failed to style site graph plot: {e}") from e

        # Save JPEG
        try:
            plt.savefig(output_path, dpi=int(dpi), format="jpeg", bbox_inches="tight")
        except Exception as e:
            raise RuntimeError(f"Failed to save site graph JPEG: {e}") from e
        finally:
            plt.close(fig)
            fig = None

        if not output_path.exists():
            raise RuntimeError(f"JPEG file was not created at {output_path}")

        size = output_path.stat().st_size
        if size < 1000:
            raise RuntimeError(f"JPEG file appears corrupt (only {size} bytes)")
        logger.info(
            f"Saved site graph visualization → {output_path} ({size / 1024:.1f} KB)"
        )
    except Exception as e:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Unexpected error during site graph map export: {e}") from e


def export_blueprint_diagram(
    blueprint_name: str,
    blueprint_def: dict[str, Any],
    net: Any,
    selected_site_path: str,
    output_path: Path,
    *,
    dpi: int = 300,
    figure_size: tuple[int, int] = (14, 6),
    seed: int = 7,
) -> None:
    """Export a two-panel diagram for a single blueprint.

    Left panel: abstract group-level adjacency from the blueprint definition.
    Right panel: concrete expanded nodes and internal links for one site instance
    (selected by the caller), with external links shown as small markers from
    internal nodes.

    Args:
        blueprint_name: Name of the blueprint.
        blueprint_def: Blueprint mapping with ``groups`` and ``adjacency`` keys.
        net: Expanded DSL network object (from ngraph) containing nodes/links.
        selected_site_path: Site path prefix to visualize (e.g., "metro1/pop2").
        output_path: Path where a JPEG will be saved.
        dpi: Image DPI for saving.
        figure_size: Matplotlib figure size.
        seed: Layout seed for deterministic positioning.

    Raises:
        ValueError: On invalid inputs or missing data in the blueprint.
        RuntimeError: If rendering or file IO fails.
    """
    fig = None
    try:
        if not isinstance(blueprint_def, dict):
            raise ValueError("blueprint_def must be a mapping")
        if "groups" not in blueprint_def or not isinstance(
            blueprint_def.get("groups"), dict
        ):
            raise ValueError("blueprint_def must include a 'groups' mapping")
        if "adjacency" not in blueprint_def or not isinstance(
            blueprint_def.get("adjacency"), (list, tuple)
        ):
            raise ValueError("blueprint_def must include an 'adjacency' list")

        # Build abstract and concrete views using helper module.
        from .blueprint_viz import build_abstract_view, collect_concrete_site

        abs_view = build_abstract_view(blueprint_def, include_self_loops=True)

        internal_nodes, node_pos, internal_links = collect_concrete_site(
            net, str(selected_site_path)
        )
        if not internal_nodes:
            raise ValueError(
                f"No expanded nodes found under site '{selected_site_path}'"
            )

        # Start figure
        fig, axes = plt.subplots(1, 2, figsize=figure_size)
        ax_abs, ax_conc = axes

        # Abstract panel drawing
        try:
            import networkx as _nx

            pos = _nx.spring_layout(abs_view.graph, seed=seed)
            _nx.draw_networkx_nodes(
                abs_view.graph, pos, node_size=900, node_color="#f0f0ff", ax=ax_abs
            )
            _nx.draw_networkx_labels(
                abs_view.graph, pos, labels=abs_view.node_labels, font_size=8, ax=ax_abs
            )
            _nx.draw_networkx_edges(
                abs_view.graph, pos, width=1.2, edge_color="#666", ax=ax_abs
            )
            _nx.draw_networkx_edge_labels(
                abs_view.graph,
                pos,
                edge_labels=abs_view.edge_labels,
                font_size=7,
                ax=ax_abs,
            )
            # Draw self-loops around nodes if present
            try:
                import math as _m

                for node, label in abs_view.self_loops:
                    if node not in pos:
                        continue
                    x, y = pos[node]
                    r = 0.15  # small loop radius in layout units
                    theta = np.linspace(0.0, 2.0 * _m.pi, 64)
                    xs = x + r * np.cos(theta)
                    ys = y + r * np.sin(theta)
                    ax_abs.plot(xs, ys, color="#888", linewidth=1.0)
                    ax_abs.text(
                        x + r * 1.1,
                        y + r * 1.1,
                        label,
                        fontsize=7,
                        ha="left",
                        va="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6
                        ),
                    )
            except Exception:
                pass
            ax_abs.set_title(f"Abstract: {blueprint_name}")
            ax_abs.set_axis_off()
        except Exception as e:  # pragma: no cover - visualization fallback
            raise RuntimeError(f"Failed to render abstract panel: {e}") from e

        # Concrete panel drawing
        try:
            # Internal links
            for s, t, _cap in internal_links:
                if s in node_pos and t in node_pos:
                    x0, y0 = node_pos[s]
                    x1, y1 = node_pos[t]
                    ax_conc.plot([x0, x1], [y0, y1], color="#4a90e2", linewidth=1.0)

            # Internal nodes
            xs = [node_pos[n][0] for n in internal_nodes if n in node_pos]
            ys = [node_pos[n][1] for n in internal_nodes if n in node_pos]
            ax_conc.scatter(
                xs, ys, s=80, c="#cc2a36", edgecolors="white", linewidths=0.6
            )
            # Labels: short local names
            for n in internal_nodes:
                if n not in node_pos:
                    continue
                x, y = node_pos[n]
                short = n.split("/", 2)[-1]
                ax_conc.text(x, y, short, fontsize=7, ha="center", va="bottom")

            # External adjacency markers intentionally omitted

            ax_conc.set_title(f"Concrete: {selected_site_path}")
            ax_conc.set_axis_off()
        except Exception as e:  # pragma: no cover - visualization fallback
            raise RuntimeError(f"Failed to render concrete panel: {e}") from e

        fig.suptitle(f"Blueprint '{blueprint_name}' example", fontsize=12)
        plt.tight_layout()

        try:
            plt.savefig(output_path, dpi=int(dpi), format="jpeg", bbox_inches="tight")
        finally:
            plt.close(fig)
            fig = None

        if not output_path.exists():
            raise RuntimeError(f"JPEG file was not created at {output_path}")
        if output_path.stat().st_size < 1000:
            raise RuntimeError("Blueprint diagram JPEG appears corrupt (<1KB)")
        logger.info("Saved blueprint diagram → %s", str(output_path))
    except Exception as e:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(
            f"Unexpected error during blueprint diagram export: {e}"
        ) from e
