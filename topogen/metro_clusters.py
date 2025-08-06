"""Metropolitan cluster processing for topology generation.

Converts Census Urban Area Centroids (UAC20) data into standardized metro clusters
using point-radius representation for downstream topology algorithms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from shapely.geometry import Point

from topogen.log_config import get_logger

if TYPE_CHECKING:
    from topogen.config import ClusteringConfig, FormattingConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class MetroCluster:
    """Metropolitan cluster with standardized point-radius representation.

    Represents a metro area as a centroid point with equivalent circular radius,
    providing consistent spatial representation for topology generation algorithms.
    """

    metro_id: str
    name: str
    uac_code: str
    land_area_km2: float
    centroid_x: float
    centroid_y: float
    radius_km: float

    @property
    def coordinates(self) -> tuple[float, float]:
        """Return coordinates as (x, y) tuple."""
        return (self.centroid_x, self.centroid_y)

    @property
    def node_key(self) -> tuple[float, float]:
        """Return node key as (x, y) tuple for graph integration."""
        return (self.centroid_x, self.centroid_y)

    @property
    def coordinates_array(self) -> np.ndarray:
        """Return coordinates as numpy array for spatial operations."""
        return np.array([self.centroid_x, self.centroid_y])

    def distance_to(self, other: MetroCluster) -> float:
        """Calculate Euclidean distance to another metro cluster.

        Args:
            other: Target metro cluster.

        Returns:
            Distance in coordinate units (typically meters for projected CRS).
        """
        dx = self.centroid_x - other.centroid_x
        dy = self.centroid_y - other.centroid_y
        return math.sqrt(dx * dx + dy * dy)

    def overlaps_with(
        self, other: MetroCluster, distance_conversion_factor: int = 1000
    ) -> bool:
        """Check if this metro cluster overlaps with another.

        Args:
            other: Target metro cluster.
            distance_conversion_factor: Factor to convert km to meters.

        Returns:
            True if circular areas overlap.
        """
        distance_m = self.distance_to(other)
        combined_radius_m = (
            self.radius_km + other.radius_km
        ) * distance_conversion_factor
        return distance_m < combined_radius_m


def load_metro_clusters(
    uac_path: Path,
    k: int,
    target_crs: str,
    clustering_config: ClusteringConfig,
    formatting_config: FormattingConfig,
    conus_boundary_path: Path | None = None,
) -> list[MetroCluster]:
    """Load metropolitan clusters from Census Urban Area Centroids data.

    Args:
        uac_path: Path to Census 2020 Urban Areas ZIP file.
        k: Number of top urban areas to select by land area.
        target_crs: Target coordinate reference system.
        clustering_config: Configuration for clustering parameters.
        formatting_config: Configuration for formatting and precision.
        conus_boundary_path: Path to CONUS boundary file for territory filtering.

    Returns:
        List of MetroCluster objects with standardized point-radius representation.

    Raises:
        FileNotFoundError: If UAC file not found.
        ValueError: If insufficient urban areas found.
    """
    logger.info(f"Loading metro clusters from: {uac_path}")
    logger.info(
        f"Target: {k} metro clusters (max radius: {clustering_config.max_uac_radius_km}km)"
    )

    if not uac_path.exists():
        raise FileNotFoundError(f"UAC file not found: {uac_path}")

    # Load UAC data and reproject to target CRS
    gdf_raw = gpd.read_file(f"zip://{uac_path}")
    logger.info(f"Loaded {len(gdf_raw):,} urban areas from UAC20 data")
    logger.info(f"UAC source CRS: {gdf_raw.crs}")

    # Validate and reproject CRS
    if gdf_raw.crs is None:
        raise ValueError("UAC data has no CRS information")

    # Filter to CONUS BEFORE reprojection to avoid coordinate issues with Alaska/Hawaii
    if conus_boundary_path is not None:
        from topogen.geo_utils import create_conus_mask

        logger.info("Filtering urban areas to continental US (before reprojection)")
        conus_mask_wgs84 = create_conus_mask(
            conus_boundary_path, "EPSG:4269"
        )  # Same as source CRS
        # Dissolve mask in case it has multiple geometries
        dissolved_mask = conus_mask_wgs84.geometry.union_all()
        gdf_filtered = gdf_raw[gdf_raw.geometry.intersects(dissolved_mask)]
        excluded_count = len(gdf_raw) - len(gdf_filtered)

        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count:,} areas outside continental US")

        gdf_raw = gdf_filtered

    # Now reproject the CONUS-only data
    gdf = gdf_raw.to_crs(target_crs)
    logger.info(f"Reprojected UAC data from {gdf_raw.crs} to {target_crs}")

    # Log coordinate range after reprojection to verify fix
    if len(gdf) > 0:
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        logger.info(
            f"Coordinate bounds after reprojection: X: [{bounds[0]:.0f}, {bounds[2]:.0f}], Y: [{bounds[1]:.0f}, {bounds[3]:.0f}]"
        )

    # Validate sufficient areas available
    if len(gdf) < k:
        conus_note = " after CONUS filtering" if conus_boundary_path else ""
        raise ValueError(
            f"Only {len(gdf)} urban areas available{conus_note}, but {k} requested"
        )

    # Handle override metro clusters
    gdf["ALAND20"] = pd.to_numeric(gdf["ALAND20"], errors="coerce")

    override_areas = pd.DataFrame()
    remaining_k = k

    if clustering_config.override_metro_clusters:
        logger.info(
            f"Processing {len(clustering_config.override_metro_clusters)} override metro clusters"
        )

        override_indices = []
        for override_pattern in clustering_config.override_metro_clusters:
            # Find metros matching the pattern (case-insensitive partial match)
            matches = gdf[
                gdf["NAME20"].str.contains(override_pattern, case=False, na=False)
            ]

            if len(matches) == 0:
                logger.warning(
                    f"Override metro pattern '{override_pattern}' matched no metros"
                )
                continue
            elif len(matches) > 1:
                # Multiple matches - take the largest
                largest_match = matches.loc[matches["ALAND20"].idxmax()]
                logger.info(
                    f"Override pattern '{override_pattern}' matched {len(matches)} metros, selecting largest: {largest_match['NAME20']}"
                )
                override_indices.append(largest_match.name)
            else:
                # Single match
                match = matches.iloc[0]
                logger.info(
                    f"Override pattern '{override_pattern}' matched: {match['NAME20']}"
                )
                override_indices.append(match.name)

        if override_indices:
            # Get override areas and remove from main pool
            override_areas = gdf.loc[override_indices].copy().reset_index(drop=True)
            gdf_remaining = gdf.drop(override_indices).copy()
            remaining_k = k - len(override_areas)

            # Log each override metro that was actually selected
            logger.info(
                f"Successfully selected {len(override_areas):,} override metros:"
            )
            for _idx, row in override_areas.iterrows():
                logger.info(
                    f"  Override: {row['NAME20']} (UAC: {row['UACE20']}, Area: {row['ALAND20']:,.0f} sq m)"
                )

            logger.info(f"Remaining {remaining_k} metros to select by size ranking")
        else:
            gdf_remaining = gdf.copy()
            logger.warning(
                "No override metros found, proceeding with size-based selection only"
            )
    else:
        gdf_remaining = gdf.copy()

    # Select remaining metros by land area
    if remaining_k > 0:
        if len(gdf_remaining) < remaining_k:
            raise ValueError(
                f"Not enough metros available for selection: need {remaining_k} more, "
                f"but only {len(gdf_remaining)} available after override selection"
            )

        size_selected = (
            gdf_remaining.sort_values("ALAND20", ascending=False)
            .head(remaining_k)
            .copy()
            .reset_index(drop=True)
        )

        # Log each dynamically selected metro for consistency with override logging
        if len(size_selected) > 0:
            logger.info(
                f"Successfully selected {len(size_selected)} metros by size ranking:"
            )
            for _idx, row in size_selected.iterrows():
                logger.info(
                    f"  ✓ Dynamic: {row['NAME20']} (UAC: {row['UACE20']}, Area: {row['ALAND20']:,.0f} sq m)"
                )

        # Combine override and size-selected metros
        if len(override_areas) > 0:
            top_areas = pd.concat([override_areas, size_selected], ignore_index=True)
        else:
            top_areas = size_selected
    else:
        # Only override metros
        top_areas = override_areas

    # Enhanced final logging with breakdown
    override_count = len(override_areas) if len(override_areas) > 0 else 0
    size_selected_count = remaining_k if remaining_k > 0 else 0

    logger.info(f"Metro selection complete: {len(top_areas)} total metros")
    if override_count > 0:
        logger.info(f"  • {override_count} override metros (forced inclusion)")
        logger.info(f"  • {size_selected_count} metros selected by size ranking")
    else:
        logger.info(
            f"  • {len(top_areas)} metros selected by size ranking (no overrides)"
        )

    # Validate that we have the expected data structure after sorting/filtering
    if len(top_areas) != k:
        raise ValueError(f"Expected {k} areas after selection, got {len(top_areas)}")
    if not top_areas.index.equals(pd.RangeIndex(k)):
        raise ValueError(
            f"DataFrame index is not sequential 0-{k - 1} after reset_index(). "
            f"Got index: {top_areas.index.tolist()}"
        )

    # Validate required columns exist
    required_cols = ["UACE20", "NAME20", "ALAND20"]
    missing_cols = [col for col in required_cols if col not in top_areas.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in UAC data: {missing_cols}")

    # Calculate representative points (safer than centroids for concave polygons)
    centroids = np.column_stack(
        [
            top_areas.geometry.representative_point().x,
            top_areas.geometry.representative_point().y,
        ]
    )
    logger.info(f"Calculated centroids for {len(centroids)} urban areas")

    # Log sample coordinates for debugging
    if len(centroids) > 0:
        sample_idx = 0
        sample_name = (
            top_areas["NAME20"].iloc[sample_idx]
            if len(top_areas) > sample_idx
            else "N/A"
        )
        logger.info(
            f"Sample metro coordinates: {sample_name} at "
            f"({centroids[sample_idx, 0]:.0f}, {centroids[sample_idx, 1]:.0f}) in {target_crs}"
        )

    # Calculate equivalent circular radius from land area
    # This standardizes differently-shaped metro areas for fair comparison
    land_areas_km2 = top_areas["ALAND20"] / formatting_config.area_conversion_factor
    radii_km = np.clip(
        np.sqrt(land_areas_km2 / math.pi),
        a_min=None,
        a_max=clustering_config.max_uac_radius_km,
    )

    logger.info(
        f"Metro radii: avg {radii_km.mean():.1f}km, range {radii_km.min():.1f}-{radii_km.max():.1f}km "
        f"(capped at {clustering_config.max_uac_radius_km}km)"
    )
    logger.debug("Using equivalent circular radius: r = sqrt(area/π)")

    # Validate data consistency before creating MetroCluster objects
    if len(centroids) != len(top_areas):
        raise ValueError(
            f"Data length mismatch: {len(centroids)} centroids vs {len(top_areas)} areas"
        )
    if len(land_areas_km2) != len(top_areas):
        raise ValueError(
            f"Data length mismatch: {len(land_areas_km2)} land areas vs {len(top_areas)} areas"
        )
    if len(radii_km) != len(top_areas):
        raise ValueError(
            f"Data length mismatch: {len(radii_km)} radii vs {len(top_areas)} areas"
        )

    # Create MetroCluster objects with UACE20 codes as stable IDs
    metro_clusters = []
    for i in range(len(centroids)):
        try:
            # Use UACE20 code directly as metro_id for stability and clarity
            uace_code = top_areas["UACE20"].iloc[i]

            # Validate UACE20 code
            if pd.isna(uace_code) or not str(uace_code).strip():
                raise ValueError(f"Invalid UACE20 code at index {i}: {uace_code}")

            cluster = MetroCluster(
                metro_id=str(uace_code).strip(),
                name=top_areas["NAME20"].iloc[i],
                uac_code=str(uace_code).strip(),
                land_area_km2=round(
                    float(land_areas_km2.iloc[i]), clustering_config.area_precision
                ),
                centroid_x=round(
                    centroids[i, 0], clustering_config.coordinate_precision
                ),
                centroid_y=round(
                    centroids[i, 1], clustering_config.coordinate_precision
                ),
                radius_km=round(float(radii_km[i]), clustering_config.area_precision),
            )
            metro_clusters.append(cluster)

        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(
                f"Failed to create metro cluster at index {i}: {e}. "
                f"This may indicate a data indexing issue. "
                f"Expected {len(centroids)} items but failed accessing index {i}."
            ) from e

    logger.info(f"Created {len(metro_clusters)} metro cluster objects")

    # Validate no duplicate metro IDs
    metro_ids = [cluster.metro_id for cluster in metro_clusters]
    if len(metro_ids) != len(set(metro_ids)):
        duplicates = [id for id in metro_ids if metro_ids.count(id) > 1]
        # Find metro names for the duplicate IDs
        duplicate_info = []
        for cluster in metro_clusters:
            if cluster.metro_id in duplicates:
                duplicate_info.append(f"{cluster.name} ({cluster.metro_id})")
        raise ValueError(f"Duplicate metro IDs found: {', '.join(duplicate_info)}")

    # Export visualization files if requested
    if clustering_config.export_clusters:
        _export_cluster_files(metro_clusters, target_crs)

    return metro_clusters


def _export_cluster_files(metro_clusters: list[MetroCluster], target_crs: str) -> None:
    """Export cluster visualization files.

    Args:
        metro_clusters: List of MetroCluster objects to export.
        target_crs: Target coordinate reference system for export.
    """
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export metro centroids as GeoJSON
    metro_data = []
    for cluster in metro_clusters:
        metro_data.append(
            {
                "metro_id": cluster.metro_id,
                "name": cluster.name,
                "uac_code": cluster.uac_code,
                "land_area_km2": cluster.land_area_km2,
                "centroid_x": cluster.centroid_x,
                "centroid_y": cluster.centroid_y,
                "radius_km": cluster.radius_km,
                "geometry": Point(cluster.centroid_x, cluster.centroid_y),
            }
        )

    if metro_data:
        # Create GeoDataFrame from metro data
        import pandas as pd

        df = pd.DataFrame(metro_data)
        metro_gdf = gpd.GeoDataFrame(df)
        metro_gdf = metro_gdf.set_crs(target_crs)
        metro_path = output_dir / "metro_clusters.geojson"
        if metro_gdf is not None:
            metro_gdf.to_file(metro_path, driver="GeoJSON")
        logger.info(
            f"Exported metro centroids: {metro_path} ({len(metro_data)} clusters)"
        )

    # Export visualization map
    from topogen.visualization import export_cluster_map

    centroids = np.array([c.coordinates for c in metro_clusters])
    jpg_path = output_dir / "metro_clusters.jpg"
    conus_boundary_path = Path("data/cb_2024_us_state_500k.zip")
    export_cluster_map(centroids, jpg_path, conus_boundary_path, target_crs)
    logger.info(f"Exported cluster map: {jpg_path}")
