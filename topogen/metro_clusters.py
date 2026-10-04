"""Select Census urban areas and represent them as points with circular radii.

Selection uses land area and explicit overrides. Points come from polygon
interiors; radii come from land area and are capped by config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Geometry
from shapely.geometry import Point

from topogen.context import RunContext
from topogen.log_config import get_logger
from topogen.naming import metro_slug

if TYPE_CHECKING:
    from topogen.config import ClusteringConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class MetroCluster:
    """Urban area represented by an interior point and a circular radius."""

    metro_id: str
    name: str
    name_orig: str
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


def load_metro_clusters(
    uac_path: Path,
    k: int,
    target_crs: str,
    clustering_config: ClusteringConfig,
    conus_boundary_path: Path | None = None,
    context: RunContext | None = None,
) -> list[MetroCluster]:
    """Select metropolitan clusters from Census urban-area polygons.

    Args:
        uac_path: Path to Census 2020 Urban Areas ZIP file.
        k: Number of top urban areas to select by land area.
        target_crs: Target coordinate reference system.
        clustering_config: Configuration for clustering parameters.
        conus_boundary_path: Path to CONUS boundary file for territory filtering.
        context: Optional destinations for GeoJSON and map exports.

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

    gdf_raw = gpd.read_file(f"zip://{uac_path}")
    logger.info(f"Loaded {len(gdf_raw):,} urban areas from UAC20 data")
    logger.info(f"UAC source CRS: {gdf_raw.crs}")

    if gdf_raw.crs is None:
        raise ValueError("UAC data has no CRS information")

    # Exclude non-contiguous states and territories before projection.
    if conus_boundary_path is not None:
        from topogen.geo_utils import create_conus_mask

        logger.info("Filtering urban areas to continental US (before reprojection)")
        conus_mask = create_conus_mask(conus_boundary_path, str(gdf_raw.crs))
        mask_geometry = cast(Geometry, conus_mask.geometry.iloc[0])
        gdf_filtered = gdf_raw[gdf_raw.geometry.intersects(mask_geometry)]
        excluded_count = len(gdf_raw) - len(gdf_filtered)

        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count:,} areas outside continental US")

        gdf_raw = gdf_filtered

    gdf = gdf_raw.to_crs(target_crs)
    logger.info(f"Reprojected UAC data from {gdf_raw.crs} to {target_crs}")

    if len(gdf) > 0:
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        logger.info(
            f"Coordinate bounds after reprojection: X: [{bounds[0]:.0f}, {bounds[2]:.0f}], Y: [{bounds[1]:.0f}, {bounds[3]:.0f}]"
        )

    if len(gdf) < k:
        conus_note = " after CONUS filtering" if conus_boundary_path else ""
        raise ValueError(
            f"Only {len(gdf)} urban areas available{conus_note}, but {k} requested"
        )

    required_cols = {"UACE20", "NAME20", "ALAND20"}
    if missing := required_cols - set(gdf.columns):
        raise ValueError(f"Missing required columns in UAC data: {sorted(missing)}")
    gdf = gdf.reset_index(drop=True)
    gdf["ALAND20"] = pd.to_numeric(gdf["ALAND20"], errors="raise")
    if not np.isfinite(gdf["ALAND20"]).all() or (gdf["ALAND20"] < 0).any():
        raise ValueError("UAC land areas must be finite and non-negative")
    selected = []
    for pattern in clustering_config.override_metro_clusters:
        matches = gdf[gdf["NAME20"].str.contains(pattern, case=False, na=False)]
        if matches.empty:
            raise ValueError(f"Override metro pattern '{pattern}' matched no metros")
        index = matches["ALAND20"].idxmax()
        if index in selected:
            raise ValueError(f"Duplicate metro override selection for '{pattern}'")
        selected.append(index)
    if len(selected) > k:
        raise ValueError(
            f"Metro overrides select {len(selected)} areas, but only {k} requested"
        )
    remainder = gdf.drop(selected).nlargest(k - len(selected), "ALAND20")
    top_areas = gdf.loc[selected + remainder.index.tolist()].reset_index(drop=True)
    logger.info("Selected %d metros (%d overrides)", len(top_areas), len(selected))

    # Calculate representative points (safer than centroids for concave polygons)
    points = top_areas.geometry.representative_point()
    centroids = np.column_stack([points.x, points.y])
    if not np.isfinite(centroids).all():
        raise ValueError("Metro coordinates must be finite")
    logger.info(f"Calculated representative points for {len(centroids)} urban areas")

    # Calculate equivalent circular radius from land area
    land_areas_km2 = top_areas["ALAND20"] / 1_000_000  # Convert m² to km²
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

    # Create MetroCluster objects with UACE20 codes as stable IDs
    metro_clusters = []
    for i in range(len(centroids)):
        try:
            uace_code = top_areas["UACE20"].iloc[i]

            if pd.isna(uace_code) or not str(uace_code).strip():
                raise ValueError(f"Invalid UACE20 code at index {i}: {uace_code}")

            original_name = top_areas["NAME20"].iloc[i]
            cluster = MetroCluster(
                metro_id=str(uace_code).strip(),
                name=metro_slug(original_name),
                name_orig=original_name,
                uac_code=str(uace_code).strip(),
                land_area_km2=round(
                    float(land_areas_km2.iloc[i]), clustering_config.area_precision
                ),
                centroid_x=round(
                    float(centroids[i, 0]), clustering_config.coordinate_precision
                ),
                centroid_y=round(
                    float(centroids[i, 1]), clustering_config.coordinate_precision
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

    metro_ids = [cluster.metro_id for cluster in metro_clusters]
    if len(metro_ids) != len(set(metro_ids)):
        duplicates = [id for id in metro_ids if metro_ids.count(id) > 1]
        duplicate_info = []
        for cluster in metro_clusters:
            if cluster.metro_id in duplicates:
                duplicate_info.append(f"{cluster.name} ({cluster.metro_id})")
        raise ValueError(f"Duplicate metro IDs found: {', '.join(duplicate_info)}")

    if clustering_config.export_clusters and context is not None:
        if conus_boundary_path is None:
            raise ValueError("Cluster maps require conus_boundary_path")
        _export_cluster_files(metro_clusters, target_crs, context, conus_boundary_path)

    return metro_clusters


def _export_cluster_files(
    metro_clusters: list[MetroCluster],
    target_crs: str,
    context: RunContext,
    conus_boundary_path: Path,
) -> None:
    """Export representative points as GeoJSON and a map at context destinations."""
    output_dir = context.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
        df = pd.DataFrame(metro_data)
        metro_gdf = gpd.GeoDataFrame(df)
        metro_gdf = metro_gdf.set_crs(target_crs)
        metro_path = context.path("metro_clusters.geojson")
        metro_gdf.to_file(metro_path, driver="GeoJSON")
        logger.info(
            f"Exported metro representative points: {metro_path} ({len(metro_data)} clusters)"
        )

    from topogen.visualization import export_cluster_map

    centroids = np.array([c.coordinates for c in metro_clusters])
    jpg_path = context.path("metro_clusters.jpg")
    export_cluster_map(centroids, jpg_path, conus_boundary_path, target_crs)
    logger.info(f"Exported cluster map: {jpg_path}")
