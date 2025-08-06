"""Geographic utilities for coordinate projections and spatial operations."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pyproj
from shapely.geometry import Point

from topogen.log_config import get_logger

logger = get_logger(__name__)

# Standard CRS definitions
WGS84 = "EPSG:4326"  # Geographic coordinate system (lat/lon)
CONUS_ALBERS = "EPSG:5070"  # NAD83 / Conus Albers Equal Area


@lru_cache(maxsize=32)
def get_transformer(src_crs: str, dst_crs: str) -> pyproj.Transformer:
    """Create a coordinate transformer between CRS.

    Args:
        src_crs: Source coordinate reference system.
        dst_crs: Destination coordinate reference system.

    Returns:
        Pyproj transformer for coordinate conversion.
    """
    return pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def transform_point(point: Point, src_crs: str, dst_crs: str) -> Point:
    """Transform a point between coordinate reference systems.

    Args:
        point: Shapely Point geometry.
        src_crs: Source CRS identifier.
        dst_crs: Destination CRS identifier.

    Returns:
        Transformed Point geometry.
    """
    transformer = get_transformer(src_crs, dst_crs)
    x, y = transformer.transform(point.x, point.y)
    return Point(x, y)


def bearing_to_offset(bearing_deg: float, radius_m: float) -> tuple[float, float]:
    """Convert bearing and radius to x,y offset in meters.

    Args:
        bearing_deg: Bearing in degrees (0=North, clockwise).
        radius_m: Distance in meters.

    Returns:
        Tuple of (x_offset, y_offset) in meters.
    """
    bearing_rad = math.radians(bearing_deg)
    x_offset = radius_m * math.sin(bearing_rad)
    y_offset = radius_m * math.cos(bearing_rad)
    return x_offset, y_offset


def point_at_bearing(
    center_point: Point, bearing_deg: float, distance_m: float
) -> Point:
    """Create a point at specified bearing and distance from center.

    Args:
        center_point: Center point (should be in projected coordinates).
        bearing_deg: Bearing in degrees (0=North, clockwise).
        distance_m: Distance in meters.

    Returns:
        New Point at the specified bearing and distance.
    """
    x_offset, y_offset = bearing_to_offset(bearing_deg, distance_m)
    return Point(center_point.x + x_offset, center_point.y + y_offset)


def create_conus_mask(conus_boundary_path: Path, target_crs: str) -> gpd.GeoDataFrame:
    """Create CONUS mask from Census state boundaries.

    Filters out non-contiguous states and territories to create a boundary
    that covers only the continental United States.

    Args:
        conus_boundary_path: Path to CONUS boundary shapefile.
        target_crs: Target coordinate reference system.

    Returns:
        GeoDataFrame with single CONUS polygon in target CRS.

    Raises:
        FileNotFoundError: If boundary file not found.
        ValueError: If no CONUS states found in boundary file.
    """
    if not conus_boundary_path.exists():
        raise FileNotFoundError(
            f"CONUS boundary file not found: {conus_boundary_path}. "
            "Download from: https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip"
        )

    # Read state boundaries
    states = gpd.read_file(f"zip://{conus_boundary_path}")

    # Filter to CONUS: exclude all non-contiguous states and territories
    EXCLUDE_FP = {
        "02",  # Alaska
        "15",  # Hawaii
        "60",  # American Samoa
        "66",  # Guam
        "69",  # Northern Mariana Islands
        "72",  # Puerto Rico
        "78",  # U.S. Virgin Islands
    }
    conus_states = states[~states["STATEFP"].isin(EXCLUDE_FP)]

    if len(conus_states) == 0:
        raise ValueError("No CONUS states found in boundary file")

    # Dissolve into single polygon
    conus_poly = conus_states.dissolve(by=None, as_index=False)

    # Reproject to target CRS
    conus_poly_target = conus_poly.to_crs(target_crs)

    logger.info(f"Created CONUS mask with {len(conus_states)} states")

    return conus_poly_target
