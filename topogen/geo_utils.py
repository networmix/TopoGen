"""Build the projected contiguous-US mask shared by filtering and map rendering."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from topogen.log_config import get_logger

logger = get_logger(__name__)


def create_conus_mask(conus_boundary_path: Path, target_crs: str) -> gpd.GeoDataFrame:
    """Dissolve Census state boundaries into a contiguous-US mask in ``target_crs``.

    Exclude Alaska, Hawaii, and territories. Return a single-row GeoDataFrame.
    Raise FileNotFoundError for a missing ZIP or ValueError if no states remain.
    """
    if not conus_boundary_path.exists():
        raise FileNotFoundError(
            f"CONUS boundary file not found: {conus_boundary_path}. "
            "Download from: https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip"
        )

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

    conus_poly = conus_states.dissolve(by=None, as_index=False)

    conus_poly_target = conus_poly.to_crs(target_crs)

    logger.info(f"Created CONUS mask with {len(conus_states)} states")

    return conus_poly_target
