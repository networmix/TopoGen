"""Tests for geographic utilities."""

import pytest
from shapely.geometry import Point

from topogen.geo_utils import (
    CONUS_ALBERS,
    WGS84,
    bearing_to_offset,
    get_transformer,
    point_at_bearing,
    transform_point,
)


def test_get_transformer() -> None:
    transformer = get_transformer(WGS84, CONUS_ALBERS)

    # Test transformation of a known point (Chicago)
    chicago_lon, chicago_lat = -87.6298, 41.8781
    x, y = transformer.transform(chicago_lon, chicago_lat)

    assert isinstance(x, float)
    assert isinstance(y, float)
    assert abs(x) > 1000
    assert abs(y) > 1000


def test_transform_point() -> None:
    # Chicago coordinates
    chicago_wgs84 = Point(-87.6298, 41.8781)

    chicago_albers = transform_point(chicago_wgs84, WGS84, CONUS_ALBERS)

    assert chicago_albers.x != chicago_wgs84.x
    assert chicago_albers.y != chicago_wgs84.y

    assert abs(chicago_albers.x) > 1000
    assert abs(chicago_albers.y) > 1000


def test_bearing_to_offset() -> None:
    # North (0 degrees)
    x, y = bearing_to_offset(0, 1000)
    assert abs(x) < 1e-10
    assert abs(y - 1000) < 1e-10

    # East (90 degrees)
    x, y = bearing_to_offset(90, 1000)
    assert abs(x - 1000) < 1e-10
    assert abs(y) < 1e-10

    # South (180 degrees)
    x, y = bearing_to_offset(180, 1000)
    assert abs(x) < 1e-10
    assert abs(y + 1000) < 1e-10

    # West (270 degrees)
    x, y = bearing_to_offset(270, 1000)
    assert abs(x + 1000) < 1e-10
    assert abs(y) < 1e-10


def test_point_at_bearing() -> None:
    center = Point(0, 0)

    # Point 1000m north
    north_point = point_at_bearing(center, 0, 1000)
    assert abs(north_point.x) < 1e-10
    assert abs(north_point.y - 1000) < 1e-10

    # Point 1000m east
    east_point = point_at_bearing(center, 90, 1000)
    assert abs(east_point.x - 1000) < 1e-10
    assert abs(east_point.y) < 1e-10


@pytest.mark.parametrize(
    "bearing,expected_x,expected_y",
    [
        (0, 0, 1),  # North
        (45, 0.707, 0.707),  # Northeast (approximate)
        (90, 1, 0),  # East
        (135, 0.707, -0.707),  # Southeast
        (180, 0, -1),  # South
        (225, -0.707, -0.707),  # Southwest
        (270, -1, 0),  # West
        (315, -0.707, 0.707),  # Northwest
    ],
)
def test_bearing_directions(
    bearing: float, expected_x: float, expected_y: float
) -> None:
    x, y = bearing_to_offset(bearing, 1.0)

    assert abs(x - expected_x) < 0.01
    assert abs(y - expected_y) < 0.01
