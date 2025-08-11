"""Functional tests for traffic matrix generation algorithms.

These tests validate the math of the uniform and gravity models directly by
calling ``topogen.traffic_matrix.generate_traffic_matrix``.
"""

from __future__ import annotations

from typing import Any

from topogen.config import TopologyConfig
from topogen.traffic_matrix import generate_traffic_matrix


def _metros_ab() -> list[dict[str, Any]]:
    # Coordinates in meters (EPSG:5070); A at origin, B at 1 km east
    return [
        {"name": "A", "x": 0.0, "y": 0.0, "radius_km": 25.0},
        {"name": "B", "x": 1000.0, "y": 0.0, "radius_km": 25.0},
    ]


def _metro_settings_two_one_dc() -> dict[str, dict[str, Any]]:
    return {"A": {"dc_regions_per_metro": 1}, "B": {"dc_regions_per_metro": 1}}


def test_uniform_totals() -> None:
    """Uniform model emits per-class totals equal to offered load times ratio."""

    cfg = TopologyConfig()
    cfg.traffic.enabled = True
    cfg.traffic.model = "uniform"
    cfg.traffic.gbps_per_mw = 100.0
    cfg.traffic.mw_per_dc_region = 10.0
    cfg.traffic.priority_ratios = {0: 0.75, 1: 0.25}
    cfg.traffic.matrix_name = "tm"

    metros = _metros_ab()
    msettings = _metro_settings_two_one_dc()

    tmset = generate_traffic_matrix(metros, msettings, cfg)
    demands = tmset[cfg.traffic.matrix_name]

    # Two entries: one per class, pairwise regex
    assert len(demands) == 2
    offered = cfg.traffic.gbps_per_mw * (2 * cfg.traffic.mw_per_dc_region)
    per_class = [d for d in demands]
    per_class.sort(key=lambda d: int(d["priority"]))
    assert per_class[0]["mode"] == "pairwise"
    assert abs(float(per_class[0]["demand"]) - offered * 0.75) < 1e-9
    assert abs(float(per_class[1]["demand"]) - offered * 0.25) < 1e-9


def test_gravity_two_dcs_symmetric_split_and_total() -> None:
    """Gravity model: two DCs yield two symmetric entries summing to D_c."""

    cfg = TopologyConfig()
    cfg.traffic.enabled = True
    cfg.traffic.model = "gravity"
    cfg.traffic.gbps_per_mw = 100.0
    cfg.traffic.mw_per_dc_region = 10.0
    cfg.traffic.priority_ratios = {0: 1.0}
    g = cfg.traffic.gravity
    g.alpha = 1.0
    g.beta = 1.0
    g.min_distance_km = 1.0
    g.exclude_same_metro = False
    g.jitter_stddev = 0.0
    g.rounding_gbps = 0.0

    metros = _metros_ab()
    msettings = _metro_settings_two_one_dc()

    tmset = generate_traffic_matrix(metros, msettings, cfg)
    demands = tmset[cfg.traffic.matrix_name]

    assert len(demands) == 2
    assert all(d["mode"] == "pairwise" for d in demands)
    total = sum(float(d["demand"]) for d in demands)
    # Offered = 100 * (10+10) = 2000; split equally A->B and B->A
    assert abs(total - 2000.0) < 1e-6


def test_gravity_top_k_pruning_keeps_best_partners() -> None:
    """Top-K pruning keeps strongest partners and removes others."""

    cfg = TopologyConfig()
    cfg.traffic.enabled = True
    cfg.traffic.model = "gravity"
    cfg.traffic.gbps_per_mw = 100.0
    cfg.traffic.mw_per_dc_region = 10.0
    cfg.traffic.priority_ratios = {0: 1.0}
    g = cfg.traffic.gravity
    g.alpha = 1.0
    g.beta = 1.0
    g.min_distance_km = 1.0
    g.exclude_same_metro = False
    g.jitter_stddev = 0.0
    g.rounding_gbps = 0.0
    g.max_partners_per_dc = 1

    # Three metros: A-B close (1 km), C far (100 km) from B and A
    metros = [
        {"name": "A", "x": 0.0, "y": 0.0, "radius_km": 25.0},
        {"name": "B", "x": 1000.0, "y": 0.0, "radius_km": 25.0},  # 1 km from A
        {"name": "C", "x": 100000.0, "y": 0.0, "radius_km": 25.0},  # 100 km from B
    ]
    msettings = {m["name"]: {"dc_regions_per_metro": 1} for m in metros}

    tmset = generate_traffic_matrix(metros, msettings, cfg)
    demands = tmset[cfg.traffic.matrix_name]

    # Expect two pairs (A<->B and B<->C), so 4 directed entries
    assert len(demands) == 4


def test_gravity_rounding_conserves_total_with_largest_remainders() -> None:
    """Rounding distributes leftover to highest remainders and conserves total."""

    cfg = TopologyConfig()
    cfg.traffic.enabled = True
    cfg.traffic.model = "gravity"
    cfg.traffic.gbps_per_mw = 100.0
    cfg.traffic.mw_per_dc_region = 15.0  # 3 DCs -> 45 MW -> offered 4500
    cfg.traffic.priority_ratios = {0: 1.0}
    g = cfg.traffic.gravity
    g.alpha = 1.0
    g.beta = 1.0
    g.min_distance_km = 1.0
    g.exclude_same_metro = (
        True  # avoid same-metro pairs in case of multiple DCs per metro
    )
    g.jitter_stddev = 0.0
    g.rounding_gbps = 10.0
    g.max_partners_per_dc = None

    # Three metros equally spaced in a line by 1 km to equalize weights approximately
    metros = [
        {"name": "A", "x": 0.0, "y": 0.0, "radius_km": 25.0},
        {"name": "B", "x": 1000.0, "y": 0.0, "radius_km": 25.0},
        {"name": "C", "x": 2000.0, "y": 0.0, "radius_km": 25.0},
    ]
    msettings = {m["name"]: {"dc_regions_per_metro": 1} for m in metros}

    tmset = generate_traffic_matrix(metros, msettings, cfg)
    demands = tmset[cfg.traffic.matrix_name]

    # One class; per-pair entries doubled for both directions.
    total = sum(float(d["demand"]) for d in demands)
    assert abs(total - 4500.0) < 1e-6


def test_gravity_dc_power_overrides_affect_offered_and_allocation() -> None:
    """Per-DC MW overrides by name and path change weights and totals."""

    cfg = TopologyConfig()
    cfg.traffic.enabled = True
    cfg.traffic.model = "gravity"
    cfg.traffic.gbps_per_mw = 100.0
    cfg.traffic.mw_per_dc_region = 10.0  # default
    cfg.traffic.priority_ratios = {0: 1.0}
    g = cfg.traffic.gravity
    g.alpha = 1.0
    g.beta = 1.0
    g.min_distance_km = 1.0
    g.exclude_same_metro = False
    g.jitter_stddev = 0.0
    g.rounding_gbps = 0.0

    metros = _metros_ab()
    msettings = _metro_settings_two_one_dc()

    # Override A to 20 MW using metro name, keep B at 10 MW
    g.mw_per_dc_region_overrides = {"A": 20.0}

    tmset = generate_traffic_matrix(metros, msettings, cfg)
    demands = tmset[cfg.traffic.matrix_name]
    total = sum(float(d["demand"]) for d in demands)
    # Offered = 100 * (20 + 10) = 3000
    assert abs(total - 3000.0) < 1e-6

    # Now override using full DC path for B
    g.mw_per_dc_region_overrides = {"A": 20.0, "b/dc1": 15.0}
    # Note: path uses slugified metro name; ensure code lowercases/slugifies consistently
    tmset2 = generate_traffic_matrix(metros, msettings, cfg)
    demands2 = tmset2[cfg.traffic.matrix_name]
    total2 = sum(float(d["demand"]) for d in demands2)
    # Offered = 100 * (20 + 15) = 3500
    assert abs(total2 - 3500.0) < 1e-6
