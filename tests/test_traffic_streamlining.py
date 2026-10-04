"""Traffic model contracts independent of inventory order and global PRNG."""

import random

from topogen.config import TopologyConfig
from topogen.traffic_matrix import generate_traffic_matrix


def traffic_case(names):
    config = TopologyConfig()
    config.traffic.model = "gravity"
    config.traffic.mw_per_dc_region = 10
    config.traffic.gbps_per_mw = 1
    config.traffic.priority_ratios = {0: 1}
    metros = [{"name": name, "x": i * 1000.0, "y": 0.0} for i, name in enumerate(names)]
    settings = {name: {"dc_regions_per_metro": 1} for name in names}
    return config, metros, settings


def test_gravity_top_k_works_for_reverse_alphabetic_inventory():
    config, metros, settings = traffic_case(["z", "y", "x"])
    config.traffic.gravity.max_partners_per_dc = 1
    demands = generate_traffic_matrix(metros, settings, config)[
        config.traffic.matrix_name
    ]
    assert len(demands) == 4
    assert abs(sum(item["demand"] for item in demands) - 30) < 1e-9


def test_gravity_jitter_is_seeded_without_touching_global_prng():
    config, metros, settings = traffic_case(["a", "b", "c", "d"])
    config.traffic.gravity.jitter_stddev = 0.8
    state = random.getstate()
    first = generate_traffic_matrix(metros, settings, config)
    second = generate_traffic_matrix(metros, settings, config)
    assert first == second
    assert random.getstate() == state
    config.output.scenario_seed += 1
    assert generate_traffic_matrix(metros, settings, config) != first


def test_zero_share_class_does_not_require_jitter_normalization():
    config, metros, settings = traffic_case(["a", "b", "c"])
    config.traffic.priority_ratios = {0: 0.0, 1: 1.0}
    config.traffic.gravity.jitter_stddev = 0.5
    demands = generate_traffic_matrix(metros, settings, config)[
        config.traffic.matrix_name
    ]
    assert {demand["priority"] for demand in demands} == {1}
    assert abs(sum(demand["demand"] for demand in demands) - 30) < 1e-9
