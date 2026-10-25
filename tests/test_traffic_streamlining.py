"""Traffic randomness is isolated and distances are metro-center distances."""

import random

from topogen.scenario.traffic import generate_traffic


def test_gravity_jitter_is_seeded_and_global_prng_unchanged(site_plan, scenario_config):
    traffic = scenario_config.traffic
    traffic.model = "gravity"
    traffic.gravity.jitter_stddev = 0.8
    state = random.getstate()
    a = generate_traffic(site_plan, traffic, 42)
    assert generate_traffic(site_plan, traffic, 42) == a
    assert random.getstate() == state
    assert generate_traffic(site_plan, traffic, 43) != a


def test_hose_samples_are_complete_sets(site_plan, scenario_config):
    scenario_config.traffic.model = "hose"
    scenario_config.traffic.samples = 3
    matrices = generate_traffic(site_plan, scenario_config.traffic, 42)
    assert [m.name for m in matrices] == ["default_1", "default_2", "default_3"]
    assert all(abs(sum(d.volume for d in m.demands) - 30) < 0.1 for m in matrices)


def test_ring_coordinates_cannot_change_gravity(site_plan, scenario_config):
    scenario_config.traffic.model = "gravity"
    a = generate_traffic(site_plan, scenario_config.traffic, 42)
    for index, (_, attrs) in enumerate(site_plan.nodes(data=True)):
        attrs.update(pos_x=index * 999999, pos_y=10000, radius_m=100000)
    assert generate_traffic(site_plan, scenario_config.traffic, 42) == a
    assert sorted({d.euclidean_km for d in a[0].demands}) == [300, 600]


def test_gravity_top_k_keeps_each_sites_strongest_partner(site_plan, scenario_config):
    from topogen.scenario.traffic import _top_pairs

    assert _top_pairs({(0, 1): 10.0, (0, 2): 1.0, (1, 2): 4.0}, 1) == {
        (0, 1): 10.0,
        (1, 2): 4.0,
    }
    for i, name in enumerate(("z", "y", "x"), 1):
        for kind in ("pop", "dc"):
            site_plan.nodes[f"metro{i}/{kind}1"]["metro_name"] = name
    traffic = scenario_config.traffic
    traffic.model = "gravity"
    traffic.gravity.max_partners_per_dc = 1
    demands = generate_traffic(site_plan, traffic, 42)[0].demands
    assert len(demands) == 4 and sum(d.volume for d in demands) == 30


def test_hose_randomness_is_observable_with_four_sites(site_plan, scenario_config):
    site_plan.add_node(
        "metro4/dc1",
        **{
            **site_plan.nodes["metro3/dc1"],
            "metro_name": "delta",
            "center_x": 900000.0,
        },
    )
    traffic = scenario_config.traffic
    traffic.model = "hose"
    traffic.samples = 3
    traffic.gravity.rounding_gbps = 0
    state = random.getstate()
    matrices = generate_traffic(site_plan, traffic, 42)
    assert random.getstate() == state
    assert matrices == generate_traffic(site_plan, traffic, 42)
    assert matrices[0].demands != matrices[1].demands
    assert matrices[0].demands != generate_traffic(site_plan, traffic, 43)[0].demands
    for matrix in matrices:
        assert len(matrix.demands) == 12
        for site in (f"metro{i}/dc1" for i in range(1, 5)):
            assert (
                abs(sum(d.volume for d in matrix.demands if d.source == site) - 10)
                < 1e-7
            )
            assert (
                abs(sum(d.volume for d in matrix.demands if d.target == site) - 10)
                < 1e-7
            )
