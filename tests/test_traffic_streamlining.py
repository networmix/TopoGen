"""Traffic randomness is isolated and distances are metro-center distances."""

import random
from dataclasses import replace

from topogen.scenario.records import Metro, Site
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
    site_plan = replace(
        site_plan,
        sites=tuple(
            replace(site, position=(index * 999999, 10000))
            for index, site in enumerate(site_plan.sites)
        ),
    )
    assert generate_traffic(site_plan, scenario_config.traffic, 42) == a
    assert sorted({d.euclidean_km for d in a[0].demands}) == [300, 600]


def test_gravity_top_k_keeps_each_sites_strongest_partner(site_plan, scenario_config):
    from topogen.scenario.traffic import _top_pairs

    assert _top_pairs({(0, 1): 10.0, (0, 2): 1.0, (1, 2): 4.0}, 1) == {
        (0, 1): 10.0,
        (1, 2): 4.0,
    }
    renamed = {
        i: replace(site_plan.sites[(i - 1) * 2].metro, name=name)
        for i, name in enumerate(("z", "y", "x"), 1)
    }
    site_plan = replace(
        site_plan,
        sites=tuple(
            replace(site, metro=renamed[site.metro.index]) for site in site_plan.sites
        ),
    )
    traffic = scenario_config.traffic
    traffic.model = "gravity"
    traffic.gravity.max_partners_per_dc = 1
    demands = generate_traffic(site_plan, traffic, 42)[0].demands
    assert len(demands) == 4 and sum(d.volume for d in demands) == 30


def test_hose_randomness_is_observable_with_four_sites(site_plan, scenario_config):
    site_plan = replace(
        site_plan,
        sites=(
            *site_plan.sites,
            Site(
                "metro4/dc1",
                Metro(4, "delta", "delta", "4", 900000.0, 0.0, 20.0),
                "dc",
                1,
                "DCRegion",
                10,
                (900000.0, 0.0),
            ),
        ),
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


def test_random_ring_changes_are_inert_for_every_model(scenario_config):
    from topogen.scenario.records import SitePlan

    rng = random.Random(20261026)
    counts = {}
    for model in ("uniform", "gravity", "hose"):
        traffic = scenario_config.traffic
        traffic.model = model
        traffic.gravity.rounding_gbps = 0
        traffic.samples = 2 if model == "hose" else 1
        counts[model] = 0
        for case in range(15):
            sites = []
            for index in range(1, rng.randint(3, 6)):
                metro = Metro(
                    index,
                    str(index),
                    str(index),
                    str(index),
                    index * 100000.0,
                    rng.uniform(0, 300000),
                    rng.uniform(1, 80),
                )
                sites.append(
                    Site(
                        f"metro{index}/dc1",
                        metro,
                        "dc",
                        1,
                        "DCRegion",
                        10,
                        (rng.uniform(-1e6, 1e6), rng.uniform(-1e6, 1e6)),
                    )
                )
            plan = SitePlan(tuple(sites), (), node_rules=())
            expected = generate_traffic(plan, traffic, case)
            assert expected and all(matrix.demands for matrix in expected)
            changed = replace(
                plan,
                sites=tuple(
                    replace(
                        site, position=(rng.uniform(-1e8, 1e8), rng.uniform(-1e8, 1e8))
                    )
                    for site in sites
                ),
            )
            assert generate_traffic(changed, traffic, case) == expected
            assert all(
                a.metro is b.metro
                for a, b in zip(plan.sites, changed.sites, strict=True)
            )
            counts[model] += 1
    assert counts == {"uniform": 15, "gravity": 15, "hose": 15}
