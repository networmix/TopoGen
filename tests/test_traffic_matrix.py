"""Traffic records preserve model rules and offer every distinct site pair."""

from fractions import Fraction

import pytest
import yaml
from ngraph import Scenario, prepare

from topogen.scenario.traffic import _quantize, generate_traffic, to_demand_sets


def test_three_site_uniform_is_six_demands_of_100(site_plan, scenario_config):
    scenario_config.traffic.mw_per_dc_region = 200
    matrices = generate_traffic(site_plan, scenario_config.traffic, 42)
    demands = matrices[0].demands
    assert len(demands) == 6
    assert {(d.source, d.target) for d in demands} == {
        (a, b)
        for a in ("metro1/dc1", "metro2/dc1", "metro3/dc1")
        for b in ("metro1/dc1", "metro2/dc1", "metro3/dc1")
        if a != b
    }
    assert all(d.volume == 100 for d in demands)
    doc = {
        "network": {
            "nodes": {
                f"{site}/n{i}": {}
                for site in site_plan
                if "/dc" in site
                for i in (1, 2)
            },
            "links": [],
        },
        "demands": to_demand_sets(matrices),
    }
    prep = prepare(Scenario.from_yaml(yaml.safe_dump(doc)), demand_sets=("default",))
    assert all(a.discarded_units == 0 for a in prep.demand_sets["default"].accounting)
    assert (
        sum(d.nominal_units for d in prep.demand_sets["default"].demands)
        * prep.topology.q
        == 600
    )


@pytest.mark.parametrize(
    "model,error",
    [
        ("uniform", "sites=1, volume=10"),
        ("gravity", "zero or non-finite total weight"),
        ("hose", "one DC exceeds"),
    ],
)
def test_one_site_refusal_keeps_other_models(site_plan, scenario_config, model, error):
    for site in list(site_plan):
        if not site.startswith("metro1/"):
            site_plan.remove_node(site)
    scenario_config.traffic.model = model
    with pytest.raises(ValueError, match=error):
        generate_traffic(site_plan, scenario_config.traffic, 42)
    site_plan.remove_node("metro1/dc1")
    assert generate_traffic(site_plan, scenario_config.traffic, 42) == ()


@pytest.mark.parametrize("model", ["gravity", "hose"])
def test_two_equal_site_classes_are_7_5_and_2_5(site_plan, scenario_config, model):
    for site in list(site_plan):
        if site.startswith("metro3/"):
            site_plan.remove_node(site)
    traffic = scenario_config.traffic
    traffic.model, traffic.priority_ratios = model, {0: 0.75, 1: 0.25}
    demands = generate_traffic(site_plan, traffic, 42)[0].demands
    assert [(d.priority, d.volume) for d in demands] == [
        (0, Fraction("7.5")),
        (0, Fraction("7.5")),
        (1, Fraction("2.5")),
        (1, Fraction("2.5")),
    ]


def test_pair_quantization_largest_remainder():
    assert _quantize({(0, 1): 5.1, (0, 2): 3, (1, 2): 1.9}, 10, 2, "floor") == {
        (1, 2): 2,
        (0, 1): 6,
        (0, 2): 2,
    }


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("floor", {(0, 1): 4, (0, 2): 4, (1, 2): 2}),
        ("nearest", {(0, 1): 4, (0, 2): 4, (1, 2): 2}),
        ("ceil", {(0, 1): 6, (0, 2): 4, (1, 2): 4}),
    ],
)
def test_pair_quantization_policy_is_observable(policy, expected):
    assert _quantize({(0, 1): 4.9, (0, 2): 3, (1, 2): 2.1}, 10, 2, policy) == expected
    # Nearest can overshoot; preserving that behavior distinguishes it from
    # floor followed by positive-residual redistribution.
    if policy == "nearest":
        assert _quantize({(0, 1): 5.1, (0, 2): 3, (1, 2): 1.9}, 10, 2, policy) == {
            (0, 1): 6,
            (0, 2): 4,
            (1, 2): 2,
        }


@pytest.mark.parametrize("counts,total", [((2, 1), 600), ((2, 2), 1200)])
def test_uniform_unequal_inventory_classes_and_devices(
    site_plan, scenario_config, counts, total
):
    for site in list(site_plan):
        if site.startswith("metro3/"):
            site_plan.remove_node(site)
    for metro, count in enumerate(counts, 1):
        if count == 2:
            site_plan.add_node(
                f"metro{metro}/dc2",
                **{**site_plan.nodes[f"metro{metro}/dc1"], "site_ordinal": 2},
            )
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region = total / sum(counts)
    traffic.priority_ratios = {0: 0.75, 1: 0.25}
    matrices = generate_traffic(site_plan, traffic, 42)
    demands = matrices[0].demands
    n = sum(counts)
    assert len(demands) == 2 * n * (n - 1)
    assert {d.volume for d in demands if d.priority == 0} == {75}
    assert {d.volume for d in demands if d.priority == 1} == {25}
    sites = [site for site in site_plan if "/dc" in site]
    doc = {
        "units": {"resolution": 0.25},
        "network": {
            "nodes": {f"{site}/n{i}": {} for site in sites for i in (1, 2)},
            "links": [],
        },
        "demands": to_demand_sets(matrices),
    }
    prepared = prepare(
        Scenario.from_yaml(yaml.safe_dump(doc)), demand_sets=("default",)
    )
    resolved = prepared.demand_sets["default"]
    assert all(not a.discarded_units for a in resolved.accounting)
    assert sum(d.nominal_units for d in resolved.demands) * prepared.topology.q == total
    for priority, ratio in ((0, 0.75), (1, 0.25)):
        selected = [d for d in resolved.demands if d.priority == priority]
        assert len(selected) == n * (n - 1) * 4
        assert {d.nominal_units * prepared.topology.q for d in selected} == {
            (75 if priority == 0 else 25) / 4
        }
        assert (
            sum(d.nominal_units for d in selected) * prepared.topology.q
            == total * ratio
        )


def test_uniform_zero_share_and_zero_power_keep_the_site_inventory(
    site_plan, scenario_config
):
    traffic = scenario_config.traffic
    traffic.priority_ratios = {0: 0, 1: 1}
    traffic.flow_policy_config = {0: "TE_MAXFLOW_UNLIM", 1: "SHORTEST_PATHS_ECMP"}
    traffic.gravity.mw_per_dc_region_overrides = {"Beta": 0, "beta": 0}
    demands = generate_traffic(site_plan, traffic, 42)[0].demands
    assert len(demands) == 6
    assert {d.priority for d in demands} == {1}
    assert {d.source for d in demands} == {
        "metro1/dc1",
        "metro2/dc1",
        "metro3/dc1",
    }
    assert {d.volume for d in demands} == {Fraction(10, 3)}
    entries = to_demand_sets(generate_traffic(site_plan, traffic, 42))["default"]
    assert all(d["preset"] == "SHORTEST_PATHS_ECMP" for d in entries)
    traffic.mw_per_dc_region = 0
    assert not generate_traffic(site_plan, traffic, 42)[0].demands
    traffic.enabled = False
    assert generate_traffic(site_plan, traffic, 42) == ()


def test_emission_retains_each_prioritys_own_preset(site_plan, scenario_config):
    traffic = scenario_config.traffic
    traffic.priority_ratios = {0: 0.75, 1: 0.25}
    traffic.flow_policy_config = {0: "TE_MAXFLOW_UNLIM", 1: "SHORTEST_PATHS_ECMP"}
    entries = to_demand_sets(generate_traffic(site_plan, traffic, 42))["default"]
    assert len(entries) == 12
    assert all(
        d["preset"] == traffic.flow_policy_config[d["priority"]] for d in entries
    )


def test_uniform_uses_the_exact_generating_rule_before_float_serialization(
    site_plan, scenario_config
):
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region = 25
    traffic.gbps_per_mw = 200
    traffic.gravity.mw_per_dc_region_overrides = {"beta": 50}
    demands = generate_traffic(site_plan, traffic, 42)[0].demands
    assert len(demands) == 6
    assert {d.volume for d in demands} == {Fraction(10000, 3)}
    assert sum(d.volume for d in demands) == 20000
    assert {
        d["volume"]
        for d in to_demand_sets(generate_traffic(site_plan, traffic, 42))["default"]
    } == {float(Fraction(10000, 3))}
    traffic.priority_ratios = {0: 0.1, 1: 0.9}
    demands = generate_traffic(site_plan, traffic, 42)[0].demands
    assert {d.volume for d in demands if d.priority == 0} == {Fraction(1000, 3)}
    assert {d.volume for d in demands if d.priority == 1} == {Fraction(3000)}


@pytest.mark.parametrize(
    "power,rate", [(9007199254740993, 0.1), (0.1, 9007199254740993)]
)
def test_uniform_does_not_round_user_integer_scalars_before_decimal_entry(
    site_plan, scenario_config, power, rate
):
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region = power
    traffic.gbps_per_mw = rate
    traffic.priority_ratios = {0: 1}
    assert {d.volume for d in generate_traffic(site_plan, traffic, 42)[0].demands} == {
        Fraction(9007199254740993, 20)
    }


def test_uniform_decimal_powers_and_rate_are_exact_before_summing(
    site_plan, scenario_config
):
    traffic = scenario_config.traffic
    traffic.gravity.mw_per_dc_region_overrides = {
        "alpha": 0.1,
        "beta": 0.2,
        "gamma": 0.3,
    }
    traffic.gbps_per_mw = 0.1
    assert {d.volume for d in generate_traffic(site_plan, traffic, 42)[0].demands} == {
        Fraction(1, 100)
    }


def test_uniform_overflow_keeps_the_named_offer_refusal(site_plan, scenario_config):
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region = traffic.gbps_per_mw = 1e308
    with pytest.raises(
        ValueError, match="Offered traffic must be finite and non-negative"
    ):
        generate_traffic(site_plan, traffic, 42)
    traffic.mw_per_dc_region = traffic.gbps_per_mw = 1e100
    assert {d.volume for d in generate_traffic(site_plan, traffic, 42)[0].demands} == {
        Fraction(5 * 10**199)
    }
