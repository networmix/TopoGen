"""Traffic records preserve model rules and offer every distinct site pair."""

from fractions import Fraction

import pytest
import yaml
from ngraph import Scenario, prepare

from topogen.scenario.document import to_demand_sets
from topogen.scenario.traffic import _quantize, generate_traffic


def test_three_site_uniform_is_six_demands_of_100(make_plan, scenario_config):
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region = 200
    plan = make_plan(traffic)
    matrices = generate_traffic(plan, traffic, 42)
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
                f"{site.id}/n{i}": {}
                for site in plan.sites
                if site.kind == "dc"
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
        ("uniform", "sites=1, volume=10.0"),
        ("gravity", "zero or non-finite total weight"),
        ("hose", "one DC exceeds"),
    ],
)
def test_one_site_refusal_keeps_other_models(make_plan, scenario_config, model, error):
    traffic = scenario_config.traffic
    traffic.model = model
    with pytest.raises(ValueError, match=error):
        generate_traffic(make_plan(traffic, (1, 0, 0)), traffic, 42)
    assert generate_traffic(make_plan(traffic, (0, 0, 0)), traffic, 42) == ()


@pytest.mark.parametrize("power", [10.5, 0.1])
def test_one_site_uniform_refusal_has_float_volume(make_plan, scenario_config, power):
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region = power
    with pytest.raises(ValueError) as error:
        generate_traffic(make_plan(traffic, (1, 0, 0)), traffic, 42)
    assert (
        str(error.value)
        == f"uniform traffic requires at least two DC sites: sites=1, volume={power!r}"
    )
    assert [
        d.volume
        for d in generate_traffic(make_plan(traffic, (1, 1, 0)), traffic, 42)[0].demands
    ] == [Fraction(repr(power))] * 2
    traffic.mw_per_dc_region = 0
    assert not generate_traffic(make_plan(traffic, (1, 0, 0)), traffic, 42)[0].demands


@pytest.mark.parametrize("model", ["gravity", "hose"])
def test_two_equal_site_classes_are_7_5_and_2_5(make_plan, scenario_config, model):
    traffic = scenario_config.traffic
    traffic.model, traffic.priority_ratios = model, {0: 0.75, 1: 0.25}
    demands = generate_traffic(make_plan(traffic, (1, 1, 0)), traffic, 42)[0].demands
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
    "policy,expected",
    [
        ("floor", {(0, 1): 4, (0, 2): 4, (1, 2): 2}),
        ("nearest", {(0, 1): 4, (0, 2): 4, (1, 2): 2}),
        ("ceil", {(0, 1): 6, (0, 2): 4, (1, 2): 4}),
    ],
)
def test_pair_quantization_policy_is_observable(policy, expected):
    assert _quantize({(0, 1): 4.9, (0, 2): 3, (1, 2): 2.1}, 10, 2, policy) == expected
    if policy == "nearest":
        assert _quantize({(0, 1): 5.1, (0, 2): 3, (1, 2): 1.9}, 10, 2, policy) == {
            (0, 1): 6,
            (0, 2): 4,
            (1, 2): 2,
        }


@pytest.mark.parametrize("counts,total", [((2, 1), 600), ((2, 2), 1200)])
def test_uniform_unequal_inventory_classes_and_devices(
    make_plan, scenario_config, counts, total
):
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region = total / sum(counts)
    traffic.priority_ratios = {0: 0.75, 1: 0.25}
    plan = make_plan(traffic, (*counts, 0))
    matrices = generate_traffic(plan, traffic, 42)
    demands = matrices[0].demands
    n = sum(counts)
    assert len(demands) == 2 * n * (n - 1)
    assert {d.volume for d in demands if d.priority == 0} == {75}
    assert {d.volume for d in demands if d.priority == 1} == {25}
    sites = [site.id for site in plan.sites if site.kind == "dc"]
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
    make_plan, scenario_config
):
    traffic = scenario_config.traffic
    traffic.priority_ratios = {0: 0, 1: 1}
    traffic.flow_policy_config = {0: "TE_MAXFLOW_UNLIM", 1: "SHORTEST_PATHS_ECMP"}
    traffic.gravity.mw_per_dc_region_overrides = {"beta": 0}
    demands = generate_traffic(make_plan(traffic), traffic, 42)[0].demands
    assert len(demands) == 6 and {d.priority for d in demands} == {1}
    assert {d.source for d in demands} == {"metro1/dc1", "metro2/dc1", "metro3/dc1"}
    assert {d.volume for d in demands} == {Fraction(10, 3)}
    assert all(
        d["preset"] == "SHORTEST_PATHS_ECMP"
        for d in to_demand_sets(generate_traffic(make_plan(traffic), traffic, 42))[
            "default"
        ]
    )
    traffic.mw_per_dc_region = 0
    assert not generate_traffic(make_plan(traffic), traffic, 42)[0].demands
    traffic.enabled = False
    assert generate_traffic(make_plan(traffic), traffic, 42) == ()


def test_emission_retains_each_prioritys_own_preset(make_plan, scenario_config):
    traffic = scenario_config.traffic
    traffic.priority_ratios = {0: 0.75, 1: 0.25}
    traffic.flow_policy_config = {0: "TE_MAXFLOW_UNLIM", 1: "SHORTEST_PATHS_ECMP"}
    entries = to_demand_sets(generate_traffic(make_plan(traffic), traffic, 42))[
        "default"
    ]
    assert len(entries) == 12
    assert all(
        d["preset"] == traffic.flow_policy_config[d["priority"]] for d in entries
    )


def test_uniform_uses_the_exact_generating_rule_before_float_serialization(
    make_plan, scenario_config
):
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region, traffic.gbps_per_mw = 25, 200
    traffic.gravity.mw_per_dc_region_overrides = {"beta": 50}
    demands = generate_traffic(make_plan(traffic), traffic, 42)[0].demands
    assert len(demands) == 6 and {d.volume for d in demands} == {Fraction(10000, 3)}
    assert sum(d.volume for d in demands) == 20000
    assert {
        d["volume"]
        for d in to_demand_sets(generate_traffic(make_plan(traffic), traffic, 42))[
            "default"
        ]
    } == {float(Fraction(10000, 3))}
    traffic.priority_ratios = {0: 0.1, 1: 0.9}
    demands = generate_traffic(make_plan(traffic), traffic, 42)[0].demands
    assert {d.volume for d in demands if d.priority == 0} == {Fraction(1000, 3)}
    assert {d.volume for d in demands if d.priority == 1} == {Fraction(3000)}


@pytest.mark.parametrize(
    "power,rate", [(9007199254740993, 0.1), (0.1, 9007199254740993)]
)
def test_uniform_does_not_round_user_integer_scalars_before_decimal_entry(
    make_plan, scenario_config, power, rate
):
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region, traffic.gbps_per_mw = power, rate
    traffic.priority_ratios = {0: 1}
    assert {
        d.volume for d in generate_traffic(make_plan(traffic), traffic, 42)[0].demands
    } == {Fraction(9007199254740993, 20)}


def test_uniform_decimal_powers_and_rate_are_exact_before_summing(
    make_plan, scenario_config
):
    traffic = scenario_config.traffic
    traffic.gravity.mw_per_dc_region_overrides = {
        "alpha": 0.1,
        "beta": 0.2,
        "gamma": 0.3,
    }
    traffic.gbps_per_mw = 0.1
    assert {
        d.volume for d in generate_traffic(make_plan(traffic), traffic, 42)[0].demands
    } == {Fraction(1, 100)}


def test_uniform_overflow_keeps_the_named_offer_refusal(make_plan, scenario_config):
    traffic = scenario_config.traffic
    traffic.mw_per_dc_region = traffic.gbps_per_mw = 1e308
    with pytest.raises(
        ValueError, match="Offered traffic must be finite and non-negative"
    ):
        generate_traffic(make_plan(traffic), traffic, 42)
    traffic.mw_per_dc_region = traffic.gbps_per_mw = 1e100
    assert {
        d.volume for d in generate_traffic(make_plan(traffic), traffic, 42)[0].demands
    } == {Fraction(5 * 10**199)}
