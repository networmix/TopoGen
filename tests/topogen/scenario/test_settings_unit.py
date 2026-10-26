"""Resolved settings own their inputs and reject invalid references."""

from dataclasses import FrozenInstanceError, asdict

import pytest
import yaml

from topogen.library import load_library
from topogen.scenario import build_scenario
from topogen.scenario.records import StripeGroups
from topogen.scenario.sites import plan_sites, read_corridors, resolve_metro_settings


def test_defaults_override_snapshot_and_nested_ownership(
    corridor_graph, scenario_config
):
    cfg = scenario_config
    cfg.build.build_defaults.inter_metro_link.attrs = {"nested": [1], "base": True}
    cfg.build.build_overrides = {
        "alpha": {
            "pop_per_metro": 3,
            "inter_metro_link": {
                "capacity": 6400,
                "attrs": {"extra": 2},
                "match": {"conditions": []},
                "mode": "one_to_one",
            },
        }
    }
    before = asdict(cfg)
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    assert settings[0].pops == 3 and settings[1].pops == 1
    link = settings[0].links["inter_metro_link"]
    assert link.capacity == 6400 and link.mode == "one_to_one"
    assert link.attrs == {"nested": (1,), "base": True, "extra": 2}
    assert link.match == {"conditions": ()}
    assert asdict(cfg) == before
    cfg.build.build_defaults.inter_metro_link.capacity = 1600
    cfg.build.build_defaults.inter_metro_link.attrs["nested"].append(99)
    cfg.build.build_overrides["alpha"]["inter_metro_link"]["attrs"]["extra"] = 3
    assert (
        link.capacity == 6400
        and link.attrs["nested"] == (1,)
        and link.attrs["extra"] == 2
    )
    assert (
        settings[1].links["inter_metro_link"].capacity
        == before["build"]["build_defaults"]["inter_metro_link"]["capacity"]
    )
    with pytest.raises(FrozenInstanceError):
        link.capacity = 0
    with pytest.raises(TypeError):
        link.attrs["nested"] = ()
    with pytest.raises(TypeError):
        settings[0].links["extra"] = link


@pytest.mark.parametrize(
    "override,error",
    [
        ({"unknown": {}}, "unknown metro"),
        ({"alpha": {"pop_per_metro": 0}}, "PoP or DC count"),
        ({"alpha": {"dc_regions_per_metro": -1}}, "PoP or DC count"),
        ({"alpha": {"inter_metro_link": {"capacity": 0}}}, "invalid inter_metro_link"),
        ({"alpha": {"intra_metro_link": {"cost": 0}}}, "invalid intra_metro_link"),
        ({"alpha": {"dc_to_pop_link": {"cost": -1}}}, "invalid dc_to_pop_link"),
    ],
)
def test_invalid_settings_do_not_mutate(
    corridor_graph, scenario_config, override, error
):
    corridors = read_corridors(corridor_graph)
    scenario_config.build.build_overrides = override
    before = asdict(scenario_config)
    with pytest.raises(ValueError, match=error):
        resolve_metro_settings(
            corridors, scenario_config.build, scenario_config.traffic
        )
    assert asdict(scenario_config) == before
    scenario_config.build.build_overrides = {}
    assert resolve_metro_settings(
        corridors, scenario_config.build, scenario_config.traffic
    )


def test_per_site_power_specificity_and_emission(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.traffic.mw_per_dc_region = 25
    cfg.build.build_overrides = {
        "alpha": {"dc_regions_per_metro": 2},
        "gamma": {"dc_regions_per_metro": 0},
    }
    cfg.traffic.gravity.mw_per_dc_region_overrides = {"alpha": 50, "alpha/dc2": 75}
    build = build_scenario(corridor_graph, cfg, load_library(None))
    assert [site.power_mw for site in build.site_plan.sites if site.kind == "dc"] == [
        50,
        75,
        25,
    ]
    assert {d.volume for d in build.traffic[0].demands} == {25}
    groups = yaml.safe_load(build.yaml)["network"]["nodes"]
    assert [
        groups[path]["attrs"]["mw_per_dc_region"]
        for path in ("metro1/dc1", "metro1/dc2", "metro2/dc1")
    ] == [50.0, 75.0, 25.0]
    assert all(
        type(groups[path]["attrs"]["mw_per_dc_region"]) is float
        for path in ("metro1/dc1", "metro1/dc2", "metro2/dc1")
    )
    cfg.traffic.gravity.mw_per_dc_region_overrides.pop("alpha/dc2")
    other = build_scenario(corridor_graph, cfg, load_library(None))
    from fractions import Fraction

    assert {d.volume for d in other.traffic[0].demands} == {Fraction(125, 6)}
    assert [site.power_mw for site in build.site_plan.sites if site.kind == "dc"] == [
        50,
        75,
        25,
    ]


@pytest.mark.parametrize(
    "key", ["alpah", "alpha/dc0", "alpha/dc2", "gamma/dc1", "missing/dc1"]
)
@pytest.mark.parametrize("enabled", [False, True])
def test_unknown_power_key_is_refused_even_without_traffic(
    corridor_graph, scenario_config, key, enabled
):
    cfg = scenario_config
    cfg.traffic.enabled = enabled
    cfg.build.build_overrides = {"gamma": {"dc_regions_per_metro": 0}}
    cfg.traffic.gravity.mw_per_dc_region_overrides = {key: 50}
    before = asdict(cfg)
    with pytest.raises(ValueError) as error:
        build_scenario(corridor_graph, cfg, load_library(None))
    assert (
        str(error.value)
        == f"DC power override references unknown metro or DC site: {[key]}"
    )
    assert asdict(cfg) == before
    cfg.traffic.gravity.mw_per_dc_region_overrides = {"gamma": 50, "alpha/dc1": 25}
    assert build_scenario(corridor_graph, cfg, load_library(None)).yaml


def test_power_resolved_once_before_traffic(corridor_graph, scenario_config):
    from topogen.scenario.traffic import generate_traffic

    cfg = scenario_config
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    plan = plan_sites(corridors, settings, StripeGroups({}))
    expected = generate_traffic(plan, cfg.traffic, 42)
    cfg.traffic.mw_per_dc_region = 999
    cfg.traffic.gravity.mw_per_dc_region_overrides = {"typo": 888}
    assert generate_traffic(plan, cfg.traffic, 42) == expected


@pytest.mark.parametrize("value", [-1, float("inf"), float("-inf"), float("nan")])
def test_invalid_power_names_the_dc_and_retries(corridor_graph, scenario_config, value):
    cfg = scenario_config
    cfg.traffic.gravity.mw_per_dc_region_overrides = {"alpha/dc1": value}
    corridors = read_corridors(corridor_graph)
    with pytest.raises(ValueError) as error:
        resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    assert str(error.value) == "DC power for alpha/dc1 must be finite and non-negative"
    cfg.traffic.gravity.mw_per_dc_region_overrides["alpha/dc1"] = 0
    assert resolve_metro_settings(corridors, cfg.build, cfg.traffic)[0].dc_power_mw == (
        0,
    )


def test_original_name_override_and_name_precedence(corridor_graph, scenario_config):
    cfg = scenario_config
    corridor_graph.nodes[0]["name_orig"] = "Original Metro, XX"
    cfg.build.build_overrides = {"original-metro": {"pop_per_metro": 2}}
    corridors = read_corridors(corridor_graph)
    assert resolve_metro_settings(corridors, cfg.build, cfg.traffic)[0].pops == 2
    cfg.build.build_overrides["alpha"] = {"pop_per_metro": 3}
    assert resolve_metro_settings(corridors, cfg.build, cfg.traffic)[0].pops == 3


def test_unknown_power_key_on_empty_graph(scenario_config):
    from topogen.scenario.records import CorridorGraph

    cfg = scenario_config
    cfg.traffic.gravity.mw_per_dc_region_overrides = {"absent": 25}
    with pytest.raises(ValueError, match="unknown metro or DC site: \\['absent'\\]"):
        resolve_metro_settings(CorridorGraph((), ()), cfg.build, cfg.traffic)
    cfg.traffic.gravity.mw_per_dc_region_overrides = {}
    assert resolve_metro_settings(CorridorGraph((), ()), cfg.build, cfg.traffic) == ()
