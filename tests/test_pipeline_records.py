"""Build stages own immutable records and cannot borrow mutable inputs."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass

import networkx as nx
import pytest

from topogen.library import Library, load_library, thaw
from topogen.scenario import build_scenario
from topogen.scenario.records import (
    Corridor,
    CorridorGraph,
    DeviceLink,
    LinkHardware,
    LinkSettings,
    Metro,
    MetroSettings,
    Site,
    SiteExpansion,
    SitePlan,
    StripeGroups,
)
from topogen.scenario.sites import (
    finalize_blueprints,
    plan_sites,
    read_corridors,
    resolve_metro_settings,
)
from topogen.scenario.striping import stripe_groups


def assert_frozen(value):
    if is_dataclass(value):
        assert not hasattr(value, "__dict__")
        for field in fields(value):
            original = getattr(value, field.name)
            with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
                setattr(value, field.name, original)
            assert_frozen(original)
    elif isinstance(value, Mapping):
        with pytest.raises(TypeError):
            value["extra"] = 99
        for child in value.values():
            assert_frozen(child)
    elif isinstance(value, tuple):
        for child in value:
            assert_frozen(child)
    else:
        assert not isinstance(value, (dict, list, nx.Graph))


def test_build_owns_every_record_and_nested_container(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.build.build_defaults.inter_metro_link.attrs = {"nested": [1, {"more": [2]}]}
    library = load_library(None)
    before = deepcopy(nx.node_link_data(corridor_graph, edges="edges"))
    config_before = asdict(cfg)
    build = build_scenario(corridor_graph, cfg, library)
    assert_frozen(build)
    assert nx.node_link_data(corridor_graph, edges="edges") == before
    assert asdict(cfg) == config_before
    cfg.build.build_defaults.inter_metro_link.attrs["nested"][1]["more"].append(99)
    corridor_graph.nodes[0]["name"] = "changed"
    assert build.site_plan.sites[0].metro.name == "alpha"
    wan = next(
        link for link in build.site_plan.links if link.kind == "inter_metro_corridor"
    )
    assert wan.attrs["nested"] == (1, {"more": (2,)})


def test_constructor_buffers_are_owned():
    metro = Metro(1, "a", "a", "a", 0, 0, 1)
    attrs = {"a": [1]}
    roles = ["core|core"]
    risks = ["r"]
    members = ["core/r1"]
    rules = [{"path": "^a$", "attrs": {"labels": [1]}}]
    position = [1.0, 2.0]
    sites = [Site("metro1/pop1", metro, "pop", 1, "b", None, position)]
    from topogen.scenario.records import SiteLink

    links = [
        SiteLink(
            "a|b|k",
            "intra_metro",
            "a",
            "b",
            1,
            1,
            800,
            attrs,
            attrs,
            "k",
            roles,
            risks,
            euclidean_km=None,
            detour_ratio=None,
        )
    ]
    link_settings = LinkSettings(800, 1, attrs, attrs, roles, attrs, "mesh")
    link_map = {"inter_metro_link": link_settings}
    powers = [25]
    setting = MetroSettings(metro, 1, 1, "b", "d", link_map, powers)
    stripe = StripeGroups({(1, "inter_metro_link", "b"): {"g1": members}})
    plan = SitePlan(sites, links, rules)
    corridors = [
        Corridor(
            metro,
            metro,
            1,
            2.0,
            risk_groups=risks,
            euclidean_km=None,
            detour_ratio=None,
        )
    ]
    corridor_graph = CorridorGraph([metro], corridors)
    devices = [DeviceLink("a", "b", "core", "core")]
    expansion = SiteExpansion({"a": devices})
    hardware = LinkHardware(devices, 800)
    attrs["a"].append(2)
    roles.append("dc|dc")
    risks.append("later")
    members.append("core/r2")
    position.append(3)
    rules[0]["attrs"]["labels"].append(2)
    sites.clear()
    links.clear()
    corridors.clear()
    devices.clear()
    powers.clear()
    link_map.clear()
    assert plan.sites[0].position == (1.0, 2.0)
    assert plan.links[0].attrs == {"a": (1,)} and plan.links[0].role_pairs == (
        "core|core",
    )
    assert plan.links[0].risk_groups == ("r",) and plan.node_rules[0]["attrs"][
        "labels"
    ] == (1,)
    assert setting.dc_power_mw == (25,) and setting.links["inter_metro_link"].attrs == {
        "a": (1,)
    }
    assert stripe.groups[1, "inter_metro_link", "b"]["g1"] == ("core/r1",)
    assert corridor_graph.corridors[0].risk_groups == ("r",)
    assert len(expansion.links["a"]) == len(hardware.device_links) == 1
    for record in (setting, stripe, plan, corridor_graph, expansion, hardware):
        assert_frozen(record)


def test_stage_records_share_metro_but_not_config(corridor_graph, scenario_config):
    scenario_config.build.build_defaults.inter_metro_link.capacity = 3200
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(
        corridors, scenario_config.build, scenario_config.traffic
    )
    blueprints = finalize_blueprints(
        settings, load_library(None), scenario_config.components.hw_component
    )
    stripes = stripe_groups(corridors, settings, blueprints)
    plan = plan_sites(corridors, settings, stripes)
    for value in (corridors, settings, blueprints, stripes, plan):
        assert_frozen(value)
    scenario_config.build.build_defaults.inter_metro_link.capacity = 1600
    assert settings[0].links["inter_metro_link"].capacity == 3200
    assert all(
        site.metro is corridors.metros[site.metro.index - 1] for site in plan.sites
    )


def test_failed_build_and_independent_concurrent_builds_keep_inputs(
    corridor_graph, scenario_config
):
    from concurrent.futures import ThreadPoolExecutor

    library = load_library(None)
    before_graph = deepcopy(nx.node_link_data(corridor_graph, edges="edges"))
    before_config = asdict(scenario_config)
    before_library = thaw(library.blueprints)
    reference = build_scenario(corridor_graph, scenario_config, library)
    broken = thaw(library.blueprints)
    broken["SingleRouter"]["links"] = [
        {"source": "missing", "target": "core", "cost": 1, "capacity": 1}
    ]
    bad_library = Library(
        broken, library.components, library.failure_policies, library.workflows
    )
    with pytest.raises(ValueError, match="SingleRouter"):
        build_scenario(corridor_graph, scenario_config, bad_library)
    assert nx.node_link_data(corridor_graph, edges="edges") == before_graph
    assert asdict(scenario_config) == before_config
    assert (
        thaw(library.blueprints) == before_library
        and thaw(bad_library.blueprints) == broken
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        builds = list(
            pool.map(
                lambda _: build_scenario(corridor_graph, scenario_config, library),
                range(2),
            )
        )
    assert all(b == reference for b in builds)
    assert (
        builds[0].site_plan is not builds[1].site_plan
        and builds[0].hardware is not builds[1].hardware
    )


def test_build_stages_run_in_contract_order(
    corridor_graph, scenario_config, monkeypatch
):
    import topogen.scenario.build as stages

    names = [
        "read_corridors",
        "resolve_metro_settings",
        "finalize_blueprints",
        "stripe_groups",
        "plan_sites",
        "expand_sites",
        "generate_traffic",
        "size_links",
        "check_components",
        "assign_hardware",
        "scenario_document",
        "to_yaml",
    ]
    observed = []
    for name in names:
        original = getattr(stages, name)

        def traced(*args, _name=name, _original=original, **kwargs):
            observed.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(stages, name, traced)
    assert stages.build_scenario(
        corridor_graph, scenario_config, load_library(None)
    ).yaml
    assert observed == names
