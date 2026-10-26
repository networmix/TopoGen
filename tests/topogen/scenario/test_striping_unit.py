"""Grouping keeps policy context while sharing the finalized device inventory."""

import pytest
from ngraph import Scenario

from topogen.library import Library, load_library, thaw
from topogen.scenario import build_scenario
from topogen.scenario.records import StripeGroups
from topogen.scenario.sites import (
    finalize_blueprints,
    plan_sites,
    read_corridors,
    resolve_metro_settings,
)
from topogen.scenario.striping import _group_devices, stripe_groups


def test_mixed_contexts_share_one_finalized_expansion(
    corridor_graph, scenario_config, monkeypatch
):
    cfg = scenario_config
    cfg.build.build_defaults.site_blueprint = "FullMesh4"
    cfg.build.build_defaults.inter_metro_link.mode = "one_to_one"
    cfg.build.build_overrides = {
        "alpha": {"inter_metro_link": {"striping": {"mode": "width", "width": 1}}},
        "beta": {"inter_metro_link": {"striping": {"mode": "width", "width": 2}}},
    }
    cfg.components.hw_component = {"core": "CoreRouter", "dc": ""}
    corridor_graph.add_edge(0, 1, key=1, length_km=300)
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    definitions = finalize_blueprints(
        settings, load_library(None), cfg.components.hw_component
    )
    original = Scenario.from_yaml
    calls = []

    def counted(text):
        result = original(text)
        calls.append(result)
        return result

    monkeypatch.setattr(Scenario, "from_yaml", counted)
    groups = stripe_groups(corridors, settings, definitions)
    assert len(calls) == 1 and len(calls[0].network.nodes) == 4
    assert groups.groups[1, "inter_metro_link", "FullMesh4"] == {
        f"g{i}": (f"core/core{i}",) for i in range(1, 5)
    }
    assert groups.groups[2, "inter_metro_link", "FullMesh4"] == {
        "g1": ("core/core1", "core/core2"),
        "g2": ("core/core3", "core/core4"),
    }
    plan = plan_sites(corridors, settings, groups)
    core2 = next(
        rule for rule in plan.node_rules if rule["path"] == r"^metro2/pop1/core/core2$"
    )
    assert core2["attrs"] == {"stripe_im_1_2": "g2", "stripe_im_2_3": "g1"}
    build = build_scenario(corridor_graph, cfg, load_library(None))
    assert [
        len(hw.device_links)
        for edge, hw in build.hardware.items()
        if "inter_metro" in edge
    ] == [1, 1, 1, 2]
    assert all(
        node.attrs["hardware"]["component"] == "CoreRouter"
        for node in calls[0].network.nodes.values()
    )


def test_no_striping_does_not_parse(corridor_graph, scenario_config, monkeypatch):
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(
        corridors, scenario_config.build, scenario_config.traffic
    )

    def refuse(*args, **kwargs):
        raise AssertionError("unexpected E1")

    monkeypatch.setattr(Scenario, "from_yaml", refuse)
    assert stripe_groups(corridors, settings, {}) == StripeGroups({})


@pytest.mark.parametrize(
    "striping,error",
    [
        ({"width": 0}, "positive divisible"),
        ({"width": True}, "positive divisible"),
        ({"width": 3}, "positive divisible"),
        ({"mode": "bogus"}, "Unknown striping.mode"),
        ({"mode": "by_attr", "attribute": "missing"}, "lacks attribute"),
    ],
)
def test_bad_groups_have_valid_sibling(striping, error):
    selected = {"a/r1": {"stripe": "z"}, "a/r2": {"stripe": "a"}}
    with pytest.raises(ValueError, match=error):
        _group_devices(selected, striping, "B")
    assert _group_devices(selected, {"width": 1}, "B") == {
        "g1": ("a/r1",),
        "g2": ("a/r2",),
    }
    assert _group_devices(
        selected, {"mode": "by_attr", "attribute": "stripe"}, "B"
    ) == {"a": ("a/r2",), "z": ("a/r1",)}
    assert list(
        _group_devices(selected, {"mode": "by_attr", "attribute": "stripe"}, "B")
    ) == ["a", "z"]
    with pytest.raises(ValueError, match="No eligible stripe devices in blueprint 'B'"):
        _group_devices({}, {"width": 1}, "B")


def test_grouping_match_and_link_kind_are_independent(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.build.build_defaults.site_blueprint = (
        cfg.build.build_defaults.dc_region_blueprint
    ) = "Two"
    cfg.build.build_defaults.inter_metro_link.striping = {"width": 1}
    cfg.build.build_defaults.inter_metro_link.match = {
        "conditions": [{"attr": "chosen", "op": "==", "value": True}]
    }
    cfg.build.build_defaults.dc_to_pop_link.striping = {"width": 2}
    base = load_library(None)
    bps = thaw(base.blueprints)
    bps["Two"] = {
        "nodes": {
            "r": {"count": 2, "template": "r{n}", "attrs": {"role": "core"}},
            "chosen": {"attrs": {"role": "core", "chosen": True}},
        }
    }
    library = Library(bps, base.components, base.failure_policies, base.workflows)
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    bps = finalize_blueprints(settings, library, {})
    # DC grouping needs a divisible count; retain two chosen devices instead of one.
    raw = thaw(bps)
    raw["Two"]["nodes"]["chosen"]["count"] = 2
    raw["Two"]["nodes"]["chosen"]["template"] = "r{n}"
    groups = stripe_groups(corridors, settings, raw)
    assert groups.groups[1, "inter_metro_link", "Two"] == {
        "g1": ("chosen/r1",),
        "g2": ("chosen/r2",),
    }
    assert groups.groups[1, "dc_to_pop_link", "Two"] == {
        "g1": ("chosen/r1", "chosen/r2"),
        "g2": ("r/r1", "r/r2"),
    }


def test_mismatched_and_conflicting_groups_refuse(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.build.build_defaults.inter_metro_link.striping = {"width": 1}
    cfg.build.build_overrides = {"beta": {"site_blueprint": "Different"}}
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    groups = {
        (1, "inter_metro_link", "SingleRouter"): {"g1": ("core/core",)},
        (1, "inter_metro_link", "Different"): {"g2": ("core/core",)},
    }
    with pytest.raises(ValueError, match="Inter-metro striping mismatch: alpha="):
        plan_sites(corridors, settings, StripeGroups(groups))
    cfg.build.build_defaults.dc_to_pop_link.striping = {"width": 2}
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    groups.update(
        {
            (1, "dc_to_pop_link", "SingleRouter"): {"g1": ("core/core",)},
            (1, "dc_to_pop_link", "DCRegion"): {"g2": ("dc/dc",)},
        }
    )
    with pytest.raises(ValueError, match="DC-to-PoP striping mismatch in alpha"):
        plan_sites(corridors, settings, StripeGroups(groups))
    cfg.build.build_defaults.dc_to_pop_link.striping = {}
    cfg.build.build_overrides = {}
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    groups = {
        (1, "inter_metro_link", "SingleRouter"): {
            "g1": ("core/core",),
            "g2": ("core/core",),
        }
    }
    with pytest.raises(
        ValueError, match="Conflicting stripe assignment for metro1/pop1/core/core"
    ):
        plan_sites(corridors, settings, StripeGroups(groups))
