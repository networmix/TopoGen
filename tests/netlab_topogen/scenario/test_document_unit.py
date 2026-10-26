"""The common writer preserves network topology, fields and scenario sections."""

import pytest
import yaml
from ngraph import Scenario

from netlab_topogen.library import Library, load_library, thaw
from netlab_topogen.scenario import build_scenario
from netlab_topogen.scenario.document import (
    network_document,
    scenario_document,
    to_yaml,
)
from netlab_topogen.scenario.sites import (
    finalize_blueprints,
    read_corridors,
    resolve_metro_settings,
)
from netlab_topogen.validation import validate_scenario


def test_common_writer_topology_and_plan_order(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.build.build_defaults.pop_per_metro = 2
    cfg.build.build_defaults.dc_regions_per_metro = 2
    cfg.build.tm_sizing.enabled = True
    cfg.components.optics = {
        "core->core": "800G-ZR+",
        "core->dc": "800G-ZR+",
        "dc->core": "800G-ZR+",
    }
    library = load_library(None)
    build = build_scenario(corridor_graph, cfg, library)
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    bps = finalize_blueprints(settings, library, cfg.components.hw_component)
    network_only = network_document(corridors, build.site_plan, bps, cfg)
    document = scenario_document(
        corridors,
        build.site_plan,
        bps,
        build.traffic,
        build.sizing,
        build.hardware,
        library,
        cfg,
    )
    parsed = yaml.safe_load(build.yaml)
    assert parsed == document
    assert list(parsed["network"]["nodes"]) == [
        site.id for site in build.site_plan.sites
    ]
    assert len(parsed["network"]["nodes"]) == 12
    assert [rule["attrs"]["site_edge"] for rule in parsed["network"]["links"]] == [
        link.id for link in build.site_plan.links
    ]
    assert [link.kind for link in build.site_plan.links] == ["intra_metro"] * 3 + [
        "dc_to_pop"
    ] * 12 + ["inter_metro_corridor"] * 12
    before = Scenario.from_yaml(to_yaml(network_only, yaml_anchors=False)).network
    after = Scenario.from_yaml(build.yaml).network
    assert list(before.nodes) == list(after.nodes)
    assert [
        (link.id, link.source, link.target, link.cost) for link in before.links.values()
    ] == [
        (link.id, link.source, link.target, link.cost) for link in after.links.values()
    ]
    assert (
        network_only["network"]["node_rules"] == parsed["network"]["node_rules"]
        if build.site_plan.node_rules
        else "node_rules" not in network_only["network"]
    )
    assert not validate_scenario(build.yaml, cfg.components).issues


def test_mixed_radii_traffic_and_capacities(corridor_graph, scenario_config):
    cfg = scenario_config
    corridor_graph.nodes[1].update(x=300000.0, y=0.0, radius_km=40.0)
    corridor_graph.nodes[2].update(x=300000.0, y=400000.0, radius_km=20.0)
    corridor_graph[0][2][0]["length_km"] = 700.0
    corridor_graph[1][2][0]["length_km"] = 400.0
    corridor_graph.add_edge(0, 1, key=1, length_km=300.0)
    cfg.build.build_defaults.site_blueprint = "FullMesh4"
    cfg.build.build_defaults.inter_metro_link.mode = "one_to_one"
    cfg.build.build_defaults.inter_metro_link.capacity = (
        cfg.build.build_defaults.dc_to_pop_link.capacity
    ) = 800
    cfg.build.build_overrides = {
        "alpha": {"inter_metro_link": {"striping": {"width": 1}}},
        "beta": {"inter_metro_link": {"striping": {"width": 2}}},
    }
    cfg.traffic.model = "gravity"
    cfg.traffic.mw_per_dc_region = 25
    cfg.traffic.gbps_per_mw = 200
    cfg.traffic.gravity.mw_per_dc_region_overrides = {"beta": 50}
    cfg.traffic.gravity.rounding_gbps = 0.01
    cfg.build.tm_sizing.enabled = True
    cfg.build.tm_sizing.headroom = 1.28
    cfg.build.tm_sizing.quantum_gbps = 1600
    cfg.components.optics = {
        "core->core": "800G-ZR+",
        "core->dc": "800G-ZR+",
        "dc->core": "800G-ZR+",
    }
    build = build_scenario(corridor_graph, cfg, load_library(None))
    # With unordered-pair step .01, half-pair volumes use step .005.
    # Remainders are BC > AC > AB, fixing declaration order independently.
    assert [
        (d.source, d.target, float(d.volume), d.euclidean_km)
        for d in build.traffic[0].demands
    ] == [
        ("metro2/dc1", "metro3/dc1", 3658.535, 400),
        ("metro3/dc1", "metro2/dc1", 3658.535, 400),
        ("metro1/dc1", "metro3/dc1", 1463.415, 500),
        ("metro3/dc1", "metro1/dc1", 1463.415, 500),
        ("metro1/dc1", "metro2/dc1", 4878.05, 300),
        ("metro2/dc1", "metro1/dc1", 4878.05, 300),
    ]
    assert [build.sizing.capacity[link.id] for link in build.site_plan.links] == [
        14400,
        19200,
        9600,
        4800,
        4800,
        1600,
        6400,
    ]
    assert [hw.capacity for hw in build.hardware.values()] == [
        3600,
        4800,
        2400,
        4800,
        4800,
        1600,
        3200,
    ]
    assert [hw.source_count for hw in build.hardware.values()] == [5, 6, 3, 6, 6, 2, 4]
    assert not validate_scenario(build.yaml, cfg.components).issues


def test_yaml_anchor_options_retain_every_field(corridor_graph, scenario_config):
    library = load_library(None)
    build = build_scenario(corridor_graph, scenario_config, library)
    document = yaml.safe_load(build.yaml)
    document["repeated"] = [document["network"], document["network"]]
    with_anchors = to_yaml(document, yaml_anchors=True)
    without_anchors = to_yaml(document, yaml_anchors=False)
    assert "&id" in with_anchors and "*id" in with_anchors
    assert "&id" not in without_anchors and "*id" not in without_anchors
    assert yaml.safe_load(with_anchors) == yaml.safe_load(without_anchors) == document


def test_finalization_dependency_and_cycle_paths(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.build.build_defaults.site_blueprint = "Outer"
    base = load_library(None)
    raw = thaw(base.blueprints)
    raw["Outer"] = {
        "nodes": {"nested": {"blueprint": "FullMesh4", "attrs": {"role": "spine"}}}
    }
    library = Library(raw, base.components, base.failure_policies, base.workflows)
    settings = resolve_metro_settings(
        read_corridors(corridor_graph), cfg.build, cfg.traffic
    )
    blueprints = finalize_blueprints(
        settings, library, {"core": "CoreRouter", "spine": "SpineRouter"}
    )
    assert list(blueprints) == ["DCRegion", "FullMesh4", "Outer"]
    assert blueprints["FullMesh4"]["nodes"]["core"]["attrs"]["hardware"] == {
        "component": "CoreRouter",
        "count": 1,
    }
    assert blueprints["Outer"]["nodes"]["nested"]["attrs"]["hardware"] == {
        "component": "SpineRouter",
        "count": 1,
    }
    raw["FullMesh4"]["nodes"]["again"] = {"blueprint": "Outer"}
    with pytest.raises(ValueError, match="Cyclic blueprint reference: Outer"):
        finalize_blueprints(
            settings,
            Library(raw, base.components, base.failure_policies, base.workflows),
            {},
        )


def test_every_corridor_metadata_field_reaches_both_writers(
    corridor_graph, scenario_config
):
    cfg = scenario_config
    cfg.build.build_defaults.inter_metro_link.attrs = {"user": {"nested": [1, 2]}}
    corridor_graph[0][1][0].update(
        euclidean_km=200, detour_ratio=1.5, risk_groups=["corridor_risk_alpha_beta"]
    )
    library = load_library(None)
    build = build_scenario(corridor_graph, cfg, library)
    link = next(link for link in build.site_plan.links if "inter_metro:1-2:" in link.id)
    assert link.euclidean_km == 200 and link.detour_ratio == 1.5
    assert link.risk_groups == ("corridor_risk_alpha_beta",)
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    bps = finalize_blueprints(settings, library, cfg.components.hw_component)
    for doc in (
        network_document(corridors, build.site_plan, bps, cfg),
        yaml.safe_load(build.yaml),
    ):
        rule = next(
            rule
            for rule in doc["network"]["links"]
            if rule["attrs"]["site_edge"] == link.id
        )
        assert rule["source"] == "metro1/pop1" and rule["target"] == "metro2/pop1"
        assert rule["pattern"] == "one_to_one" and rule["cost"] == 300
        assert rule["risk_groups"] == ["corridor_risk_alpha_beta"]
        attrs = rule["attrs"]
        assert attrs["source_metro"] == "alpha" and attrs["target_metro"] == "beta"
        assert attrs["distance_km"] == 300 and type(attrs["distance_km"]) is int
        assert attrs["euclidean_km"] == 200.0 and type(attrs["euclidean_km"]) is float
        assert attrs["detour_ratio"] == 1.5 and type(attrs["detour_ratio"]) is float
        assert attrs["user"] == {"nested": [1, 2]}
        assert attrs["adjacency_id"] == "inter_metro:1-2:path0"
    cfg.corridors.risk_groups.enabled = False
    doc = network_document(corridors, build.site_plan, bps, cfg)
    assert "risk_groups" not in doc and all(
        "risk_groups" not in rule for rule in doc["network"]["links"]
    )


def test_role_pair_union_keeps_roles_from_every_pair(corridor_graph, scenario_config):
    from netlab_topogen.scenario.records import LinkSettings
    from netlab_topogen.scenario.sites import role_match

    params = LinkSettings(
        100, 1, {}, {"fallback": True}, ("leaf|core", "spine|dc"), {}, "mesh"
    )
    assert role_match(params) == {
        "conditions": [
            {"attr": "role", "op": "==", "value": role}
            for role in ("core", "dc", "leaf", "spine")
        ],
        "logic": "or",
    }
    params = LinkSettings(100, 1, {}, {"fallback": True}, (), {}, "mesh")
    assert role_match(params) == {"fallback": True}
    cfg = scenario_config
    cfg.traffic.enabled = False
    cfg.build.build_defaults.dc_regions_per_metro = 0
    cfg.build.build_defaults.site_blueprint = "TwoRoles"
    cfg.build.build_defaults.inter_metro_link.role_pairs = ["core|core", "leaf|leaf"]
    base = load_library(None)
    raw = thaw(base.blueprints)
    raw["TwoRoles"] = {
        "nodes": {role: {"attrs": {"role": role}} for role in ("core", "leaf", "spine")}
    }
    library = Library(raw, base.components, base.failure_policies, base.workflows)
    build = build_scenario(corridor_graph, cfg, library)
    network = Scenario.from_yaml(build.yaml).network
    assert len(network.links) == 6
    assert (
        sorted(
            (
                network.nodes[link.source].attrs["role"],
                network.nodes[link.target].attrs["role"],
            )
            for link in network.links.values()
        )
        == [("core", "core")] * 3 + [("leaf", "leaf")] * 3
    )


def test_failure_policy_refusal_lists_the_given_library(scenario_config):
    from netlab_topogen.scenario.document import _failure_policies

    library = Library({}, {}, {"UserPolicy": {}}, {})
    scenario_config.failure_policies.assignments.default = "missing"
    with pytest.raises(ValueError) as error:
        _failure_policies(scenario_config, [], library)
    assert (
        str(error.value)
        == "Default failure policy 'missing' not found. Available failure policies: ['UserPolicy']"
    )
    scenario_config.failure_policies.assignments.default = "UserPolicy"
    assert _failure_policies(scenario_config, [], library) == {"UserPolicy": {}}
