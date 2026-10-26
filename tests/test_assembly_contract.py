"""Hardware division, both endpoints and reachable empty-expansion failures."""

from fractions import Fraction

import pytest
import yaml

from netlab_topogen.library import Library, load_library, thaw
from netlab_topogen.scenario import build_scenario
from netlab_topogen.validation import validate_scenario


def hardware_case(corridor_graph, config, optic, devices=1):
    corridor_graph.remove_node(2)
    config.build.build_defaults.site_blueprint = "Core"
    config.build.build_defaults.inter_metro_link.mode = "one_to_one"
    config.components.optics = {"core->core": "O", "core->dc": "O", "dc->core": "O"}
    base = load_library(None)
    blueprints = thaw(base.blueprints)
    blueprints["Core"] = {
        "nodes": {
            "core": {"count": devices, "template": "core{n}", "attrs": {"role": "core"}}
        },
        "links": [],
    }
    components = thaw(base.components)
    components["O"] = {"capacity": optic, "ports": 0}
    return Library(blueprints, components, base.failure_policies, base.workflows)


def test_49_optics_runs_through_assignment_emission_and_validation(
    corridor_graph, scenario_config
):
    cfg = scenario_config
    cfg.traffic.mw_per_dc_region = 1
    cfg.traffic.gbps_per_mw = 60.0005
    cfg.build.tm_sizing.enabled = True
    cfg.build.tm_sizing.headroom = 1
    cfg.build.tm_sizing.quantum_gbps = 0
    cfg.build.tm_sizing.respect_min_base_capacity = False
    library = hardware_case(corridor_graph, cfg, 1.2245)
    build = build_scenario(corridor_graph, cfg, library)
    wan = [hw for edge, hw in build.hardware.items() if "inter_metro" in edge]
    assert len(wan) == 1 and wan[0].capacity == Fraction("60.0005")
    assert (wan[0].source_count, wan[0].target_count) == (49, 49)
    assert not validate_scenario(build.yaml, cfg.components).issues
    doc = yaml.safe_load(build.yaml)
    for rule in doc["network"]["links"]:
        assert type(rule["capacity"]) is float
        assert type(rule["attrs"]["target_capacity"]) is float
        for assignment in rule["attrs"]["hardware"].values():
            assert type(assignment["count"]) is float
    for link in doc["network"]["links"]:
        if link["attrs"]["link_type"] == "inter_metro_corridor":
            link["attrs"]["hardware"]["source"]["count"] = 48
    assert any(
        "capacity shortfall" in s
        for s in validate_scenario(yaml.safe_dump(doc), cfg.components).issues
    )


@pytest.mark.parametrize(
    "budget,optic,devices,count",
    [
        (1, 0.1111111111111111, 3, 3),
        (775, 36.9047619047619, 3, 7),
        (2**48, 1 / 64, 1, 2**54),
        (10**14, 1e-290, 1, 10**304),
        (10**16, 9999999999999999, 1, 1),
        (6117552732154677, 21465.09354, 5, 57000010001),
    ],
)
def test_written_device_and_held_optic_boundaries(
    corridor_graph, scenario_config, budget, optic, devices, count
):
    cfg = scenario_config
    cfg.build.build_defaults.inter_metro_link.capacity = budget
    library = hardware_case(corridor_graph, cfg, optic, devices)
    build = build_scenario(corridor_graph, cfg, library)
    wan = [hw for edge, hw in build.hardware.items() if "inter_metro" in edge]
    assert len(wan) == 1 and wan[0].capacity == Fraction(budget, devices)
    assert wan[0].source_count == wan[0].target_count == count
    assert not validate_scenario(build.yaml, cfg.components).issues


@pytest.mark.parametrize("budget,optic", [(8106479329266893, 0.9), (10**14, 1e-300)])
def test_unwritable_count_refuses_by_link_end_optic(
    corridor_graph, scenario_config, budget, optic
):
    cfg = scenario_config
    cfg.build.build_defaults.inter_metro_link.capacity = budget
    library = hardware_case(corridor_graph, cfg, optic)
    with pytest.raises(
        ValueError,
        match="Adjacency .* source optic 'O': optic count .* cannot be written exactly",
    ):
        build_scenario(corridor_graph, cfg, library)


def test_empty_internal_blueprint_rule_refuses_in_e2(corridor_graph, scenario_config):
    base = load_library(None)
    bps = thaw(base.blueprints)
    bps["SingleRouter"]["links"] = [
        {"source": "missing", "target": "core", "capacity": 1, "cost": 1}
    ]
    library = Library(bps, base.components, base.failure_policies, base.workflows)
    with pytest.raises(ValueError, match="SingleRouter"):
        build_scenario(corridor_graph, scenario_config, library)


def test_all_self_internal_blueprint_rule_refuses_in_e2(
    corridor_graph, scenario_config
):
    base = load_library(None)
    bps = thaw(base.blueprints)
    bps["SingleRouter"]["links"] = [
        {"source": "core", "target": "core", "capacity": 1, "cost": 1}
    ]
    library = Library(bps, base.components, base.failure_policies, base.workflows)
    with pytest.raises(ValueError) as error:
        build_scenario(corridor_graph, scenario_config, library)
    assert (
        str(error.value) == "blueprint 'SingleRouter' link rule 0 expands to no links"
    )
    assert build_scenario(corridor_graph, scenario_config, base).hardware


def test_unreferenced_empty_site_refuses_in_e2(corridor_graph, scenario_config):
    corridor_graph.remove_nodes_from((1, 2))
    scenario_config.build.build_defaults.dc_regions_per_metro = 0
    scenario_config.traffic.enabled = False
    base = load_library(None)
    bps = thaw(base.blueprints)
    bps["SingleRouter"] = {"nodes": {}, "links": []}
    library = Library(bps, base.components, base.failure_policies, base.workflows)
    with pytest.raises(ValueError) as error:
        build_scenario(corridor_graph, scenario_config, library)
    assert str(error.value) == "Site 'metro1/pop1' expands to no devices"
    assert (
        len(build_scenario(corridor_graph, scenario_config, base).site_plan.sites) == 1
    )


@pytest.mark.parametrize("json_roundtrip", [False, True], ids=["direct", "json"])
def test_self_corridor_has_named_refusal(tmp_path, json_roundtrip):
    import networkx as nx

    from netlab_topogen.config import TopologyConfig
    from netlab_topogen.corridors import validate_corridor_graph
    from netlab_topogen.integrated_graph import load_from_json, save_to_json

    config = TopologyConfig()
    config.traffic.enabled = False
    config.build.tm_sizing.enabled = False
    config.corridors.risk_groups.enabled = False
    config.build.build_defaults.dc_regions_per_metro = 0
    config.build.build_defaults.pop_per_metro = 1
    graph = nx.MultiGraph()
    nodes = [(0.0, 0.0), (100000.0, 0.0)]
    for index, node in enumerate(nodes):
        graph.add_node(
            node,
            node_type="metro",
            name=f"m{index}",
            name_orig=f"m{index}",
            metro_id=str(index),
            x=node[0],
            y=node[1],
            radius_km=10.0,
        )
    graph.add_edge(
        *nodes,
        key=0,
        edge_type="corridor",
        length_km=100.0,
        geometry=nodes,
    )

    def build_input(source, label):
        validate_corridor_graph(source, nodes, config.validation)
        if not json_roundtrip:
            return source
        path = tmp_path / f"{label}.json"
        save_to_json(
            source, path, config.projection.target_crs, config.output.formatting
        )
        loaded, crs = load_from_json(path)
        assert crs == config.projection.target_crs
        return loaded

    broken = graph.copy()
    broken.add_edge(
        nodes[0],
        nodes[0],
        key=0,
        edge_type="corridor",
        length_km=20.0,
        geometry=[nodes[0], nodes[0]],
    )
    library = load_library(None)
    with pytest.raises(ValueError) as error:
        build_scenario(build_input(broken, "self-corridor"), config, library)
    assert str(error.value) == (
        "Adjacency 'metro1/pop1|metro1/pop1|inter_metro:1-1:path0:1-1' "
        "expands to no device links"
    )
    control = build_scenario(build_input(graph, "ordinary-corridor"), config, library)
    assert tuple(control.hardware) == (
        "metro1/pop1|metro2/pop1|inter_metro:1-2:path0:1-1",
    )


def test_exact_derived_step_changes_230_to_220(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.traffic.mw_per_dc_region = 200
    cfg.build.tm_sizing.enabled = True
    cfg.build.tm_sizing.headroom = 1
    cfg.build.tm_sizing.quantum_gbps = 10
    cfg.build.tm_sizing.alpha_dc_to_pop = 1.1
    cfg.build.tm_sizing.respect_min_base_capacity = False
    library = hardware_case(corridor_graph, cfg, 10)
    build = build_scenario(corridor_graph, cfg, library)
    dc = [h for edge, h in build.hardware.items() if "dc_to_pop" in edge]
    assert len(dc) == 2 and all(
        h.capacity == 220 and h.source_count == h.target_count == 22 for h in dc
    )
    assert not validate_scenario(build.yaml, cfg.components).issues


@pytest.mark.parametrize("striped", [False, True])
def test_stripe_match_sees_finalized_platforms(
    corridor_graph, scenario_config, striped
):
    cfg = scenario_config
    cfg.build.build_defaults.site_blueprint = "FullMesh4"
    cfg.build.build_defaults.inter_metro_link.striping = (
        {"mode": "width", "width": 2} if striped else {}
    )
    cfg.build.build_defaults.inter_metro_link.mode = "one_to_one"
    cfg.build.build_defaults.inter_metro_link.match = {
        "conditions": [
            {"attr": "hardware.component", "op": "==", "value": "CoreRouter"}
        ]
    }
    cfg.components.hw_component = {"core": "CoreRouter", "dc": ""}
    build = build_scenario(corridor_graph, cfg, load_library(None))
    assert all(
        len(hw.device_links) == (2 if striped else 4)
        for edge, hw in build.hardware.items()
        if "inter_metro" in edge
    )
    assert "800G-DR4" in yaml.safe_load(build.yaml)["components"]
    assert not validate_scenario(build.yaml, cfg.components).issues


def test_directional_optics_are_assigned_to_their_own_end(
    corridor_graph, scenario_config
):
    base = load_library(None)
    components = thaw(base.components)
    components.update(
        small={"capacity": 100, "ports": 0}, large={"capacity": 200, "ports": 0}
    )
    library = Library(
        base.blueprints, components, base.failure_policies, base.workflows
    )
    cfg = scenario_config
    cfg.components.optics = {
        "core->core": "large",
        "core->dc": "small",
        "dc->core": "large",
    }
    build = build_scenario(corridor_graph, cfg, library)
    dc = [hw for edge, hw in build.hardware.items() if "dc_to_pop" in edge]
    assert len(dc) == 3
    assert all(
        (hw.source_optic, hw.source_count, hw.target_optic, hw.target_count)
        == ("small", 4, "large", 2)
        for hw in dc
    )
    assert not validate_scenario(build.yaml, cfg.components).issues


def test_expansion_stages_are_batched_and_private_tags_do_not_escape(
    corridor_graph, scenario_config, monkeypatch
):
    from ngraph import Scenario

    cfg = scenario_config
    cfg.build.build_defaults.site_blueprint = "FullMesh4"
    cfg.build.build_defaults.inter_metro_link.striping = {"mode": "width", "width": 2}
    cfg.build.build_defaults.inter_metro_link.mode = "one_to_one"
    original = Scenario.from_yaml
    calls = []

    def counted(*args, **kwargs):
        parsed = original(*args, **kwargs)
        calls.append(parsed)
        return parsed

    monkeypatch.setattr(Scenario, "from_yaml", counted)
    library = load_library(None)
    build = build_scenario(corridor_graph, cfg, library)
    assert len(calls) == 2
    assert "_netlab_topogen_rule" not in build.yaml
    assert all(name.startswith("stripe_probe") for name in calls[0].network.nodes)
    assert len(calls[1].network.nodes) > len(calls[0].network.nodes)
    assert not validate_scenario(build.yaml, cfg.components).issues
    assert len(calls) == 3
    assert all(
        "_netlab_topogen_rule" not in link.attrs
        for link in calls[2].network.links.values()
    )


@pytest.mark.parametrize("empty_end", ["source", "target", "match"])
def test_empty_site_adjacency_is_refused_before_hardware_assignment(
    corridor_graph, scenario_config, empty_end
):
    from dataclasses import replace

    from netlab_topogen.scenario.hardware import expand_sites
    from netlab_topogen.scenario.records import StripeGroups
    from netlab_topogen.scenario.sites import (
        finalize_blueprints,
        plan_sites,
        read_corridors,
        resolve_metro_settings,
    )

    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(
        corridors, scenario_config.build, scenario_config.traffic
    )
    blueprints = finalize_blueprints(
        settings, load_library(None), scenario_config.components.hw_component
    )
    plan = plan_sites(corridors, settings, StripeGroups({}))
    first = plan.links[0]
    broken_blueprints = thaw(blueprints)
    if empty_end == "match":
        broken = replace(
            plan,
            links=(
                replace(
                    first,
                    match={
                        "conditions": [{"attr": "role", "op": "==", "value": "missing"}]
                    },
                ),
                *plan.links[1:],
            ),
        )
    else:
        broken_blueprints["Empty"] = {"nodes": {}, "links": []}
        site_id = getattr(first, empty_end)
        broken = replace(
            plan,
            sites=tuple(
                replace(site, blueprint="Empty") if site.id == site_id else site
                for site in plan.sites
            ),
        )
    with pytest.raises(ValueError) as error:
        expand_sites(corridors, broken, broken_blueprints, scenario_config)
    assert first.source in str(error.value) and first.target in str(error.value)
    links = expand_sites(corridors, plan, blueprints, scenario_config).links[first.id]
    assert len(links) == 1
    assert links[0].source_role == "core" and links[0].target_role == "dc"


def test_combined_exact_uniform_sizing_keeps_the_intended_step(
    corridor_graph, scenario_config
):
    cfg = scenario_config
    corridor_graph[0][2][0]["length_km"] = 600
    corridor_graph.add_edge(0, 1, key=1, edge_type="corridor", length_km=450)
    cfg.build.build_defaults.site_blueprint = "FullMesh4"
    cfg.build.build_defaults.inter_metro_link.mode = "one_to_one"
    cfg.build.build_defaults.inter_metro_link.capacity = 800
    cfg.build.build_defaults.dc_to_pop_link.capacity = 800
    cfg.build.build_overrides = {
        "alpha": {"inter_metro_link": {"striping": {"mode": "width", "width": 1}}},
        "beta": {"inter_metro_link": {"striping": {"mode": "width", "width": 2}}},
    }
    cfg.build.tm_sizing.enabled = True
    cfg.build.tm_sizing.headroom = 1.28
    cfg.build.tm_sizing.quantum_gbps = 1600
    cfg.traffic.mw_per_dc_region = 25
    cfg.traffic.gbps_per_mw = 200
    cfg.traffic.gravity.mw_per_dc_region_overrides = {"beta": 50}
    cfg.components.optics = {
        "core->core": "800G-ZR+",
        "core->dc": "800G-ZR+",
        "dc->core": "800G-ZR+",
    }
    build = build_scenario(corridor_graph, cfg, load_library(None))
    assert {d.volume for d in build.traffic[0].demands} == {Fraction(10000, 3)}
    assert sorted(build.sizing.load.values()) == [0, Fraction(5000, 3), 5000, 5000]
    assert sorted(build.sizing.capacity.values()) == [
        800,
        3200,
        6400,
        6400,
        12800,
        12800,
        17600,
    ]
    dc = [hw for edge, hw in build.hardware.items() if "dc_to_pop" in edge]
    assert sorted(hw.capacity for hw in dc) == [3200, 3200, 4400]
    assert sorted(hw.source_count for hw in dc) == [4, 4, 6]
    assert not validate_scenario(build.yaml, cfg.components).issues


@pytest.mark.parametrize("capacity", [0, -5, float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("end", ["source", "target"])
def test_nonpositive_optic_error_names_adjacency_link_end_and_optic(capacity, end):
    from netlab_topogen.config import ComponentsConfig
    from netlab_topogen.scenario.hardware import assign_hardware
    from netlab_topogen.scenario.records import (
        DeviceLink,
        SiteExpansion,
        SiteLink,
        SitePlan,
        Sizing,
    )

    plan = SitePlan(
        (),
        (
            SiteLink(
                "A|B|k",
                "dc_to_pop",
                "A",
                "B",
                1,
                1,
                10,
                {},
                {},
                "k",
                (),
                (),
                None,
                None,
            ),
        ),
        (),
    )
    edge = "A|B|k"
    expansion = SiteExpansion({edge: (DeviceLink("A/core", "B/dc", "core", "dc"),)})
    sizes = Sizing({}, {edge: Fraction(10)})
    components = {"OZ": {"capacity": capacity}, "good": {"capacity": 10}}
    cfg = ComponentsConfig(
        optics={
            "core->dc": "OZ" if end == "source" else "good",
            "dc->core": "OZ" if end == "target" else "good",
        }
    )
    with pytest.raises(ValueError) as error:
        assign_hardware(plan, sizes, expansion, Library({}, components, {}, {}), cfg)
    assert (
        str(error.value)
        == f"Adjacency 'A|B|k' A/core->B/dc {end} optic 'OZ': optic capacity must be finite and positive, got {float(capacity)!r}"
    )
    components["OZ"]["capacity"] = 10
    assert (
        assign_hardware(plan, sizes, expansion, Library({}, components, {}, {}), cfg)[
            edge
        ].source_count
        == 1
    )


def test_unknown_assigned_optic_has_a_named_refusal(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.components.optics = {
        "core->core": "Missing",
        "core->dc": "Missing",
        "dc->core": "Missing",
    }
    with pytest.raises(ValueError) as error:
        build_scenario(corridor_graph, cfg, load_library(None))
    assert str(error.value) == "Unknown components: ['Missing']"
    cfg.components.optics = {
        "core->core": "800G-ZR+",
        "core->dc": "800G-ZR+",
        "dc->core": "800G-ZR+",
    }
    assert build_scenario(corridor_graph, cfg, load_library(None)).yaml


def test_unknown_components_are_complete_sorted_and_precede_hardware(
    corridor_graph, scenario_config, monkeypatch
):
    import netlab_topogen.scenario.build as stages

    base = load_library(None)
    blueprints = thaw(base.blueprints)
    blueprints["SingleRouter"]["nodes"]["core"]["attrs"]["hardware"] = {
        "component": "NoBlueprintPlatform",
        "count": 1,
    }
    library = Library(
        blueprints, base.components, base.failure_policies, base.workflows
    )
    cfg = scenario_config
    cfg.components.hw_component = {"spine": "NoPlatform", "dc": ""}
    cfg.components.optics = {
        "core->core": "NoOptA",
        "core->dc": "NoOptB",
        "dc->core": "NoOptB",
    }

    def unexpected(*args, **kwargs):
        raise AssertionError("hardware ran before the complete component check")

    monkeypatch.setattr(stages, "assign_hardware", unexpected)
    with pytest.raises(ValueError) as error:
        build_scenario(corridor_graph, cfg, library)
    assert (
        str(error.value)
        == "Unknown components: ['NoBlueprintPlatform', 'NoOptA', 'NoOptB', 'NoPlatform']"
    )
    cfg.build.tm_sizing.enabled = True
    cfg.traffic.enabled = False
    with pytest.raises(ValueError) as error:
        build_scenario(corridor_graph, cfg, library)
    assert (
        str(error.value)
        == "TM sizing is enabled but traffic generation is disabled in configuration"
    )


def test_two_unknown_optics_are_reported_together(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.components.optics = {
        "core->core": "NoOptA",
        "core->dc": "NoOptB",
        "dc->core": "NoOptB",
    }
    with pytest.raises(ValueError) as error:
        build_scenario(corridor_graph, cfg, load_library(None))
    assert str(error.value) == "Unknown components: ['NoOptA', 'NoOptB']"
