"""Preparation refusals, endpoint accounting and ordered audits on written text."""

from copy import deepcopy

import pytest
import yaml

from topogen.config import ComponentsConfig
from topogen.library import load_library
from topogen.scenario import build_scenario
from topogen.validation import validate_scenario


def tiny():
    return {
        "network": {
            "nodes": {
                "A": {"attrs": {"role": "core"}},
                "B": {"attrs": {"role": "core"}},
            },
            "links": [{"source": "A", "target": "B", "capacity": 10, "cost": 1}],
        }
    }


def validate(doc, components=None):
    return validate_scenario(
        yaml.safe_dump(doc, sort_keys=False), components or ComponentsConfig()
    )


def test_zero_cost_is_refused_in_prepare_with_message():
    doc = tiny()
    doc["network"]["links"][0]["cost"] = 0
    result = validate(doc)
    assert result.prepared is None
    assert len(result.issues) == 1 and "cost must be in [1,2^62)" in result.issues[0]
    doc["network"]["links"][0]["cost"] = 1
    assert not validate(doc).issues


def test_all_sets_bound_even_when_not_used():
    doc = tiny()
    doc["demands"] = {
        "unused": [
            {"source": "^(A|B)$", "target": "^B$", "volume": 2, "mode": "combine"}
        ]
    }
    result = validate(doc)
    assert result.prepared is None
    assert (
        len(result.issues) == 1
        and "non-diagonal source/target overlap" in result.issues[0]
    )
    doc["demands"]["unused"][0]["source"] = "^A$"
    assert not validate(doc).issues


@pytest.mark.parametrize(
    "value",
    [
        {1: "bad"},
        {"nested": {2: "bad"}},
        {"x": float("nan")},
        {"x": [float("inf")]},
        {"x": float("-inf")},
    ],
)
def test_bad_attrs_refuse_at_parse(value):
    doc = tiny()
    doc["network"]["links"][0]["attrs"] = value
    result = validate(doc)
    assert result.prepared is None and len(result.issues) == 1
    assert (
        "JSON object keys must be strings" in result.issues[0]
        or "expected finite JSON data" in result.issues[0]
    )
    doc["network"]["links"][0]["attrs"] = {"valid": [1, True, None, {"x": 1.5}]}
    assert not validate(doc).issues


def test_discarded_units_are_issues():
    doc = tiny()
    doc["demands"] = {"tm": [{"source": "^A$", "target": "^A$", "volume": 2}]}
    result = validate(doc)
    assert result.prepared is not None
    assert len(result.issues) == 1 and "discarded units" in result.issues[0]


@pytest.mark.parametrize("reference", ["demand_set", "failure_policy"])
def test_workflow_references_fail_in_ngraph(reference):
    doc = tiny()
    doc["workflow"] = [
        {
            "type": "TrafficMatrixPlacement",
            "name": "p",
            "demand_set": "tm",
            reference: "missing",
        }
    ]
    doc["demands"] = {"tm": [{"source": "^A$", "target": "^B$", "volume": 1}]}
    result = validate(doc)
    assert result.prepared is None and len(result.issues) == 1
    assert "missing" in result.issues[0]


def test_empty_nodes_keep_a_home():
    result = validate({"network": {"nodes": {}, "links": []}})
    assert result.issues
    assert "nodes" in result.issues[0]


def test_build_metadata_has_one_source(corridor_graph, scenario_config):
    doc = yaml.safe_load(
        build_scenario(corridor_graph, scenario_config, load_library(None)).yaml
    )
    for index, (_, metro) in enumerate(corridor_graph.nodes(data=True), 1):
        pop = doc["network"]["nodes"][f"metro{index}/pop[1-1]"]["attrs"]
        dc = doc["network"]["nodes"][f"metro{index}/dc[1-1]"]["attrs"]
        for key in (
            "metro_name",
            "metro_name_orig",
            "metro_id",
            "location_x",
            "location_y",
        ):
            assert pop[key] == dc[key]
        assert (pop["location_x"], pop["location_y"]) == (metro["x"], metro["y"])
        assert dc["mw_per_dc_region"] == scenario_config.traffic.mw_per_dc_region
        assert dc["gbps_per_mw"] == scenario_config.traffic.gbps_per_mw
    bad = deepcopy(corridor_graph)
    del bad.nodes[0]["x"]
    with pytest.raises(ValueError, match="missing required attributes"):
        build_scenario(bad, scenario_config, load_library(None))


def dc_case(volume=40, combine=False):
    names = [f"metro{i}/dc1/n{j}" for i in (1, 2) for j in (1, 2)]
    nodes = {name: {"attrs": {"role": "dc"}} for name in names}
    nodes["metro1/pop1/n"] = {"attrs": {"role": "core"}}
    links = [
        {"source": name, "target": "metro1/pop1/n", "capacity": 20, "cost": 1}
        for name in names
    ]
    return {
        "network": {"nodes": nodes, "links": links},
        "demands": {
            "tm": [
                {
                    "source": "^metro1/dc1/.*",
                    "target": "^metro2/dc1/.*",
                    "volume": volume,
                    "mode": "combine" if combine else "pairwise",
                }
            ]
        },
    }


@pytest.mark.parametrize("combine", [False, True])
def test_dc_accounting_is_per_site_not_per_leaf(combine):
    assert not validate(dc_case(40, combine)).issues
    result = validate(dc_case(41, combine))
    assert len(result.issues) == 2
    assert (
        "metro1/dc1 egress demand 41 exceeds adjacency capacity 40" in result.issues[0]
    )
    assert (
        "metro2/dc1 ingress demand 41 exceeds adjacency capacity 40" in result.issues[1]
    )


def test_dc_sets_stay_separate():
    doc = dc_case(30)
    doc["demands"]["second"] = deepcopy(doc["demands"]["tm"])
    assert not validate(doc).issues
    doc["demands"]["tm"] += deepcopy(doc["demands"]["second"])
    assert len(validate(doc).issues) == 2


def test_optional_platforms_and_exact_optic_coverage():
    doc = tiny()
    doc["components"] = {"O": {"capacity": 1.2245, "ports": 1}}
    link = doc["network"]["links"][0]
    link["capacity"] = 60.0005
    link["attrs"] = {
        "hardware": {
            end: {"component": "O", "count": 49.0} for end in ("source", "target")
        }
    }
    components = ComponentsConfig(hw_component={"core": ""})
    assert not validate(doc, components).issues
    link["attrs"]["hardware"]["target"]["count"] = 48
    result = validate(doc, components)
    assert len(result.issues) == 1 and "hardware capacity shortfall" in result.issues[0]


def test_isolated_and_role_issues_have_deterministic_order():
    doc = tiny()
    doc["network"]["nodes"]["isolated"] = {}
    issues = validate(doc).issues
    assert len(issues) == 2
    assert issues[0].startswith("1 isolated") and issues[1].startswith("node roles:")


@pytest.mark.parametrize("disabled", ["link", "node"])
def test_dc_available_capacity_excludes_disabled_objects(disabled):
    doc = dc_case(30)
    assert not validate(doc).issues
    if disabled == "link":
        doc["network"]["links"][0]["disabled"] = True
    else:
        doc["network"]["nodes"]["metro1/dc1/n1"]["disabled"] = True
    issues = validate(doc).issues
    assert (
        len(issues) == 1
        and "metro1/dc1 egress demand 30 exceeds adjacency capacity 20" in issues[0]
    )


@pytest.mark.parametrize("container", [[], ["role"], "core", 7])
def test_attrs_container_must_be_an_object(container):
    doc = tiny()
    doc["network"]["links"][0]["attrs"] = container
    refused = validate(doc)
    assert refused.prepared is None and len(refused.issues) == 1
    assert "attrs" in refused.issues[0]
    doc["network"]["links"][0]["attrs"] = {"valid": container}
    assert not validate(doc).issues


@pytest.mark.parametrize(
    "section,value",
    [
        ("network", ["bad"]),
        ("nodes", ["bad"]),
        ("nodes", {1: {}}),
        ("nodes", {"A": ["bad"]}),
        ("failures", ["bad"]),
        ("demands", ["bad"]),
    ],
)
def test_old_container_shape_checks_have_a_netgraph_home(section, value):
    from ngraph import Scenario

    doc = tiny()
    if section == "nodes":
        doc["network"][section] = value
    else:
        doc[section] = value
    text = yaml.safe_dump(doc)
    with pytest.raises(Exception) as upstream:
        Scenario.from_yaml(text)
    result = validate_scenario(text, ComponentsConfig())
    assert result.prepared is None
    assert result.issues == (f"ngraph scenario: {upstream.value}",)
    assert not validate(tiny()).issues


@pytest.mark.parametrize("capacity,delta", [(20, 2e-8), (1e-5, 5e-10)])
def test_dc_capacity_preserves_relative_and_absolute_tolerance(capacity, delta):
    doc = dc_case(2 * capacity + delta)
    doc["units"] = {"resolution": 2**-32}
    for link in doc["network"]["links"]:
        link["capacity"] = capacity
    result = validate(doc)
    assert result.prepared is not None
    demands = result.prepared.demand_sets["tm"].demands
    assert (
        sum(d.nominal_units for d in demands) * result.prepared.topology.q
        > 2 * capacity
    )
    assert not result.issues
    doc["demands"]["tm"][0]["volume"] = (
        2 * capacity + max(1e-9, 2 * capacity * 1e-9) * 3
    )
    assert len(validate(doc).issues) == 2


def test_nonempty_platform_mapping_requires_installed_platforms():
    doc = tiny()
    doc["components"] = {"P": {"capacity": 20, "ports": 2}}
    components = ComponentsConfig(hw_component={"core": "P"})
    assert validate(doc, components).issues == tuple(
        f"node hardware: role 'core' mapped in components.hw_component but node '{name}' has no hardware assignment"
        for name in ("A", "B")
    )
    for node in doc["network"]["nodes"].values():
        node["attrs"]["hardware"] = {"component": "P", "count": 1}
    assert not validate(doc, components).issues


@pytest.mark.parametrize(
    "capacity,volume,excess",
    [
        (1001, 1001, False),
        (1003, 1003, False),
        (33333, 33333, False),
        (100, 100.02, True),
        (4097, 4097.8194, True),
        (20001, 20005.0002, True),
    ],
)
def test_dc_audit_uses_declared_volume_with_automatic_units(
    corridor_graph, scenario_config, capacity, volume, excess
):
    corridor_graph.remove_node(2)
    cfg = scenario_config
    cfg.build.tm_sizing.enabled = False
    cfg.build.build_defaults.dc_to_pop_link.capacity = capacity
    cfg.build.build_defaults.inter_metro_link.capacity = 1_000_000
    cfg.traffic.mw_per_dc_region = 1
    cfg.traffic.gbps_per_mw = volume
    build = build_scenario(corridor_graph, cfg, load_library(None))
    assert "units" not in yaml.safe_load(build.yaml)
    result = validate_scenario(build.yaml, cfg.components)
    assert result.prepared is not None
    assert result.prepared.topology.q > 0.01
    if excess:
        assert result.issues == tuple(
            f"dc capacity: demand set 'default', metro{site}/dc1 {direction} demand {volume:,.6g} exceeds adjacency capacity {capacity:,.6g}"
            for direction in ("egress", "ingress")
            for site in (1, 2)
        )
    else:
        assert not result.issues


@pytest.mark.parametrize("optic_capacity", [0, -5])
def test_platform_and_optics_issues_have_full_ordered_messages(optic_capacity):
    doc = tiny()
    doc["components"] = {
        "OZ": {"capacity": optic_capacity, "ports": 1},
        "P": {"capacity": 100, "ports": 100},
    }
    doc["network"]["links"][0]["attrs"] = {
        "hardware": {
            end: {"component": "OZ", "count": 1} for end in ("source", "target")
        }
    }
    result = validate(doc, ComponentsConfig(hw_component={"core": "P"}))
    assert result.issues == (
        "node hardware: role 'core' mapped in components.hw_component but node 'A' has no hardware assignment",
        "node hardware: role 'core' mapped in components.hw_component but node 'B' has no hardware assignment",
        f"optics (blueprint): link 'A|B|0' source optic 'OZ': capacity must be positive, got {float(optic_capacity)!r} - 1 link ends",
        f"optics (blueprint): link 'A|B|0' target optic 'OZ': capacity must be positive, got {float(optic_capacity)!r} - 1 link ends",
    )
    doc["components"]["OZ"]["capacity"] = 10
    for node in doc["network"]["nodes"].values():
        node["attrs"]["hardware"] = {"component": "P", "count": 1}
    assert not validate(doc, ComponentsConfig(hw_component={"core": "P"})).issues


def test_combine_spanning_sites_keeps_guaranteed_load_rule():
    doc = {
        "network": {
            "nodes": {
                **{
                    f"metro{i}/dc1/n{j}": {"attrs": {"role": "dc"}}
                    for i in (1, 2)
                    for j in (1, 2)
                },
                "metro3/dc1/n": {"attrs": {"role": "dc"}},
                "core": {"attrs": {"role": "core"}},
            },
            "links": [
                *(
                    {
                        "source": f"metro{i}/dc1/n{j}",
                        "target": "core",
                        "capacity": 5,
                        "cost": 1,
                    }
                    for i in (1, 2)
                    for j in (1, 2)
                ),
                {"source": "metro3/dc1/n", "target": "core", "capacity": 40, "cost": 1},
            ],
        },
        "demands": {
            "tm": [
                {
                    "source": "^(metro[12])/dc1/.*",
                    "target": "^metro3/dc1/.*",
                    "volume": 30,
                    "mode": "combine",
                }
            ]
        },
    }
    assert not validate(doc).issues
    doc["demands"]["tm"][0]["group_mode"] = "per_group"
    assert validate(doc).issues == (
        "dc capacity: demand set 'tm', metro1/dc1 egress demand 15 exceeds adjacency capacity 10",
        "dc capacity: demand set 'tm', metro2/dc1 egress demand 15 exceeds adjacency capacity 10",
    )
    doc["demands"]["tm"][0]["volume"] = 20
    assert not validate(doc).issues
