"""Exercise emitted scenarios through NetGraph's real parser and workflows."""

from collections import Counter
from pathlib import Path

import networkx as nx
import pytest
import yaml
from ngraph.dsl.blueprints.expand import expand_network_dsl
from ngraph.scenario import Scenario

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.config import TopologyConfig
from topogen.scenario_builder import build_scenario
from topogen.validation import validate_scenario_yaml

REPO = Path(__file__).resolve().parents[2]


def _graph():
    graph = nx.Graph()
    for index, name in enumerate(["alpha", "beta", "gamma"]):
        graph.add_node(
            name,
            node_type="metro",
            name=name,
            metro_id=name,
            x=index * 300_000.0,
            y=0.0,
            radius_km=20.0,
        )
    for source, target in [("alpha", "beta"), ("beta", "gamma"), ("alpha", "gamma")]:
        graph.add_edge(
            source,
            target,
            edge_type="corridor",
            length_km=300.0,
            risk_groups=[f"corridor_risk_{source}_{target}"],
        )
    return graph


@pytest.mark.parametrize("model", ["uniform", "gravity", "hose"])
@pytest.mark.parametrize("matrix_name", ["default", "custom_traffic"])
def test_generated_scenario_executes_current_netgraph(
    tmp_path, monkeypatch, model, matrix_name
):
    # No cwd/lib overrides: exercise the installed package's built-in libraries.
    monkeypatch.chdir(tmp_path)
    config = TopologyConfig()
    config.traffic.matrix_name = matrix_name
    config.traffic.model = model
    config.traffic.gbps_per_mw = 1.0
    config.traffic.mw_per_dc_region = 10.0
    config.traffic.priority_ratios = {0: 1.0}
    config.traffic.flow_policy_config = {0: "SHORTEST_PATHS_ECMP"}
    config.build.build_defaults.dc_regions_per_metro = 1
    scenario_yaml = build_scenario(_graph(), config)
    scenario = Scenario.from_yaml(scenario_yaml)
    assert scenario.demand_set.get_set(matrix_name)
    assert scenario.failure_policy_set.policies
    assert not validate_scenario_yaml(scenario_yaml)
    # Keep the full workflow, with a small Monte Carlo sample for the test.
    for step in scenario.workflow:
        if hasattr(step, "iterations"):
            step.iterations = 3
    scenario.run()
    results = scenario.results.to_dict()["steps"]
    assert set(results) == {
        "network_statistics",
        "msd_baseline",
        "tm_placement",
        "cost_power",
    }
    assert results["msd_baseline"]["data"]["alpha_star"] > 0


@pytest.mark.parametrize("example", sorted((REPO / "examples").glob("*.yml")))
def test_example_scenario_runs_with_current_netgraph(example, monkeypatch):
    monkeypatch.chdir(REPO)
    config = TopologyConfig.from_yaml(example)
    graph = _graph()
    # Exercise the actual example DC overrides and optics with synthetic corridors.
    for node, name in zip(
        graph.nodes,
        ["new-york-jersey-city-newark", "columbus", "washington-arlington"],
        strict=True,
    ):
        graph.nodes[node]["name"] = name
    scenario_yaml = build_scenario(graph, config)
    assert not validate_scenario_yaml(
        scenario_yaml,
        hw_component_map=config.components.hw_component,
        optics_map=config.components.optics,
    )
    scenario = Scenario.from_yaml(scenario_yaml)
    for step in scenario.workflow:
        if hasattr(step, "iterations"):
            step.iterations = 3
    scenario.run()
    assert scenario.results.to_dict()["steps"]["msd_baseline"]["data"]["alpha_star"] > 0


@pytest.mark.parametrize("samples", [3, 10])
def test_shipped_sample_workflows_execute(samples, monkeypatch):
    monkeypatch.chdir(REPO)
    config = TopologyConfig()
    config.traffic.model = "hose"
    config.traffic.samples = samples
    config.traffic.matrix_name = "baseline_traffic_matrix"
    config.traffic.gbps_per_mw = 1.0
    config.traffic.mw_per_dc_region = 10.0
    config.traffic.priority_ratios = {0: 1.0}
    config.build.build_defaults.dc_regions_per_metro = 1
    config.workflows.assignments.default = f"hose_samples_{samples}"
    scenario_yaml = build_scenario(_graph(), config)
    assert not validate_scenario_yaml(scenario_yaml)
    scenario = Scenario.from_yaml(scenario_yaml)
    for step in scenario.workflow:
        if hasattr(step, "iterations"):
            step.iterations = 3
    scenario.run()
    assert len(scenario.results.to_dict()["steps"]) == 2 * samples + 2


def test_validator_reports_netgraph_workflow_error():
    step = {"type": "NetworkStats", "name": "analysis", "unknown_option": True}
    data = {
        "network": {
            "nodes": {"a": {}, "b": {}},
            "links": [{"source": "a", "target": "b", "capacity": 100, "cost": 1}],
        },
        "demands": {"default": [{"source": "a", "target": "b", "volume": 10}]},
        "workflow": [step],
    }
    issues = validate_scenario_yaml(yaml.safe_dump(data))
    assert any(
        "Unrecognized key" in issue and "unknown_option" in issue for issue in issues
    )


def test_validator_rejects_unknown_failure_field():
    data = {
        "network": {
            "nodes": {"a": {}, "b": {}},
            "links": [{"source": "a", "target": "b", "capacity": 100, "cost": 1}],
        },
        "failures": {
            "custom": {
                "modes": [
                    {
                        "weight": 1,
                        "rules": [
                            {
                                "scope": "link",
                                "mode": "choice",
                                "count": 1,
                                "unknown_option": True,
                            }
                        ],
                    }
                ]
            }
        },
    }
    issues = validate_scenario_yaml(yaml.safe_dump(data))
    assert any("unknown_option" in issue for issue in issues)


@pytest.mark.parametrize(
    "blueprint,nodes,intra,inter",
    [("Dragonfly_A3H2G7", 21, 21, 21), ("DragonFly_CustomG4", 16, 24, 16)],
)
def test_dragonfly_expansion_preserves_fabric(blueprint, nodes, intra, inter):
    net = expand_network_dsl(
        {
            "blueprints": {blueprint: get_builtin_blueprints()[blueprint]},
            "network": {"nodes": {"site": {"blueprint": blueprint}}},
        }
    )
    assert len(net.nodes) == nodes
    counts = Counter(link.attrs["link_type"] for link in net.links.values())
    assert counts == {"intra_group": intra, "inter_group": inter}
    graph = nx.Graph((link.source, link.target) for link in net.links.values())
    assert nx.is_connected(graph)


@pytest.mark.parametrize(
    "policy",
    [
        "empty",
        "single_random_link_failure",
        "mc_baseline",
        "mc_fabric_heavy",
        "fabric_only",
    ],
)
def test_shipped_failure_policies_run(policy, monkeypatch):
    monkeypatch.chdir(REPO)
    config = TopologyConfig()
    config.failure_policies.assignments.default = policy
    config.traffic.gbps_per_mw = 1.0
    config.traffic.mw_per_dc_region = 10.0
    scenario = Scenario.from_yaml(build_scenario(_graph(), config))
    for step in scenario.workflow:
        if hasattr(step, "iterations"):
            step.iterations = 3
            step.failure_policy = policy
    scenario.run()
    assert scenario.results.to_dict()["steps"]["tm_placement"]["data"]["baseline"]
