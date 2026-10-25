"""Generated scenarios run through the public prepare/run result boundary."""

import io
import json
from pathlib import Path

import pytest
import yaml
from ngraph import Scenario, prepare, run, write_json

from topogen.config import TopologyConfig
from topogen.corridors import corridor_risk_name
from topogen.library import load_library
from topogen.scenario import build_scenario
from topogen.validation import validate_scenario

REPO = Path(__file__).resolve().parents[2]


def execute(text, policy=None):
    doc = yaml.safe_load(text)
    for step in doc["workflow"]:
        if "iterations" in step:
            step["iterations"] = 3
            step["workers"] = 1
            if policy is not None:
                step["failure_policy"] = policy
    scenario = Scenario.from_yaml(yaml.safe_dump(doc))
    stream = io.StringIO()
    write_json(run(prepare(scenario, demand_sets=tuple(scenario.demand_sets))), stream)
    return {step["name"]: step for step in json.loads(stream.getvalue())["steps"]}


@pytest.mark.parametrize(
    "model,alpha", [("uniform", 20), ("gravity", 16.65625), ("hose", 20)]
)
@pytest.mark.parametrize("matrix", ["default", "custom_traffic"])
def test_models_run(corridor_graph, scenario_config, model, alpha, matrix):
    scenario_config.traffic.model = model
    scenario_config.traffic.matrix_name = matrix
    scenario_config.traffic.flow_policy_config = {0: "SHORTEST_PATHS_ECMP"}
    # The reference default uses two PoPs per metro.
    scenario_config.build.build_defaults.pop_per_metro = 2
    build = build_scenario(corridor_graph, scenario_config, load_library(None))
    assert not validate_scenario(build.yaml, scenario_config.components).issues
    results = execute(build.yaml)
    assert results["msd_baseline"]["result"]["alpha_feasible"]["value"] == alpha
    placement = results["tm_placement"]["result"]
    assert (
        placement["alpha"]["value"]
        == results["msd_baseline"]["result"]["alpha_feasible"]["value"]
    )
    assert placement["outputs"]["cost_distribution"] is True
    assert placement["outputs"]["native"]["edge_flows"] is False
    assert any(
        row.get("cost_distribution", {}).get("bins")
        for case in placement["cases"]
        for row in case["states"]["converged"]["rows"]
    )
    assert list(results) == [
        "network_statistics",
        "msd_baseline",
        "tm_placement",
        "cost_power",
    ]


@pytest.mark.parametrize("example", sorted((REPO / "examples").glob("*.yml")))
def test_examples_run(example, corridor_graph):
    cfg = TopologyConfig.from_yaml(example)
    for node, name in zip(
        corridor_graph,
        ["new-york-jersey-city-newark", "columbus", "washington-arlington"],
        strict=True,
    ):
        corridor_graph.nodes[node]["name"] = name
    for u, v, k, data in corridor_graph.edges(keys=True, data=True):
        data["risk_groups"] = [
            corridor_risk_name(
                "corridor_risk",
                corridor_graph.nodes[u]["name"],
                corridor_graph.nodes[v]["name"],
                k,
            )
        ]
    build = build_scenario(corridor_graph, cfg, load_library(REPO / "lib"))
    assert not validate_scenario(build.yaml, cfg.components).issues
    assert (
        execute(build.yaml)["msd_baseline"]["result"]["alpha_feasible"]["value"]
        == 1.6875
    )


@pytest.mark.parametrize("samples", [3, 10])
def test_sampled_workflows_run(samples, corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.traffic.model = "hose"
    cfg.traffic.samples = samples
    cfg.traffic.matrix_name = "baseline_traffic_matrix"
    cfg.workflows.assignments.default = f"hose_samples_{samples}"
    build = build_scenario(corridor_graph, cfg, load_library(REPO / "lib"))
    assert not validate_scenario(build.yaml, cfg.components).issues
    assert len(execute(build.yaml)) == 2 * samples + 2


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
def test_failure_policies_run(policy, corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.failure_policies.assignments.default = policy
    build = build_scenario(corridor_graph, cfg, load_library(REPO / "lib"))
    result = execute(build.yaml, policy)["tm_placement"]["result"]
    assert sum(case["case"]["occurrences"] for case in result["cases"]) == 3
