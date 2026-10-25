"""Workflow outputs and dependencies preserve their meanings on the public API."""

import io
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError
from ngraph import Scenario, prepare, run, write_json

from topogen.library import load_library


def execute(outputs=None, workers=1, trace=False, capacity=10):
    doc = {
        "seed": 42,
        "units": {"resolution": 2**-10},
        "network": {
            "nodes": {"A": {}, "B": {}},
            "links": [{"source": "A", "target": "B", "capacity": capacity, "cost": 3}],
        },
        "demands": {
            "tm": [
                {
                    "source": "^A$",
                    "target": "^B$",
                    "volume": 2,
                    "preset": "SHORTEST_PATHS_ECMP",
                }
            ]
        },
        "workflow": [
            {
                "type": "MaximumSupportedDemand",
                "name": "msd",
                "demand_set": "tm",
                "alpha_tolerance": 0.05,
            },
            {
                "type": "TrafficMatrixPlacement",
                "name": "p",
                "demand_set": "tm",
                "alpha_from_step": "msd",
                "workers": workers,
                "failure_trace": trace,
                "outputs": outputs or {},
            },
        ],
    }
    stream = io.StringIO()
    scenario = Scenario.from_yaml(yaml.safe_dump(doc))
    write_json(run(prepare(scenario, demand_sets=("tm",))), stream)
    return json.loads(stream.getvalue())


@pytest.mark.parametrize("directory", [None, Path("lib")])
def test_builtin_and_repository_workflow_definitions(directory):
    library = load_library(directory)
    assert set(library.workflows) == (
        {"design_analysis_brief", "hose_samples_3", "hose_samples_10"}
        if directory is not None
        else {"design_analysis_brief"}
    )
    for definition in library.workflows.values():
        for step in definition.steps:
            if step["type"] == "MaximumSupportedDemand":
                assert step["alpha_tolerance"] == 0.05
            if step["type"] == "TrafficMatrixPlacement":
                assert step["workers"] == "auto"
                assert step["outputs"] == {"cost_distribution": True}
                assert step["alpha_from_step"].startswith("msd")
    assert not library.workflows["design_analysis_brief"].steps[2]["failure_trace"]
    if directory is not None:
        assert library.workflows["hose_samples_3"].steps[2]["failure_trace"]


def test_feasible_alpha_cost_bins_and_edge_flows():
    payload = execute({"cost_distribution": True, "edge_flows": True})
    assert payload["steps"][0]["result"]["alpha_feasible"]["value"] == 5
    p = payload["steps"][1]
    assert p["question"]["alpha_source"] == {"kind": "msd_step", "step": "msd"}
    assert p["result"]["alpha"]["value"] == 5
    row = p["result"]["cases"][0]["states"]["converged"]["rows"][0]
    assert row["result"]["delivered_units"] == 10240
    assert row["cost_distribution"]["bins"] == [
        {"cost": 3, "physical_path_cost": 3, "delivered_units": 10240}
    ]
    assert sorted(x["units"] for x in row["result"]["edge_flows"]) == [0, 10240]


def test_details_is_not_cost_distribution():
    row = execute({"details": True})["steps"][1]["result"]["cases"][0]["states"][
        "converged"
    ]["rows"][0]
    assert "cost_distribution" not in row
    assert "edge_flows" not in row["result"]


def test_workers_preserve_semantic_results():
    a, b = execute(workers=1), execute(workers=2)
    pa, pb = a["steps"][1]["result"], b["steps"][1]["result"]
    assert (pa["metadata"]["workers"], pb["metadata"]["workers"]) == (1, 2)
    pa.pop("metadata")
    pb.pop("metadata")
    assert pa == pb


@pytest.mark.parametrize("workers", [0, -1, 1.5])
def test_bad_workers_refused(workers):
    with pytest.raises((ValueError, ValidationError)):
        execute(workers=workers)


def test_alpha_tolerance_controls_a_non_boundary_optimum():
    payload = execute(capacity=10.37)
    # q=1/1024: capacity10619 units. The bracket ends at5.15625/5.1875.
    assert payload["steps"][0]["result"]["alpha_feasible"]["value"] == 5.15625


def test_failure_trace_and_occurrences():
    def one(trace):
        doc = {
            "seed": 42,
            "network": {
                "nodes": {"A": {}, "B": {}},
                "links": [{"source": "A", "target": "B", "capacity": 10, "cost": 1}],
            },
            "demands": {"tm": [{"source": "^A$", "target": "^B$", "volume": 2}]},
            "failures": {
                "one": {
                    "modes": [
                        {
                            "weight": 1,
                            "rules": [{"scope": "link", "mode": "choice", "count": 1}],
                        }
                    ]
                }
            },
            "workflow": [
                {
                    "type": "TrafficMatrixPlacement",
                    "name": "p",
                    "demand_set": "tm",
                    "failure_policy": "one",
                    "iterations": 3,
                    "workers": 1,
                    "failure_trace": trace,
                }
            ],
        }
        stream = io.StringIO()
        scenario = Scenario.from_yaml(yaml.safe_dump(doc))
        write_json(run(prepare(scenario, demand_sets=("tm",))), stream)
        return json.loads(stream.getvalue())["steps"][0]["result"]["cases"]

    a, b = one(True), one(False)
    assert len(a) == len(b) == 2
    assert a[0]["case"]["kind"] == b[0]["case"]["kind"] == "baseline"
    assert a[0]["case"]["occurrences"] == b[0]["case"]["occurrences"] == 0
    a, b = a[1:], b[1:]
    assert a[0]["case"]["occurrences"] == b[0]["case"]["occurrences"] == 3
    assert (
        a[0]["case"]["pattern"]
        == b[0]["case"]["pattern"]
        == {"excluded_nodes": [], "excluded_links": ["A|B|0"]}
    )
    assert a[0]["case"]["trace"]["draw_index"] == 0
    assert a[0]["case"]["trace"]["expanded_links"] == []
    assert a[0]["case"]["trace"]["selections"] == [
        {"rule_index": 0, "scope": "link", "selected": ["A|B|0"]}
    ]
    assert a[0]["case"]["trace"]["selections"]
    assert "trace" not in b[0]["case"]
