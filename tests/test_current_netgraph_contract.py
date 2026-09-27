"""Reject invalid NetGraph inputs before TopoGen emits a scenario."""

from pathlib import Path

import pytest
import yaml

from topogen.config import TopologyConfig
from topogen.scenario.policies import _build_workflow_section
from topogen.traffic_matrix import generate_traffic_matrix


@pytest.mark.parametrize(
    "extra,error",
    [
        ({"placement_rounds": "auto"}, "Unrecognized key"),
        ({"parallelism": 1.5}, "parallelism must be"),
        (
            {"alpha": 1.0, "alpha_from_step": "msd"},
            "Set either alpha or alpha_from_step, not both",
        ),
    ],
)
def test_custom_workflow_rejected_before_emission(tmp_path, monkeypatch, extra, error):
    monkeypatch.chdir(tmp_path)
    library = tmp_path / "lib"
    library.mkdir()
    step = {"type": "TrafficMatrixPlacement", "name": "placement", **extra}
    (library / "workflows.yml").write_text(yaml.safe_dump({"custom": [step]}))
    config = TopologyConfig()
    config.workflows.assignments.default = "custom"
    with pytest.raises(ValueError, match=f"Workflow 'custom': .*{error}"):
        _build_workflow_section(config)


@pytest.mark.parametrize("policy", [1, True, "1", "", "  ", "unknown"])
def test_yaml_flow_policy_requires_preset_name(tmp_path, policy):
    path = tmp_path / "config.yml"
    data = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "examples/small_baseline.yml"
        ).read_text()
    )
    data["traffic"]["flow_policy_config"] = {0: policy}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="flow_policy_config"):
        TopologyConfig.from_yaml(path)


@pytest.mark.parametrize("model", ["uniform", "gravity", "hose"])
@pytest.mark.parametrize("policy", [1, "1", "", "unknown"])
def test_mutated_flow_policy_rejected_before_generation(model, policy):
    config = TopologyConfig()
    config.traffic.model = model
    config.traffic.flow_policy_config = {0: policy}
    metros = [
        {"name": "a", "x": 0, "y": 0},
        {"name": "b", "x": 1000, "y": 0},
    ]
    settings = {name: {"dc_regions_per_metro": 1} for name in ["a", "b"]}
    with pytest.raises(ValueError, match="flow_policy_config"):
        generate_traffic_matrix(metros, settings, config)
