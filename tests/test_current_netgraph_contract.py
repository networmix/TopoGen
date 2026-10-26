"""Configuration and library boundaries reject unsupported declarations."""

from pathlib import Path

import pytest
import yaml

from topogen.config import TopologyConfig
from topogen.library import load_library


@pytest.mark.parametrize(
    "extra",
    [
        {"placement_rounds": "auto"},
        {"workers": 1.5},
        {"alpha": 1, "alpha_from_step": "msd"},
    ],
)
def test_custom_workflow_is_checked_when_selected(
    tmp_path, extra, corridor_graph, scenario_config
):
    from topogen.scenario import build_scenario
    from topogen.validation import validate_scenario

    step = {"type": "TrafficMatrixPlacement", "name": "p", "demand_set": "tm", **extra}
    (tmp_path / "workflows.yml").write_text(yaml.safe_dump({"custom": [step]}))
    library = load_library(tmp_path)
    scenario_config.traffic.matrix_name = "tm"
    unused = build_scenario(corridor_graph, scenario_config, library)
    assert not validate_scenario(unused.yaml, scenario_config.components).issues
    scenario_config.workflows.assignments.default = "custom"
    selected = build_scenario(corridor_graph, scenario_config, library)
    issues = validate_scenario(selected.yaml, scenario_config.components).issues
    assert len(issues) == 1 and next(iter(extra)) in issues[0]
    (tmp_path / "workflows.yml").write_text(
        yaml.safe_dump(
            {
                "custom": [
                    {"type": "TrafficMatrixPlacement", "name": "p", "demand_set": "tm"}
                ]
            }
        )
    )
    selected = build_scenario(corridor_graph, scenario_config, load_library(tmp_path))
    assert not validate_scenario(selected.yaml, scenario_config.components).issues


@pytest.mark.parametrize("policy", [1, True, "1", "", "  ", "unknown", "TE_WCMP_UNLIM"])
def test_config_requires_current_preset_name(policy):
    raw = yaml.safe_load(Path("examples/small_baseline.yml").read_text())
    raw["traffic"]["flow_policy_config"] = {0: policy}
    with pytest.raises(ValueError, match="flow_policy_config"):
        TopologyConfig._from_dict(raw)


@pytest.mark.parametrize("value", ["EQUAL_BALANCED", "PROPORTIONAL", "unknown"])
def test_removed_sizing_key_is_unknown_even_when_disabled(value):
    raw = yaml.safe_load(Path("examples/small_baseline.yml").read_text())
    raw["build"]["tm_sizing"].update(enabled=False, flow_placement=value)
    with pytest.raises(ValueError, match="Additional properties are not allowed"):
        TopologyConfig._from_dict(raw)

    from dataclasses import fields

    sizing_type = type(TopologyConfig().build.tm_sizing)
    known = {field.name for field in fields(sizing_type)}
    unknown = set(raw["build"]["tm_sizing"]) - known
    assert len(unknown) == 1
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        sizing_type(**{unknown.pop(): value})


@pytest.mark.parametrize(
    "preset", ["shortest_paths_ecmp", " SHORTEST_PATHS_ECMP ", "Shortest_Paths_Ecmp"]
)
def test_preset_spelling_is_canonical_at_load(preset):
    from ngraph import FlowPolicyPreset

    raw = yaml.safe_load(Path("examples/small_baseline.yml").read_text())
    raw["traffic"]["flow_policy_config"] = {0: preset}
    with pytest.raises(ValueError) as error:
        TopologyConfig._from_dict(raw)
    assert (
        str(error.value)
        == f"traffic.flow_policy_config.0: expected one of {list(FlowPolicyPreset.__members__)}, got {preset!r}"
    )
    for name in FlowPolicyPreset.__members__:
        raw["traffic"]["flow_policy_config"] = {0: name}
        assert TopologyConfig._from_dict(raw).traffic.flow_policy_config == {0: name}
