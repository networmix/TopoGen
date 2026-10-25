"""Tests for the workflows library module."""

from __future__ import annotations

from topogen.workflows_lib import get_builtin_workflows


def test_builtin_failure_sampling_follows_scenario_seed(tmp_path, monkeypatch):
    from ngraph.utils.seed_manager import SeedManager
    from ngraph.workflow.parse import build_workflow_steps

    monkeypatch.chdir(tmp_path)

    def placement_seed(scenario_seed):
        seeds = SeedManager(scenario_seed)
        steps = build_workflow_steps(
            get_builtin_workflows()["design_analysis_brief"],
            derive_seed=lambda name: seeds.derive_seed("workflow_step", name),
        )
        placement = next(step for step in steps if step.name == "tm_placement")
        assert placement.seed == seeds.derive_seed("workflow_step", "tm_placement")
        return placement.seed

    first = placement_seed(7)
    assert placement_seed(7) == first
    assert placement_seed(42) != first


class TestGetBuiltinWorkflows:
    def test_returns_dict(self) -> None:
        workflows = get_builtin_workflows()
        assert isinstance(workflows, dict)

    def test_contains_expected_workflows(self) -> None:
        workflows = get_builtin_workflows()
        assert isinstance(workflows, dict) and len(workflows) >= 1
        any_name, any_steps = next(iter(workflows.items()))
        assert isinstance(any_name, str) and isinstance(any_steps, list)

    def test_returns_deep_copy(self) -> None:
        workflows1 = get_builtin_workflows()
        workflows2 = get_builtin_workflows()
        wf_name = next(iter(workflows1.keys()))
        workflows1[wf_name][0]["mutated"] = True
        assert "mutated" not in workflows2[wf_name][0]
