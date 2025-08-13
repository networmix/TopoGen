"""Tests for the simplified workflows library module."""

from __future__ import annotations

from topogen.workflows_lib import get_builtin_workflows


class TestGetBuiltinWorkflows:
    """Tests for get_builtin_workflows."""

    def test_returns_dict(self) -> None:
        workflows = get_builtin_workflows()
        assert isinstance(workflows, dict)

    def test_contains_expected_workflows(self) -> None:
        workflows = get_builtin_workflows()
        # Should contain at least one workflow mapping name -> list[steps]
        assert isinstance(workflows, dict) and len(workflows) >= 1
        any_name, any_steps = next(iter(workflows.items()))
        assert isinstance(any_name, str) and isinstance(any_steps, list)

    def test_returns_deep_copy(self) -> None:
        workflows1 = get_builtin_workflows()
        workflows2 = get_builtin_workflows()
        wf_name = next(iter(workflows1.keys()))
        workflows1[wf_name][0]["mutated"] = True
        assert "mutated" not in workflows2[wf_name][0]
