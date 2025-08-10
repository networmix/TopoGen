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
        assert "capacity_analysis" in workflows

    def test_returns_deep_copy(self) -> None:
        workflows1 = get_builtin_workflows()
        workflows2 = get_builtin_workflows()
        workflows1["capacity_analysis"][0]["mutated"] = True
        assert "mutated" not in workflows2["capacity_analysis"][0]

    def test_basic_capacity_analysis_structure(self) -> None:
        workflows = get_builtin_workflows()
        workflow = workflows["capacity_analysis"]
        assert len(workflow) == 2
        assert workflow[0]["step_type"] == "NetworkStats"
        assert workflow[1]["step_type"] == "CapacityEnvelopeAnalysis"
        envelope = workflow[1]
        assert envelope["source_path"] == "(metro[0-9]+/dc[0-9]+)"
        assert envelope["sink_path"] == "(metro[0-9]+/dc[0-9]+)"
        assert envelope["mode"] in {"pairwise", "combine"}
