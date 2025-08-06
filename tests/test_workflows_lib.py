"""Tests for the workflows library module."""

import pytest

from topogen.workflows_lib import (
    get_builtin_workflow,
    get_builtin_workflows,
    get_fast_workflows,
    get_workflows_by_step_type,
    list_builtin_workflow_names,
    validate_workflow_steps,
)


class TestGetBuiltinWorkflows:
    """Test get_builtin_workflows function."""

    def test_returns_dict(self):
        """Test that the function returns a dictionary."""
        workflows = get_builtin_workflows()
        assert isinstance(workflows, dict)

    def test_contains_expected_workflows(self):
        """Test that standard workflows are included."""
        workflows = get_builtin_workflows()

        expected_workflows = [
            "basic_capacity_analysis",
            "comprehensive_resilience_analysis",
            "fast_network_analysis",
            "network_stats_only",
        ]

        for workflow_name in expected_workflows:
            assert workflow_name in workflows

    def test_returns_deep_copy(self):
        """Test that modifications don't affect the original."""
        workflows1 = get_builtin_workflows()
        workflows2 = get_builtin_workflows()

        # Modify one copy
        workflows1["basic_capacity_analysis"][0]["test"] = True

        # Original should be unchanged
        assert "test" not in workflows2["basic_capacity_analysis"][0]

    def test_workflow_structure(self):
        """Test that workflows have the expected structure."""
        workflows = get_builtin_workflows()

        for name, steps in workflows.items():
            assert isinstance(steps, list), f"Workflow {name} should be a list"
            assert len(steps) > 0, f"Workflow {name} should not be empty"

            for i, step in enumerate(steps):
                assert isinstance(step, dict), (
                    f"Workflow {name} step {i} should be a dict"
                )
                assert "step_type" in step, (
                    f"Workflow {name} step {i} should have 'step_type'"
                )
                assert isinstance(step["step_type"], str)


class TestGetBuiltinWorkflow:
    """Test get_builtin_workflow function."""

    def test_valid_workflow_name(self):
        """Test retrieving a valid workflow."""
        workflow = get_builtin_workflow("basic_capacity_analysis")
        assert isinstance(workflow, list)
        assert len(workflow) > 0
        assert workflow[0]["step_type"] == "NetworkStats"

    def test_invalid_workflow_name(self):
        """Test that unknown workflow names raise KeyError."""
        with pytest.raises(KeyError, match="Unknown built-in workflow 'nonexistent'"):
            get_builtin_workflow("nonexistent")

    def test_returns_deep_copy(self):
        """Test that modifications don't affect the original."""
        workflow1 = get_builtin_workflow("basic_capacity_analysis")
        workflow2 = get_builtin_workflow("basic_capacity_analysis")

        # Modify one copy
        workflow1[0]["test"] = True

        # Original should be unchanged
        assert "test" not in workflow2[0]


class TestListBuiltinWorkflowNames:
    """Test list_builtin_workflow_names function."""

    def test_returns_list(self):
        """Test that the function returns a list."""
        names = list_builtin_workflow_names()
        assert isinstance(names, list)

    def test_contains_expected_names(self):
        """Test that expected workflow names are included."""
        names = list_builtin_workflow_names()

        expected_names = [
            "basic_capacity_analysis",
            "comprehensive_resilience_analysis",
            "fast_network_analysis",
        ]

        for name in expected_names:
            assert name in names

    def test_sorted_output(self):
        """Test that names are returned in sorted order."""
        names = list_builtin_workflow_names()
        assert names == sorted(names)


class TestGetWorkflowsByStepType:
    """Test get_workflows_by_step_type function."""

    def test_network_stats_workflows(self):
        """Test filtering for workflows with NetworkStats steps."""
        stats_workflows = get_workflows_by_step_type("NetworkStats")

        # Should include workflows that have NetworkStats
        assert len(stats_workflows) > 0

        # Verify the workflows actually contain NetworkStats steps
        for name, steps in stats_workflows.items():
            has_stats_step = any(
                step.get("step_type") == "NetworkStats" for step in steps
            )
            assert has_stats_step, f"Workflow {name} should contain NetworkStats step"

    def test_capacity_envelope_workflows(self):
        """Test filtering for workflows with CapacityEnvelopeAnalysis steps."""
        envelope_workflows = get_workflows_by_step_type("CapacityEnvelopeAnalysis")

        # Should include workflows that have CapacityEnvelopeAnalysis
        assert len(envelope_workflows) > 0

        # Verify the workflows actually contain CapacityEnvelopeAnalysis steps
        for name, steps in envelope_workflows.items():
            has_envelope_step = any(
                step.get("step_type") == "CapacityEnvelopeAnalysis" for step in steps
            )
            assert has_envelope_step, (
                f"Workflow {name} should contain CapacityEnvelopeAnalysis step"
            )

    def test_nonexistent_step_type(self):
        """Test filtering for non-existent step type."""
        workflows = get_workflows_by_step_type("NonexistentStepType")
        assert isinstance(workflows, dict)
        assert len(workflows) == 0

    def test_returns_deep_copy(self):
        """Test that modifications don't affect the original."""
        workflows1 = get_workflows_by_step_type("NetworkStats")
        workflows2 = get_workflows_by_step_type("NetworkStats")

        if workflows1:
            first_workflow_name = list(workflows1.keys())[0]
            workflows1[first_workflow_name][0]["test"] = True

            # Original should be unchanged
            assert "test" not in workflows2[first_workflow_name][0]


class TestGetFastWorkflows:
    """Test get_fast_workflows function."""

    def test_returns_dict(self):
        """Test that the function returns a dictionary."""
        fast_workflows = get_fast_workflows()
        assert isinstance(fast_workflows, dict)

    def test_fast_characteristics(self):
        """Test that returned workflows have fast characteristics."""
        fast_workflows = get_fast_workflows()

        for name, steps in fast_workflows.items():
            has_fast_characteristics = False

            # Check for fast characteristics
            for step in steps:
                if step.get("step_type") == "CapacityEnvelopeAnalysis":
                    iterations = step.get("iterations", 1000)
                    shortest_path = step.get("shortest_path", False)
                    mode = step.get("mode", "pairwise")

                    if iterations <= 100 or shortest_path or mode == "combine":
                        has_fast_characteristics = True
                        break

            # Or if workflow only has NetworkStats (no heavy analysis)
            analysis_steps = [
                s for s in steps if s.get("step_type") == "CapacityEnvelopeAnalysis"
            ]
            if not analysis_steps:
                has_fast_characteristics = True

            assert has_fast_characteristics, (
                f"Workflow {name} should have fast characteristics"
            )

    def test_includes_fast_network_analysis(self):
        """Test that fast_network_analysis is included."""
        fast_workflows = get_fast_workflows()
        assert "fast_network_analysis" in fast_workflows

    def test_includes_network_stats_only(self):
        """Test that network_stats_only is included."""
        fast_workflows = get_fast_workflows()
        assert "network_stats_only" in fast_workflows

    def test_returns_deep_copy(self):
        """Test that modifications don't affect the original."""
        workflows1 = get_fast_workflows()
        workflows2 = get_fast_workflows()

        if workflows1:
            first_workflow_name = list(workflows1.keys())[0]
            workflows1[first_workflow_name][0]["test"] = True

            # Original should be unchanged
            assert "test" not in workflows2[first_workflow_name][0]


class TestValidateWorkflowSteps:
    """Test validate_workflow_steps function."""

    def test_valid_workflow(self):
        """Test validation of a valid workflow."""
        steps = [
            {"step_type": "NetworkStats", "name": "stats"},
            {
                "step_type": "CapacityEnvelopeAnalysis",
                "source_path": "^core",
                "sink_path": "^core",
                "mode": "pairwise",
            },
        ]

        # Should not raise an exception
        validate_workflow_steps(steps)

    def test_empty_workflow(self):
        """Test validation of empty workflow."""
        with pytest.raises(ValueError, match="Workflow cannot be empty"):
            validate_workflow_steps([])

    def test_non_list_workflow(self):
        """Test validation of non-list workflow."""
        with pytest.raises(ValueError, match="Workflow must be a list of steps"):
            validate_workflow_steps({"not": "a list"})

    def test_non_dict_step(self):
        """Test validation of non-dict step."""
        steps = [{"step_type": "NetworkStats"}, "invalid step"]

        with pytest.raises(ValueError, match="Step 1 must be a dictionary"):
            validate_workflow_steps(steps)

    def test_missing_step_type(self):
        """Test validation of step without step_type."""
        steps = [
            {"name": "stats"},  # Missing step_type
        ]

        with pytest.raises(
            ValueError, match="Step 0 missing required 'step_type' field"
        ):
            validate_workflow_steps(steps)

    def test_non_string_step_type(self):
        """Test validation of non-string step_type."""
        steps = [
            {"step_type": 123},  # step_type should be string
        ]

        with pytest.raises(ValueError, match="Step 0 'step_type' must be a string"):
            validate_workflow_steps(steps)

    def test_capacity_envelope_missing_fields(self):
        """Test validation of CapacityEnvelopeAnalysis with missing fields."""
        steps = [
            {
                "step_type": "CapacityEnvelopeAnalysis",
                "source_path": "^core",
                # Missing sink_path and mode
            }
        ]

        with pytest.raises(
            ValueError,
            match="CapacityEnvelopeAnalysis step 0 missing required field 'sink_path'",
        ):
            validate_workflow_steps(steps)


class TestBuiltinWorkflowContent:
    """Test the content and structure of specific built-in workflows."""

    def test_basic_capacity_analysis_structure(self):
        """Test basic_capacity_analysis workflow structure."""
        workflow = get_builtin_workflow("basic_capacity_analysis")

        assert len(workflow) == 2
        assert workflow[0]["step_type"] == "NetworkStats"
        assert workflow[1]["step_type"] == "CapacityEnvelopeAnalysis"

        # Check capacity envelope step has required fields
        envelope_step = workflow[1]
        assert "source_path" in envelope_step
        assert "sink_path" in envelope_step
        assert "mode" in envelope_step
        assert envelope_step["baseline"] is True

    def test_comprehensive_resilience_analysis_structure(self):
        """Test comprehensive_resilience_analysis workflow structure."""
        workflow = get_builtin_workflow("comprehensive_resilience_analysis")

        # Should have multiple analysis steps
        assert len(workflow) >= 3
        assert workflow[0]["step_type"] == "NetworkStats"

        # Should have multiple CapacityEnvelopeAnalysis steps
        envelope_steps = [
            s for s in workflow if s["step_type"] == "CapacityEnvelopeAnalysis"
        ]
        assert len(envelope_steps) >= 2

        # Should have baseline and failure analysis
        baseline_steps = [s for s in envelope_steps if s.get("baseline")]
        failure_steps = [s for s in envelope_steps if not s.get("baseline")]
        assert len(baseline_steps) >= 1
        assert len(failure_steps) >= 1

    def test_fast_network_analysis_structure(self):
        """Test fast_network_analysis workflow structure."""
        workflow = get_builtin_workflow("fast_network_analysis")

        assert len(workflow) == 2
        assert workflow[0]["step_type"] == "NetworkStats"
        assert workflow[1]["step_type"] == "CapacityEnvelopeAnalysis"

        # Should have fast characteristics
        envelope_step = workflow[1]
        iterations = envelope_step.get("iterations", 1000)
        shortest_path = envelope_step.get("shortest_path", False)
        mode = envelope_step.get("mode", "pairwise")

        assert iterations <= 100 or shortest_path or mode == "combine"

    def test_network_stats_only_structure(self):
        """Test network_stats_only workflow structure."""
        workflow = get_builtin_workflow("network_stats_only")

        assert len(workflow) == 1
        assert workflow[0]["step_type"] == "NetworkStats"

    def test_all_workflows_are_valid(self):
        """Test that all built-in workflows pass validation."""
        workflows = get_builtin_workflows()

        for name, steps in workflows.items():
            try:
                validate_workflow_steps(steps)
            except ValueError as e:
                pytest.fail(f"Built-in workflow '{name}' failed validation: {e}")

    def test_workflow_step_names_are_unique(self):
        """Test that steps within workflows have unique names when specified."""
        workflows = get_builtin_workflows()

        for workflow_name, steps in workflows.items():
            step_names = []
            for step in steps:
                if "name" in step:
                    step_names.append(step["name"])

            # Check for duplicates
            if step_names:
                assert len(step_names) == len(set(step_names)), (
                    f"Workflow '{workflow_name}' has duplicate step names: {step_names}"
                )
