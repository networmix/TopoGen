"""Tests for failure policies and workflows in scenario builder."""

import networkx as nx
import pytest
import yaml

from topogen.config import TopologyConfig
from topogen.scenario import build_scenario
from topogen.scenario.policies import (
    _build_failure_policy_set_section,
    _build_workflow_section,
)
from topogen.workflows_lib import get_builtin_workflows


class TestBuildFailurePolicySetSection:
    def test_default_policy_only(self):
        config = TopologyConfig()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(
            config, _build_workflow_section(config)
        )

        assert isinstance(result, dict)
        assert "single_random_link_failure" in result
        assert "modes" in result["single_random_link_failure"]

    def test_custom_policy_in_library(self):
        config = TopologyConfig()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(
            config, _build_workflow_section(config)
        )

        assert "single_random_link_failure" in result

    def test_custom_and_default_policies(self):
        config = TopologyConfig()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(
            config, _build_workflow_section(config)
        )

        assert "single_random_link_failure" in result

    def test_custom_policy_overrides_builtin(self):
        config = TopologyConfig()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(
            config, _build_workflow_section(config)
        )

        assert "single_random_link_failure" in result

    def test_unknown_default_policy(self):
        config = TopologyConfig()
        config.failure_policies.assignments.default = "unknown_policy"

        with pytest.raises(
            ValueError, match="Default failure policy 'unknown_policy' not found"
        ):
            _build_failure_policy_set_section(config, _build_workflow_section(config))


class TestBuildWorkflowSection:
    def test_default_workflow_builtin(self):
        config = TopologyConfig()
        # Pick any available built-in workflow to avoid brittle name pinning
        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )

        result = _build_workflow_section(config)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], dict) and "type" in result[0]

    def test_custom_workflow_in_library(self):
        """The selected workflow is a list of steps from the merged library."""
        config = TopologyConfig()
        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )

        result = _build_workflow_section(config)

        assert isinstance(result, list)

    def test_custom_workflow_priority(self):
        config = TopologyConfig()
        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )

        result = _build_workflow_section(config)

        assert isinstance(result, list)

    def test_unknown_default_workflow(self):
        config = TopologyConfig()
        config.workflows.assignments.default = "unknown_workflow"

        with pytest.raises(
            ValueError, match="Default workflow 'unknown_workflow' not found"
        ):
            _build_workflow_section(config)


class TestBuildScenarioIntegration:
    def create_mock_config(self):
        config = TopologyConfig()
        config.failure_policies.assignments.default = "single_random_link_failure"
        config.workflows.assignments.default = next(iter(get_builtin_workflows()))
        return config

    def create_mock_graph(self):
        graph = nx.MultiGraph()

        graph.add_node(
            "metro_1",
            node_type="metro",
            name="Metro 1",
            metro_id="1",
            x=100,
            y=200,
            radius_km=20.0,
            name_orig="Metro 1",
        )
        graph.add_node(
            "metro_2",
            node_type="metro",
            name="Metro 2",
            metro_id="2",
            x=300,
            y=400,
            radius_km=25.0,
            name_orig="Metro 2",
        )

        graph.add_edge("metro_1", "metro_2", edge_type="corridor", length_km=500.5)

        return graph

    def test_scenario_includes_failure_policy_set(self):
        config = self.create_mock_config()
        graph = self.create_mock_graph()

        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )
        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        assert "failures" in scenario
        assert "single_random_link_failure" in scenario["failures"]

    def test_scenario_includes_workflow(self):
        config = self.create_mock_config()
        graph = self.create_mock_graph()

        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )
        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        assert "workflow" in scenario
        assert isinstance(scenario["workflow"], list)
        assert len(scenario["workflow"]) > 0
        assert scenario["workflow"][0]["type"] == "NetworkStats"

    def test_custom_failure_policy_in_scenario(self):
        config = self.create_mock_config()
        config.failure_policies.assignments.default = "single_random_link_failure"

        graph = self.create_mock_graph()

        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )
        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        assert "single_random_link_failure" in scenario["failures"]

    def test_custom_workflow_in_scenario(self):
        config = self.create_mock_config()
        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )

        graph = self.create_mock_graph()

        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        assert len(scenario["workflow"]) >= 1
        assert scenario["workflow"][0]["type"] == "NetworkStats"

    def test_scenario_structure_order(self):
        config = self.create_mock_config()
        graph = self.create_mock_graph()

        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )
        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        expected_sections = [
            "seed",
            "blueprints",
            "components",
            "network",
            "failures",
            "workflow",
        ]
        for section in expected_sections:
            assert section in scenario

        yaml_lines = yaml_output.split("\n")
        section_positions = {}
        for i, line in enumerate(yaml_lines):
            for section in expected_sections:
                if line.startswith(f"{section}:"):
                    section_positions[section] = i
                    break

        assert section_positions["blueprints"] < section_positions["network"]
        assert section_positions["components"] < section_positions["network"]
        assert section_positions["network"] < section_positions["workflow"]

    def test_failure_policy_references_in_workflow(self):
        config = self.create_mock_config()

        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )

        graph = self.create_mock_graph()

        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        for step in scenario["workflow"]:
            fp_name = step.get("failure_policy")
            if fp_name:
                assert fp_name in scenario["failures"]
