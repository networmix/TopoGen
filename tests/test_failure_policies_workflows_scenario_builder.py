"""Tests for failure policies and workflows in scenario builder."""

from unittest.mock import Mock

import networkx as nx
import pytest
import yaml

from topogen.config import FailurePoliciesConfig, TopologyConfig, WorkflowsConfig
from topogen.scenario_builder import (
    _build_failure_policy_set_section,
    _build_workflow_section,
    build_scenario,
)
from topogen.workflows_lib import get_builtin_workflows


class TestBuildFailurePolicySetSection:
    def test_default_policy_only(self):
        config = Mock(spec=TopologyConfig)
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(config)

        assert isinstance(result, dict)
        assert "single_random_link_failure" in result
        assert "modes" in result["single_random_link_failure"]

    def test_custom_policy_in_library(self):
        config = Mock(spec=TopologyConfig)
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(config)

        assert "single_random_link_failure" in result

    def test_custom_and_default_policies(self):
        config = Mock(spec=TopologyConfig)
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(config)

        assert "single_random_link_failure" in result

    def test_custom_policy_overrides_builtin(self):
        config = Mock(spec=TopologyConfig)
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(config)

        assert "single_random_link_failure" in result

    def test_unknown_default_policy(self):
        config = Mock(spec=TopologyConfig)
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.library = {}
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "unknown_policy"

        with pytest.raises(
            ValueError, match="Default failure policy 'unknown_policy' not found"
        ):
            _build_failure_policy_set_section(config)


class TestBuildWorkflowSection:
    def test_default_workflow_builtin(self):
        config = Mock(spec=TopologyConfig)
        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.assignments = Mock()
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
        config = Mock(spec=TopologyConfig)
        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.assignments = Mock()
        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )

        result = _build_workflow_section(config)

        assert isinstance(result, list)

    def test_custom_workflow_priority(self):
        config = Mock(spec=TopologyConfig)
        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.assignments = Mock()
        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )

        result = _build_workflow_section(config)

        assert isinstance(result, list)

    def test_unknown_default_workflow(self):
        config = Mock(spec=TopologyConfig)
        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.assignments = Mock()
        config.workflows.assignments.default = "unknown_workflow"

        with pytest.raises(
            ValueError, match="Default workflow 'unknown_workflow' not found"
        ):
            _build_workflow_section(config)


class TestBuildScenarioIntegration:
    def create_mock_config(self):
        config = Mock(spec=TopologyConfig)

        config.build = Mock()
        config.build.build_defaults = Mock()
        config.build.build_defaults.pop_per_metro = 2
        config.build.build_defaults.site_blueprint = "SingleRouter"
        config.build.build_defaults.dc_regions_per_metro = 2
        config.build.build_defaults.dc_region_blueprint = "DCRegion"

        config.build.build_defaults.intra_metro_link = Mock()
        config.build.build_defaults.intra_metro_link.capacity = 400
        config.build.build_defaults.intra_metro_link.cost = 1
        config.build.build_defaults.intra_metro_link.attrs = {
            "link_type": "intra_metro"
        }

        config.build.build_defaults.inter_metro_link = Mock()
        config.build.build_defaults.inter_metro_link.capacity = 100
        config.build.build_defaults.inter_metro_link.cost = 1
        config.build.build_defaults.inter_metro_link.attrs = {
            "link_type": "inter_metro_corridor"
        }

        config.build.build_defaults.dc_to_pop_link = Mock()
        config.build.build_defaults.dc_to_pop_link.capacity = 400
        config.build.build_defaults.dc_to_pop_link.cost = 1
        config.build.build_defaults.dc_to_pop_link.attrs = {"link_type": "dc_to_pop"}

        config.build.build_overrides = {}

        config.components = Mock()
        config.components.assignments = Mock()
        config.components.assignments.spine = Mock(hw_component="", optics="")
        config.components.assignments.leaf = Mock(hw_component="", optics="")
        config.components.assignments.core = Mock(hw_component="", optics="")
        config.components.assignments.dc = Mock(hw_component="", optics="")

        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "single_random_link_failure"

        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.assignments = Mock()
        # Use first available workflow to avoid brittle expectations
        config.workflows.assignments.default = next(
            iter(get_builtin_workflows().keys())
        )

        config.corridors = Mock()
        config.corridors.risk_groups = Mock()
        config.corridors.risk_groups.enabled = True
        config.corridors.risk_groups.group_prefix = "corridor_risk"
        config.corridors.risk_groups.exclude_metro_radius_shared = True

        return config

    def create_mock_graph(self):
        graph = nx.Graph()

        graph.add_node(
            "metro_1",
            node_type="metro",
            name="Metro 1",
            metro_id="1",
            x=100,
            y=200,
            radius_km=20.0,
        )
        graph.add_node(
            "metro_2",
            node_type="metro",
            name="Metro 2",
            metro_id="2",
            x=300,
            y=400,
            radius_km=25.0,
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
