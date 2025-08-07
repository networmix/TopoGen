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


class TestBuildFailurePolicySetSection:
    """Test _build_failure_policy_set_section function."""

    def test_default_policy_only(self):
        """Test with default policy only."""
        config = Mock(spec=TopologyConfig)
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.library = {}
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(config)

        assert isinstance(result, dict)
        assert "single_random_link_failure" in result
        assert "rules" in result["single_random_link_failure"]

    def test_custom_policy_in_library(self):
        """Test with custom policy in library."""
        config = Mock(spec=TopologyConfig)
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.library = {
            "custom_policy": {
                "attrs": {"description": "Custom test policy"},
                "rules": [{"entity_scope": "link", "rule_type": "all"}],
            }
        }
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "custom_policy"

        result = _build_failure_policy_set_section(config)

        assert "custom_policy" in result
        assert result["custom_policy"]["attrs"]["description"] == "Custom test policy"
        assert len(result["custom_policy"]["rules"]) == 1

    def test_custom_and_default_policies(self):
        """Test with both custom and default policies."""
        config = Mock(spec=TopologyConfig)
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.library = {
            "custom_policy": {
                "rules": [{"entity_scope": "node", "rule_type": "choice", "count": 2}]
            }
        }
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(config)

        # Should have both custom and default policies
        assert "custom_policy" in result
        assert "single_random_link_failure" in result
        assert result["custom_policy"]["rules"][0]["count"] == 2

    def test_custom_policy_overrides_builtin(self):
        """Test that custom policy overrides built-in with same name."""
        config = Mock(spec=TopologyConfig)
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.library = {
            "single_random_link_failure": {
                "attrs": {"description": "Custom override"},
                "rules": [{"entity_scope": "link", "rule_type": "choice", "count": 5}],
            }
        }
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "single_random_link_failure"

        result = _build_failure_policy_set_section(config)

        assert (
            result["single_random_link_failure"]["attrs"]["description"]
            == "Custom override"
        )
        assert result["single_random_link_failure"]["rules"][0]["count"] == 5

    def test_unknown_default_policy(self):
        """Test error when default policy is unknown."""
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
    """Test _build_workflow_section function."""

    def test_default_workflow_builtin(self):
        """Test with built-in default workflow."""
        config = Mock(spec=TopologyConfig)
        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.library = {}
        config.workflows.assignments = Mock()
        config.workflows.assignments.default = "basic_capacity_analysis"

        result = _build_workflow_section(config)

        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["step_type"] == "NetworkStats"

    def test_custom_workflow_in_library(self):
        """Test with custom workflow in library."""
        custom_workflow = [
            {"step_type": "NetworkStats", "name": "custom_stats"},
            {"step_type": "CapacityProbe", "name": "custom_probe"},
        ]

        config = Mock(spec=TopologyConfig)
        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.library = {"custom_workflow": custom_workflow}
        config.workflows.assignments = Mock()
        config.workflows.assignments.default = "custom_workflow"

        result = _build_workflow_section(config)

        assert result == custom_workflow
        assert result[1]["step_type"] == "CapacityProbe"

    def test_custom_workflow_priority(self):
        """Test that custom workflow takes priority over built-in with same name."""
        custom_workflow = [{"step_type": "NetworkStats", "name": "overridden_stats"}]

        config = Mock(spec=TopologyConfig)
        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.library = {"basic_capacity_analysis": custom_workflow}
        config.workflows.assignments = Mock()
        config.workflows.assignments.default = "basic_capacity_analysis"

        result = _build_workflow_section(config)

        assert result == custom_workflow
        assert result[0]["name"] == "overridden_stats"

    def test_unknown_default_workflow(self):
        """Test error when default workflow is unknown."""
        config = Mock(spec=TopologyConfig)
        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.library = {}
        config.workflows.assignments = Mock()
        config.workflows.assignments.default = "unknown_workflow"

        with pytest.raises(
            ValueError, match="Default workflow 'unknown_workflow' not found"
        ):
            _build_workflow_section(config)


class TestBuildScenarioIntegration:
    """Test integration of failure policies and workflows in build_scenario."""

    def create_mock_config(self):
        """Create a mock configuration with default values."""
        config = Mock(spec=TopologyConfig)

        # Build configuration
        config.build = Mock()
        config.build.build_defaults = Mock()
        config.build.build_defaults.sites_per_metro = 2
        config.build.build_defaults.site_blueprint = "SingleRouter"
        config.build.build_defaults.dc_regions_per_metro = 2
        config.build.build_defaults.dc_region_blueprint = "DCRegion"

        # Link parameter defaults
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

        # Components configuration
        config.components = Mock()
        config.components.library = {}
        config.components.assignments = Mock()
        config.components.assignments.spine = Mock(hw_component="", optics="")
        config.components.assignments.leaf = Mock(hw_component="", optics="")
        config.components.assignments.core = Mock(hw_component="", optics="")
        config.components.assignments.dc = Mock(hw_component="", optics="")
        config.components.assignments.blueprint_overrides = {}

        # Failure policies configuration
        config.failure_policies = Mock(spec=FailurePoliciesConfig)
        config.failure_policies.library = {}
        config.failure_policies.assignments = Mock()
        config.failure_policies.assignments.default = "single_random_link_failure"

        # Workflows configuration
        config.workflows = Mock(spec=WorkflowsConfig)
        config.workflows.library = {}
        config.workflows.assignments = Mock()
        config.workflows.assignments.default = "basic_capacity_analysis"

        # Corridors configuration
        config.corridors = Mock()
        config.corridors.risk_groups = Mock()
        config.corridors.risk_groups.enabled = True
        config.corridors.risk_groups.group_prefix = "corridor_risk"
        config.corridors.risk_groups.exclude_metro_radius_shared = True

        return config

    def create_mock_graph(self):
        """Create a mock metro graph."""
        graph = nx.Graph()

        # Add metro nodes
        graph.add_node(
            "metro_1", node_type="metro", name="Metro 1", metro_id="1", x=100, y=200
        )
        graph.add_node(
            "metro_2", node_type="metro", name="Metro 2", metro_id="2", x=300, y=400
        )

        # Add corridor edge
        graph.add_edge("metro_1", "metro_2", edge_type="corridor", length_km=500.5)

        return graph

    def test_scenario_includes_failure_policy_set(self):
        """Test that generated scenario includes failure_policy_set section."""
        config = self.create_mock_config()
        graph = self.create_mock_graph()

        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        assert "failure_policy_set" in scenario
        assert "single_random_link_failure" in scenario["failure_policy_set"]

    def test_scenario_includes_workflow(self):
        """Test that generated scenario includes workflow section."""
        config = self.create_mock_config()
        graph = self.create_mock_graph()

        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        assert "workflow" in scenario
        assert isinstance(scenario["workflow"], list)
        assert len(scenario["workflow"]) > 0
        assert scenario["workflow"][0]["step_type"] == "NetworkStats"

    def test_custom_failure_policy_in_scenario(self):
        """Test that custom failure policies appear in the scenario."""
        config = self.create_mock_config()
        config.failure_policies.library = {
            "custom_policy": {
                "attrs": {"description": "Test policy"},
                "rules": [{"entity_scope": "node", "rule_type": "all"}],
            }
        }
        config.failure_policies.assignments.default = "custom_policy"

        graph = self.create_mock_graph()

        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        assert "custom_policy" in scenario["failure_policy_set"]
        assert (
            scenario["failure_policy_set"]["custom_policy"]["attrs"]["description"]
            == "Test policy"
        )

    def test_custom_workflow_in_scenario(self):
        """Test that custom workflows appear in the scenario."""
        custom_workflow = [{"step_type": "NetworkStats", "name": "test_stats"}]

        config = self.create_mock_config()
        config.workflows.library = {"test_workflow": custom_workflow}
        config.workflows.assignments.default = "test_workflow"

        graph = self.create_mock_graph()

        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        assert len(scenario["workflow"]) == 1
        assert scenario["workflow"][0]["name"] == "test_stats"

    def test_scenario_structure_order(self):
        """Test that scenario sections are in the expected order."""
        config = self.create_mock_config()
        graph = self.create_mock_graph()

        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        # Check that all expected sections are present
        expected_sections = [
            "blueprints",
            "components",
            "network",
            "failure_policy_set",
            "workflow",
        ]
        for section in expected_sections:
            assert section in scenario

        # Check YAML ordering (approximately - YAML dict ordering may vary)
        yaml_lines = yaml_output.split("\n")
        section_positions = {}
        for i, line in enumerate(yaml_lines):
            for section in expected_sections:
                if line.startswith(f"{section}:"):
                    section_positions[section] = i
                    break

        # Verify basic ordering constraints
        assert section_positions["blueprints"] < section_positions["network"]
        assert section_positions["components"] < section_positions["network"]
        assert section_positions["network"] < section_positions["workflow"]

    def test_failure_policy_references_in_workflow(self):
        """Test that workflow steps can reference failure policies."""
        config = self.create_mock_config()
        config.failure_policies.library = {
            "test_policy": {
                "rules": [{"entity_scope": "link", "rule_type": "choice", "count": 2}]
            }
        }

        # Workflow that references the failure policy
        custom_workflow = [
            {"step_type": "NetworkStats"},
            {
                "step_type": "CapacityEnvelopeAnalysis",
                "name": "test_analysis",
                "source_path": "^core",
                "sink_path": "^core",
                "mode": "pairwise",
                "failure_policy": "test_policy",
            },
        ]
        config.workflows.library = {"test_workflow": custom_workflow}
        config.workflows.assignments.default = "test_workflow"

        graph = self.create_mock_graph()

        yaml_output = build_scenario(graph, config)
        scenario = yaml.safe_load(yaml_output)

        # Both sections should be present and consistent
        assert "test_policy" in scenario["failure_policy_set"]

        envelope_step = None
        for step in scenario["workflow"]:
            if step.get("step_type") == "CapacityEnvelopeAnalysis":
                envelope_step = step
                break

        assert envelope_step is not None
        assert envelope_step["failure_policy"] == "test_policy"
