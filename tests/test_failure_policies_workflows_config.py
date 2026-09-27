"""Tests for failure policies and workflows configuration parsing."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from topogen.config import (
    FailurePoliciesConfig,
    FailurePolicyAssignments,
    TopologyConfig,
    WorkflowAssignments,
    WorkflowsConfig,
)
from topogen.scenario.policies import _build_workflow_section


class TestFailurePolicyAssignments:
    def test_default_values(self):
        assignments = FailurePolicyAssignments()
        assert assignments.default == "single_random_link_failure"
        assert assignments.scenario_overrides == {}

    def test_custom_values(self):
        overrides = {"test_scenario": {"failure_policy": "custom_policy"}}
        assignments = FailurePolicyAssignments(
            default="dual_random_link_failure", scenario_overrides=overrides
        )
        assert assignments.default == "dual_random_link_failure"
        assert assignments.scenario_overrides == overrides


class TestFailurePoliciesConfig:
    def test_default_values(self):
        config = FailurePoliciesConfig()
        assert isinstance(config.assignments, FailurePolicyAssignments)
        assert config.assignments.default == "single_random_link_failure"

    def test_custom_values(self):
        assignments = FailurePolicyAssignments(default="custom_policy")
        config = FailurePoliciesConfig(assignments=assignments)
        assert config.assignments == assignments


class TestWorkflowAssignments:
    def test_default_values(self):
        assignments = WorkflowAssignments()
        assert assignments.default == "design_analysis_brief"
        assert assignments.scenario_overrides == {}

    def test_custom_values(self):
        overrides = {"test_scenario": {"workflow": "custom_workflow"}}
        assignments = WorkflowAssignments(
            default="fast_network_analysis", scenario_overrides=overrides
        )
        assert assignments.default == "fast_network_analysis"
        assert assignments.scenario_overrides == overrides


class TestWorkflowsConfig:
    def test_default_values(self):
        config = WorkflowsConfig()
        assert isinstance(config.assignments, WorkflowAssignments)
        assert config.assignments.default == "design_analysis_brief"

    def test_custom_values(self):
        assignments = WorkflowAssignments(default="custom_workflow")
        config = WorkflowsConfig(assignments=assignments)
        assert config.assignments == assignments


class TestConfigurationParsing:
    def create_temp_config(self, config_dict: Dict[str, Any]) -> Path:
        """Create a temporary configuration file."""
        import yaml

        # Add all required sections if not present
        if "data_sources" not in config_dict:
            config_dict["data_sources"] = {
                "uac_polygons": "data/tl_2020_us_uac20.zip",
                "tiger_roads": "data/tl_2024_us_primaryroads.zip",
                "conus_boundary": "data/cb_2024_us_state_500k.zip",
            }

        if "projection" not in config_dict:
            config_dict["projection"] = {"target_crs": "EPSG:5070"}

        if "clustering" not in config_dict:
            config_dict["clustering"] = {
                "metro_clusters": 30,
                "max_uac_radius_km": 100.0,
                "export_clusters": False,
                "export_integrated_graph": False,
                "coordinate_precision": 1,
                "area_precision": 2,
            }

        if "highway_processing" not in config_dict:
            config_dict["highway_processing"] = {
                "min_edge_length_km": 0.05,
                "snap_precision_m": 10.0,
                "highway_classes": ["S1100", "S1200"],
                "min_cycle_nodes": 3,
                "filter_largest_component": True,
                "validation_sample_size": 5,
            }

        if "corridors" not in config_dict:
            config_dict["corridors"] = {
                "k_paths": 1,
                "k_nearest": 3,
                "max_edge_km": 600.0,
                "max_corridor_distance_km": 1000.0,
                "risk_groups": {
                    "enabled": True,
                    "group_prefix": "corridor_risk",
                    "exclude_metro_radius_shared": True,
                },
            }

        if "validation" not in config_dict:
            config_dict["validation"] = {
                "max_metro_highway_distance_km": 10.0,
                "require_connected": True,
                "max_degree_threshold": 1000,
                "high_degree_warning": 20,
                "min_largest_component_fraction": 0.5,
            }

        if "output" not in config_dict:
            config_dict["output"] = {
                "scenario_metadata": {
                    "title": "Continental US Backbone Topology",
                    "description": "Generated backbone topology based on population density and highway infrastructure",
                    "version": "1.0",
                },
                "formatting": {
                    "json_indent": 2,
                },
            }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(config_dict, f)
            return Path(f.name)

    def test_empty_failure_policies_section(self):
        config_dict = {"failure_policies": {"assignments": {}}}

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert isinstance(config.failure_policies, FailurePoliciesConfig)
            assert (
                config.failure_policies.assignments.default
                == "single_random_link_failure"
            )
        finally:
            config_path.unlink()

    def test_empty_workflows_section(self):
        config_dict = {"workflows": {"assignments": {}}}

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert isinstance(config.workflows, WorkflowsConfig)
            assert config.workflows.assignments.default == "design_analysis_brief"
            assert _build_workflow_section(config)[0]["type"] == "NetworkStats"
        finally:
            config_path.unlink()

    def test_failure_policies_assignments_parsing(self):
        config_dict = {
            "failure_policies": {
                "assignments": {
                    "default": "dual_random_link_failure",
                    "scenario_overrides": {
                        "test_scenario": {"failure_policy": "custom_policy"}
                    },
                }
            }
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert (
                config.failure_policies.assignments.default
                == "dual_random_link_failure"
            )
            assert (
                "test_scenario"
                in config.failure_policies.assignments.scenario_overrides
            )
            assert (
                config.failure_policies.assignments.scenario_overrides["test_scenario"][
                    "failure_policy"
                ]
                == "custom_policy"
            )
        finally:
            config_path.unlink()

    def test_workflows_assignments_parsing(self):
        config_dict = {
            "workflows": {
                "assignments": {
                    "default": "fast_network_analysis",
                    "scenario_overrides": {
                        "test_scenario": {"workflow": "custom_workflow"}
                    },
                }
            }
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert config.workflows.assignments.default == "fast_network_analysis"
            assert "test_scenario" in config.workflows.assignments.scenario_overrides
            assert (
                config.workflows.assignments.scenario_overrides["test_scenario"][
                    "workflow"
                ]
                == "custom_workflow"
            )
        finally:
            config_path.unlink()

    def test_none_values_handling(self):
        config_dict = {
            "failure_policies": {"assignments": {"scenario_overrides": None}},
            "workflows": {"assignments": {"scenario_overrides": None}},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert config.failure_policies.assignments.scenario_overrides == {}
            assert config.workflows.assignments.scenario_overrides == {}
        finally:
            config_path.unlink()

    def test_missing_sections(self):
        config_dict = {}

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert isinstance(config.failure_policies, FailurePoliciesConfig)
            assert (
                config.failure_policies.assignments.default
                == "single_random_link_failure"
            )
            assert isinstance(config.workflows, WorkflowsConfig)
            assert config.workflows.assignments.default == "design_analysis_brief"
            assert _build_workflow_section(config)[0]["type"] == "NetworkStats"
        finally:
            config_path.unlink()

    def test_invalid_failure_policies_type(self):
        config_dict = {"failure_policies": "not a dict"}

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(
                ValueError,
                match="'failure_policies' configuration section must be a dictionary",
            ):
                TopologyConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_invalid_workflows_type(self):
        config_dict = {"workflows": "not a dict"}

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(
                ValueError,
                match="'workflows' configuration section must be a dictionary",
            ):
                TopologyConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_invalid_library_type(self):
        config_dict = {"failure_policies": {"assignments": "not a dict"}}

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(
                ValueError, match="'failure_policies.assignments' must be a dictionary"
            ):
                TopologyConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_invalid_assignments_type(self):
        config_dict = {"workflows": {"assignments": "not a dict"}}

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(
                ValueError, match="'workflows.assignments' must be a dictionary"
            ):
                TopologyConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_complete_configuration(self):
        config_dict = {
            "failure_policies": {"assignments": {"default": "custom_failure"}},
            "workflows": {"assignments": {"default": "custom_workflow"}},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert config.failure_policies.assignments.default == "custom_failure"
            assert config.workflows.assignments.default == "custom_workflow"
        finally:
            config_path.unlink()
