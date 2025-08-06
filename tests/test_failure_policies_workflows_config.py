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


class TestFailurePolicyAssignments:
    """Test FailurePolicyAssignments dataclass."""

    def test_default_values(self):
        """Test default values."""
        assignments = FailurePolicyAssignments()
        assert assignments.default == "single_random_link_failure"
        assert assignments.scenario_overrides == {}

    def test_custom_values(self):
        """Test custom values."""
        overrides = {"test_scenario": {"failure_policy": "custom_policy"}}
        assignments = FailurePolicyAssignments(
            default="dual_random_link_failure", scenario_overrides=overrides
        )
        assert assignments.default == "dual_random_link_failure"
        assert assignments.scenario_overrides == overrides


class TestFailurePoliciesConfig:
    """Test FailurePoliciesConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = FailurePoliciesConfig()
        assert config.library == {}
        assert isinstance(config.assignments, FailurePolicyAssignments)
        assert config.assignments.default == "single_random_link_failure"

    def test_custom_values(self):
        """Test custom values."""
        library = {"custom_policy": {"rules": []}}
        assignments = FailurePolicyAssignments(default="custom_policy")

        config = FailurePoliciesConfig(library=library, assignments=assignments)
        assert config.library == library
        assert config.assignments == assignments


class TestWorkflowAssignments:
    """Test WorkflowAssignments dataclass."""

    def test_default_values(self):
        """Test default values."""
        assignments = WorkflowAssignments()
        assert assignments.default == "basic_capacity_analysis"
        assert assignments.scenario_overrides == {}

    def test_custom_values(self):
        """Test custom values."""
        overrides = {"test_scenario": {"workflow": "custom_workflow"}}
        assignments = WorkflowAssignments(
            default="fast_network_analysis", scenario_overrides=overrides
        )
        assert assignments.default == "fast_network_analysis"
        assert assignments.scenario_overrides == overrides


class TestWorkflowsConfig:
    """Test WorkflowsConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = WorkflowsConfig()
        assert config.library == {}
        assert isinstance(config.assignments, WorkflowAssignments)
        assert config.assignments.default == "basic_capacity_analysis"

    def test_custom_values(self):
        """Test custom values."""
        library = {"custom_workflow": [{"step_type": "NetworkStats"}]}
        assignments = WorkflowAssignments(default="custom_workflow")

        config = WorkflowsConfig(library=library, assignments=assignments)
        assert config.library == library
        assert config.assignments == assignments


class TestConfigurationParsing:
    """Test parsing of failure policies and workflows in configuration files."""

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
                "pop_blueprint": {
                    "sites_per_metro": 4,
                    "cores_per_pop": 2,
                    "internal_pattern": "mesh",
                },
                "scenario_metadata": {
                    "title": "Continental US Backbone Topology",
                    "description": "Generated backbone topology based on population density and highway infrastructure",
                    "version": "1.0",
                },
                "formatting": {
                    "json_indent": 2,
                    "distance_conversion_factor": 1000,
                    "area_conversion_factor": 1000000,
                },
            }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(config_dict, f)
            return Path(f.name)

    def test_empty_failure_policies_section(self):
        """Test parsing empty failure_policies section."""
        config_dict = {"failure_policies": {}}

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert isinstance(config.failure_policies, FailurePoliciesConfig)
            assert config.failure_policies.library == {}
            assert (
                config.failure_policies.assignments.default
                == "single_random_link_failure"
            )
        finally:
            config_path.unlink()

    def test_empty_workflows_section(self):
        """Test parsing empty workflows section."""
        config_dict = {"workflows": {}}

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert isinstance(config.workflows, WorkflowsConfig)
            assert config.workflows.library == {}
            assert config.workflows.assignments.default == "basic_capacity_analysis"
        finally:
            config_path.unlink()

    def test_failure_policies_library_parsing(self):
        """Test parsing failure_policies.library section."""
        config_dict = {
            "failure_policies": {
                "library": {
                    "custom_policy": {
                        "attrs": {"description": "Custom policy"},
                        "rules": [
                            {"entity_scope": "link", "rule_type": "choice", "count": 3}
                        ],
                    }
                }
            }
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert "custom_policy" in config.failure_policies.library
            policy = config.failure_policies.library["custom_policy"]
            assert policy["attrs"]["description"] == "Custom policy"
            assert len(policy["rules"]) == 1
            assert policy["rules"][0]["count"] == 3
        finally:
            config_path.unlink()

    def test_failure_policies_assignments_parsing(self):
        """Test parsing failure_policies.assignments section."""
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

    def test_workflows_library_parsing(self):
        """Test parsing workflows.library section."""
        config_dict = {
            "workflows": {
                "library": {
                    "custom_workflow": [
                        {"step_type": "NetworkStats", "name": "custom_stats"},
                        {
                            "step_type": "CapacityEnvelopeAnalysis",
                            "name": "custom_analysis",
                            "source_path": "^core",
                            "sink_path": "^core",
                            "mode": "pairwise",
                            "iterations": 50,
                        },
                    ]
                }
            }
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            assert "custom_workflow" in config.workflows.library
            workflow = config.workflows.library["custom_workflow"]
            assert len(workflow) == 2
            assert workflow[0]["step_type"] == "NetworkStats"
            assert workflow[1]["iterations"] == 50
        finally:
            config_path.unlink()

    def test_workflows_assignments_parsing(self):
        """Test parsing workflows.assignments section."""
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
        """Test handling of None values (empty YAML sections)."""
        config_dict = {
            "failure_policies": {
                "library": None,
                "assignments": {"scenario_overrides": None},
            },
            "workflows": {"library": None, "assignments": {"scenario_overrides": None}},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            # None values should be converted to empty dicts
            assert config.failure_policies.library == {}
            assert config.failure_policies.assignments.scenario_overrides == {}
            assert config.workflows.library == {}
            assert config.workflows.assignments.scenario_overrides == {}
        finally:
            config_path.unlink()

    def test_missing_sections(self):
        """Test that missing sections use defaults."""
        config_dict = {}  # No failure_policies or workflows sections

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            # Should have default values
            assert isinstance(config.failure_policies, FailurePoliciesConfig)
            assert (
                config.failure_policies.assignments.default
                == "single_random_link_failure"
            )
            assert isinstance(config.workflows, WorkflowsConfig)
            assert config.workflows.assignments.default == "basic_capacity_analysis"
        finally:
            config_path.unlink()

    def test_invalid_failure_policies_type(self):
        """Test error handling for invalid failure_policies type."""
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
        """Test error handling for invalid workflows type."""
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
        """Test error handling for invalid library type."""
        config_dict = {"failure_policies": {"library": "not a dict"}}

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(
                ValueError, match="'failure_policies.library' must be a dictionary"
            ):
                TopologyConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_invalid_assignments_type(self):
        """Test error handling for invalid assignments type."""
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
        """Test parsing a complete configuration with both sections."""
        config_dict = {
            "failure_policies": {
                "library": {
                    "custom_failure": {
                        "rules": [{"entity_scope": "node", "rule_type": "all"}]
                    }
                },
                "assignments": {"default": "custom_failure"},
            },
            "workflows": {
                "library": {"custom_workflow": [{"step_type": "NetworkStats"}]},
                "assignments": {"default": "custom_workflow"},
            },
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = TopologyConfig.from_yaml(config_path)

            # Check failure policies
            assert "custom_failure" in config.failure_policies.library
            assert config.failure_policies.assignments.default == "custom_failure"

            # Check workflows
            assert "custom_workflow" in config.workflows.library
            assert config.workflows.assignments.default == "custom_workflow"
        finally:
            config_path.unlink()
