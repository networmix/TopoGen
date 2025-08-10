"""Tests for component functionality in scenario builder."""

import pytest

from topogen.config import (
    ComponentAssignment,
    ComponentAssignments,
    ComponentsConfig,
    TopologyConfig,
)
from topogen.scenario_builder import (
    _build_blueprints_section,
    _build_components_section,
)


class TestComponentsScenarioBuilder:
    """Test component-related scenario builder functionality."""

    def create_test_config(self) -> TopologyConfig:
        """Create a test configuration with component assignments."""
        config = TopologyConfig()

        # Set up component assignments
        config.components = ComponentsConfig(
            assignments=ComponentAssignments(
                spine=ComponentAssignment(hw_component="CoreRouter", optics="800G-ZR+"),
                leaf=ComponentAssignment(hw_component="CoreRouter", optics="800G-ZR+"),
                core=ComponentAssignment(hw_component="CoreRouter", optics="800G-ZR+"),
                dc=ComponentAssignment(hw_component="CoreRouter", optics="800G-ZR+"),
            ),
        )

        return config

    def test_build_components_section_basic(self):
        """Test building components section with basic assignments."""
        config = self.create_test_config()
        used_blueprints = {"SingleRouter", "FullMesh4"}

        components = _build_components_section(config, used_blueprints)

        assert isinstance(components, dict)
        assert len(components) > 0

        # Should include referenced components
        assert "CoreRouter" in components
        assert "800G-ZR+" in components

    def test_build_components_section_with_role_assignments(self):
        """Test building components section with role assignments only."""
        config = self.create_test_config()
        used_blueprints = {"Clos_64_256"}

        components = _build_components_section(config, used_blueprints)

        assert isinstance(components, dict)
        assert "CoreRouter" in components
        assert "800G-ZR+" in components

    def test_build_components_section_uses_merged_library(self):
        """Smoke test that merged library is used (built-ins at minimum)."""
        config = self.create_test_config()
        used_blueprints = {"Clos_64_256"}
        components = _build_components_section(config, used_blueprints)
        assert "CoreRouter" in components

    def test_build_components_section_missing_component_warning(self):
        """Test warning when referenced component is not found."""
        config = self.create_test_config()

        # Reference a non-existent component
        config.components.assignments.spine.hw_component = "NonExistentChassis"

        used_blueprints = {"Clos_64_256"}

        # Should not raise exception but may log warning
        components = _build_components_section(config, used_blueprints)
        assert isinstance(components, dict)

    def test_build_blueprints_section_basic(self):
        """Test building blueprints section with component assignments."""
        config = self.create_test_config()
        used_blueprints = {"SingleRouter"}

        blueprints = _build_blueprints_section(used_blueprints, config)

        assert isinstance(blueprints, dict)
        assert "SingleRouter" in blueprints

        # Check that component assignment was added
        single_router = blueprints["SingleRouter"]
        assert "groups" in single_router

        core_group = single_router["groups"]["core"]
        assert "attrs" in core_group
        assert core_group["attrs"]["hw_component"] == "CoreRouter"

    def test_build_blueprints_section_with_role_assignments(self):
        """Test building blueprints section uses role assignments only."""
        config = self.create_test_config()
        used_blueprints = {"Clos_64_256"}

        blueprints = _build_blueprints_section(used_blueprints, config)

        assert "Clos_64_256" in blueprints

        clos_blueprint = blueprints["Clos_64_256"]
        spine_group = clos_blueprint["groups"]["spine"]
        leaf_group = clos_blueprint["groups"]["leaf"]

        # Should use role-based components
        assert spine_group["attrs"]["hw_component"] == "CoreRouter"
        assert leaf_group["attrs"]["hw_component"] == "CoreRouter"

    def test_build_blueprints_section_preserves_existing_attrs(self):
        """Test that building blueprints preserves existing node attributes."""
        config = self.create_test_config()
        used_blueprints = {"Clos_64_256"}

        blueprints = _build_blueprints_section(used_blueprints, config)

        clos_blueprint = blueprints["Clos_64_256"]
        spine_group = clos_blueprint["groups"]["spine"]

        # Should preserve original attributes
        assert spine_group["attrs"]["role"] == "spine"
        assert spine_group["attrs"]["hw_type"] == "spine_chassis"
        assert spine_group["attrs"]["tier"] == "spine"

        # And add new component assignment
        assert spine_group["attrs"]["hw_component"] == "CoreRouter"

    def test_build_blueprints_section_unknown_blueprint(self):
        """Test error when unknown blueprint is requested."""
        config = self.create_test_config()
        used_blueprints = {"UnknownBlueprint"}

        with pytest.raises(ValueError, match="Unknown blueprint: UnknownBlueprint"):
            _build_blueprints_section(used_blueprints, config)

    def test_build_blueprints_section_no_role_assignment(self):
        """Test blueprint with role that has no component assignment."""
        config = TopologyConfig()  # Empty config, no assignments
        used_blueprints = {"SingleRouter"}

        blueprints = _build_blueprints_section(used_blueprints, config)

        assert "SingleRouter" in blueprints

        # Should not add hw_component if no assignment
        core_group = blueprints["SingleRouter"]["groups"]["core"]
        assert "hw_component" not in core_group["attrs"]

    def test_build_blueprints_section_role_inference(self):
        """Test that role is correctly inferred from group name."""
        config = self.create_test_config()

        # Test with FullMesh4 which has 'core' group
        used_blueprints = {"FullMesh4"}

        blueprints = _build_blueprints_section(used_blueprints, config)

        full_mesh = blueprints["FullMesh4"]
        core_group = full_mesh["groups"]["core"]

        # Should use default core assignment since no override
        assert core_group["attrs"]["hw_component"] == "CoreRouter"

    def test_components_section_includes_only_referenced(self):
        """Test that components section includes only referenced components."""
        config = self.create_test_config()

        # Library no longer provided via config; this is a smoke test of filtered output

        used_blueprints = {"SingleRouter"}
        components = _build_components_section(config, used_blueprints)

        # Should not include unused components
        assert "UnusedChassis1" not in components
        assert "UnusedChassis2" not in components
        assert "UnusedOptic1" not in components

        # Should include referenced ones
        assert "CoreRouter" in components
        assert "800G-ZR+" in components

    def test_empty_used_blueprints(self):
        """Test behavior with empty used blueprints set."""
        config = self.create_test_config()
        used_blueprints = set()

        components = _build_components_section(config, used_blueprints)
        blueprints = _build_blueprints_section(used_blueprints, config)

        # Should handle empty sets gracefully
        assert isinstance(components, dict)
        assert isinstance(blueprints, dict)
        assert len(blueprints) == 0
