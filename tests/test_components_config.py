"""Tests for component configuration functionality."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from topogen.config import ComponentAssignment, ComponentsConfig, TopologyConfig


class TestComponentsConfig:
    """Test component configuration parsing and validation."""

    def test_component_assignment_defaults(self):
        """Test ComponentAssignment with default values."""
        assignment = ComponentAssignment()

        assert assignment.hw_component == ""
        assert assignment.optics == ""

    def test_component_assignment_with_values(self):
        """Test ComponentAssignment with specified values."""
        assignment = ComponentAssignment(hw_component="SpineChassis", optics="400G-LR4")

        assert assignment.hw_component == "SpineChassis"
        assert assignment.optics == "400G-LR4"

    def test_components_config_defaults(self):
        """Test ComponentsConfig with default values."""
        config = ComponentsConfig()

        assert config.library == {}
        assert isinstance(config.assignments.spine, ComponentAssignment)
        assert isinstance(config.assignments.leaf, ComponentAssignment)
        assert isinstance(config.assignments.core, ComponentAssignment)
        assert config.assignments.blueprint_overrides == {}

    def test_parse_empty_components_config(self):
        """Test parsing configuration with empty components section."""
        config_data = {
            "data_sources": {
                "uac_polygons": "test.zip",
                "tiger_roads": "test.zip",
                "conus_boundary": "test.zip",
            },
            "projection": {"target_crs": "EPSG:5070"},
            "clustering": {"metro_clusters": 25},
            "highway_processing": {},
            "corridors": {},
            "validation": {},
            "output": {"scenario_metadata": {}, "formatting": {}},
            "components": {},
        }

        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = TopologyConfig.from_yaml(config_path)
            assert isinstance(config.components, ComponentsConfig)
            assert config.components.library == {}
        finally:
            config_path.unlink()

    def test_parse_components_with_library(self):
        """Test parsing configuration with custom component library."""
        config_data = {
            "data_sources": {
                "uac_polygons": "test.zip",
                "tiger_roads": "test.zip",
                "conus_boundary": "test.zip",
            },
            "projection": {"target_crs": "EPSG:5070"},
            "clustering": {"metro_clusters": 25},
            "highway_processing": {},
            "corridors": {},
            "validation": {},
            "output": {"scenario_metadata": {}, "formatting": {}},
            "components": {
                "library": {
                    "CustomChassis": {
                        "component_type": "chassis",
                        "cost": 50000.0,
                        "power_watts": 2000.0,
                    }
                }
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = TopologyConfig.from_yaml(config_path)
            assert "CustomChassis" in config.components.library
            assert config.components.library["CustomChassis"]["cost"] == 50000.0
        finally:
            config_path.unlink()

    def test_parse_components_with_assignments(self):
        """Test parsing configuration with component assignments."""
        config_data = {
            "data_sources": {
                "uac_polygons": "test.zip",
                "tiger_roads": "test.zip",
                "conus_boundary": "test.zip",
            },
            "projection": {"target_crs": "EPSG:5070"},
            "clustering": {"metro_clusters": 25},
            "highway_processing": {},
            "corridors": {},
            "validation": {},
            "output": {"scenario_metadata": {}, "formatting": {}},
            "components": {
                "assignments": {
                    "spine": {"hw_component": "SpineChassis", "optics": "400G-LR4"},
                    "leaf": {"hw_component": "LeafChassis", "optics": "400G-SR8"},
                    "core": {"hw_component": "CoreRouter", "optics": "100G-LR4"},
                }
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = TopologyConfig.from_yaml(config_path)

            assert config.components.assignments.spine.hw_component == "SpineChassis"
            assert config.components.assignments.spine.optics == "400G-LR4"
            assert config.components.assignments.leaf.hw_component == "LeafChassis"
            assert config.components.assignments.leaf.optics == "400G-SR8"
            assert config.components.assignments.core.hw_component == "CoreRouter"
            assert config.components.assignments.core.optics == "100G-LR4"
        finally:
            config_path.unlink()

    def test_parse_components_with_blueprint_overrides(self):
        """Test parsing configuration with blueprint overrides."""
        config_data = {
            "data_sources": {
                "uac_polygons": "test.zip",
                "tiger_roads": "test.zip",
                "conus_boundary": "test.zip",
            },
            "projection": {"target_crs": "EPSG:5070"},
            "clustering": {"metro_clusters": 25},
            "highway_processing": {},
            "corridors": {},
            "validation": {},
            "output": {"scenario_metadata": {}, "formatting": {}},
            "components": {
                "assignments": {
                    "blueprint_overrides": {
                        "Clos_64_256": {
                            "spine": {
                                "hw_component": "HighEndSpineChassis",
                                "optics": "400G-LR4",
                            },
                            "leaf": {
                                "hw_component": "HighEndLeafChassis",
                                "optics": "400G-SR8",
                            },
                        }
                    }
                }
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = TopologyConfig.from_yaml(config_path)

            overrides = config.components.assignments.blueprint_overrides
            assert "Clos_64_256" in overrides

            clos_overrides = overrides["Clos_64_256"]
            assert "spine" in clos_overrides
            assert "leaf" in clos_overrides

            assert clos_overrides["spine"].hw_component == "HighEndSpineChassis"
            assert clos_overrides["spine"].optics == "400G-LR4"
            assert clos_overrides["leaf"].hw_component == "HighEndLeafChassis"
            assert clos_overrides["leaf"].optics == "400G-SR8"
        finally:
            config_path.unlink()

    def test_parse_components_invalid_library_type(self):
        """Test parsing configuration with invalid library type."""
        config_data = {
            "data_sources": {
                "uac_polygons": "test.zip",
                "tiger_roads": "test.zip",
                "conus_boundary": "test.zip",
            },
            "projection": {"target_crs": "EPSG:5070"},
            "clustering": {"metro_clusters": 25},
            "highway_processing": {},
            "corridors": {},
            "validation": {},
            "output": {"scenario_metadata": {}, "formatting": {}},
            "components": {
                "library": "invalid_type"  # Should be dict
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ValueError, match="'components.library' must be a dictionary"
            ):
                TopologyConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_parse_components_invalid_assignments_type(self):
        """Test parsing configuration with invalid assignments type."""
        config_data = {
            "data_sources": {
                "uac_polygons": "test.zip",
                "tiger_roads": "test.zip",
                "conus_boundary": "test.zip",
            },
            "projection": {"target_crs": "EPSG:5070"},
            "clustering": {"metro_clusters": 25},
            "highway_processing": {},
            "corridors": {},
            "validation": {},
            "output": {"scenario_metadata": {}, "formatting": {}},
            "components": {
                "assignments": "invalid_type"  # Should be dict
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            config_path = Path(f.name)

        try:
            with pytest.raises(
                ValueError, match="'components.assignments' must be a dictionary"
            ):
                TopologyConfig.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_parse_components_none_library(self):
        """Test parsing configuration with None library (empty YAML section)."""
        config_yaml = """
data_sources:
  uac_polygons: test.zip
  tiger_roads: test.zip
  conus_boundary: test.zip
projection:
  target_crs: EPSG:5070
clustering:
  metro_clusters: 25
highway_processing: {}
corridors: {}
validation: {}
output:

  scenario_metadata: {}
  formatting: {}
components:
  library:  # This becomes None in YAML
  assignments: {}
"""

        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(config_yaml)
            config_path = Path(f.name)

        try:
            config = TopologyConfig.from_yaml(config_path)
            # Should handle None library gracefully
            assert config.components.library == {}
        finally:
            config_path.unlink()
