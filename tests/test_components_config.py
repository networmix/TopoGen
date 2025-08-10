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

        assert isinstance(config.assignments.spine, ComponentAssignment)
        assert isinstance(config.assignments.leaf, ComponentAssignment)
        assert isinstance(config.assignments.core, ComponentAssignment)

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
        finally:
            config_path.unlink()

    def test_parse_components_ignores_library(self):
        """Test that inline component library is ignored by parser."""
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
            "components": {"library": {"CustomChassis": {"component_type": "chassis"}}},
        }

        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = TopologyConfig.from_yaml(config_path)
            # No inline library retained in config
            assert hasattr(config.components, "assignments")
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

    # Blueprint overrides removed

    def test_parse_components_invalid_library_type(self):
        """Inline library is ignored; parser should not raise."""
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
            cfg = TopologyConfig.from_yaml(config_path)
            assert isinstance(cfg.components, ComponentsConfig)
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
            # Presence of assignments struct is sufficient
            assert hasattr(config.components, "assignments")
        finally:
            config_path.unlink()
