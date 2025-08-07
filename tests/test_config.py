"""Tests for configuration management."""

from pathlib import Path

import pytest
import yaml

from topogen.config import TopologyConfig


def test_config_from_yaml(tmp_path: Path) -> None:
    """Test loading configuration from YAML file."""
    # Create test config file
    config_data = {
        "clustering": {
            "metro_clusters": 25,
            "export_clusters": True,
            "coordinate_precision": 1,
            "area_precision": 2,
            "max_uac_radius_km": 100.0,
        },
        "data_sources": {
            "uac_polygons": "test_uac.zip",
            "tiger_roads": "test_tiger.zip",
            "conus_boundary": "test_conus.zip",
        },
        "projection": {"target_crs": "EPSG:3857"},
        "highway_processing": {
            "min_edge_length_km": 0.05,
            "snap_precision_m": 10.0,
            "highway_classes": ["S1100", "S1200"],
            "min_cycle_nodes": 3,
            "filter_largest_component": True,
            "validation_sample_size": 5,
        },
        "validation": {
            "max_metro_highway_distance_km": 10.0,
            "require_connected": True,
            "max_degree_threshold": 1000,
            "high_degree_warning": 20,
        },
        "corridors": {
            "k_paths": 1,
            "k_nearest": 3,
            "max_edge_km": 600.0,
            "max_corridor_distance_km": 1000.0,
        },
        "output": {
            "scenario_metadata": {
                "title": "Test Topology",
                "description": "Test description",
                "version": "1.0",
            },
            "formatting": {
                "json_indent": 2,
            },
        },
    }

    config_path = tmp_path / "test_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Load configuration
    config = TopologyConfig.from_yaml(config_path)

    # Verify values
    assert config.clustering.metro_clusters == 25
    assert config.clustering.export_clusters is True
    assert config.projection.target_crs == "EPSG:3857"
    assert config.data_sources.uac_polygons == Path("test_uac.zip")


def test_config_defaults() -> None:
    """Test default configuration values."""
    config = TopologyConfig()

    assert config.clustering.metro_clusters == 30
    assert config.clustering.export_clusters is False  # Default should be False


def test_export_clusters_config() -> None:
    """Test that export_clusters configuration works correctly."""
    config = TopologyConfig.from_yaml(Path("config.yml"))

    # The current config has export_clusters set to true
    assert config.clustering.export_clusters is True


def test_dc_regions_config_defaults() -> None:
    """Test DC Region configuration defaults."""
    config = TopologyConfig()

    # Test build defaults
    assert config.build.build_defaults.dc_regions_per_metro == 2
    assert config.build.build_defaults.dc_region_blueprint == "DCRegion"

    # Test dc_to_pop_link defaults
    assert config.build.build_defaults.dc_to_pop_link.capacity == 400
    assert config.build.build_defaults.dc_to_pop_link.cost == 1
    assert config.build.build_defaults.dc_to_pop_link.attrs["link_type"] == "dc_to_pop"

    # Test component assignments
    assert hasattr(config.components.assignments, "dc")
    assert config.components.assignments.dc.hw_component == ""
    assert config.components.assignments.dc.optics == ""


def test_config_validation_missing_files() -> None:
    """Test configuration validation with missing data files."""
    from topogen.config import DataSources

    # Create config with missing UAC file
    config = TopologyConfig()
    config.data_sources = DataSources(
        uac_polygons=Path("data/nonexistent_uac.zip"),
        tiger_roads=config.data_sources.tiger_roads,
        conus_boundary=config.data_sources.conus_boundary,
    )

    # Should raise ValueError for missing files
    with pytest.raises(ValueError, match="UAC polygons file not found"):
        config.validate()


def test_config_validation_invalid_params() -> None:
    """Test configuration validation with invalid parameters."""
    from topogen.config import ClusteringConfig

    # Test negative metro_clusters
    config = TopologyConfig()
    config.clustering = ClusteringConfig(metro_clusters=0)
    with pytest.raises(ValueError, match="metro_clusters must be positive"):
        config.validate()


def test_config_summary() -> None:
    """Test configuration summary generation."""
    config = TopologyConfig()
    summary = config.summary()

    assert "TOPOLOGY GENERATOR CONFIGURATION" in summary
    assert "Metro Clusters: ~30" in summary


def test_config_file_not_found() -> None:
    """Test handling of missing configuration file."""
    non_existent_path = Path("non_existent_config.yml")

    with pytest.raises(FileNotFoundError):
        TopologyConfig.from_yaml(non_existent_path)


def test_config_invalid_yaml(tmp_path: Path) -> None:
    """Test handling of invalid YAML."""
    config_path = tmp_path / "invalid_config.yml"

    # Write invalid YAML
    with open(config_path, "w") as f:
        f.write("invalid: yaml: content: [unclosed")

    with pytest.raises(yaml.YAMLError):
        TopologyConfig.from_yaml(config_path)
