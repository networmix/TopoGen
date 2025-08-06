"""Pytest configuration and shared fixtures for TopologyGenerator tests."""

import pytest


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        "clustering": {
            "metro_clusters": 5,
            "max_uac_radius_km": 100.0,
            "export_clusters": False,
            "coordinate_precision": 1,
            "area_precision": 2,
        },
        "data_sources": {
            "uac_polygons": "data/test_uac.zip",
            "tiger_roads": "data/test_roads.zip",
            "conus_boundary": "data/test_conus.zip",
        },
        "projection": {"target_crs": "EPSG:5070"},
        "highway_processing": {
            "min_edge_length_km": 1.0,
            "snap_precision_m": 10.0,
            "highway_classes": ["S1100", "S1200"],
            "min_cycle_nodes": 3,
            "filter_largest_component": True,
            "validation_sample_size": 5,
        },
        "corridors": {
            "k_paths": 3,
            "k_nearest": 3,
            "max_edge_km": 600.0,
            "max_corridor_distance_km": 1000.0,
        },
        "validation": {
            "max_metro_highway_distance_km": 10.0,
            "require_connected": True,
            "max_degree_threshold": 1000,
            "high_degree_warning": 20,
        },
        "output": {
            "pop_blueprint": {
                "sites_per_metro": 4,
                "cores_per_pop": 2,
                "internal_pattern": "mesh",
            },
            "scenario_metadata": {
                "title": "Test Backbone Topology",
                "description": "Test topology for validation",
                "version": "1.0",
            },
            "formatting": {
                "json_indent": 2,
                "distance_conversion_factor": 1000,
                "area_conversion_factor": 1000000,
            },
        },
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create a temporary configuration file for testing."""
    import yaml

    config_file = tmp_path / "test_config.yml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    return config_file


@pytest.fixture
def invalid_config_file(tmp_path):
    """Create an invalid YAML configuration file for testing."""
    config_file = tmp_path / "invalid_config.yml"
    config_file.write_text("invalid: yaml: content: [unclosed")
    return config_file


@pytest.fixture
def topology_config(temp_config_file):
    """Create a complete TopologyConfig object for testing."""
    from topogen.config import TopologyConfig

    return TopologyConfig.from_yaml(temp_config_file)


@pytest.fixture
def clustering_config(topology_config):
    """Provide ClusteringConfig for testing."""
    return topology_config.clustering


@pytest.fixture
def highway_config(topology_config):
    """Provide HighwayProcessingConfig for testing."""
    return topology_config.highway_processing


@pytest.fixture
def validation_config(topology_config):
    """Provide ValidationConfig for testing."""
    return topology_config.validation


@pytest.fixture
def formatting_config(topology_config):
    """Provide FormattingConfig for testing."""
    return topology_config.output.formatting


@pytest.fixture
def corridors_config(topology_config):
    """Provide CorridorsConfig for testing."""
    return topology_config.corridors
