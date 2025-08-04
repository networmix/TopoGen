"""Pytest configuration and shared fixtures for TopologyGenerator tests."""

import pytest


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        "random_seed": 42,
        "metro_clusters": 5,
        "ring_radius_factor": 0.8,
        "k_shortest_paths": 3,
        "waxman_alpha": 0.25,
        "waxman_beta": 0.7,
        "budget_multiplier": 1.5,
        "target_avg_degree": 4,
        "data_sources": {
            "population_raster": "data/test_population.tif",
            "osm_highways": "data/test_highways.osm.pbf",
            "tiger_roads": "data/test_roads.zip",
        },
        "projection": {"target_crs": "EPSG:5070"},
        "highway_processing": {
            "snap_tolerance": 30,
            "min_edge_length": 1000,
            "densify_interval": 200,
        },
        "validation": {
            "max_metro_highway_distance": 10,
            "require_connected": True,
        },
        "output": {
            "pop_blueprint": {"cores_per_pop": 2, "internal_pattern": "mesh"},
            "scenario_metadata": {
                "title": "Test Backbone Topology",
                "description": "Test topology for validation",
                "version": "1.0",
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
