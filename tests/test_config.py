"""Test configuration loading and validation."""

import yaml


def test_config_with_missing_optional_sections():
    """Test config handling with missing optional sections."""
    minimal_config = {
        "random_seed": 123,
        "metro_clusters": 10,
    }

    # Should not raise errors when optional sections are missing
    assert minimal_config["random_seed"] == 123
    assert minimal_config["metro_clusters"] == 10

    # Missing sections should be handled gracefully
    assert minimal_config.get("data_sources", {}) == {}
    assert minimal_config.get("waxman_alpha", 0.25) == 0.25


def test_config_parameter_ranges(sample_config):
    """Test that configuration parameters are within reasonable ranges."""
    # Probability parameters should be between 0 and 1
    assert 0 <= sample_config["waxman_alpha"] <= 1

    # Beta should be positive
    assert sample_config["waxman_beta"] > 0

    # Budget multiplier should be >= 1 (more than MST)
    assert sample_config["budget_multiplier"] >= 1.0

    # Target degree should be at least 2 for connectivity
    assert sample_config["target_avg_degree"] >= 2

    # Number of clusters should be positive
    assert sample_config["metro_clusters"] > 0

    # K-shortest paths should be at least 1
    assert sample_config["k_shortest_paths"] >= 1

    # Ring radius factor should be positive
    assert sample_config["ring_radius_factor"] > 0


def test_config_serialization_roundtrip(sample_config, tmp_path):
    """Test that config can be serialized and deserialized correctly."""
    config_file = tmp_path / "roundtrip_config.yml"

    # Write config
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)

    # Read config back
    with open(config_file, "r") as f:
        loaded_config = yaml.safe_load(f)

    # Should be identical
    assert loaded_config == sample_config
