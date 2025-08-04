"""Test the command-line interface functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from topogen import cli


def test_cli_help():
    """Test that help command works."""
    with patch("sys.argv", ["topogen", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 0


def test_cli_build_help():
    """Test that build subcommand help works."""
    with patch("sys.argv", ["topogen", "build", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 0


def test_cli_build_dry_run(temp_config_file, capsys):
    """Test dry run mode with valid configuration."""
    with patch(
        "sys.argv", ["topogen", "build", "--config", str(temp_config_file), "--dry-run"]
    ):
        cli.main()

    captured = capsys.readouterr()
    assert "TOPOLOGY GENERATOR CONFIGURATION" in captured.out
    assert "DRY RUN MODE - NO FILES WILL BE CREATED" in captured.out


def test_cli_build_dry_run_verbose(temp_config_file, capsys):
    """Test dry run mode with verbose logging."""
    with patch(
        "sys.argv",
        [
            "topogen",
            "--verbose",
            "build",
            "--config",
            str(temp_config_file),
            "--dry-run",
        ],
    ):
        cli.main()

    captured = capsys.readouterr()
    assert "Debug logging enabled" in captured.err


def test_cli_build_dry_run_quiet(temp_config_file, capsys):
    """Test dry run mode with quiet logging."""
    with patch(
        "sys.argv",
        ["topogen", "--quiet", "build", "--config", str(temp_config_file), "--dry-run"],
    ):
        cli.main()

    captured = capsys.readouterr()
    # In quiet mode, only output should be printed, not INFO level logs
    assert "TOPOLOGY GENERATOR CONFIGURATION" in captured.out
    # Should not see INFO logs in stderr
    assert "Loading configuration from:" not in captured.err


def test_cli_build_missing_config():
    """Test error handling for missing configuration file."""
    with patch("sys.argv", ["topogen", "build", "--config", "nonexistent.yml"]):
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 1


def test_cli_build_invalid_config(invalid_config_file):
    """Test error handling for invalid YAML configuration."""
    with patch("sys.argv", ["topogen", "build", "--config", str(invalid_config_file)]):
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 1


def test_cli_build_custom_output(temp_config_file, tmp_path):
    """Test build command with custom output path."""
    output_file = tmp_path / "custom_output.yaml"

    with patch(
        "sys.argv",
        [
            "topogen",
            "build",
            "--config",
            str(temp_config_file),
            "--out",
            str(output_file),
        ],
    ):
        cli.main()

    # Check that placeholder file was created
    assert output_file.exists()
    content = yaml.safe_load(output_file.read_text())
    assert content["metadata"]["generated_by"] == "topogen"
    assert content["metadata"]["status"] == "placeholder"


def test_cli_build_creates_output_directory(temp_config_file, tmp_path):
    """Test that build command creates output directory if it doesn't exist."""
    output_dir = tmp_path / "new_scenarios"
    output_file = output_dir / "topology.yaml"

    with patch(
        "sys.argv",
        [
            "topogen",
            "build",
            "--config",
            str(temp_config_file),
            "--out",
            str(output_file),
        ],
    ):
        cli.main()

    assert output_dir.exists()
    assert output_file.exists()


def test_load_config_success(temp_config_file):
    """Test successful configuration loading."""
    config = cli._load_config(temp_config_file)
    assert config["random_seed"] == 42
    assert config["metro_clusters"] == 5
    assert "data_sources" in config


def test_load_config_file_not_found():
    """Test configuration loading with missing file."""
    with pytest.raises(FileNotFoundError):
        cli._load_config(Path("nonexistent.yml"))


def test_load_config_invalid_yaml(invalid_config_file):
    """Test configuration loading with invalid YAML."""
    with pytest.raises(yaml.YAMLError):
        cli._load_config(invalid_config_file)


def test_placeholder_scenario_structure(temp_config_file, tmp_path):
    """Test that generated placeholder scenario has correct structure."""
    output_file = tmp_path / "test_output.yaml"

    # Actually build the scenario using the real function
    cli._build_topology(temp_config_file, output_file, dry_run=False)

    # Load and verify the generated structure
    content = yaml.safe_load(output_file.read_text())

    # Verify top-level structure
    assert "metadata" in content
    assert "blueprints" in content
    assert "network" in content

    # Verify metadata has required fields
    metadata = content["metadata"]
    assert metadata["generated_by"] == "topogen"
    assert metadata["status"] == "placeholder"
    assert "config_file" in metadata

    # Verify blueprint structure
    blueprints = content["blueprints"]
    assert "PoP_Site" in blueprints

    pop_site = blueprints["PoP_Site"]
    assert "groups" in pop_site
    assert "adjacency" in pop_site

    # Verify groups have required structure
    groups = pop_site["groups"]
    assert "core[1-2]" in groups

    core_group = groups["core[1-2]"]
    assert "node_count" in core_group
    assert "name_template" in core_group

    # Verify adjacency has expected format
    adjacency = pop_site["adjacency"]
    assert isinstance(adjacency, list)
    assert len(adjacency) > 0

    # Verify network sections exist (even if empty for placeholder)
    network = content["network"]
    assert "groups" in network
    assert "adjacency" in network


def test_main_with_argv():
    """Test main function with explicit argv."""
    with patch("sys.argv", ["original"]):  # Set up sys.argv
        with pytest.raises(SystemExit):
            cli.main(["--help"])


def test_main_without_argv():
    """Test main function using sys.argv."""
    with patch("sys.argv", ["topogen", "--help"]):
        with pytest.raises(SystemExit):
            cli.main()


def test_module_entry_point():
    """Test that module can be called with python -m topogen."""
    # Test that the __main__ module has the correct structure
    from topogen import __main__

    # Read the __main__ module source to verify it calls main()
    main_file = Path(__main__.__file__)
    content = main_file.read_text()

    # Verify that __main__ imports and calls the main function
    assert "from topogen.cli import main" in content
    assert "main()" in content
    assert '__name__ == "__main__"' in content
