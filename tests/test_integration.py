"""Integration tests for the topology generator."""

import subprocess
import sys

import yaml


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_cli_subprocess_help(self):
        """Test CLI help via subprocess (real entry point)."""
        result = subprocess.run(
            [sys.executable, "-m", "topogen", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Generate realistic network topologies" in result.stdout
        assert "build" in result.stdout

    def test_cli_subprocess_build_help(self):
        """Test CLI build help via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "topogen", "build", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--out" in result.stdout
        assert "--dry-run" in result.stdout

    def test_cli_subprocess_dry_run(self, temp_config_file):
        """Test CLI dry run via subprocess."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "topogen",
                "build",
                "--config",
                str(temp_config_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "TOPOLOGY GENERATOR CONFIGURATION" in result.stdout
        assert "DRY RUN MODE" in result.stdout
        assert "Random Seed: 42" in result.stdout

    def test_cli_subprocess_verbose(self, temp_config_file):
        """Test CLI verbose mode via subprocess."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "topogen",
                "--verbose",
                "build",
                "--config",
                str(temp_config_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Debug logging enabled" in result.stderr

    def test_cli_subprocess_quiet(self, temp_config_file):
        """Test CLI quiet mode via subprocess."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "topogen",
                "--quiet",
                "build",
                "--config",
                str(temp_config_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Should have output but minimal stderr
        assert "TOPOLOGY GENERATOR CONFIGURATION" in result.stdout
        # Should not have info-level log messages in stderr
        assert "Loading configuration from:" not in result.stderr

    def test_cli_subprocess_missing_config(self):
        """Test CLI error handling via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "topogen", "build", "--config", "nonexistent.yml"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "ERROR: File not found" in result.stdout

    def test_full_build_workflow(self, temp_config_file, tmp_path):
        """Test complete build workflow with file output."""
        output_file = tmp_path / "test_topology.yaml"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "topogen",
                "build",
                "--config",
                str(temp_config_file),
                "--out",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Placeholder scenario written to:" in result.stdout
        assert output_file.exists()

        # Verify output file structure
        content = yaml.safe_load(output_file.read_text())
        assert "metadata" in content
        assert "blueprints" in content
        assert "network" in content
        assert content["metadata"]["generated_by"] == "topogen"


class TestConfigurationIntegration:
    """Integration tests for configuration handling."""

    def test_default_config_location(self, tmp_path):
        """Test that CLI uses default config.yml if present."""
        # Create config.yml in temp directory
        config_content = {
            "random_seed": 999,
            "metro_clusters": 15,
            "data_sources": {"population_raster": "default_test.tif"},
        }
        config_file = tmp_path / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        # Set up environment to include current directory in Python path
        import os

        env = os.environ.copy()
        current_dir = os.getcwd()
        python_path = env.get("PYTHONPATH", "")
        if python_path:
            env["PYTHONPATH"] = f"{current_dir}:{python_path}"
        else:
            env["PYTHONPATH"] = current_dir

        # Run CLI from that directory
        result = subprocess.run(
            [sys.executable, "-m", "topogen", "build", "--dry-run"],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Random Seed: 999" in result.stdout
        assert "Metro Clusters: ~15" in result.stdout

    def test_custom_config_location(self, sample_config, tmp_path):
        """Test CLI with custom config file location."""
        # Create config in custom location
        custom_config = sample_config.copy()
        custom_config["random_seed"] = 777

        config_file = tmp_path / "custom" / "my_config.yml"
        config_file.parent.mkdir(parents=True)

        with open(config_file, "w") as f:
            yaml.dump(custom_config, f)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "topogen",
                "build",
                "--config",
                str(config_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Random Seed: 777" in result.stdout

    def test_config_with_all_sections(self, tmp_path):
        """Test configuration with all possible sections."""
        comprehensive_config = {
            "random_seed": 12345,
            "metro_clusters": 25,
            "ring_radius_factor": 0.9,
            "k_shortest_paths": 5,
            "waxman_alpha": 0.3,
            "waxman_beta": 0.8,
            "budget_multiplier": 2.0,
            "target_avg_degree": 6,
            "data_sources": {
                "population_raster": "data/comprehensive_population.tif",
                "osm_highways": "data/comprehensive_highways.osm.pbf",
                "tiger_roads": "data/comprehensive_roads.zip",
                "extra_corridors": "data/extra/",
            },
            "projection": {"target_crs": "EPSG:5070"},
            "highway_processing": {
                "snap_tolerance": 50,
                "min_edge_length": 2000,
                "densify_interval": 100,
            },
            "validation": {
                "max_metro_highway_distance": 15,
                "require_connected": True,
            },
            "output": {
                "pop_blueprint": {"cores_per_pop": 4, "internal_pattern": "ring"},
                "scenario_metadata": {
                    "title": "Comprehensive Test Topology",
                    "description": "Full-featured test configuration",
                    "version": "2.0",
                },
            },
        }

        config_file = tmp_path / "comprehensive_config.yml"
        with open(config_file, "w") as f:
            yaml.dump(comprehensive_config, f, default_flow_style=False, indent=2)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "topogen",
                "build",
                "--config",
                str(config_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Random Seed: 12345" in result.stdout
        assert "Metro Clusters: ~25" in result.stdout
        assert "Ring Radius Factor: 0.9" in result.stdout
        assert "Waxman Alpha: 0.3" in result.stdout
        assert "comprehensive_population.tif" in result.stdout


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_missing_required_command(self):
        """Test error when no command is provided."""
        result = subprocess.run(
            [sys.executable, "-m", "topogen"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2  # argparse error
        assert "required" in result.stderr.lower()

    def test_invalid_command(self):
        """Test error with invalid command."""
        result = subprocess.run(
            [sys.executable, "-m", "topogen", "invalid_command"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2  # argparse error

    def test_malformed_yaml_config(self, tmp_path):
        """Test error handling with malformed YAML."""
        bad_config = tmp_path / "bad_config.yml"
        bad_config.write_text("invalid: yaml: content: [unclosed")

        result = subprocess.run(
            [sys.executable, "-m", "topogen", "build", "--config", str(bad_config)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "ERROR: Invalid configuration" in result.stdout

    def test_permission_error_simulation(self, temp_config_file, tmp_path):
        """Test handling of file permission errors."""
        # Create a directory where the file should go, but make it non-writable
        output_dir = tmp_path / "readonly"
        output_dir.mkdir()
        output_file = output_dir / "topology.yaml"

        # On Unix systems, we can test permission errors
        import os
        import stat

        if os.name == "posix":  # Unix-like systems
            # Make directory read-only
            os.chmod(output_dir, stat.S_IREAD | stat.S_IEXEC)

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "topogen",
                        "build",
                        "--config",
                        str(temp_config_file),
                        "--out",
                        str(output_file),
                    ],
                    capture_output=True,
                    text=True,
                )

                # Should fail due to permission error
                assert result.returncode == 1
                assert "ERROR:" in result.stdout

            finally:
                # Restore permissions so cleanup works
                os.chmod(output_dir, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)


class TestOutputGeneration:
    """Integration tests for output file generation."""

    def test_output_directory_creation(self, temp_config_file, tmp_path):
        """Test that output directories are created automatically."""
        nested_output = tmp_path / "deeply" / "nested" / "output" / "topology.yaml"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "topogen",
                "build",
                "--config",
                str(temp_config_file),
                "--out",
                str(nested_output),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert nested_output.exists()
        assert nested_output.parent.exists()

    def test_output_file_overwrite(self, temp_config_file, tmp_path):
        """Test that existing output files are overwritten."""
        output_file = tmp_path / "topology.yaml"

        # Create existing file with different content
        output_file.write_text("existing content")
        assert output_file.read_text() == "existing content"

        # Run CLI to overwrite
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "topogen",
                "build",
                "--config",
                str(temp_config_file),
                "--out",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # File should be overwritten with new content
        content = yaml.safe_load(output_file.read_text())
        assert "metadata" in content
        assert content["metadata"]["generated_by"] == "topogen"

    def test_yaml_output_validity(self, temp_config_file, tmp_path):
        """Test that generated YAML is valid and well-formed."""
        output_file = tmp_path / "topology.yaml"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "topogen",
                "build",
                "--config",
                str(temp_config_file),
                "--out",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Load and validate YAML structure
        content = yaml.safe_load(output_file.read_text())

        # Should have top-level sections
        assert isinstance(content, dict)
        assert "metadata" in content
        assert "blueprints" in content
        assert "network" in content

        # Metadata should be complete
        metadata = content["metadata"]
        assert "generated_by" in metadata
        assert "config_file" in metadata
        assert "status" in metadata

        # Blueprints should have PoP_Site
        blueprints = content["blueprints"]
        assert "PoP_Site" in blueprints

        pop_site = blueprints["PoP_Site"]
        assert "groups" in pop_site
        assert "adjacency" in pop_site

        # Network should have empty sections (placeholder)
        network = content["network"]
        assert "groups" in network
        assert "adjacency" in network
