"""Command-line interface for TopologyGenerator."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from topogen.logging import get_logger, set_global_log_level

logger = get_logger(__name__)


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file has invalid YAML.
    """
    logger.info(f"Loading configuration from: {config_path}")

    try:
        config_text = config_path.read_text()
        config = yaml.safe_load(config_text)
        logger.info("✓ Configuration loaded successfully")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration file: {e}")
        raise


def _print_config_summary(config: dict[str, Any]) -> None:
    """Print a summary of the loaded configuration.

    Args:
        config: Configuration dictionary to summarize.
    """
    print("\n" + "=" * 60)
    print("TOPOLOGY GENERATOR CONFIGURATION")
    print("=" * 60)

    # General parameters
    print("\n1. GENERAL PARAMETERS")
    print("-" * 30)
    print(f"   Random Seed: {config.get('random_seed', 42)}")
    print(f"   Metro Clusters: ~{config.get('metro_clusters', 30)}")

    # Geographic parameters
    print("\n2. GEOGRAPHIC PARAMETERS")
    print("-" * 30)
    print(f"   Ring Radius Factor: {config.get('ring_radius_factor', 0.8)}")
    print(f"   K-Shortest Paths: {config.get('k_shortest_paths', 3)}")

    # Network generation parameters
    print("\n3. NETWORK GENERATION")
    print("-" * 30)
    print(f"   Waxman Alpha: {config.get('waxman_alpha', 0.25)}")
    print(f"   Waxman Beta: {config.get('waxman_beta', 0.7)}")
    print(f"   Budget Multiplier: {config.get('budget_multiplier', 1.5)}")
    print(f"   Target Avg Degree: {config.get('target_avg_degree', 4)}")

    # Data sources
    print("\n4. DATA SOURCES")
    print("-" * 30)
    data_sources = config.get("data_sources", {})
    for source_name, source_path in data_sources.items():
        print(f"   {source_name}: {source_path}")

    print("\n" + "=" * 60)


def _build_topology(
    config_path: Path,
    output_path: Path,
    dry_run: bool = False,
) -> None:
    """Build topology and export NetGraph scenario.

    Args:
        config_path: Path to configuration YAML file.
        output_path: Path where NetGraph scenario should be written.
        dry_run: If True, parse config and echo parameters without building.
    """
    logger.info("Starting topology generation")

    try:
        # Load and validate configuration
        config = _load_config(config_path)

        # Print configuration summary
        _print_config_summary(config)

        if dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN MODE - NO FILES WILL BE CREATED")
            print("=" * 60)
            print(f"\nWould create output: {output_path}")

            # Create output directory if it doesn't exist (in dry run, just check)
            output_dir = output_path.parent
            if not output_dir.exists():
                print(f"Would create directory: {output_dir}")
            else:
                print(f"Output directory exists: {output_dir}")

            logger.info("Dry run completed successfully")
            return

        # TODO: Implement actual topology generation
        # This is where the main algorithm will be implemented:
        # 1. Load geographic data
        # 2. Generate metro clusters
        # 3. Build highway graph
        # 4. Select corridors
        # 5. Apply MST + Waxman sampling
        # 6. Export to NetGraph YAML

        print("\n" + "=" * 60)
        print("TOPOLOGY GENERATION")
        print("=" * 60)
        print("\n⚠️  Topology generation not yet implemented")
        print("   This will be the main algorithm pipeline:")
        print("   1. Load population and highway data")
        print("   2. Generate metro clusters via K-means")
        print("   3. Build hybrid highway graph")
        print("   4. Select inter-metro corridors")
        print("   5. Apply MST + Waxman sampling")
        print("   6. Export NetGraph scenario with blueprints")

        # Create output directory
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        # Placeholder: write a minimal scenario file
        placeholder_scenario = {
            "metadata": {
                "generated_by": "topogen",
                "config_file": str(config_path),
                "status": "placeholder",
            },
            "blueprints": {
                "PoP_Site": {
                    "groups": {
                        "core[1-2]": {
                            "node_count": 1,
                            "name_template": "core{node_num}",
                        }
                    },
                    "adjacency": [
                        {"source": "core[1]", "target": "core[2]", "pattern": "mesh"}
                    ],
                }
            },
            "network": {"groups": {}, "adjacency": []},
        }

        # Write placeholder scenario
        with open(output_path, "w") as f:
            yaml.dump(placeholder_scenario, f, default_flow_style=False, indent=2)

        print(f"\n✅ Placeholder scenario written to: {output_path}")
        print("   (Implementation of topology generation algorithm pending)")
        logger.info("Topology generation completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ ERROR: File not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Configuration error: {e}")
        print(f"❌ ERROR: Invalid configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to build topology: {type(e).__name__}: {e}")
        print(f"❌ ERROR: Failed to build topology: {type(e).__name__}: {e}")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``topogen`` command.

    Args:
        argv: Optional list of command-line arguments. If ``None``, ``sys.argv``
            is used.
    """
    parser = argparse.ArgumentParser(
        prog="topogen",
        description="Generate realistic network topologies for backbone analysis",
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Enable quiet mode (WARNING+ only)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build command
    build_parser = subparsers.add_parser(
        "build", help="Build topology and export NetGraph scenario"
    )
    build_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yml"),
        help="Path to configuration YAML file (default: config.yml)",
    )
    build_parser.add_argument(
        "--out",
        type=Path,
        default=Path("scenarios/us_backbone.yaml"),
        help="Output path for NetGraph scenario (default: scenarios/us_backbone.yaml)",
    )
    build_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and echo parameters without building",
    )

    args = parser.parse_args(argv)

    # Configure logging based on arguments
    if args.verbose:
        set_global_log_level(logging.DEBUG)
        logger.debug("Debug logging enabled")
    elif args.quiet:
        set_global_log_level(logging.WARNING)
    else:
        set_global_log_level(logging.INFO)

    if args.command == "build":
        _build_topology(
            args.config,
            args.out,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
