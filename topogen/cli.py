"""Command line interface for backbone topology generation."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from topogen.config import TopologyConfig
from topogen.log_config import get_logger

logger = get_logger(__name__)


@contextmanager
def Timer(description: str):
    """Context manager for timing operations with both print and log output.

    Args:
        description: Operation description for timing messages.

    Yields:
        None: Context manager yields nothing.
    """
    print(f"🔄 {description}...")
    logger.info(f"Starting {description}")
    start = time.time()
    try:
        yield
        elapsed = time.time() - start
        print(f"✅ {description} (completed in {elapsed:.1f}s)")
        logger.info(f"Completed {description} in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ {description} (failed after {elapsed:.1f}s)")
        logger.error(f"Failed {description} after {elapsed:.1f}s: {e}")
        raise


def _load_config(config_path: Path) -> TopologyConfig:
    """Load and validate configuration.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Loaded and validated configuration object.

    Raises:
        SystemExit: If configuration loading or validation fails.
    """
    try:
        config = TopologyConfig.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print(f"💡 Create one with: cp config.yml {config_path}")
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(2)  # Config problem
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        print(f"❌ Configuration error: {e}")
        print(f"💡 Check YAML syntax in: {config_path}")
        sys.exit(2)  # Config problem


def build_command(args: argparse.Namespace) -> None:
    """Build continental US backbone topology.

    Args:
        args: Parsed command line arguments containing config and output paths.
    """
    try:
        config_path = Path(args.config)
        config_obj = _load_config(config_path)
        # Compute output directory and scenario path.
        # If -o is a directory, write '<stem>_scenario.yml' inside it.
        # If -o is a file, use it directly and treat its parent as output dir.
        prefix_path = getattr(config_obj, "_source_path", config_path)
        stem = Path(prefix_path).stem if isinstance(prefix_path, Path) else "scenario"
        if getattr(args, "output", None):
            output_arg = Path(args.output)
            if output_arg.suffix.lower() in {".yml", ".yaml"}:
                output_dir = output_arg.parent
                output_path = output_arg
            else:
                output_dir = output_arg
                output_path = output_dir / f"{stem}_scenario.yml"
        else:
            output_dir = Path.cwd()
            output_path = output_dir / f"{stem}_scenario.yml"
        # Persist chosen output directory on config for downstream artefacts
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            config_obj._output_dir = output_dir  # type: ignore[attr-defined]
        except Exception:
            pass

        # Attach optional debug directory to config for downstream use
        if getattr(args, "debug_dir", None):
            try:
                debug_dir = Path(args.debug_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                config_obj._debug_dir = debug_dir
                # Provide a stable stem to downstream exporters
                config_obj._source_stem = Path(config_path).stem
            except Exception:
                # Non-fatal: continue without debug directory
                pass

        # Run the pipeline with timing
        with Timer("Topology generation pipeline"):
            scenario_yaml = _run_pipeline(
                config_obj, output_path, print_yaml=args.print
            )

        if args.print:
            print("\n" + "=" * 60)
            print("GENERATED SCENARIO YAML:")
            print("=" * 60)
            print(scenario_yaml)
        else:
            print(f"🎉 SUCCESS! Generated topology: {output_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ File not found: {e}")
        print("💡 Check data file paths in configuration")
        sys.exit(3)  # Validation failure
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print("💡 Check input data quality and configuration parameters")
        sys.exit(3)  # Validation failure
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print("💡 Use -v for detailed error information")
        sys.exit(1)  # Runtime error


def _run_pipeline(
    config: TopologyConfig, output_path: Path, print_yaml: bool = False
) -> str:
    """Execute topology generation from integrated graph.

    Args:
        config: Topology configuration object.
        output_path: Path for output scenario file.
        print_yaml: Whether to print YAML to stdout instead of validation.

    Returns:
        Generated scenario YAML string.

    Notes:
        Terminates the process via sys.exit(1) when the integrated graph is
        missing. When validation is enabled and finds issues, raises a
        ValueError (caller maps to exit code 3) after printing issue details.
    """
    from topogen import load_from_json
    from topogen.scenario_builder import build_scenario

    # Check for integrated graph in configured output dir (fallback to CWD)
    source_path = getattr(config, "_source_path", None)
    prefix = Path(source_path).stem if isinstance(source_path, Path) else "scenario"
    output_dir = getattr(config, "_output_dir", None)
    base_dir = Path(output_dir) if isinstance(output_dir, (str, Path)) else Path.cwd()
    graph_path = base_dir / f"{prefix}_integrated_graph.json"
    if not graph_path.exists():
        print("❌ No integrated graph found!")
        print("   Run generation first: python -m topogen generate")
        sys.exit(1)

    print("Topology Generation Pipeline")
    print("=" * 50)

    # Load integrated graph
    print("🔄 Loading integrated graph...")
    graph, crs = load_from_json(graph_path)

    print(f"📊 Graph loaded: {len(graph.nodes):,} nodes, {len(graph.edges):,} edges")

    # Count metro and highway nodes
    metro_nodes = [
        n
        for n, d in graph.nodes(data=True)
        if d.get("node_type") in ["metro", "metro+highway"]
    ]
    highway_nodes = [n for n in graph.nodes() if n not in metro_nodes]
    print(f"   Metro nodes: {len(metro_nodes)}")
    print(f"   Highway nodes: {len(highway_nodes)}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build NetGraph scenario
    with Timer("Generate NetGraph scenario"):
        scenario_yaml = build_scenario(graph, config)

    # Write scenario to file
    with Timer(f"Write scenario to {output_path}"):
        with open(output_path, "w") as f:
            f.write(scenario_yaml)

    print(f"\n📄 Scenario written to: {output_path}")

    # Validate the generated scenario (unless just printing)
    if not print_yaml:
        from topogen.validation import validate_scenario_yaml

        print("🔄 Validating generated scenario...")
        # Pass streamlined component mappings from configuration into validation
        comp_obj = getattr(config, "components", None)
        hw_map = (
            getattr(comp_obj, "hw_component", None) if comp_obj is not None else None
        )
        optics_map = getattr(comp_obj, "optics", None) if comp_obj is not None else None
        # Be strict about unexpected types to avoid silently disabling audits
        if hw_map is not None and not isinstance(hw_map, dict):
            raise ValueError(
                "'components.hw_component' must be a mapping when provided"
            )
        if optics_map is not None and not isinstance(optics_map, dict):
            raise ValueError("'components.optics' must be a mapping when provided")
        issues = validate_scenario_yaml(
            scenario_yaml,
            integrated_graph_path=graph_path,
            run_ngraph=True,
            hw_component_map=hw_map,
            optics_map=optics_map,
        )
        if issues:
            print("❌ Scenario validation found issues:")
            for s in issues:
                print(f"   - {s}")
            print("   Scenario file generated but has issues")
            # Fail the pipeline with a validation error
            raise ValueError("Scenario validation failed")
        else:
            print("✅ Scenario validation passed")

    return scenario_yaml


def _run_generation(config: TopologyConfig) -> None:
    """Execute integrated graph generation pipeline.

    Args:
        config: Topology configuration object.
    """
    from topogen import build_integrated_graph, save_to_json

    print("Integrated Graph Generation Pipeline")
    print("=" * 50)

    # Artefacts in configured output directory (fallback to CWD)
    cfg_out = getattr(config, "_output_dir", None)
    output_dir = Path(cfg_out) if isinstance(cfg_out, (str, Path)) else Path.cwd()

    # Output path for integrated graph
    source_path = getattr(config, "_source_path", None)
    prefix = Path(source_path).stem if isinstance(source_path, Path) else "scenario"
    graph_output = output_dir / f"{prefix}_integrated_graph.json"

    print(f"   Urban areas: {config.clustering.metro_clusters}")
    print(f"   UAC data: {config.data_sources.uac_polygons}")
    print(f"   Highway data: {config.data_sources.tiger_roads}")

    # Build integrated graph with timing
    with Timer("Build integrated metro + highway graph"):
        graph = build_integrated_graph(config)

    # Save integrated graph to JSON with timing
    with Timer("Save integrated graph"):
        save_to_json(
            graph, graph_output, config.projection.target_crs, config.output.formatting
        )

    print("\n🎉 Generation complete!")
    print(f"📁 Integrated graph: {graph_output}")
    print(f"📊 Graph summary: {len(graph.nodes):,} nodes, {len(graph.edges):,} edges")
    print("🔗 Ready for topology generation with:")
    print(
        f"   python -m topogen build -c {getattr(config, '_source_path', 'config.yml')}"
        f" -o {output_dir}"
    )


def generate_command(args: argparse.Namespace) -> None:
    """Generate integrated graph from raw datasets.

    Args:
        args: Parsed command line arguments containing config file path.
    """
    try:
        config_path = Path(args.config)
        config_obj = _load_config(config_path)
        # If output directory provided, persist on config
        if getattr(args, "output", None):
            try:
                out_dir = Path(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)
                config_obj._output_dir = out_dir  # type: ignore[attr-defined]
            except Exception:
                pass

        # Run generation pipeline
        _run_generation(config_obj)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)


def info_command(args: argparse.Namespace) -> None:
    """Show configuration and data source information.

    Args:
        args: Parsed command line arguments containing config file path.
    """
    try:
        config_path = Path(args.config)
        config_obj = _load_config(config_path)

        print("TopoGen Configuration")
        print("=" * 30)
        print(f"Metro clusters: {config_obj.clustering.metro_clusters}")
        print(f"Target CRS: {config_obj.projection.target_crs}")

        print("\nData Sources")
        print("=" * 20)
        print(f"UAC polygons: {config_obj.data_sources.uac_polygons}")
        print(f"TIGER roads: {config_obj.data_sources.tiger_roads}")

        # Check data source availability
        print("\nData Availability")
        print("=" * 20)

        uac_path = Path(config_obj.data_sources.uac_polygons)
        tiger_path = Path(config_obj.data_sources.tiger_roads)

        uac_status = "✅" if uac_path.exists() else "❌"
        tiger_status = "✅" if tiger_path.exists() else "❌"

        print(f"UAC data: {uac_status} {uac_path}")
        print(f"TIGER roads: {tiger_status} {tiger_path}")

        if not uac_path.exists() or not tiger_path.exists():
            print("\n⚠️  Missing data files - download required before generation")

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


def main() -> None:
    """Parse command line arguments and execute the appropriate subcommand.

    Configures logging, parses CLI arguments, and dispatches to the correct
    command function (build, generate, or info).
    """
    parser = argparse.ArgumentParser(
        prog="topogen",
        description="Generate continental US backbone topologies from highway infrastructure and urban area data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (logs only)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser(
        "build", help="Build continental US backbone topology"
    )
    build_parser.add_argument(
        "config",
        nargs="?",
        default="config.yml",
        help="Configuration file path (default: config.yml)",
    )
    build_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output YAML scenario file. Defaults to '<config_stem>_scenario.yml' in CWD.",
    )
    build_parser.add_argument(
        "--print",
        action="store_true",
        help="Print generated YAML to stdout for debugging",
    )
    build_parser.add_argument(
        "--debug-dir",
        default=None,
        help=(
            "Optional directory to write debug artifacts (e.g., traffic matrix "
            "internals as JSON) when -v is enabled"
        ),
    )

    build_parser.set_defaults(func=build_command)

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Build integrated metro + highway graph"
    )
    generate_parser.add_argument(
        "config",
        nargs="?",
        default="config.yml",
        help="Configuration file path (default: config.yml)",
    )
    generate_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output directory for integrated graph JSON and preview JPEG. "
            "Defaults to CWD."
        ),
    )

    generate_parser.set_defaults(func=generate_command)

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show configuration and data source information"
    )
    info_parser.add_argument(
        "config",
        nargs="?",
        default="config.yml",
        help="Configuration file path (default: config.yml)",
    )
    info_parser.set_defaults(func=info_command)

    # Parse arguments and dispatch
    args = parser.parse_args()

    # Configure logging based on arguments
    import logging

    from topogen.log_config import set_global_log_level

    # Determine log level from flags
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    set_global_log_level(log_level)

    # Suppress print output if --quiet is set
    if args.quiet:
        import builtins

        builtins.print = lambda *args, **kwargs: None

    if not hasattr(args, "func") or args.func is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
