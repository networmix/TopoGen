"""Command line interface for backbone topology generation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import NamedTemporaryFile

from topogen.config import TopologyConfig
from topogen.context import RunContext
from topogen.log_config import get_logger

logger = get_logger(__name__)


@contextmanager
def Timer(description: str):
    """Print and log elapsed time on completion or failure."""
    print(f"🔄 {description}...")
    logger.info(f"Starting {description}")
    start = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - start
        print(f"✅ {description} (completed in {elapsed:.1f}s)")
        logger.info(f"Completed {description} in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"❌ {description} (failed after {elapsed:.1f}s)")
        logger.error(f"Failed {description} after {elapsed:.1f}s: {e}")
        raise


def _load_config(config_path: Path) -> TopologyConfig:
    """Load and validate a YAML config, exiting with code 2 on failure."""
    try:
        config = TopologyConfig.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print(f"💡 Create one with: cp examples/small_baseline.yml {config_path}")
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(2)  # Config problem
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        print(f"❌ Configuration error: {e}")
        print(f"💡 Check YAML syntax in: {config_path}")
        sys.exit(2)  # Config problem


def build_command(args: argparse.Namespace) -> None:
    """Build and validate a NetGraph scenario from a saved graph."""
    try:
        config_path = Path(args.config)
        config_obj = _load_config(config_path)
        output_arg = Path(args.output) if args.output else Path.cwd()
        if output_arg.suffix.lower() in {".yml", ".yaml"}:
            output_dir, output_path = output_arg.parent, output_arg
        else:
            output_dir = output_arg
            output_path = output_dir / f"{config_path.stem}_scenario.yml"
        context = RunContext(
            output_dir,
            config_path.stem,
            Path(args.debug_dir) if args.debug_dir else None,
        )

        with Timer("Topology generation pipeline"):
            scenario_yaml = _run_pipeline(config_obj, output_path, context=context)

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


def _write_scenario(path: Path, content: str) -> None:
    """Replace a scenario atomically, only after it has passed validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _run_pipeline(
    config: TopologyConfig, output_path: Path, *, context: RunContext
) -> str:
    """Build, validate, then publish scenario YAML from the saved corridor graph."""
    from topogen import load_from_json
    from topogen.scenario import build_scenario
    from topogen.validation import validate_scenario_yaml

    graph_path = context.path("integrated_graph.json")
    if not graph_path.exists():
        raise FileNotFoundError(
            f"No integrated graph found: {graph_path}. Run topogen generate first."
        )
    graph, crs = load_from_json(graph_path)
    if crs != config.projection.target_crs:
        raise ValueError(
            f"Saved graph CRS {crs} differs from configured {config.projection.target_crs}; regenerate the integrated graph"
        )
    print(f"Graph loaded: {len(graph.nodes):,} metros, {len(graph.edges):,} corridors")
    with Timer("Generate NetGraph scenario"):
        scenario_yaml = build_scenario(graph, config, context=context)
    issues = validate_scenario_yaml(
        scenario_yaml,
        integrated_graph_path=graph_path,
        hw_component_map=config.components.hw_component,
        optics_map=config.components.optics,
    )
    if issues:
        raise ValueError("Scenario validation failed:\n" + "\n".join(issues))
    print("✅ Scenario validation passed")
    _write_scenario(output_path, scenario_yaml)
    print(f"Scenario written to: {output_path}")
    return scenario_yaml


def _run_generation(config: TopologyConfig, context: RunContext) -> None:
    """Generate and save the corridor graph at the explicit artifact destination."""
    from topogen import build_integrated_graph, save_to_json

    config.validate()
    context.output_dir.mkdir(parents=True, exist_ok=True)
    with Timer("Generate a metro-to-metro corridor graph"):
        graph = build_integrated_graph(config, context=context)
    graph_output = context.path("integrated_graph.json")
    save_to_json(
        graph, graph_output, config.projection.target_crs, config.output.formatting
    )
    print(f"Integrated graph: {graph_output}")
    print(f"Graph summary: {len(graph.nodes):,} nodes, {len(graph.edges):,} edges")


def generate_command(args: argparse.Namespace) -> None:
    """Generate a corridor graph from the configured geographic datasets."""
    try:
        config_path = Path(args.config)
        config_obj = _load_config(config_path)
        context = RunContext(
            Path(args.output) if args.output else Path.cwd(),
            config_path.stem,
        )
        _run_generation(config_obj, context)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)


def info_command(args: argparse.Namespace) -> None:
    """Print the configuration summary and data file availability."""
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
        print(f"CONUS boundary: {config_obj.data_sources.conus_boundary}")

        print("\nData Availability")
        print("=" * 20)

        uac_path = Path(config_obj.data_sources.uac_polygons)
        tiger_path = Path(config_obj.data_sources.tiger_roads)
        boundary_path = Path(config_obj.data_sources.conus_boundary)

        uac_status = "✅" if uac_path.exists() else "❌"
        tiger_status = "✅" if tiger_path.exists() else "❌"

        print(f"UAC data: {uac_status} {uac_path}")
        print(f"TIGER roads: {tiger_status} {tiger_path}")

        print(
            f"CONUS boundary: {'✅' if boundary_path.exists() else '❌'} {boundary_path}"
        )

        if not all(path.exists() for path in (uac_path, tiger_path, boundary_path)):
            print("\n⚠️  Missing data files - download required before generation")

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


def main() -> None:
    """Parse arguments, configure logging, and dispatch the selected command."""
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

    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser(
        "build", help="Build a NetGraph scenario from a saved corridor graph"
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
        help="Output directory or YAML file (default: '<config_stem>_scenario.yml' in CWD).",
    )
    build_parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the validated scenario YAML to stdout",
    )
    build_parser.add_argument(
        "--debug-dir",
        default=None,
        help=("Optional directory to write the generated traffic matrices as JSON"),
    )

    build_parser.set_defaults(func=build_command)

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate a metro-to-metro corridor graph"
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

    args = parser.parse_args()

    import logging

    from topogen.log_config import set_global_log_level

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    set_global_log_level(log_level)

    if args.func is None:
        parser.print_help()
        sys.exit(1)

    if args.quiet:
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            args.func(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
