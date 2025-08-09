"""Command-line interface for NetGraph."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ngraph.explorer import NetworkExplorer
from ngraph.logging import get_logger, set_global_log_level
from ngraph.profiling import PerformanceProfiler, PerformanceReporter
from ngraph.report import ReportGenerator
from ngraph.scenario import Scenario

logger = get_logger(__name__)


def _format_table(headers: List[str], rows: List[List[str]], min_width: int = 8) -> str:
    """Format data as a simple ASCII table.

    Args:
        headers: Column headers
        rows: Data rows
        min_width: Minimum column width

    Returns:
        Formatted table string
    """
    if not rows:
        return ""

    # Calculate column widths
    all_data = [headers] + rows
    col_widths = []
    for col_idx in range(len(headers)):
        max_width = max(len(str(row[col_idx])) for row in all_data)
        col_widths.append(max(max_width, min_width))

    # Format rows
    def format_row(row_data: List[str]) -> str:
        return "   " + " | ".join(
            f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row_data)
        )

    # Build table
    lines = []
    lines.append(format_row(headers))
    lines.append("   " + "-+-".join("-" * width for width in col_widths))
    for row in rows:
        lines.append(format_row(row))

    return "\n".join(lines)


def _inspect_scenario(path: Path, detail: bool = False) -> None:
    """Inspect a scenario file, validate it, and show key characteristics.

    Args:
        path: Scenario YAML file.
        detail: Whether to show detailed information including sample node names.
    """
    logger.info(f"Inspecting scenario from: {path}")

    try:
        # Load and validate scenario
        yaml_text = path.read_text()
        logger.info("✓ YAML file loaded successfully")

        scenario = Scenario.from_yaml(yaml_text)
        logger.info("✓ Scenario validated and loaded successfully")

        # Show scenario metadata
        print("\n" + "=" * 60)
        print("NETGRAPH SCENARIO INSPECTION")
        print("=" * 60)

        print("\n1. SCENARIO METADATA")
        print("-" * 30)
        if scenario.seed is not None:
            print(f"   Seed: {scenario.seed} (deterministic)")
            print(
                "   All workflow step seeds are derived deterministically from scenario seed"
            )
        else:
            print("   Seed: None (non-deterministic)")
            print("   Workflow step seeds will be random on each run")

        # Network Analysis
        print("\n2. NETWORK STRUCTURE")
        print("-" * 30)

        network = scenario.network
        nodes = network.nodes
        links = network.links

        print(f"   Total Nodes: {len(nodes):,}")
        print(f"   Total Links: {len(links):,}")

        # Show enabled/disabled breakdown
        enabled_nodes = [n for n in nodes.values() if not n.disabled]
        disabled_nodes = [n for n in nodes.values() if n.disabled]
        enabled_links = [link for link in links.values() if not link.disabled]
        disabled_links = [link for link in links.values() if link.disabled]

        print(f"   Enabled Nodes: {len(enabled_nodes):,}")
        if disabled_nodes:
            print(f"   Disabled Nodes: {len(disabled_nodes):,}")

        print(f"   Enabled Links: {len(enabled_links):,}")
        if disabled_links:
            print(f"   Disabled Links: {len(disabled_links):,}")

        # Network hierarchy analysis
        if nodes:
            # Suppress the "Analyzing..." log message during inspect for cleaner output
            original_level = logger.level
            logger.setLevel(logging.WARNING)
            try:
                explorer = NetworkExplorer.explore_network(
                    network, scenario.components_library
                )
                print("\n   Network Hierarchy:")
                explorer.print_tree(
                    max_depth=3 if not detail else None,
                    skip_leaves=not detail,
                    detailed=detail,
                )
            except Exception as e:
                print(f"   Network Hierarchy: Unable to analyze ({e})")
            finally:
                logger.setLevel(original_level)

        # Show complete node and link tables in detail mode
        if detail:
            # Nodes table
            if nodes:
                print("\n   Nodes:")
                node_rows = []
                for node_name in sorted(nodes.keys()):
                    node = nodes[node_name]
                    status = "disabled" if node.disabled else "enabled"

                    # Calculate total capacity and link count for this node
                    node_capacity = 0
                    node_link_count = 0
                    for link in links.values():
                        if link.source == node_name or link.target == node_name:
                            if not link.disabled:
                                node_capacity += link.capacity
                                node_link_count += 1

                    capacity_str = f"{node_capacity:,.0f}" if node_capacity > 0 else "0"

                    node_rows.append(
                        [node_name, status, capacity_str, str(node_link_count)]
                    )

                node_table = _format_table(
                    ["Node", "Status", "Tot. Capacity", "Links"], node_rows
                )
                print(node_table)

            # Links table
            if links:
                print("\n   Links:")
                link_rows = []
                for _link_id, link in links.items():
                    status = "disabled" if link.disabled else "enabled"
                    capacity = f"{link.capacity:,.0f}"

                    # Get cost if available
                    cost = ""
                    if hasattr(link, "cost") and link.cost:
                        cost = f"{link.cost:,.0f}"
                    elif hasattr(link, "attrs") and link.attrs and "cost" in link.attrs:
                        cost = f"{link.attrs['cost']:,.0f}"

                    link_rows.append([link.source, link.target, status, capacity, cost])

                link_table = _format_table(
                    ["Source", "Target", "Status", "Capacity", "Cost"], link_rows
                )
                print(link_table)

        # Link capacity analysis as table
        if links:
            link_caps = [link.capacity for link in enabled_links]
            if link_caps:
                print("\n   Link Capacity Statistics:")
                cap_table = _format_table(
                    ["Metric", "Value"],
                    [
                        ["Min", f"{min(link_caps):,.1f}"],
                        ["Max", f"{max(link_caps):,.1f}"],
                        ["Mean", f"{sum(link_caps) / len(link_caps):,.1f}"],
                        ["Total", f"{sum(link_caps):,.1f}"],
                    ],
                )
                print(cap_table)

        # Node capacity analysis
        if nodes and links:
            print("\n   Node Capacity Statistics:")
            node_capacities = []
            for node_name in nodes.keys():
                node_capacity = 0
                for link in enabled_links:
                    if link.source == node_name or link.target == node_name:
                        node_capacity += link.capacity
                if node_capacity > 0:  # Only include nodes with links
                    node_capacities.append(node_capacity)

            if node_capacities:
                node_cap_table = _format_table(
                    ["Metric", "Value"],
                    [
                        ["Min", f"{min(node_capacities):,.1f}"],
                        ["Max", f"{max(node_capacities):,.1f}"],
                        ["Mean", f"{sum(node_capacities) / len(node_capacities):,.1f}"],
                        ["Total", f"{sum(node_capacities):,.1f}"],
                    ],
                )
                print(node_cap_table)

        # Risk Groups Analysis
        print("\n3. RISK GROUPS")
        print("-" * 30)
        if network.risk_groups:
            print(f"   Total: {len(network.risk_groups)}")
            if detail:
                # Show all risk groups in detail mode
                for rg_name, rg in network.risk_groups.items():
                    status = "disabled" if rg.disabled else "enabled"
                    print(f"     {rg_name} ({status})")
            else:
                # Show first 5 risk groups, then summary
                risk_items = list(network.risk_groups.items())[:5]
                for rg_name, rg in risk_items:
                    status = "disabled" if rg.disabled else "enabled"
                    print(f"     {rg_name} ({status})")
                if len(network.risk_groups) > 5:
                    remaining = len(network.risk_groups) - 5
                    print(f"     ... and {remaining} more")
        else:
            print("   Total: 0")

        # Components Library
        print("\n4. COMPONENTS LIBRARY")
        print("-" * 30)
        comp_count = len(scenario.components_library.components)
        print(f"   Total: {comp_count}")
        if scenario.components_library.components:
            if detail:
                # Show all components in detail mode
                for comp_name in scenario.components_library.components.keys():
                    print(f"     {comp_name}")
            else:
                # Show first 5 components, then summary
                comp_items = list(scenario.components_library.components.keys())[:5]
                for comp_name in comp_items:
                    print(f"     {comp_name}")
                if comp_count > 5:
                    remaining = comp_count - 5
                    print(f"     ... and {remaining} more")

        # Failure Policies Analysis
        print("\n5. FAILURE POLICIES")
        print("-" * 30)
        policy_count = len(scenario.failure_policy_set.policies)
        print(f"   Total: {policy_count}")
        if scenario.failure_policy_set.policies:
            policy_items = list(scenario.failure_policy_set.policies.items())[:5]
            for policy_name, policy in policy_items:
                rules_count = len(policy.rules)
                print(
                    f"     {policy_name}: {rules_count} rule{'s' if rules_count != 1 else ''}"
                )
                if detail and rules_count > 0:
                    for i, rule in enumerate(policy.rules[:3]):  # Show first 3 rules
                        print(f"       {i + 1}. {rule.entity_scope} {rule.rule_type}")
                    if rules_count > 3:
                        print(f"       ... and {rules_count - 3} more rules")
            if policy_count > 5:
                remaining = policy_count - 5
                print(f"     ... and {remaining} more")

        # Traffic Matrices Analysis
        print("\n6. TRAFFIC MATRICES")
        print("-" * 30)
        matrix_count = len(scenario.traffic_matrix_set.matrices)
        print(f"   Total: {matrix_count}")
        if scenario.traffic_matrix_set.matrices:
            matrix_items = list(scenario.traffic_matrix_set.matrices.items())[:5]
            for matrix_name, demands in matrix_items:
                demand_count = len(demands)
                print(
                    f"     {matrix_name}: {demand_count} demand{'s' if demand_count != 1 else ''}"
                )
                if detail and demands:
                    for i, demand in enumerate(demands[:3]):  # Show first 3 demands
                        print(
                            f"       {i + 1}. {demand.source_path} → {demand.sink_path} ({demand.demand})"
                        )
                    if demand_count > 3:
                        print(f"       ... and {demand_count - 3} more demands")
            if matrix_count > 5:
                remaining = matrix_count - 5
                print(f"     ... and {remaining} more")

        # Helper: collect step path-like fields and summarize node matches
        def _collect_step_path_fields(step: Any) -> list[tuple[str, str]]:
            fields: list[tuple[str, str]] = []
            for key, value in step.__dict__.items():
                if key.startswith("_"):
                    continue
                if not isinstance(value, str):
                    continue
                if not value.strip():
                    continue
                if key.endswith("_path") or key.endswith("_regex"):
                    fields.append((key, value))
            return fields

        def _summarize_node_matches(
            step: Any,
            net: Any,
        ) -> Dict[str, Dict[str, Any]]:
            summary: Dict[str, Dict[str, Any]] = {}
            fields = _collect_step_path_fields(step)
            if not fields:
                return summary
            for name, pattern in fields:
                try:
                    groups = net.select_node_groups_by_path(pattern)
                except Exception as exc:  # pragma: no cover (defensive)
                    summary[name] = {
                        "pattern": pattern,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    continue

                group_labels = list(groups.keys())
                total_nodes = sum(len(nodes) for nodes in groups.values())
                enabled_nodes = sum(
                    1 for nodes in groups.values() for nd in nodes if not nd.disabled
                )
                summary[name] = {
                    "pattern": pattern,
                    "groups": len(group_labels),
                    "nodes": total_nodes,
                    "enabled_nodes": enabled_nodes,
                    "labels": group_labels[:5],  # preview up to 5 labels
                }
            return summary

        # Workflow Analysis as table
        print("\n7. WORKFLOW STEPS")
        print("-" * 30)
        step_count = len(scenario.workflow)
        print(f"   Total: {step_count}")
        if scenario.workflow:
            if not detail:
                # Simple table format for basic view
                workflow_rows = []
                for i, step in enumerate(scenario.workflow):
                    step_name = step.name or f"step_{i + 1}"
                    step_type = step.__class__.__name__
                    determinism = (
                        "deterministic" if scenario.seed is not None else "random"
                    )
                    workflow_rows.append(
                        [str(i + 1), step_name, step_type, determinism]
                    )

                workflow_table = _format_table(
                    ["#", "Name", "Type", "Seed"], workflow_rows
                )
                print(workflow_table)

                # Node selection preview for steps with path-like fields
                print("\n   Node selection preview:")
                any_preview = False
                for i, step in enumerate(scenario.workflow):
                    match_info = _summarize_node_matches(step, network)
                    if not match_info:
                        continue
                    any_preview = True
                    parts: list[str] = []
                    for field_name, info in match_info.items():
                        if "error" in info:
                            parts.append(f"{field_name}: ERROR {info['error']}")
                        else:
                            parts.append(
                                f"{field_name}: {info['groups']} groups, {info['nodes']} nodes ({info['enabled_nodes']} enabled)"
                            )
                    label = step.name or step.__class__.__name__
                    print(f"     {i + 1}. {label}: " + "; ".join(parts))
                if not any_preview:
                    print("     (no node selection fields in workflow steps)")
            else:
                # Detailed view with parameters
                for i, step in enumerate(scenario.workflow):
                    step_name = step.name or f"step_{i + 1}"
                    step_type = step.__class__.__name__
                    determinism = (
                        "deterministic" if scenario.seed is not None else "random"
                    )
                    seed_info = (
                        f" (seed: {step.seed}, {determinism})"
                        if step.seed is not None
                        else f" ({determinism})"
                    )
                    print(f"     {i + 1}. {step_name} ({step_type}){seed_info}")

                    # Show step-specific parameters if detail mode
                    step_dict = step.__dict__
                    param_rows = []
                    for key, value in step_dict.items():
                        if key not in ["name", "seed"] and not key.startswith("_"):
                            param_rows.append([key, str(value)])

                    if param_rows:
                        param_table = _format_table(["Parameter", "Value"], param_rows)
                        # Indent the table
                        indented_table = "\n".join(
                            f"        {line}" for line in param_table.split("\n")
                        )
                        print(indented_table)

                    # Show node selection matches for path-like fields
                    match_info = _summarize_node_matches(step, network)
                    if match_info:
                        rows: list[List[str]] = []
                        for field_name, info in match_info.items():
                            if "error" in info:
                                rows.append(
                                    [
                                        field_name,
                                        info["pattern"],
                                        "ERROR",
                                        info["error"],
                                    ]
                                )
                            else:
                                label_preview = (
                                    ", ".join(info["labels"]) if info["labels"] else "-"
                                )
                                rows.append(
                                    [
                                        field_name,
                                        info["pattern"],
                                        f"{info['groups']} groups / {info['nodes']} nodes",
                                        f"{info['enabled_nodes']} enabled; labels: {label_preview}",
                                    ]
                                )
                        if rows:
                            print("        Node matches:")
                            match_table = _format_table(
                                ["Field", "Pattern", "Matches", "Details"], rows
                            )
                            indented = "\n".join(
                                f"        {line}" for line in match_table.split("\n")
                            )
                            print(indented)
                        # Emphasize empty matches
                        if all(
                            (info.get("nodes", 0) == 0) and ("error" not in info)
                            for info in match_info.values()
                        ):
                            print(
                                "        WARNING: No nodes matched for the given patterns"
                            )

        print("\n" + "=" * 60)
        print("INSPECTION COMPLETE")
        print("=" * 60)

        if scenario.workflow:
            print(
                f"\nReady to run with {len(scenario.workflow)} workflow step{'s' if len(scenario.workflow) != 1 else ''}"
            )
            print(f"Usage: python -m ngraph run {path}")
        else:
            print("\nNo workflow steps defined")
            print(
                "This scenario can be used for network analysis but has no automated workflow"
            )

        logger.info("Scenario inspection completed successfully")

    except FileNotFoundError:
        print(f"❌ ERROR: Scenario file not found: {path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to inspect scenario: {e}")
        print("❌ ERROR: Failed to inspect scenario")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)


def _run_scenario(
    path: Path,
    output: Path,
    no_results: bool,
    stdout: bool,
    keys: Optional[list[str]] = None,
    profile: bool = False,
) -> None:
    """Run a scenario file and export results as JSON by default.

    Args:
        path: Scenario YAML file.
        output: Path where JSON results should be written.
        no_results: Whether to disable results file generation.
        stdout: Whether to also print results to stdout.
        keys: Optional list of workflow step names to include. When ``None`` all steps are
            exported.
        profile: Whether to enable performance profiling with CPU analysis.
    """
    logger.info(f"Loading scenario from: {path}")

    try:
        yaml_text = path.read_text()
        scenario = Scenario.from_yaml(yaml_text)

        if profile:
            logger.info("Performance profiling enabled")
            # Initialize detailed profiler
            profiler = PerformanceProfiler()

            # Start scenario-level profiling
            profiler.start_scenario()

            logger.info("Starting scenario execution with profiling")

            # Enable child-process profiling for parallel workflows
            child_profile_dir = Path("worker_profiles")
            child_profile_dir.mkdir(exist_ok=True)
            os.environ["NGRAPH_PROFILE_DIR"] = str(child_profile_dir.resolve())
            logger.info(f"Worker profiles will be saved to: {child_profile_dir}")

            # Manual execution of workflow steps with profiling
            for step in scenario.workflow:
                step_name = step.name or step.__class__.__name__
                step_type = step.__class__.__name__

                with profiler.profile_step(step_name, step_type):
                    step.execute(scenario)

                # Merge any worker profiles generated by this step
                if child_profile_dir.exists():
                    profiler.merge_child_profiles(child_profile_dir, step_name)

            logger.info("Scenario execution completed successfully")

            # End scenario profiling and analyze results
            profiler.end_scenario()
            profiler.analyze_performance()

            # Clean up any remaining worker profile files
            if child_profile_dir.exists():
                remaining_files = list(child_profile_dir.glob("*.pstats"))
                if remaining_files:
                    logger.debug(
                        f"Cleaning up {len(remaining_files)} remaining profile files"
                    )
                    for f in remaining_files:
                        try:
                            f.unlink()
                        except Exception:
                            pass
                try:
                    child_profile_dir.rmdir()  # Remove dir if empty
                except Exception:
                    pass

            # Generate and display performance report
            reporter = PerformanceReporter(profiler.results)
            performance_report = reporter.generate_report()
            print("\n" + performance_report)

        else:
            logger.info("Starting scenario execution")
            scenario.run()
            logger.info("Scenario execution completed successfully")
            print("✅ Scenario execution completed")

        # Export JSON results by default unless disabled
        if not no_results:
            logger.info("Serializing results to JSON")
            results_dict: Dict[str, Dict[str, Any]] = scenario.results.to_dict()

            if keys:
                filtered: Dict[str, Dict[str, Any]] = {}
                for step, data in results_dict.items():
                    if step in keys:
                        filtered[step] = data
                results_dict = filtered

            json_str = json.dumps(results_dict, indent=2, default=str)

            logger.info(f"Writing results to: {output}")
            output.write_text(json_str)
            logger.info("Results written successfully")
            print(f"✅ Results written to: {output}")

            if stdout:
                print(json_str)
        elif stdout:
            # Print to stdout even without file export
            results_dict: Dict[str, Dict[str, Any]] = scenario.results.to_dict()
            if keys:
                filtered: Dict[str, Dict[str, Any]] = {}
                for step, data in results_dict.items():
                    if step in keys:
                        filtered[step] = data
                results_dict = filtered
            json_str = json.dumps(results_dict, indent=2, default=str)
            print(json_str)

    except FileNotFoundError:
        logger.error(f"Scenario file not found: {path}")
        print(f"❌ ERROR: Scenario file not found: {path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to run scenario: {type(e).__name__}: {e}")
        print(f"❌ ERROR: Failed to run scenario: {type(e).__name__}: {e}")
        sys.exit(1)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the ``ngraph`` command.

    Args:
        argv: Optional list of command-line arguments. If ``None``, ``sys.argv``
            is used.
    """
    parser = argparse.ArgumentParser(
        prog="ngraph",
        description="Run and analyze network scenarios.",
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress console output (logs only)"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="Available commands",
        metavar="{run,inspect,report}",
        help="Available commands",
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a scenario")
    run_parser.add_argument("scenario", type=Path, help="Path to scenario YAML")
    run_parser.add_argument(
        "--results",
        "-r",
        type=Path,
        default=Path("results.json"),
        help="Export results to JSON file (default: results.json)",
    )
    run_parser.add_argument(
        "--no-results",
        action="store_true",
        help="Disable results file generation",
    )
    run_parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print results to stdout",
    )
    run_parser.add_argument(
        "--keys",
        "-k",
        nargs="+",
        help="Filter output to these workflow step names",
    )
    run_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling with CPU analysis and bottleneck detection",
    )

    # Inspect command
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect and validate a scenario"
    )
    inspect_parser.add_argument("scenario", type=Path, help="Path to scenario YAML")
    inspect_parser.add_argument(
        "--detail",
        "-d",
        action="store_true",
        help="Show detailed information including complete node/link tables and step parameters",
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Generate analysis reports from results file"
    )
    report_parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=Path("results.json"),
        help="Path to results JSON file (default: results.json)",
    )
    report_parser.add_argument(
        "--notebook",
        "-n",
        type=Path,
        help="Output path for Jupyter notebook (default: analysis.ipynb)",
    )
    report_parser.add_argument(
        "--html",
        type=Path,
        nargs="?",
        const=Path("analysis.html"),
        help="Generate HTML report (default: analysis.html if no path specified)",
    )
    report_parser.add_argument(
        "--include-code",
        action="store_true",
        help="Include code cells in HTML output (default: report without code)",
    )

    # Determine effective arguments (support both direct calls and module entrypoint)
    effective_args = sys.argv[1:] if argv is None else argv

    # If no arguments are provided, show help and exit cleanly
    if not effective_args:
        parser.print_help()
        raise SystemExit(0)

    args = parser.parse_args(effective_args)

    # Configure logging based on arguments
    if args.verbose:
        set_global_log_level(logging.DEBUG)
        logger.debug("Debug logging enabled")
    elif args.quiet:
        set_global_log_level(logging.WARNING)
    else:
        set_global_log_level(logging.INFO)

    if args.command == "run":
        _run_scenario(
            args.scenario,
            args.results,
            args.no_results,
            args.stdout,
            args.keys,
            args.profile,
        )
    elif args.command == "inspect":
        _inspect_scenario(args.scenario, args.detail)
    elif args.command == "report":
        _generate_report(
            args.results,
            args.notebook,
            args.html,
            args.include_code,
        )


def _generate_report(
    results_path: Path,
    notebook_path: Optional[Path],
    html_path: Optional[Path],
    include_code: bool,
) -> None:
    """Generate analysis reports from results file.

    Args:
        results_path: Path to results.json file.
        notebook_path: Output path for notebook (default: analysis.ipynb).
        html_path: Output path for HTML report (None = no HTML).
        include_code: Whether to include code cells in HTML output.
    """
    logger.info(f"Generating report from: {results_path}")

    try:
        # Initialize report generator
        generator = ReportGenerator(results_path)
        generator.load_results()

        # Generate notebook
        notebook_output = notebook_path or Path("analysis.ipynb")
        generated_notebook = generator.generate_notebook(notebook_output)
        print(f"✅ Notebook generated: {generated_notebook}")

        # Generate HTML if requested
        if html_path:
            generated_html = generator.generate_html_report(
                notebook_path=generated_notebook,
                html_path=html_path,
                include_code=include_code,
            )
            print(f"✅ HTML report generated: {generated_html}")

    except FileNotFoundError as e:
        logger.error(f"Results file not found: {e}")
        print(f"❌ ERROR: Results file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid results data: {e}")
        print(f"❌ ERROR: Invalid results data: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Report generation failed: {e}")
        print(f"❌ ERROR: Report generation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ ERROR: Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
