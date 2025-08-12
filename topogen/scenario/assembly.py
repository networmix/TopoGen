"""Scenario assembly orchestrator and YAML post-processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.log_config import get_logger

from .config import _determine_metro_settings
from .graph_pipeline import (
    assign_per_link_capacity,
    build_site_graph,
    to_network_sections,
)
from .libraries import _build_blueprints_section, _build_components_section
from .network import _extract_metros_from_graph
from .policies import _build_failure_policy_set_section, _build_workflow_section
from .risk import _build_risk_groups_section, _build_traffic_matrix_section

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    import networkx as nx

    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def build_scenario(graph: "nx.Graph", config: "TopologyConfig") -> str:
    """Build a NetGraph scenario YAML from an integrated metro-highway graph.

    Transforms each metro node into a hierarchical site structure using blueprint
    templates, preserving corridor connectivity between metros.
    """
    logger.info("Building NetGraph scenario from integrated graph")

    metros = _extract_metros_from_graph(graph)
    logger.info(f"Found {len(metros)} metro nodes")

    metro_settings = _determine_metro_settings(metros, config)
    max_sites = max((s["pop_per_metro"] for s in metro_settings.values()), default=1)
    max_dc_regions = max(
        (s["dc_regions_per_metro"] for s in metro_settings.values()), default=0
    )
    logger.info(f"Maximum sites per metro: {max_sites}")
    logger.info(f"Maximum DC regions per metro: {max_dc_regions}")

    used_blueprints = set(s["site_blueprint"] for s in metro_settings.values())
    used_blueprints.update(
        s["dc_region_blueprint"]
        for s in metro_settings.values()
        if s["dc_regions_per_metro"] > 0
    )
    builtin_blueprints = get_builtin_blueprints()
    for bp_name in used_blueprints:
        if bp_name not in builtin_blueprints:
            available = ", ".join(sorted(builtin_blueprints.keys()))
            raise ValueError(f"Unknown blueprint '{bp_name}'. Available: {available}")

    scenario: dict[str, Any] = {}

    try:
        scenario_seed = int(getattr(config.output, "scenario_seed", 42))
    except Exception:
        scenario_seed = 42
    scenario["seed"] = scenario_seed

    scenario["blueprints"] = _build_blueprints_section(used_blueprints, config)
    scenario["components"] = _build_components_section(config, used_blueprints)

    # New graph-based pipeline
    logger.info("Building site-level MultiGraph")
    G = build_site_graph(metros, metro_settings, graph, config)
    logger.info("Assigning capacities from defaults split by adjacency")
    assign_per_link_capacity(G, config)
    logger.info("Serializing network sections from MultiGraph")
    groups, adjacency = to_network_sections(G, metros, metro_settings, config)
    scenario["network"] = {"groups": groups, "adjacency": adjacency}

    # Persist the site-level network graph as an artefact
    try:
        from pathlib import Path

        from .graph_pipeline import save_site_graph_json

        output_dir = Path.cwd()
        prefix_path = getattr(config, "_source_path", None)
        stem = Path(prefix_path).stem if isinstance(prefix_path, Path) else "scenario"
        network_graph_path = output_dir / f"{stem}_network_graph.json"
        logger.info(f"Saving site-level network graph to {network_graph_path}")
        json_indent = int(
            getattr(
                getattr(config, "output", object()), "formatting", object()
            ).json_indent  # type: ignore[attr-defined]
            if hasattr(getattr(config, "output", object()), "formatting")
            else 2
        )
        save_site_graph_json(G, network_graph_path, json_indent=json_indent)
    except Exception as e:  # pragma: no cover - best-effort artefact save
        logger.warning(f"Failed to save site-level network graph: {e}")

    risk_groups = _build_risk_groups_section(graph, config)
    if risk_groups:
        scenario["risk_groups"] = risk_groups

    scenario["failure_policy_set"] = _build_failure_policy_set_section(config)
    traffic_section = _build_traffic_matrix_section(metros, metro_settings, config)
    if traffic_section:
        scenario["traffic_matrix_set"] = traffic_section
    scenario["workflow"] = _build_workflow_section(config)

    try:
        emit_anchors = bool(config.output.formatting.yaml_anchors)
    except Exception:
        emit_anchors = True
    if emit_anchors:
        yaml_output = yaml.safe_dump(
            scenario, sort_keys=False, default_flow_style=False
        )
    else:

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):  # type: ignore[override]
                return True

        yaml_output = yaml.dump(
            scenario,
            Dumper=NoAliasDumper,
            sort_keys=False,
            default_flow_style=False,
        )

    yaml_output = _add_adjacency_comments(yaml_output)
    logger.info("Generated NetGraph scenario YAML")
    return yaml_output


def _add_adjacency_comments(yaml_content: str) -> str:
    """Add section comments to the adjacency section of the YAML."""
    lines = yaml_content.split("\n")
    result_lines: list[str] = []
    in_adjacency = False
    intra_metro_added = False
    inter_metro_added = False
    for i, line in enumerate(lines):
        if line.strip() == "adjacency:" and not in_adjacency:
            in_adjacency = True
            result_lines.append(line)
            continue
        if (
            in_adjacency
            and line
            and not line.startswith(" ")
            and not line.startswith("-")
        ):
            in_adjacency = False
        if in_adjacency and line.strip().startswith("- source:"):
            link_type = None
            for j in range(i, min(i + 15, len(lines))):
                if "link_type: intra_metro" in lines[j]:
                    link_type = "intra_metro"
                    break
                elif "link_type: inter_metro_corridor" in lines[j]:
                    link_type = "inter_metro"
                    break
            if link_type == "intra_metro" and not intra_metro_added:
                result_lines.append(
                    "  # Intra-metro adjacency (connectivity within each metro's sites)"
                )
                intra_metro_added = True
            elif link_type == "inter_metro" and not inter_metro_added:
                result_lines.append(
                    "  # Inter-metro corridor connectivity (backbone links between metros)"
                )
                inter_metro_added = True
        result_lines.append(line)
    return "\n".join(result_lines)
