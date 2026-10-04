"""Assemble NetGraph scenario sections and serialize them to YAML."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from topogen.context import RunContext
from topogen.log_config import get_logger
from topogen.traffic_matrix import generate_traffic_matrix

from .artifacts import export_artifacts
from .config import _determine_metro_settings
from .expansion import resolve_network
from .graph_pipeline import build_site_graph
from .libraries import _build_blueprints_section, _build_components_section
from .network import _extract_metros_from_graph, to_network_sections
from .policies import _build_failure_policy_set_section, _build_workflow_section
from .risk import _build_risk_groups_section
from .sizing import tm_based_size_capacities
from .traffic import to_demand_sets

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    import networkx as nx

    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def _emit_yaml(scenario: dict[str, Any], *, yaml_anchors: bool = True) -> str:
    """Serialize a scenario with optional YAML anchors."""
    if yaml_anchors:
        yaml_output = yaml.safe_dump(
            scenario, sort_keys=False, default_flow_style=False
        )
    else:

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        yaml_output = yaml.dump(
            scenario,
            Dumper=NoAliasDumper,
            sort_keys=False,
            default_flow_style=False,
        )
    logger.info("Generated NetGraph scenario YAML")
    return yaml_output


def build_scenario(
    graph: "nx.MultiGraph",
    config: "TopologyConfig",
    *,
    context: RunContext | None = None,
) -> str:
    """Build NetGraph scenario YAML from a metro-to-metro corridor graph.

    Expand metros into PoP and DC sites, size links, and resolve hardware and
    scenario libraries. Write site-graph JSON when a run context is supplied,
    and export maps when configured. The caller validates the returned YAML
    with ``validate_scenario_yaml``; this function does not run workflows.
    """
    metros = _extract_metros_from_graph(graph)
    metro_settings = _determine_metro_settings(metros, config)
    used_blueprints = {
        settings[key]
        for settings in metro_settings.values()
        for count, key in (
            ("pop_per_metro", "site_blueprint"),
            ("dc_regions_per_metro", "dc_region_blueprint"),
        )
        if settings[count] > 0
    }
    blueprints = _build_blueprints_section(used_blueprints, config)
    site_graph = build_site_graph(metros, metro_settings, graph, blueprints)
    traffic_matrices = generate_traffic_matrix(metros, metro_settings, config)
    tm_based_size_capacities(
        site_graph, metros, metro_settings, config, traffic_matrices
    )

    groups, adjacency = to_network_sections(site_graph, metros, metro_settings, config)
    scenario: dict[str, Any] = {
        "seed": config.output.scenario_seed,
        "blueprints": blueprints,
        "components": _build_components_section(config, blueprints),
        "network": {"nodes": groups, "links": adjacency},
    }
    node_rules = site_graph.graph.get("node_overrides", [])
    if node_rules:
        scenario["network"]["node_rules"] = node_rules
    risk_groups = _build_risk_groups_section(graph, config)
    if risk_groups:
        scenario["risk_groups"] = risk_groups
    workflow = _build_workflow_section(config)
    scenario["failures"] = _build_failure_policy_set_section(config, workflow)
    if traffic_matrices:
        scenario["demands"] = to_demand_sets(traffic_matrices)
    scenario["workflow"] = workflow

    network = resolve_network(site_graph, scenario, config.components.optics)
    if context is not None:
        export_artifacts(
            site_graph,
            network,
            config,
            context,
            traffic_matrices,
        )
    return _emit_yaml(scenario, yaml_anchors=config.output.formatting.yaml_anchors)
