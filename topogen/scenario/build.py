"""Build immutable scenario results and publish validated text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from topogen.config import TopologyConfig
from topogen.library import Library, thaw

from .config import _determine_metro_settings
from .graph_pipeline import build_site_graph
from .hardware import assign_hardware, expand_sites
from .libraries import _build_blueprints_section, _build_components_section
from .network import _extract_metros_from_graph, to_network_sections
from .policies import _build_failure_policy_set_section, _build_workflow_section
from .publication import write_atomic
from .records import Build, readonly_graph
from .risk import _build_risk_groups_section
from .sizing import size_links
from .traffic import generate_traffic, to_demand_sets


def _emit_yaml(document: dict[str, Any], *, yaml_anchors: bool) -> str:
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    return yaml.dump(
        document,
        Dumper=yaml.SafeDumper if yaml_anchors else NoAliasDumper,
        sort_keys=False,
        default_flow_style=False,
    )


def build_scenario(
    graph: nx.MultiGraph, config: TopologyConfig, library: Library
) -> Build:
    """Build scenario text and records, without reading or writing files."""
    metros = _extract_metros_from_graph(graph)
    settings = _determine_metro_settings(metros, config)
    used = {
        value[key]
        for value in settings.values()
        for count, key in (
            ("pop_per_metro", "site_blueprint"),
            ("dc_regions_per_metro", "dc_region_blueprint"),
        )
        if value[count] > 0
    }
    blueprints = _build_blueprints_section(used, config, library)
    plan = readonly_graph(build_site_graph(metros, settings, graph, blueprints))
    components = _build_components_section(config, blueprints, library)

    def document(sizing=None, hardware=None):
        groups, links = to_network_sections(
            plan, metros, settings, config, sizing, hardware
        )
        network: dict[str, Any] = {"nodes": groups, "links": links}
        if rules := plan.graph.get("node_overrides"):
            network["node_rules"] = thaw(rules)
        result: dict[str, Any] = {
            "seed": config.output.scenario_seed,
            "blueprints": blueprints,
            "components": components,
            "network": network,
        }
        if risks := _build_risk_groups_section(graph, config):
            result["risk_groups"] = risks
        return result

    expansion = expand_sites(plan, document())
    traffic = generate_traffic(plan, config.traffic, config.output.scenario_seed)
    sizing = size_links(plan, traffic, config)
    hardware = assign_hardware(plan, sizing, expansion, library, config.components)
    result = document(sizing, hardware)
    workflow = _build_workflow_section(config, library)
    result["failures"] = _build_failure_policy_set_section(config, workflow, library)
    if traffic:
        result["demands"] = to_demand_sets(traffic)
    result["workflow"] = workflow
    return Build(
        _emit_yaml(result, yaml_anchors=config.output.formatting.yaml_anchors),
        traffic,
        sizing,
        hardware,
        plan,
    )


def write_scenario(path: Path, text: str) -> None:
    """Atomically publish the supplied complete scenario text."""
    write_atomic(
        path, lambda temporary: temporary.write_text(text, encoding="utf-8") and None
    )
