"""Build immutable scenario results and publish validated text."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from netlab_topogen.config import TopologyConfig
from netlab_topogen.library import Library

from .document import check_components, scenario_document, to_yaml
from .hardware import assign_hardware, expand_sites
from .publication import write_atomic
from .records import Build
from .sites import (
    finalize_blueprints,
    plan_sites,
    read_corridors,
    resolve_metro_settings,
)
from .sizing import size_links
from .striping import stripe_groups
from .traffic import generate_traffic


def build_scenario(
    graph: nx.MultiGraph, config: TopologyConfig, library: Library
) -> Build:
    """Build scenario text and records, without reading or writing files."""
    corridors = read_corridors(graph)
    settings = resolve_metro_settings(corridors, config.build, config.traffic)
    blueprints = finalize_blueprints(settings, library, config.components.hw_component)
    stripes = stripe_groups(corridors, settings, blueprints)
    plan = plan_sites(corridors, settings, stripes)
    expansion = expand_sites(corridors, plan, blueprints, config)
    traffic = generate_traffic(plan, config.traffic, config.output.scenario_seed)
    sizing = size_links(plan, traffic, config)
    check_components(config.components, blueprints, library)
    hardware = assign_hardware(plan, sizing, expansion, library, config.components)
    document = scenario_document(
        corridors, plan, blueprints, traffic, sizing, hardware, library, config
    )
    return Build(
        to_yaml(document, yaml_anchors=config.output.formatting.yaml_anchors),
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
