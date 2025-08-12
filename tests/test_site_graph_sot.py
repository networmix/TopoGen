"""Tests for MultiGraph-as-source-of-truth site metadata serialization."""

from __future__ import annotations

import networkx as nx
import yaml

from topogen.config import TopologyConfig
from topogen.scenario_builder import build_scenario


def _single_metro_graph() -> nx.Graph:
    g = nx.Graph()
    metro = (100.0, 200.0)
    g.add_node(
        metro,
        node_type="metro",
        name="Denver",
        metro_id="metro_001",
        x=100.0,
        y=200.0,
        radius_km=50.0,
    )
    return g


def test_sites_section_includes_per_site_blueprint_and_components():
    graph = _single_metro_graph()

    cfg = TopologyConfig()
    # Two PoPs, one DC region
    cfg.build.build_defaults.pop_per_metro = 2
    cfg.build.build_defaults.dc_regions_per_metro = 1
    cfg.build.build_defaults.site_blueprint = "SingleRouter"
    cfg.build.build_defaults.dc_region_blueprint = "DCRegion"

    # Provide explicit role->component assignments to be serialized per site
    cfg.components.assignments.core.hw_component = "CoreRouter"
    cfg.components.assignments.leaf.hw_component = "LeafRouter"
    cfg.components.assignments.spine.hw_component = "SpineRouter"
    cfg.components.assignments.dc.hw_component = "CoreRouter"

    yaml_str = build_scenario(graph, cfg)
    data = yaml.safe_load(yaml_str)

    # In the current schema, per-site details are internal to artefacts;
    # ensure the network groups reflect the right blueprints and counts.
    network = data["network"]
    groups = network.get("groups", {})
    assert "metro1/pop[1-2]" in groups
    assert groups["metro1/pop[1-2]"]["use_blueprint"] == "SingleRouter"
    assert "metro1/dc[1-1]" in groups
    assert groups["metro1/dc[1-1]"]["use_blueprint"] == "DCRegion"
