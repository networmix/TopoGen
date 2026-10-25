"""Tests for serialization of site metadata from the MultiGraph."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

from topogen.config import TopologyConfig
from topogen.library import load_library
from topogen.scenario import build_scenario


def _single_metro_graph() -> nx.Graph:
    g = nx.MultiGraph()
    metro = (100.0, 200.0)
    g.add_node(
        metro,
        node_type="metro",
        name="Denver",
        metro_id="metro_001",
        x=100.0,
        y=200.0,
        radius_km=50.0,
        name_orig="Denver",
    )
    return g


def test_sites_section_includes_per_site_blueprint_and_components():
    graph = _single_metro_graph()

    cfg = TopologyConfig()
    cfg.traffic.enabled = False
    # Two PoPs, one DC region
    cfg.build.build_defaults.pop_per_metro = 2
    cfg.build.build_defaults.dc_regions_per_metro = 1
    cfg.build.build_defaults.site_blueprint = "SingleRouter"
    cfg.build.build_defaults.dc_region_blueprint = "DCRegion"

    # Select any available workflow so the scenario builder can proceed without assuming names
    cfg.workflows.assignments.default = next(
        iter(load_library(Path("lib")).workflows.keys())
    )
    yaml_str = build_scenario(graph, cfg, load_library(Path("lib"))).yaml
    data = yaml.safe_load(yaml_str)

    # Network groups must reflect the configured blueprints and site counts.
    network = data["network"]
    groups = network.get("nodes", {})
    assert "metro1/pop[1-2]" in groups
    assert groups["metro1/pop[1-2]"]["blueprint"] == "SingleRouter"
    assert "metro1/dc[1-1]" in groups
    assert groups["metro1/dc[1-1]"]["blueprint"] == "DCRegion"
