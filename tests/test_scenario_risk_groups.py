"""Functional tests for risk groups in scenario generation."""

from pathlib import Path

import networkx as nx
import pytest
import yaml

from netlab_topogen.config import (
    BuildConfig,
    BuildDefaults,
    CorridorsConfig,
    RiskGroupsConfig,
    TopologyConfig,
)
from netlab_topogen.library import load_library
from netlab_topogen.scenario import build_scenario
from netlab_topogen.scenario.document import _risk_groups
from netlab_topogen.scenario.sites import read_corridors


class TestScenarioRiskGroups:
    def test_risk_groups_section_generation(self):
        graph = nx.MultiGraph()
        metro1 = (100.0, 200.0)
        metro2 = (200.0, 300.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="denver-aurora",
            metro_id="23527",
            radius_km=30.0,
            name_orig="denver-aurora",
            x=100.0,
            y=200.0,
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="kansas-city",
            metro_id="43912",
            radius_km=25.0,
            name_orig="kansas-city",
            x=200.0,
            y=300.0,
        )

        graph.add_edge(
            metro1,
            metro2,
            edge_type="corridor",
            length_km=500.0,
            risk_groups=["corridor_risk_denver-aurora_kansas-city"],
        )

        config = TopologyConfig()
        config.corridors = CorridorsConfig()
        config.corridors.risk_groups = RiskGroupsConfig(enabled=True)

        risk_groups = _risk_groups(read_corridors(graph), config)

        assert len(risk_groups) == 1
        rg = risk_groups[0]
        assert rg["name"] == "corridor_risk_denver-aurora_kansas-city"
        assert rg["attrs"]["type"] == "corridor_risk"
        # distance_km should be present and equal to ceil(length_km)
        assert rg["attrs"]["distance_km"] == 500

    def test_risk_groups_in_scenario_yaml(self):
        graph = nx.MultiGraph()
        metro1 = (100.0, 200.0)
        metro2 = (200.0, 300.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="denver-aurora",
            name_orig="Denver--Aurora, CO",
            metro_id="23527",
            x=100.0,
            y=200.0,
            radius_km=35.0,
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="kansas-city",
            name_orig="Kansas City, MO--KS",
            metro_id="43912",
            x=200.0,
            y=300.0,
            radius_km=30.0,
        )

        graph.add_edge(
            metro1,
            metro2,
            edge_type="corridor",
            length_km=500.0,
            capacity=400,
            metro_a="23527",
            metro_b="43912",
            risk_groups=["corridor_risk_denver-aurora_kansas-city"],
        )

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(pop_per_metro=2, site_blueprint="SingleRouter")
        )
        config.corridors = CorridorsConfig()
        config.corridors.risk_groups = RiskGroupsConfig(enabled=True)

        config.workflows.assignments.default = next(
            iter(load_library(Path("lib")).workflows.keys())
        )
        yaml_str = build_scenario(graph, config, load_library(Path("lib"))).yaml
        scenario_data = yaml.safe_load(yaml_str)

        assert "risk_groups" in scenario_data
        risk_groups = scenario_data["risk_groups"]
        assert len(risk_groups) == 1
        assert risk_groups[0]["name"] == "corridor_risk_denver-aurora_kansas-city"
        assert risk_groups[0]["attrs"]["distance_km"] == 500

        adjacency = scenario_data["network"]["links"]
        corridor_links = [
            adj
            for adj in adjacency
            if adj.get("attrs", {}).get("link_type") == "inter_metro_corridor"
        ]
        assert len(corridor_links) > 0

        corridor_link = corridor_links[0]
        assert "risk_groups" in corridor_link
        assert "corridor_risk_denver-aurora_kansas-city" in corridor_link["risk_groups"]

    def test_multiple_risk_groups_per_link(self):
        graph = nx.MultiGraph()
        metro1 = (100.0, 200.0)
        metro2 = (200.0, 300.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="metro1",
            metro_id="001",
            x=100.0,
            y=200.0,
            radius_km=25.0,
            name_orig="metro1",
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="metro2",
            metro_id="002",
            x=200.0,
            y=300.0,
            radius_km=25.0,
            name_orig="metro2",
        )

        # Add corridor with multiple risk groups (shared infrastructure)
        graph.add_edge(
            metro1,
            metro2,
            edge_type="corridor",
            length_km=150.0,
            capacity=400,
            risk_groups=[
                "corridor_risk_metro1_metro2",
                "corridor_risk_metro1_metro2_path1",
                "corridor_risk_metro1_metro2_path2",
            ],
        )

        for key, length in ((1, 100.0), (2, 200.0)):
            graph.add_edge(
                metro1, metro2, key=key, edge_type="corridor", length_km=length
            )

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(pop_per_metro=1, site_blueprint="SingleRouter")
        )
        config.corridors = CorridorsConfig()
        config.corridors.risk_groups = RiskGroupsConfig(enabled=True)

        config.workflows.assignments.default = next(
            iter(load_library(Path("lib")).workflows.keys())
        )
        yaml_str = build_scenario(graph, config, load_library(Path("lib"))).yaml
        scenario_data = yaml.safe_load(yaml_str)

        assert len(scenario_data["risk_groups"]) == 3
        risk_group_names = {rg["name"] for rg in scenario_data["risk_groups"]}
        assert "corridor_risk_metro1_metro2" in risk_group_names
        assert "corridor_risk_metro1_metro2_path1" in risk_group_names
        assert "corridor_risk_metro1_metro2_path2" in risk_group_names
        assert [rg["attrs"]["distance_km"] for rg in scenario_data["risk_groups"]] == [
            150,
            100,
            200,
        ]

        corridor_links = [
            adj
            for adj in scenario_data["network"]["links"]
            if adj.get("attrs", {}).get("link_type") == "inter_metro_corridor"
        ]
        corridor_link = corridor_links[0]
        link_risk_groups = corridor_link["risk_groups"]
        assert len(link_risk_groups) == 3
        assert set(link_risk_groups) == risk_group_names

    def test_risk_groups_disabled(self):
        graph = nx.MultiGraph()
        metro1 = (100.0, 200.0)
        metro2 = (200.0, 300.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="metro1",
            metro_id="001",
            radius_km=25.0,
            name_orig="metro1",
            x=100.0,
            y=200.0,
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="metro2",
            metro_id="002",
            radius_km=25.0,
            name_orig="metro2",
            x=200.0,
            y=300.0,
        )
        graph.add_edge(metro1, metro2, edge_type="corridor", length_km=100.0)

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(pop_per_metro=1, site_blueprint="SingleRouter")
        )
        config.corridors = CorridorsConfig()
        config.corridors.risk_groups = RiskGroupsConfig(enabled=False)

        config.workflows.assignments.default = next(
            iter(load_library(Path("lib")).workflows.keys())
        )
        yaml_str = build_scenario(graph, config, load_library(Path("lib"))).yaml
        scenario_data = yaml.safe_load(yaml_str)

        assert "risk_groups" not in scenario_data

        adjacency = scenario_data["network"]["links"]
        for adj in adjacency:
            assert "risk_groups" not in adj

    def test_unknown_risk_owner_is_rejected(self):
        graph = nx.MultiGraph()
        graph.add_node(
            0,
            node_type="metro",
            name="a",
            name_orig="a",
            metro_id="a",
            x=0.0,
            y=0.0,
            radius_km=1.0,
        )
        graph.add_node(
            1,
            node_type="metro",
            name="b",
            name_orig="b",
            metro_id="b",
            x=1000.0,
            y=0.0,
            radius_km=1.0,
        )
        graph.add_edge(0, 1, length_km=10.0, risk_groups=["unknown"])
        with pytest.raises(ValueError, match="no owning corridor path"):
            _risk_groups(read_corridors(graph), TopologyConfig())

    def test_metro_name_attributes_in_scenario(self):
        graph = nx.MultiGraph()
        metro1 = (100.0, 200.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="denver-aurora",
            name_orig="Denver--Aurora, CO",
            metro_id="23527",
            x=100.0,
            y=200.0,
            radius_km=35.0,
        )

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(pop_per_metro=1, site_blueprint="SingleRouter")
        )

        config.workflows.assignments.default = next(
            iter(load_library(Path("lib")).workflows.keys())
        )
        yaml_str = build_scenario(graph, config, load_library(Path("lib"))).yaml
        scenario_data = yaml.safe_load(yaml_str)

        groups = scenario_data["network"]["nodes"]
        metro_group = list(groups.values())[0]

        attrs = metro_group["attrs"]
        assert attrs["metro_name"] == "denver-aurora"  # Sanitized (primary)
        assert attrs["metro_name_orig"] == "Denver--Aurora, CO"  # Original (display)
