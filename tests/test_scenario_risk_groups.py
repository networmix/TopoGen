"""Functional tests for risk groups in scenario generation."""

import networkx as nx
import yaml

from topogen.config import (
    BuildConfig,
    BuildDefaults,
    CorridorsConfig,
    RiskGroupsConfig,
    TopologyConfig,
)
from topogen.scenario_builder import _build_risk_groups_section, build_scenario


class TestScenarioRiskGroups:
    """Test risk group functionality in scenario generation."""

    def test_risk_groups_section_generation(self):
        """Test that risk groups section is properly generated."""
        # Create test graph with metro nodes and risk groups
        graph = nx.Graph()
        metro1 = (100.0, 200.0)
        metro2 = (200.0, 300.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="denver-aurora",
            metro_id="23527",
            radius_km=30.0,
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="kansas-city",
            metro_id="43912",
            radius_km=25.0,
        )

        # Add corridor edge with risk groups
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

        risk_groups = _build_risk_groups_section(graph, config)

        assert len(risk_groups) == 1
        rg = risk_groups[0]
        assert rg["name"] == "corridor_risk_denver-aurora_kansas-city"
        assert rg["attrs"]["type"] == "corridor_risk"

    def test_risk_groups_in_scenario_yaml(self):
        """Test that risk groups appear in generated scenario YAML."""
        # Create test graph
        graph = nx.Graph()
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

        # Add corridor edge with risk groups
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

        # Configure scenario
        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(
                sites_per_metro=2, site_blueprint="SingleRouter"
            )
        )
        config.corridors = CorridorsConfig()
        config.corridors.risk_groups = RiskGroupsConfig(enabled=True)

        # Generate scenario
        yaml_str = build_scenario(graph, config)
        scenario_data = yaml.safe_load(yaml_str)

        # Check risk groups section exists
        assert "risk_groups" in scenario_data
        risk_groups = scenario_data["risk_groups"]
        assert len(risk_groups) == 1
        assert risk_groups[0]["name"] == "corridor_risk_denver-aurora_kansas-city"

        # Check risk groups are assigned to links
        adjacency = scenario_data["network"]["adjacency"]
        corridor_links = [
            adj
            for adj in adjacency
            if adj.get("link_params", {}).get("attrs", {}).get("link_type")
            == "inter_metro_corridor"
        ]
        assert len(corridor_links) > 0

        corridor_link = corridor_links[0]
        assert "risk_groups" in corridor_link["link_params"]
        assert (
            "corridor_risk_denver-aurora_kansas-city"
            in corridor_link["link_params"]["risk_groups"]
        )

    def test_multiple_risk_groups_per_link(self):
        """Test that links can have multiple risk groups assigned."""
        graph = nx.Graph()
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
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="metro2",
            metro_id="002",
            x=200.0,
            y=300.0,
            radius_km=25.0,
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
                "corridor_risk_metro1_metro3",
                "corridor_risk_metro2_metro3",
            ],
        )

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(
                sites_per_metro=1, site_blueprint="SingleRouter"
            )
        )
        config.corridors = CorridorsConfig()
        config.corridors.risk_groups = RiskGroupsConfig(enabled=True)

        yaml_str = build_scenario(graph, config)
        scenario_data = yaml.safe_load(yaml_str)

        # Should have all 3 risk groups defined
        assert len(scenario_data["risk_groups"]) == 3
        risk_group_names = {rg["name"] for rg in scenario_data["risk_groups"]}
        assert "corridor_risk_metro1_metro2" in risk_group_names
        assert "corridor_risk_metro1_metro3" in risk_group_names
        assert "corridor_risk_metro2_metro3" in risk_group_names

        # Link should have all 3 risk groups assigned
        corridor_links = [
            adj
            for adj in scenario_data["network"]["adjacency"]
            if adj.get("link_params", {}).get("attrs", {}).get("link_type")
            == "inter_metro_corridor"
        ]
        corridor_link = corridor_links[0]
        link_risk_groups = corridor_link["link_params"]["risk_groups"]
        assert len(link_risk_groups) == 3
        assert set(link_risk_groups) == risk_group_names

    def test_risk_groups_disabled(self):
        """Test that no risk groups are generated when disabled."""
        graph = nx.Graph()
        metro1 = (100.0, 200.0)
        metro2 = (200.0, 300.0)

        graph.add_node(
            metro1, node_type="metro", name="metro1", metro_id="001", radius_km=25.0
        )
        graph.add_node(
            metro2, node_type="metro", name="metro2", metro_id="002", radius_km=25.0
        )
        graph.add_edge(metro1, metro2, edge_type="corridor", length_km=100.0)

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(
                sites_per_metro=1, site_blueprint="SingleRouter"
            )
        )
        config.corridors = CorridorsConfig()
        config.corridors.risk_groups = RiskGroupsConfig(enabled=False)  # Disabled

        yaml_str = build_scenario(graph, config)
        scenario_data = yaml.safe_load(yaml_str)

        # Should not have risk groups section
        assert "risk_groups" not in scenario_data

        # Links should not have risk groups
        adjacency = scenario_data["network"]["adjacency"]
        for adj in adjacency:
            link_params = adj.get("link_params", {})
            assert "risk_groups" not in link_params

    def test_risk_groups_only_on_corridor_edges(self):
        """Test that risk groups are only collected from metro-to-metro edges."""
        graph = nx.Graph()
        metro1 = (100.0, 200.0)
        metro2 = (200.0, 300.0)
        highway = (300.0, 400.0)

        graph.add_node(
            metro1, node_type="metro", name="metro1", metro_id="001", radius_km=20.0
        )
        graph.add_node(
            metro2, node_type="metro", name="metro2", metro_id="002", radius_km=20.0
        )
        graph.add_node(highway, node_type="highway")  # Not a metro

        # Metro-to-metro edge with risk groups
        graph.add_edge(
            metro1,
            metro2,
            edge_type="corridor",
            risk_groups=["corridor_risk_metro1_metro2"],
        )

        # Metro-to-highway edge with risk groups (should be ignored)
        graph.add_edge(
            metro1, highway, edge_type="metro_anchor", risk_groups=["should_be_ignored"]
        )

        config = TopologyConfig()
        config.corridors = CorridorsConfig()
        config.corridors.risk_groups = RiskGroupsConfig(enabled=True)

        risk_groups = _build_risk_groups_section(graph, config)

        # Should only find the metro-to-metro risk group
        assert len(risk_groups) == 1
        assert risk_groups[0]["name"] == "corridor_risk_metro1_metro2"

    def test_metro_name_attributes_in_scenario(self):
        """Test that both sanitized and original metro names appear in scenario."""
        graph = nx.Graph()
        metro1 = (100.0, 200.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="denver-aurora",  # Sanitized
            name_orig="Denver--Aurora, CO",  # Original
            metro_id="23527",
            x=100.0,
            y=200.0,
            radius_km=35.0,
        )

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(
                sites_per_metro=1, site_blueprint="SingleRouter"
            )
        )

        yaml_str = build_scenario(graph, config)
        scenario_data = yaml.safe_load(yaml_str)

        # Find the metro group
        groups = scenario_data["network"]["groups"]
        metro_group = list(groups.values())[0]

        # Should have both sanitized and original names
        attrs = metro_group["attrs"]
        assert attrs["metro_name"] == "denver-aurora"  # Sanitized (primary)
        assert attrs["metro_name_orig"] == "Denver--Aurora, CO"  # Original (display)
