"""Integration tests for the build functionality."""

import networkx as nx
import pytest
import yaml

from topogen.config import TopologyConfig
from topogen.scenario_builder import build_scenario


class TestBuildIntegration:
    """Integration tests for the complete build process."""

    @pytest.fixture
    def sample_integrated_graph(self):
        """Create a sample integrated graph for testing."""
        graph = nx.Graph()

        # Add metro nodes
        denver = (100.0, 200.0)
        slc = (150.0, 250.0)
        phoenix = (75.0, 150.0)

        graph.add_node(
            denver,
            node_type="metro",
            name="Denver",
            metro_id="metro_001",
            x=100.0,
            y=200.0,
            radius_km=50.0,
        )
        graph.add_node(
            slc,
            node_type="metro+highway",
            name="Salt Lake City",
            metro_id="metro_002",
            x=150.0,
            y=250.0,
            radius_km=40.0,
        )
        graph.add_node(
            phoenix,
            node_type="metro",
            name="Phoenix",
            metro_id="metro_003",
            x=75.0,
            y=150.0,
            radius_km=45.0,
        )

        # Add some highway nodes
        highway1 = (125.0, 225.0)
        highway2 = (90.0, 180.0)
        graph.add_node(highway1, node_type="highway")
        graph.add_node(highway2, node_type="highway")

        # Add corridor edges between metros
        graph.add_edge(denver, slc, length_km=530.0, capacity=400, edge_type="corridor")
        graph.add_edge(
            denver, phoenix, length_km=890.0, capacity=400, edge_type="corridor"
        )
        graph.add_edge(
            slc, phoenix, length_km=650.0, capacity=400, edge_type="corridor"
        )

        # Add highway edges
        graph.add_edge(denver, highway1, length_km=25.0)
        graph.add_edge(highway1, slc, length_km=30.0)
        graph.add_edge(phoenix, highway2, length_km=20.0)

        return graph

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        config = TopologyConfig()
        config.build.build_defaults.pop_per_metro = 2
        config.build.build_defaults.site_blueprint = "SingleRouter"
        config.build.build_overrides = {
            "Denver": {"pop_per_metro": 4, "site_blueprint": "Clos_64_256"},
            "Salt Lake City": {"site_blueprint": "FullMesh4"},
        }
        return config

    def test_build_scenario_complete(self, sample_integrated_graph, sample_config):
        """Test complete scenario building with realistic data."""
        yaml_str = build_scenario(sample_integrated_graph, sample_config)

        # Parse the YAML to ensure it's valid
        scenario_data = yaml.safe_load(yaml_str)

        # Validate top-level structure
        assert "blueprints" in scenario_data
        assert "network" in scenario_data

        # Validate blueprints section
        blueprints = scenario_data["blueprints"]
        expected_blueprints = {"SingleRouter", "Clos_64_256", "FullMesh4", "DCRegion"}
        assert set(blueprints.keys()) == expected_blueprints

        # Validate network structure
        network = scenario_data["network"]
        assert "groups" in network
        assert "adjacency" in network

        # Should have 6 groups (3 metros x 2 types: PoPs + DC regions)
        groups = network["groups"]
        assert len(groups) == 6

        # Separate PoP and DC groups
        pop_groups = [g for g in groups.keys() if "/pop[" in g]
        dc_groups = [g for g in groups.keys() if "/dc[" in g]
        assert len(pop_groups) == 3  # One per metro
        assert len(dc_groups) == 3  # One per metro

        # Check groups use bracket expansion
        for group_name, group_def in groups.items():
            assert "metro" in group_name
            if "/pop[" in group_name:
                assert "pop[1-4]" in group_name  # Max sites is 4 (Denver)
            elif "/dc[" in group_name:
                assert "dc[1-2]" in group_name  # Default DC regions
            assert "use_blueprint" in group_def
            assert "attrs" in group_def

            # Check attrs contain metro information
            attrs = group_def["attrs"]
            assert "metro_name" in attrs
            assert "metro_id" in attrs
            assert "location_x" in attrs
            assert "location_y" in attrs

        # Check adjacency rules
        adjacency = network["adjacency"]
        assert len(adjacency) > 0

        # Should have both intra-metro and inter-metro adjacency
        intra_metro_rules = [adj for adj in adjacency if "intra_metro" in str(adj)]
        inter_metro_rules = [adj for adj in adjacency if "inter_metro" in str(adj)]

        # Denver (4 sites) and Salt Lake City (2 sites) should have intra-metro mesh
        # Phoenix (2 sites) should also have intra-metro mesh
        assert len(intra_metro_rules) == 3  # All 3 metros have >1 site

        # Should have inter-metro corridor connectivity
        assert len(inter_metro_rules) == 3  # 3 corridor edges in sample graph

    def test_scenario_yaml_format(self, sample_integrated_graph, sample_config):
        """Test that generated YAML follows expected format."""
        yaml_str = build_scenario(sample_integrated_graph, sample_config)

        # Should be valid YAML
        scenario_data = yaml.safe_load(yaml_str)
        assert scenario_data is not None

        # Check specific formatting expectations
        lines = yaml_str.split("\n")

        # Should start with blueprints section
        assert any(line.strip() == "blueprints:" for line in lines)

        # Should have network section
        assert any(line.strip() == "network:" for line in lines)

        # Should have groups and adjacency subsections
        assert any(line.strip() == "groups:" for line in lines)
        assert any(line.strip() == "adjacency:" for line in lines)

    def test_metro_override_application(self, sample_integrated_graph, sample_config):
        """Test that metro overrides are correctly applied."""
        yaml_str = build_scenario(sample_integrated_graph, sample_config)
        scenario_data = yaml.safe_load(yaml_str)

        # Check that correct blueprints are used based on overrides
        blueprints = scenario_data["blueprints"]

        # Should include all blueprints referenced in config
        assert "SingleRouter" in blueprints  # Default for Phoenix
        assert "Clos_64_256" in blueprints  # Override for Denver
        assert "FullMesh4" in blueprints  # Override for Salt Lake City

        # Check group configurations match overrides
        groups = scenario_data["network"]["groups"]

        # Denver should support 4 sites (from override)
        denver_group = None
        for _group_name, group_def in groups.items():
            if group_def["attrs"]["metro_name"] == "Denver":
                denver_group = group_def
                break

        assert denver_group is not None
        assert denver_group["use_blueprint"] == "Clos_64_256"

        # Salt Lake City should use FullMesh4 blueprint
        slc_group = None
        for _group_name, group_def in groups.items():
            if group_def["attrs"]["metro_name"] == "Salt Lake City":
                slc_group = group_def
                break

        assert slc_group is not None
        assert slc_group["use_blueprint"] == "FullMesh4"

    def test_corridor_connectivity_preservation(
        self, sample_integrated_graph, sample_config
    ):
        """Test that corridor connectivity between metros is preserved."""
        yaml_str = build_scenario(sample_integrated_graph, sample_config)
        scenario_data = yaml.safe_load(yaml_str)

        adjacency = scenario_data["network"]["adjacency"]

        # Find inter-metro corridor rules
        corridor_rules = []
        for rule in adjacency:
            link_params = rule.get("link_params", {})
            attrs = link_params.get("attrs", {})
            if attrs.get("link_type") == "inter_metro_corridor":
                corridor_rules.append(rule)

        # Should have 3 corridor connections (triangle of metros)
        assert len(corridor_rules) == 3

        # Check that corridor distances are preserved
        for rule in corridor_rules:
            attrs = rule["link_params"]["attrs"]
            distance = attrs["distance_km"]
            source_metro = attrs["source_metro"]
            target_metro = attrs["target_metro"]

            # Validate distance matches expected corridor lengths
            if {source_metro, target_metro} == {"Denver", "Salt Lake City"}:
                assert distance == 530.0
            elif {source_metro, target_metro} == {"Denver", "Phoenix"}:
                assert distance == 890.0
            elif {source_metro, target_metro} == {"Salt Lake City", "Phoenix"}:
                assert distance == 650.0
            else:
                pytest.fail(f"Unexpected metro pair: {source_metro} - {target_metro}")

    def test_empty_graph_handling(self):
        """Test handling of empty or minimal graphs."""
        # Create config with no overrides for empty graph
        config = TopologyConfig()
        config.build.build_defaults.pop_per_metro = 2
        config.build.build_defaults.site_blueprint = "SingleRouter"
        config.build.build_overrides = {}

        # Empty graph
        empty_graph = nx.Graph()
        yaml_str = build_scenario(empty_graph, config)
        scenario_data = yaml.safe_load(yaml_str)

        assert "blueprints" in scenario_data
        assert "network" in scenario_data
        assert len(scenario_data["network"]["groups"]) == 0
        assert len(scenario_data["network"]["adjacency"]) == 0

    def test_single_metro_graph(self):
        """Test handling of graph with single metro."""
        # Create config with overrides only for Denver
        config = TopologyConfig()
        config.build.build_defaults.pop_per_metro = 2
        config.build.build_defaults.site_blueprint = "SingleRouter"
        config.build.build_overrides = {
            "Denver": {"pop_per_metro": 4, "site_blueprint": "Clos_64_256"}
        }

        graph = nx.Graph()
        metro = (100.0, 200.0)
        graph.add_node(
            metro,
            node_type="metro",
            name="Denver",
            metro_id="metro_001",
            x=100.0,
            y=200.0,
            radius_km=50.0,
        )

        yaml_str = build_scenario(graph, config)
        scenario_data = yaml.safe_load(yaml_str)

        # Should have two groups: one PoP group and one DC group
        groups = scenario_data["network"]["groups"]
        assert len(groups) == 2

        pop_groups = [g for g in groups.keys() if "/pop[" in g]
        dc_groups = [g for g in groups.keys() if "/dc[" in g]
        assert len(pop_groups) == 1
        assert len(dc_groups) == 1

        # Should have intra-metro adjacency but no inter-metro
        adjacency = scenario_data["network"]["adjacency"]
        intra_rules = [adj for adj in adjacency if "intra_metro" in str(adj)]
        inter_rules = [adj for adj in adjacency if "inter_metro" in str(adj)]

        assert len(intra_rules) == 1  # Denver has 4 sites from override
        assert len(inter_rules) == 0  # No other metros to connect to
