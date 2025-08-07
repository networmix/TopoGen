"""Tests for the scenario builder module."""

import networkx as nx
import pytest
import yaml

from topogen.config import BuildConfig, BuildDefaults, TopologyConfig
from topogen.scenario_builder import (
    _determine_metro_settings,
    _extract_corridor_edges,
    _extract_metros_from_graph,
    build_scenario,
)


class TestScenarioBuilder:
    """Test cases for the scenario builder functionality."""

    def test_extract_metros_from_graph(self):
        """Test metro extraction from integrated graph."""
        # Create test graph with metro nodes
        graph = nx.Graph()
        graph.add_node(
            (100.0, 200.0),
            node_type="metro",
            name="Denver",
            metro_id="metro_001",
            x=100.0,
            y=200.0,
            radius_km=50.0,
        )
        graph.add_node(
            (150.0, 250.0),
            node_type="metro+highway",
            name="Salt Lake City",
            metro_id="metro_002",
            x=150.0,
            y=250.0,
            radius_km=40.0,
        )
        graph.add_node(
            (200.0, 300.0),
            node_type="highway",  # Not a metro
        )

        metros = _extract_metros_from_graph(graph)

        assert len(metros) == 2

        denver = next(m for m in metros if m["name"] == "Denver")
        assert denver["metro_id"] == "metro_001"
        assert denver["x"] == 100.0
        assert denver["y"] == 200.0
        assert denver["radius_km"] == 50.0

        slc = next(m for m in metros if m["name"] == "Salt Lake City")
        assert slc["metro_id"] == "metro_002"

    def test_extract_metros_missing_attributes(self):
        """Test metro extraction fails with missing required attributes."""
        graph = nx.Graph()
        graph.add_node(
            (100.0, 200.0),
            node_type="metro",
            # Missing 'name' and 'metro_id'
            x=100.0,
            y=200.0,
        )

        with pytest.raises(ValueError, match="missing required attribute"):
            _extract_metros_from_graph(graph)

    def test_determine_metro_settings_defaults(self):
        """Test metro settings determination with defaults only."""
        metros = [
            {"name": "Denver", "metro_id": "metro_001"},
            {"name": "Salt Lake City", "metro_id": "metro_002"},
        ]

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(
                sites_per_metro=3, site_blueprint="TestBlueprint"
            ),
            build_overrides={},
        )

        settings = _determine_metro_settings(metros, config)

        assert len(settings) == 2
        assert settings["Denver"]["sites_per_metro"] == 3
        assert settings["Denver"]["site_blueprint"] == "TestBlueprint"
        assert settings["Salt Lake City"]["sites_per_metro"] == 3
        assert settings["Salt Lake City"]["site_blueprint"] == "TestBlueprint"

    def test_determine_metro_settings_with_overrides(self):
        """Test metro settings determination with overrides."""
        metros = [
            {"name": "Denver", "metro_id": "metro_001"},
            {"name": "Salt Lake City", "metro_id": "metro_002"},
        ]

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(
                sites_per_metro=2, site_blueprint="SingleRouter"
            ),
            build_overrides={
                "Denver": {"sites_per_metro": 4, "site_blueprint": "Clos_64_256"},
                "Salt Lake City": {
                    "site_blueprint": "FullMesh4"
                },  # Only blueprint override
            },
        )

        settings = _determine_metro_settings(metros, config)

        # Denver should have both overrides
        assert settings["Denver"]["sites_per_metro"] == 4
        assert settings["Denver"]["site_blueprint"] == "Clos_64_256"

        # Salt Lake City should have blueprint override but default sites
        assert settings["Salt Lake City"]["sites_per_metro"] == 2
        assert settings["Salt Lake City"]["site_blueprint"] == "FullMesh4"

    def test_determine_metro_settings_unknown_metro(self):
        """Test metro settings fails for unknown metro in overrides."""
        metros = [{"name": "Denver", "metro_id": "metro_001"}]

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(),
            build_overrides={"NonExistentMetro": {"sites_per_metro": 4}},
        )

        with pytest.raises(ValueError, match="unknown metro 'NonExistentMetro'"):
            _determine_metro_settings(metros, config)

    def test_determine_metro_settings_invalid_sites(self):
        """Test metro settings fails for invalid sites_per_metro."""
        metros = [{"name": "Denver", "metro_id": "metro_001"}]

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(),
            build_overrides={
                "Denver": {"sites_per_metro": 0}  # Invalid
            },
        )

        with pytest.raises(ValueError, match="invalid sites_per_metro"):
            _determine_metro_settings(metros, config)

    def test_determine_metro_settings_dc_regions(self):
        """Test metro settings with DC regions configuration."""
        metros = [
            {"name": "Denver", "metro_id": "metro_001"},
            {"name": "Salt Lake City", "metro_id": "metro_002"},
        ]

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(
                sites_per_metro=2,
                site_blueprint="SingleRouter",
                dc_regions_per_metro=2,
                dc_region_blueprint="DCRegion",
            ),
            build_overrides={
                "Denver": {
                    "dc_regions_per_metro": 3,
                    "dc_region_blueprint": "SingleRouter",  # Override to different blueprint
                },
            },
        )

        settings = _determine_metro_settings(metros, config)

        # Denver should have DC region overrides
        assert settings["Denver"]["dc_regions_per_metro"] == 3
        assert settings["Denver"]["dc_region_blueprint"] == "SingleRouter"
        assert "dc_to_pop_link" in settings["Denver"]

        # Salt Lake City should have defaults
        assert settings["Salt Lake City"]["dc_regions_per_metro"] == 2
        assert settings["Salt Lake City"]["dc_region_blueprint"] == "DCRegion"
        assert "dc_to_pop_link" in settings["Salt Lake City"]

    def test_determine_metro_settings_invalid_dc_regions(self):
        """Test metro settings fails for invalid dc_regions_per_metro."""
        metros = [{"name": "Denver", "metro_id": "metro_001"}]

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(),
            build_overrides={
                "Denver": {"dc_regions_per_metro": -1}  # Invalid
            },
        )

        with pytest.raises(ValueError, match="invalid dc_regions_per_metro"):
            _determine_metro_settings(metros, config)

    def test_extract_corridor_edges(self):
        """Test corridor edge extraction from graph."""
        graph = nx.Graph()

        # Add metro nodes
        metro1 = (100.0, 200.0)
        metro2 = (150.0, 250.0)
        highway = (200.0, 300.0)

        graph.add_node(metro1, node_type="metro", name="Metro1")
        graph.add_node(metro2, node_type="metro+highway", name="Metro2")
        graph.add_node(highway, node_type="highway")

        # Add edges
        graph.add_edge(
            metro1, metro2, length_km=100.0, capacity=400, edge_type="corridor"
        )
        graph.add_edge(metro1, highway, length_km=50.0)  # Not metro-to-metro

        corridors = _extract_corridor_edges(graph)

        assert len(corridors) == 1
        corridor = corridors[0]
        assert corridor["source"] == metro1
        assert corridor["target"] == metro2
        assert corridor["length_km"] == 100.0
        assert corridor["capacity"] == 400

    def test_build_scenario_basic(self):
        """Test basic scenario building functionality."""
        # Create minimal test graph
        graph = nx.Graph()
        metro1 = (100.0, 200.0)
        metro2 = (150.0, 250.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="Denver",
            metro_id="metro_001",
            x=100.0,
            y=200.0,
            radius_km=50.0,
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="Salt Lake City",
            metro_id="metro_002",
            x=150.0,
            y=250.0,
            radius_km=40.0,
        )

        # Add corridor between metros
        graph.add_edge(metro1, metro2, length_km=100.0, capacity=400)

        # Create minimal config
        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(
                sites_per_metro=2, site_blueprint="SingleRouter"
            ),
            build_overrides={},
        )

        # Build scenario
        yaml_str = build_scenario(graph, config)

        # Parse and validate YAML
        scenario_data = yaml.safe_load(yaml_str)

        assert "blueprints" in scenario_data
        assert "network" in scenario_data

        # Check blueprints
        assert "SingleRouter" in scenario_data["blueprints"]

        # Check network structure
        network = scenario_data["network"]
        assert "groups" in network
        assert "adjacency" in network

        # Should have groups for metros
        groups = network["groups"]
        pop_groups = [g for g in groups.keys() if "/pop[" in g]
        dc_groups = [g for g in groups.keys() if "/dc[" in g]
        assert len(pop_groups) == 2  # Two metros with PoPs
        assert len(dc_groups) == 2  # Two metros with DC regions

        # Groups should use bracket expansion
        group_names = list(groups.keys())
        assert any("metro1" in name for name in group_names)
        assert any("metro2" in name for name in group_names)

    def test_build_scenario_unknown_blueprint(self):
        """Test scenario building fails for unknown blueprint."""
        graph = nx.Graph()
        graph.add_node(
            (100.0, 200.0),
            node_type="metro",
            name="Denver",
            metro_id="metro_001",
            x=100.0,
            y=200.0,
        )

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(site_blueprint="NonExistentBlueprint"),
            build_overrides={},
        )

        with pytest.raises(
            ValueError, match="Unknown blueprint 'NonExistentBlueprint'"
        ):
            build_scenario(graph, config)

    def test_build_scenario_with_dc_regions(self):
        """Test scenario building with DC regions enabled."""
        # Create minimal test graph
        graph = nx.Graph()
        metro1 = (100.0, 200.0)
        metro2 = (150.0, 250.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="Denver",
            metro_id="metro_001",
            x=100.0,
            y=200.0,
            radius_km=50.0,
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="Salt Lake City",
            metro_id="metro_002",
            x=150.0,
            y=250.0,
            radius_km=40.0,
        )
        graph.add_edge(metro1, metro2, length_km=100.0, capacity=400)

        # Create config with DC regions enabled
        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(
                sites_per_metro=2,
                site_blueprint="SingleRouter",
                dc_regions_per_metro=2,
                dc_region_blueprint="DCRegion",
            )
        )

        scenario_yaml = build_scenario(graph, config)
        scenario_dict = yaml.safe_load(scenario_yaml)

        # Should have both PoP and DC region groups
        groups = scenario_dict["network"]["groups"]
        pop_groups = [g for g in groups.keys() if "/pop[" in g]
        dc_groups = [g for g in groups.keys() if "/dc[" in g]

        assert len(pop_groups) == 2  # One per metro
        assert len(dc_groups) == 2  # One per metro

        # DC groups should use DCRegion blueprint
        for dc_group_name in dc_groups:
            dc_group = groups[dc_group_name]
            assert dc_group["use_blueprint"] == "DCRegion"
            assert "node_type" in dc_group["attrs"]
            assert dc_group["attrs"]["node_type"] == "dc_region"

        # Should have DC-to-PoP adjacency rules
        adjacency = scenario_dict["network"]["adjacency"]
        dc_pop_links = [
            adj
            for adj in adjacency
            if "/dc[" in adj["source"] and "/pop[" in adj["target"]
        ]
        assert len(dc_pop_links) == 2  # One per metro

        # Verify DC-to-PoP link parameters
        for link in dc_pop_links:
            assert link["pattern"] == "mesh"
            assert link["link_params"]["attrs"]["link_type"] == "dc_to_pop"
