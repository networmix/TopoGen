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
            radius_km=10.0,
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
                pop_per_metro=3, site_blueprint="TestBlueprint"
            ),
            build_overrides={},
        )

        settings = _determine_metro_settings(metros, config)

        assert len(settings) == 2
        assert settings["Denver"]["pop_per_metro"] == 3
        assert settings["Denver"]["site_blueprint"] == "TestBlueprint"
        assert settings["Salt Lake City"]["pop_per_metro"] == 3
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
                pop_per_metro=2, site_blueprint="SingleRouter"
            ),
            build_overrides={
                "Denver": {"pop_per_metro": 4, "site_blueprint": "Clos_64_256"},
                "Salt Lake City": {
                    "site_blueprint": "FullMesh4"
                },  # Only blueprint override
            },
        )

        settings = _determine_metro_settings(metros, config)

        # Denver should have both overrides
        assert settings["Denver"]["pop_per_metro"] == 4
        assert settings["Denver"]["site_blueprint"] == "Clos_64_256"

        # Salt Lake City should have blueprint override but default sites
        assert settings["Salt Lake City"]["pop_per_metro"] == 2
        assert settings["Salt Lake City"]["site_blueprint"] == "FullMesh4"

    def test_determine_metro_settings_unknown_metro(self):
        """Test metro settings fails for unknown metro in overrides."""
        metros = [{"name": "Denver", "metro_id": "metro_001"}]

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(),
            build_overrides={"NonExistentMetro": {"pop_per_metro": 4}},
        )

        with pytest.raises(ValueError, match="unknown metro 'NonExistentMetro'"):
            _determine_metro_settings(metros, config)

    def test_determine_metro_settings_invalid_sites(self):
        """Test metro settings fails for invalid pop_per_metro."""
        metros = [{"name": "Denver", "metro_id": "metro_001"}]

        config = TopologyConfig()
        config.build = BuildConfig(
            build_defaults=BuildDefaults(),
            build_overrides={
                "Denver": {"pop_per_metro": 0}  # Invalid
            },
        )

        with pytest.raises(ValueError, match="invalid pop_per_metro"):
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
                pop_per_metro=2,
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

        graph.add_node(metro1, node_type="metro", name="Metro1", radius_km=10.0)
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
                pop_per_metro=2, site_blueprint="SingleRouter"
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
            radius_km=10.0,
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
                pop_per_metro=2,
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

    def test_gravity_traffic_totals_and_structure(self):
        """Gravity model emits explicit fixed per-pair entries with conserved totals."""
        # Build simple graph with two metros and 1 DC each for determinism
        graph = nx.Graph()
        m1 = (0.0, 0.0)
        m2 = (300_000.0, 0.0)  # 300 km apart in EPSG:5070 meters
        graph.add_node(
            m1,
            node_type="metro",
            name="A",
            metro_id="m1",
            x=m1[0],
            y=m1[1],
            radius_km=10.0,
        )
        graph.add_node(
            m2,
            node_type="metro",
            name="B",
            metro_id="m2",
            x=m2[0],
            y=m2[1],
            radius_km=10.0,
        )
        graph.add_edge(m1, m2, length_km=300.0, capacity=100)

        cfg = TopologyConfig()
        cfg.build = BuildConfig(
            build_defaults=BuildDefaults(
                pop_per_metro=1,
                site_blueprint="SingleRouter",
                dc_regions_per_metro=1,
                dc_region_blueprint="DCRegion",
            )
        )
        # Traffic: gravity model, deterministic (no jitter, no rounding)
        cfg.traffic.enabled = True
        cfg.traffic.model = "gravity"
        cfg.traffic.gbps_per_mw = 100.0
        cfg.traffic.mw_per_dc_region = 10.0
        cfg.traffic.priority_ratios = {0: 1.0}
        cfg.traffic.gravity.alpha = 1.0
        cfg.traffic.gravity.beta = 1.0
        cfg.traffic.gravity.min_distance_km = 1.0
        cfg.traffic.gravity.exclude_same_metro = False
        cfg.traffic.gravity.jitter_stddev = 0.0
        cfg.traffic.gravity.rounding_gbps = 0.0

        yaml_str = build_scenario(graph, cfg)
        scenario = yaml.safe_load(yaml_str)

        assert "traffic_matrix_set" in scenario
        tm = scenario["traffic_matrix_set"][cfg.traffic.matrix_name]
        # Two symmetric entries A->B and B->A for one class
        assert len(tm) == 2
        assert all(e["mode"] == "pairwise" for e in tm)
        total = sum(float(e["demand"]) for e in tm)
        # Offered = gbps_per_mw * sum(MW) = 100 * 20 = 2000; split by class 1.0
        # We split equally A->B and B->A, so total across both = 2000
        assert abs(total - 2000.0) < 1e-6

    def test_hw_capacity_allocation_round_robin(self):
        """HW-aware allocation distributes extra capacity to inter-metro links."""
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
            radius_km=10.0,
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="Salt Lake City",
            metro_id="metro_002",
            x=150.0,
            y=250.0,
            radius_km=10.0,
        )

        # Base corridor capacity per POP pair is 100
        graph.add_edge(metro1, metro2, length_km=100.0, capacity=100)

        # Configuration: 1 POP per metro, no DCs. Base intra=0 avoids reservation.
        cfg = TopologyConfig()
        cfg.build = BuildConfig(
            build_defaults=BuildDefaults(
                pop_per_metro=1,
                site_blueprint="SingleRouter",
                dc_regions_per_metro=0,
                intra_metro_link=BuildDefaults().intra_metro_link,
                inter_metro_link=BuildDefaults().inter_metro_link,
                dc_to_pop_link=BuildDefaults().dc_to_pop_link,
            ),
            build_overrides={},
        )
        # Enable HW capacity allocation
        cfg.build.capacity_allocation.enabled = True

        # Assign built-in component CoreRouter for simplicity; capacity is ample
        cfg.components.assignments.core.hw_component = "CoreRouter"

        # Build scenario and check updated capacity on adjacency
        yaml_str = build_scenario(graph, cfg)
        scenario = yaml.safe_load(yaml_str)
        net = scenario["network"]

        # Expect exactly one per-pair adjacency with final capacity
        match = [
            a
            for a in net["adjacency"]
            if a.get("pattern") == "one_to_one"
            and a.get("source") == "metro1/pop1"
            and a.get("target") == "metro2/pop1"
        ]
        assert len(match) == 1
        # With ample platform capacity, capacity should be >= base (100)
        assert match[0]["link_params"]["capacity"] >= 100
        # No link_overrides expected in this minimal case
        assert net.get("link_overrides", []) == []

    def test_hw_capacity_overcommit_raises(self):
        """Overcommit should raise when base > platform and flag is set."""
        graph = nx.Graph()
        metro1 = (100.0, 200.0)
        metro2 = (150.0, 250.0)

        graph.add_node(
            metro1,
            node_type="metro",
            name="A",
            metro_id="m1",
            radius_km=10.0,
        )
        graph.add_node(
            metro2,
            node_type="metro",
            name="B",
            metro_id="m2",
            radius_km=10.0,
        )
        # Base corridor capacity 300 per POP pair
        graph.add_edge(metro1, metro2, capacity=300)

        cfg = TopologyConfig()
        cfg.build.capacity_allocation.enabled = True
        # 1 POP per metro
        cfg.build.build_defaults.pop_per_metro = 1
        # With built-in large capacity, overcommit should not raise
        cfg.components.assignments.core.hw_component = "CoreRouter"
        build_scenario(graph, cfg)
