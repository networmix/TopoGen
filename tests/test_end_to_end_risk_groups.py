"""End-to-end integration tests for risk group functionality."""

import networkx as nx
import yaml

from topogen.config import (
    BuildConfig,
    BuildDefaults,
    CorridorsConfig,
    RiskGroupsConfig,
    TopologyConfig,
)
from topogen.corridors import (
    CorridorPath,
    extract_corridor_graph,
)
from topogen.corridors import (
    assign_risk_groups as assign_risk_groups_to_corridors,
)
from topogen.metro_clusters import MetroCluster
from topogen.scenario_builder import build_scenario


class TestEndToEndRiskGroups:
    """Test complete risk group workflow from corridor discovery to scenario generation."""

    def test_complete_risk_group_workflow(self):
        """Test complete workflow: corridor discovery → risk assignment → scenario generation."""
        # Step 1: Create test highway network with metro anchors
        highway_graph = nx.Graph()

        # Highway backbone: A --- B --- C --- D
        highway_graph.add_edge((0.0, 0.0), (500.0, 0.0), length_km=500.0)  # A-B
        highway_graph.add_edge((500.0, 0.0), (1000.0, 0.0), length_km=500.0)  # B-C
        highway_graph.add_edge((1000.0, 0.0), (1500.0, 0.0), length_km=500.0)  # C-D

        # Branch from B: B --- E
        highway_graph.add_edge((500.0, 0.0), (500.0, 500.0), length_km=500.0)  # B-E

        # Step 2: Create metros
        metros = [
            MetroCluster(
                "metro1",
                "denver-aurora",
                "Denver--Aurora, CO",
                "001",
                100.0,
                0.0,
                0.0,
                25.0,
            ),
            MetroCluster(
                "metro2",
                "kansas-city",
                "Kansas City, MO--KS",
                "002",
                100.0,
                1000.0,
                0.0,
                25.0,
            ),
            MetroCluster(
                "metro3", "omaha", "Omaha, NE--IA", "003", 100.0, 1500.0, 0.0, 25.0
            ),
            MetroCluster(
                "metro4",
                "minneapolis-st-paul",
                "Minneapolis--St. Paul, MN",
                "004",
                100.0,
                500.0,
                500.0,
                25.0,
            ),
        ]

        # Step 3: Add metro nodes to highway graph
        for metro in metros:
            highway_graph.add_node(
                metro.node_key,
                node_type="metro",
                name=metro.name,
                name_orig=metro.name_orig,
                metro_id=metro.metro_id,
                x=metro.centroid_x,
                y=metro.centroid_y,
                radius_km=metro.radius_km,
            )

            # Add anchor edges (metros connected to highway nodes)
            anchor_point = metro.node_key  # For simplicity, metro is at highway node
            if anchor_point in highway_graph.nodes:
                # Already added as highway node, just update type
                highway_graph.nodes[anchor_point]["node_type"] = "metro+highway"

        # Step 4: Simulate corridor discovery by adding corridor tags
        # Denver → Kansas City corridor (uses A-B-C)
        highway_graph[(0.0, 0.0)][(500.0, 0.0)]["corridor"] = [
            {
                "metro_a": "metro1",
                "metro_b": "metro2",
                "path_index": 0,
                "distance_km": 1000.0,
            }
        ]
        highway_graph[(500.0, 0.0)][(1000.0, 0.0)]["corridor"] = [
            {
                "metro_a": "metro1",
                "metro_b": "metro2",
                "path_index": 0,
                "distance_km": 1000.0,
            }
        ]

        # Kansas City → Omaha corridor (uses C-D)
        highway_graph[(1000.0, 0.0)][(1500.0, 0.0)]["corridor"] = [
            {
                "metro_a": "metro2",
                "metro_b": "metro3",
                "path_index": 0,
                "distance_km": 500.0,
            }
        ]

        # Kansas City → Minneapolis corridor (uses B-E, shares B-C with Denver-KC)
        highway_graph[(500.0, 0.0)][(1000.0, 0.0)]["corridor"].append(
            {
                "metro_a": "metro2",
                "metro_b": "metro4",
                "path_index": 0,
                "distance_km": 707.0,
            }
        )
        highway_graph[(500.0, 0.0)][(500.0, 500.0)]["corridor"] = [
            {
                "metro_a": "metro2",
                "metro_b": "metro4",
                "path_index": 0,
                "distance_km": 707.0,
            }
        ]

        # Step 5: Assign risk groups
        config = CorridorsConfig()
        config.risk_groups = RiskGroupsConfig(
            enabled=True,
            group_prefix="corridor_risk",
            exclude_metro_radius_shared=False,  # Include all for this test
        )

        assign_risk_groups_to_corridors(highway_graph, metros, config)

        # Step 6: Prepare minimal registry for extraction (one shortest path per pair)
        registry = {}
        # metro1-metro2 via two edges total 1000km
        pid_12 = ("metro1", "metro2", 0)
        registry[pid_12] = CorridorPath(
            metros=("metro1", "metro2"),
            path_index=0,
            nodes=[metros[0].node_key, metros[1].node_key],
            edges=[((0.0, 0.0), (500.0, 0.0)), ((500.0, 0.0), (1000.0, 0.0))],
            segment_ids=[],
            length_km=1000.0,
            geometry=[metros[0].node_key, metros[1].node_key],
        )
        # metro2-metro3 via one edge 500km
        pid_23 = ("metro2", "metro3", 0)
        registry[pid_23] = CorridorPath(
            metros=("metro2", "metro3"),
            path_index=0,
            nodes=[metros[1].node_key, metros[2].node_key],
            edges=[((1000.0, 0.0), (1500.0, 0.0))],
            segment_ids=[],
            length_km=500.0,
            geometry=[metros[1].node_key, metros[2].node_key],
        )
        # metro2-metro4 via 2 edges ~707km
        pid_24 = ("metro2", "metro4", 0)
        registry[pid_24] = CorridorPath(
            metros=("metro2", "metro4"),
            path_index=0,
            nodes=[metros[1].node_key, metros[3].node_key],
            edges=[((500.0, 0.0), (1000.0, 0.0)), ((500.0, 0.0), (500.0, 500.0))],
            segment_ids=[],
            length_km=707.0,
            geometry=[metros[1].node_key, metros[3].node_key],
        )
        highway_graph.graph["corridor_paths"] = registry
        # Tag edges with path membership for risk aggregation
        highway_graph[(0.0, 0.0)][(500.0, 0.0)]["corridor_path_ids"] = {pid_12}
        highway_graph[(500.0, 0.0)][(1000.0, 0.0)]["corridor_path_ids"] = {
            pid_12,
            pid_24,
        }
        highway_graph[(1000.0, 0.0)][(1500.0, 0.0)]["corridor_path_ids"] = {pid_23}
        highway_graph[(500.0, 0.0)][(500.0, 500.0)]["corridor_path_ids"] = {pid_24}

        corridor_graph = extract_corridor_graph(highway_graph, metros)

        # Step 7: Generate scenario
        scenario_config = TopologyConfig()
        scenario_config.build = BuildConfig(
            build_defaults=BuildDefaults(pop_per_metro=2, site_blueprint="SingleRouter")
        )
        scenario_config.corridors = config

        yaml_str = build_scenario(corridor_graph, scenario_config)
        scenario_data = yaml.safe_load(yaml_str)

        # Step 8: Verify end-to-end results

        # Should have risk groups section
        assert "risk_groups" in scenario_data
        risk_groups = {rg["name"] for rg in scenario_data["risk_groups"]}

        # Should have risk groups for each corridor
        assert "corridor_risk_denver-aurora_kansas-city" in risk_groups
        assert "corridor_risk_kansas-city_omaha" in risk_groups
        assert "corridor_risk_kansas-city_minneapolis-st-paul" in risk_groups

        # Should have metro groups with correct naming (both PoPs and DC regions)
        groups = scenario_data["network"]["groups"]
        metro_groups = [
            g for g in groups.values() if "metro_name" in g.get("attrs", {})
        ]
        assert len(metro_groups) == 8  # 4 metros x 2 types (PoPs + DC regions)

        # Check metro name attributes
        metro_names = {g["attrs"]["metro_name"] for g in metro_groups}
        assert "denver-aurora" in metro_names
        assert "kansas-city" in metro_names
        assert "omaha" in metro_names
        assert "minneapolis-st-paul" in metro_names

        # Should have corridor adjacencies with risk groups
        adjacency = scenario_data["network"]["adjacency"]
        corridor_links = [
            adj
            for adj in adjacency
            if adj.get("link_params", {}).get("attrs", {}).get("link_type")
            == "inter_metro_corridor"
        ]

        assert len(corridor_links) >= 3  # At least 3 corridors

        # Find the Kansas City - Denver corridor (should have shared risk)
        kc_denver_link = None
        for link in corridor_links:
            attrs = link["link_params"]["attrs"]
            if (
                attrs["source_metro"] == "kansas-city"
                and attrs["target_metro"] == "denver-aurora"
            ) or (
                attrs["source_metro"] == "denver-aurora"
                and attrs["target_metro"] == "kansas-city"
            ):
                kc_denver_link = link
                break

        assert kc_denver_link is not None

        # Kansas City - Denver should have shared risk from Minneapolis corridor
        kc_denver_risks = set(kc_denver_link["link_params"]["risk_groups"])
        assert "corridor_risk_denver-aurora_kansas-city" in kc_denver_risks
        assert (
            "corridor_risk_kansas-city_minneapolis-st-paul" in kc_denver_risks
        )  # Shared risk

    def test_metro_radius_exclusion_integration(self):
        """Test that metro radius exclusion works in the complete workflow."""
        # Create highway network with edges near metro
        highway_graph = nx.Graph()

        # Highway edge very close to metro center (passes through center at y=1000)
        close_edge = (
            (990.0, 1000.0),
            (1010.0, 1000.0),
        )  # EPSG:5070 coordinates in meters

        highway_graph.add_edge(*close_edge, length_km=20.0)
        highway_graph.add_edge(
            (1200000.0, 1000.0), (2000000.0, 1000.0), length_km=800.0
        )

        # Add metros
        metros = [
            MetroCluster(
                "metro1", "test-metro", "Test Metro", "001", 100.0, 1000.0, 1000.0, 50.0
            ),  # 50km radius
            MetroCluster(
                "metro2", "far-metro", "Far Metro", "002", 100.0, 2000.0, 1000.0, 25.0
            ),
        ]

        for metro in metros:
            highway_graph.add_node(
                metro.node_key,
                node_type="metro",
                name=metro.name,
                metro_id=metro.metro_id,
                x=metro.centroid_x,
                y=metro.centroid_y,
                radius_km=metro.radius_km,
            )

        # Add corridor tags to both edges
        highway_graph[close_edge[0]][close_edge[1]]["corridor"] = [
            {
                "metro_a": "metro1",
                "metro_b": "metro2",
                "path_index": 0,
                "distance_km": 1010.0,
            }
        ]
        highway_graph[(1200000.0, 1000.0)][(2000000.0, 1000.0)]["corridor"] = [
            {
                "metro_a": "metro1",
                "metro_b": "metro2",
                "path_index": 0,
                "distance_km": 1010.0,
            }
        ]

        # Configure with metro radius exclusion enabled
        config = CorridorsConfig()
        config.risk_groups = RiskGroupsConfig(
            enabled=True,
            exclude_metro_radius_shared=True,  # Enable exclusion
        )

        assign_risk_groups_to_corridors(highway_graph, metros, config)

        # Close edge should NOT have risk groups (within metro radius)
        close_edge_data = highway_graph[close_edge[0]][close_edge[1]]
        assert (
            "risk_groups" not in close_edge_data
            or len(close_edge_data.get("risk_groups", [])) == 0
        )

        # Far edge should have risk groups
        far_edge_data = highway_graph[(1200000.0, 1000.0)][(2000000.0, 1000.0)]
        assert "risk_groups" in far_edge_data
        assert len(far_edge_data["risk_groups"]) > 0

        # Minimal registry for this pair
        pid = ("metro1", "metro2", 0)
        highway_graph.graph["corridor_paths"] = {
            pid: CorridorPath(
                metros=("metro1", "metro2"),
                path_index=0,
                nodes=[metros[0].node_key, metros[1].node_key],
                edges=[(close_edge[0], close_edge[1])],
                segment_ids=[],
                length_km=1010.0,
                geometry=[metros[0].node_key, metros[1].node_key],
            )
        }
        # Tag only far edge with path membership to keep its risks
        highway_graph[(1200000.0, 1000.0)][(2000000.0, 1000.0)]["corridor_path_ids"] = {
            pid
        }

        corridor_graph = extract_corridor_graph(highway_graph, metros)

        scenario_config = TopologyConfig()
        scenario_config.build = BuildConfig(
            build_defaults=BuildDefaults(pop_per_metro=1, site_blueprint="SingleRouter")
        )
        scenario_config.corridors = config

        yaml_str = build_scenario(corridor_graph, scenario_config)
        scenario_data = yaml.safe_load(yaml_str)

        # Should still have risk groups in scenario (from far edge)
        assert "risk_groups" in scenario_data
        assert len(scenario_data["risk_groups"]) > 0

    def test_risk_group_naming_consistency_end_to_end(self):
        """Test that risk group naming is consistent throughout the complete workflow."""
        # Create simple 3-metro network
        highway_graph = nx.Graph()

        metros = [
            MetroCluster(
                "01171",
                "albuquerque",
                "Albuquerque, NM",
                "01171",
                100.0,
                0.0,
                0.0,
                25.0,
            ),
            MetroCluster(
                "23527",
                "denver-aurora",
                "Denver--Aurora, CO",
                "23527",
                100.0,
                1000000.0,
                0.0,
                25.0,
            ),
            MetroCluster(
                "43912",
                "kansas-city",
                "Kansas City, MO--KS",
                "43912",
                100.0,
                2000000.0,
                0.0,
                25.0,
            ),
        ]

        for metro in metros:
            highway_graph.add_node(
                metro.node_key,
                node_type="metro",
                name=metro.name,
                metro_id=metro.metro_id,
            )

        # Add highway edges connecting metros
        highway_graph.add_edge(
            (100000.0, 0.0),
            (900000.0, 0.0),
            corridor=[
                {
                    "metro_a": "01171",
                    "metro_b": "23527",
                    "path_index": 0,
                    "distance_km": 1000.0,
                }
            ],
        )
        highway_graph.add_edge(
            (1100000.0, 0.0),
            (1900000.0, 0.0),
            corridor=[
                {
                    "metro_a": "23527",
                    "metro_b": "43912",
                    "path_index": 0,
                    "distance_km": 1000.0,
                }
            ],
        )

        # Edge shared by both corridors (creates shared risk)
        highway_graph.add_edge(
            (500000.0, 0.0),
            (1500000.0, 0.0),
            corridor=[
                {
                    "metro_a": "01171",
                    "metro_b": "23527",
                    "path_index": 1,
                    "distance_km": 1000.0,
                },
                {
                    "metro_a": "23527",
                    "metro_b": "43912",
                    "path_index": 1,
                    "distance_km": 1000.0,
                },
            ],
        )

        config = CorridorsConfig()
        config.risk_groups = RiskGroupsConfig(
            enabled=True, exclude_metro_radius_shared=False
        )

        assign_risk_groups_to_corridors(highway_graph, metros, config)
        # Registry for shared-risk paths
        pid_12_0 = ("01171", "23527", 0)
        pid_12_1 = ("01171", "23527", 1)
        pid_23_1 = ("23527", "43912", 1)
        highway_graph.graph["corridor_paths"] = {
            pid_12_0: CorridorPath(
                metros=("01171", "23527"),
                path_index=0,
                nodes=[metros[0].node_key, metros[1].node_key],
                edges=[((100000.0, 0.0), (900000.0, 0.0))],
                segment_ids=[],
                length_km=1000.0,
                geometry=[metros[0].node_key, metros[1].node_key],
            ),
            pid_12_1: CorridorPath(
                metros=("01171", "23527"),
                path_index=1,
                nodes=[metros[0].node_key, metros[1].node_key],
                edges=[((500000.0, 0.0), (1500000.0, 0.0))],
                segment_ids=[],
                length_km=1000.0,
                geometry=[metros[0].node_key, metros[1].node_key],
            ),
            pid_23_1: CorridorPath(
                metros=("23527", "43912"),
                path_index=1,
                nodes=[metros[1].node_key, metros[2].node_key],
                edges=[((500000.0, 0.0), (1500000.0, 0.0))],
                segment_ids=[],
                length_km=1000.0,
                geometry=[metros[1].node_key, metros[2].node_key],
            ),
        }

        # Tag edges for chosen paths
        highway_graph[(100000.0, 0.0)][(900000.0, 0.0)]["corridor_path_ids"] = {
            pid_12_0
        }
        highway_graph[(1100000.0, 0.0)][(1900000.0, 0.0)]["corridor_path_ids"] = {
            pid_23_1
        }
        # Shared edge carries both second paths
        highway_graph[(500000.0, 0.0)][(1500000.0, 0.0)]["corridor_path_ids"] = {
            pid_12_1,
            pid_23_1,
        }

        corridor_graph = extract_corridor_graph(highway_graph, metros)

        scenario_config = TopologyConfig()
        scenario_config.build = BuildConfig(
            build_defaults=BuildDefaults(pop_per_metro=1, site_blueprint="SingleRouter")
        )
        scenario_config.corridors = config

        yaml_str = build_scenario(corridor_graph, scenario_config)
        scenario_data = yaml.safe_load(yaml_str)

        # Check that risk group names use sanitized metro names consistently
        risk_group_names = {rg["name"] for rg in scenario_data["risk_groups"]}

        # Should use alphabetically sorted sanitized names
        assert "corridor_risk_albuquerque_denver-aurora" in risk_group_names
        assert "corridor_risk_denver-aurora_kansas-city" in risk_group_names

        # Should handle multiple paths
        assert "corridor_risk_albuquerque_denver-aurora_path1" in risk_group_names
        assert "corridor_risk_denver-aurora_kansas-city_path1" in risk_group_names

        # Check adjacency links have correct risk groups assigned
        adjacency = scenario_data["network"]["adjacency"]
        corridor_links = [
            adj
            for adj in adjacency
            if adj.get("link_params", {}).get("attrs", {}).get("link_type")
            == "inter_metro_corridor"
        ]

        # Find the Denver-Kansas City link
        denver_kc_link = None
        for link in corridor_links:
            attrs = link["link_params"]["attrs"]
            if (
                attrs["source_metro"] == "denver-aurora"
                and attrs["target_metro"] == "kansas-city"
            ):
                denver_kc_link = link
                break
            elif (
                attrs["source_metro"] == "kansas-city"
                and attrs["target_metro"] == "denver-aurora"
            ):
                denver_kc_link = link
                break

        assert denver_kc_link is not None

        # Should have both its own risk group and shared risk from Albuquerque corridor
        link_risks = set(denver_kc_link["link_params"]["risk_groups"])
        assert "corridor_risk_denver-aurora_kansas-city" in link_risks
        assert (
            "corridor_risk_albuquerque_denver-aurora_path1" in link_risks
        )  # Shared infrastructure (path1 only)
