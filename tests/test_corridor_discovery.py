"""Functional tests for corridor discovery logic."""

import networkx as nx
import pytest

from topogen.config import CorridorsConfig
from topogen.corridors import (
    CorridorPath,
    add_corridors,
    extract_corridor_graph,
)
from topogen.metro_clusters import MetroCluster


class TestCorridorDiscovery:
    def test_basic_corridor_discovery(self):
        graph = nx.Graph()

        # Highway path: A --- B --- C
        graph.add_edge(
            (0.0, 0.0),
            (500.0, 0.0),
            length_km=500.0,
            geometry=[(0.0, 0.0), (500.0, 0.0)],
        )
        graph.add_edge(
            (500.0, 0.0),
            (1000.0, 0.0),
            length_km=500.0,
            geometry=[(500.0, 0.0), (1000.0, 0.0)],
        )

        metros = [
            MetroCluster("metro1", "metro-a", "Metro A", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster(
                "metro2", "metro-b", "Metro B", "002", 100.0, 1000.0, 0.0, 25.0
            ),
        ]

        # Add metro nodes and anchor edges (metro-to-anchor Euclidean edges)
        for metro in metros:
            graph.add_node(
                metro.node_key,
                node_type="metro",
                name=metro.name,
                name_orig=metro.name_orig,
                metro_id=metro.metro_id,
                x=metro.centroid_x,
                y=metro.centroid_y,
                radius_km=metro.radius_km,
            )
            # Anchor at the same coordinate as metro for this test
            anchor_point = metro.node_key
            # Euclidean distance in km (zero in this case)
            graph.add_edge(
                metro.node_key,
                anchor_point,
                edge_type="metro_anchor",
                length_km=0.0,
                geometry=[metro.node_key, anchor_point],
            )

        config = CorridorsConfig()
        config.k_paths = 1
        config.k_nearest = 5
        config.max_edge_km = 2000.0
        config.max_corridor_distance_km = 2000.0

        add_corridors(graph, metros, config)

        corridor_edges = 0
        for _u, _v, data in graph.edges(data=True):
            if "corridor" in data and data["corridor"]:
                corridor_edges += 1
                corridor_info = data["corridor"][0]
                assert "metro_a" in corridor_info
                assert "metro_b" in corridor_info
                assert "path_index" in corridor_info
                assert "distance_km" in corridor_info

        assert corridor_edges == 2

    def test_corridor_discovery_no_path(self):
        graph = nx.Graph()
        graph.add_edge(
            (0.0, 0.0),
            (100.0, 0.0),
            length_km=100.0,
            geometry=[(0.0, 0.0), (100.0, 0.0)],
        )
        graph.add_edge(
            (2000.0, 0.0),
            (2100.0, 0.0),
            length_km=100.0,
            geometry=[(2000.0, 0.0), (2100.0, 0.0)],
        )  # Disconnected

        metros = [
            MetroCluster("metro1", "metro-a", "Metro A", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster(
                "metro2", "metro-b", "Metro B", "002", 100.0, 2000.0, 0.0, 25.0
            ),
        ]

        for metro in metros:
            graph.add_node(
                metro.node_key,
                node_type="metro",
                name=metro.name,
                name_orig=metro.name_orig,
                metro_id=metro.metro_id,
                x=metro.centroid_x,
                y=metro.centroid_y,
                radius_km=metro.radius_km,
            )
            anchor_point = metro.node_key
            graph.add_edge(
                metro.node_key,
                anchor_point,
                edge_type="metro_anchor",
                length_km=0.0,
                geometry=[metro.node_key, anchor_point],
            )

        config = CorridorsConfig()
        config.k_paths = 1
        config.k_nearest = 5
        config.max_edge_km = 3000.0
        config.max_corridor_distance_km = 3000.0

        with pytest.raises(
            ValueError, match="No corridors found - corridor discovery failed"
        ):
            add_corridors(graph, metros, config)

    def test_corridor_distance_limiting(self):
        graph = nx.Graph()

        # Very long path: 5000km total
        for i in range(10):
            start = (i * 500000.0, 0.0)  # 500km intervals in EPSG:5070 meters
            end = ((i + 1) * 500000.0, 0.0)
            graph.add_edge(start, end, length_km=500.0, geometry=[start, end])

        metros = [
            MetroCluster("metro1", "metro-a", "Metro A", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster(
                "metro2", "metro-b", "Metro B", "002", 100.0, 5000000.0, 0.0, 25.0
            ),
        ]

        for metro in metros:
            graph.add_node(
                metro.node_key,
                node_type="metro",
                name=metro.name,
                name_orig=metro.name_orig,
                metro_id=metro.metro_id,
                x=metro.centroid_x,
                y=metro.centroid_y,
                radius_km=metro.radius_km,
            )
            anchor_point = metro.node_key
            graph.add_edge(
                metro.node_key,
                anchor_point,
                edge_type="metro_anchor",
                length_km=0.0,
                geometry=[metro.node_key, anchor_point],
            )

        config = CorridorsConfig()
        config.k_paths = 1
        config.max_corridor_distance_km = 3000.0  # Shorter than actual distance

        with pytest.raises(
            ValueError, match="No adjacent metro pairs found for corridor discovery"
        ):
            add_corridors(graph, metros, config)

    def test_multiple_paths_discovery(self):
        graph = nx.Graph()

        # Path 1: A --- B --- D
        graph.add_edge(
            (0.0, 0.0),
            (500.0, 100.0),
            length_km=510.0,
            geometry=[(0.0, 0.0), (500.0, 100.0)],
        )
        graph.add_edge(
            (500.0, 100.0),
            (1000.0, 0.0),
            length_km=510.0,
            geometry=[(500.0, 100.0), (1000.0, 0.0)],
        )

        # Path 2: A --- C --- D (shorter)
        graph.add_edge(
            (0.0, 0.0),
            (500.0, -100.0),
            length_km=510.0,
            geometry=[(0.0, 0.0), (500.0, -100.0)],
        )
        graph.add_edge(
            (500.0, -100.0),
            (1000.0, 0.0),
            length_km=510.0,
            geometry=[(500.0, -100.0), (1000.0, 0.0)],
        )

        metros = [
            MetroCluster("metro1", "metro-a", "Metro A", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster(
                "metro2", "metro-d", "Metro D", "002", 100.0, 1000.0, 0.0, 25.0
            ),
        ]

        for metro in metros:
            graph.add_node(
                metro.node_key,
                node_type="metro",
                name=metro.name,
                name_orig=metro.name_orig,
                metro_id=metro.metro_id,
                x=metro.centroid_x,
                y=metro.centroid_y,
                radius_km=metro.radius_km,
            )
            anchor_point = metro.node_key
            graph.add_edge(
                metro.node_key,
                anchor_point,
                edge_type="metro_anchor",
                length_km=0.0,
                geometry=[metro.node_key, anchor_point],
            )

        config = CorridorsConfig()
        config.k_paths = 2
        config.k_nearest = 5
        config.max_edge_km = 2000.0
        config.max_corridor_distance_km = 2000.0

        add_corridors(graph, metros, config)

        path_indices = set()
        for _u, _v, data in graph.edges(data=True):
            if "corridor" in data and data["corridor"]:
                for corridor_info in data["corridor"]:
                    path_indices.add(corridor_info["path_index"])

        assert 0 in path_indices
        assert 1 in path_indices

    def test_corridor_distance_equals_path_length(self):
        """Corridor distance should equal the shortest path length across highway edges."""
        # Highway network: A -- B -- C, each edge 100 km
        # Metro centroids: A at (0, 0), C at (100000, 0) => 100 km straight-line
        # Shortest path A->C = 200 km (via B)
        graph = nx.Graph()

        A = (0.0, 0.0)
        B = (60000.0, 80000.0)  # detour up (arbitrary coords, units in meters)
        C = (100000.0, 0.0)

        graph.add_edge(A, B, length_km=100.0, geometry=[A, B])
        graph.add_edge(B, C, length_km=100.0, geometry=[B, C])

        metros = [
            MetroCluster(
                "metroA", "metro-a", "Metro A", "001", 100.0, A[0], A[1], 25.0
            ),
            MetroCluster(
                "metroC", "metro-c", "Metro C", "002", 100.0, C[0], C[1], 25.0
            ),
        ]

        for metro in metros:
            graph.add_node(
                metro.node_key,
                node_type="metro",
                name=metro.name,
                name_orig=metro.name_orig,
                metro_id=metro.metro_id,
                x=metro.centroid_x,
                y=metro.centroid_y,
                radius_km=metro.radius_km,
            )
            # Anchors coincide with metro coordinates in this test
            anchor_point = metro.node_key
            graph.add_edge(
                metro.node_key,
                anchor_point,
                edge_type="metro_anchor",
                length_km=0.0,
                geometry=[metro.node_key, anchor_point],
            )

        config = CorridorsConfig()
        config.k_paths = 1
        config.k_nearest = 1
        config.max_edge_km = 1000.0
        config.max_corridor_distance_km = 1000.0

        add_corridors(graph, metros, config)

        euclidean_km = 100.0  # straight-line between A and C (100km)
        # Sum of edge lengths along the shortest path (A->B->C)
        path = nx.shortest_path(graph, A, C, weight="length_km")
        path_km = sum(
            graph[path[i]][path[i + 1]]["length_km"] for i in range(len(path) - 1)
        )

        assert path_km == 200.0

        # Verify corridor tags recorded the path length, not the euclidean distance
        tagged = 0
        for _u, _v, data in graph.edges(data=True):
            if "corridor" in data and data["corridor"]:
                tagged += 1
                info = data["corridor"][0]
                assert info["distance_km"] == path_km
                assert info["distance_km"] != euclidean_km

        assert tagged == 2

    def test_path_length_filtering_over_euclidean(self):
        """Filter by path length even when Euclidean separation is below threshold."""
        graph = nx.Graph()

        # Construct a 2-edge path totaling 1200 km between endpoints 800 km apart
        A = (0.0, 0.0)
        B = (400000.0, 300000.0)  # arbitrary detour coordinate (meters)
        C = (800000.0, 0.0)

        graph.add_edge(A, B, length_km=600.0, geometry=[A, B])
        graph.add_edge(B, C, length_km=600.0, geometry=[B, C])

        metros = [
            MetroCluster(
                "metroA", "metro-a", "Metro A", "001", 100.0, A[0], A[1], 25.0
            ),
            MetroCluster(
                "metroC", "metro-c", "Metro C", "002", 100.0, C[0], C[1], 25.0
            ),
        ]

        # Add metro nodes and zero-length anchors at same coords as highway endpoints
        for metro in metros:
            graph.add_node(
                metro.node_key,
                node_type="metro",
                name=metro.name,
                name_orig=metro.name_orig,
                metro_id=metro.metro_id,
                x=metro.centroid_x,
                y=metro.centroid_y,
                radius_km=metro.radius_km,
            )
            anchor_point = metro.node_key
            graph.add_edge(
                metro.node_key,
                anchor_point,
                edge_type="metro_anchor",
                length_km=0.0,
                geometry=[metro.node_key, anchor_point],
            )

        # Euclidean ~800 km; path = 1200 km
        config = CorridorsConfig()
        config.k_paths = 1
        config.k_nearest = 1
        config.max_edge_km = 1000.0  # allow adjacency by Euclidean
        config.max_corridor_distance_km = 1000.0  # but disallow by path length

        with pytest.raises(
            ValueError, match="No corridors found - corridor discovery failed"
        ):
            add_corridors(graph, metros, config)


class TestCorridorGraphExtraction:
    def test_basic_corridor_graph_extraction(self):
        full_graph = nx.Graph()

        metro1_coords = (100.0, 200.0)
        metro2_coords = (200.0, 300.0)

        full_graph.add_node(
            metro1_coords,
            node_type="metro",
            name="metro-a",
            metro_id="001",
            x=100.0,
            y=200.0,
            radius_km=25.0,
            name_orig="metro-a",
        )
        full_graph.add_node(
            metro2_coords,
            node_type="metro",
            name="metro-b",
            metro_id="002",
            x=200.0,
            y=300.0,
            radius_km=25.0,
            name_orig="metro-b",
        )

        full_graph.add_edge(
            (150.0, 220.0),
            (180.0, 280.0),
            length_km=50.0,
            corridor=[
                {
                    "metro_a": "001",
                    "metro_b": "002",
                    "path_index": 0,
                    "distance_km": 141.4,
                }
            ],
            risk_groups=["corridor_risk_metro-a_metro-b"],
            geometry=[(150.0, 220.0), (180.0, 280.0)],
        )

        metros = [
            MetroCluster("001", "metro-a", "Metro A", "001", 100.0, 100.0, 200.0, 25.0),
            MetroCluster("002", "metro-b", "Metro B", "002", 100.0, 200.0, 300.0, 25.0),
        ]

        # Populate the corridor path registry.
        path_id = ("001", "002", 0)
        full_graph.graph["corridor_paths"] = {
            path_id: CorridorPath(
                edges=[((150.0, 220.0), (180.0, 280.0))],
                segment_ids=[],
                length_km=141.4,
                geometry=[metro1_coords, metro2_coords],
            )
        }

        corridor_graph = extract_corridor_graph(full_graph, metros)

        assert len(corridor_graph.nodes) == 2
        assert metro1_coords in corridor_graph.nodes
        assert metro2_coords in corridor_graph.nodes

        assert len(corridor_graph.edges) == 1

        edge_data = corridor_graph[metro1_coords][metro2_coords][0]
        assert edge_data["edge_type"] == "corridor"
        assert edge_data["length_km"] == 141.4
        assert edge_data["metro_a"] == "001"
        assert edge_data["metro_b"] == "002"
        assert "corridor_risk_metro-a_metro-b" in edge_data["risk_groups"]

        # Coordinates are in meters; euclidean distance in km ~ 0.1414
        assert "euclidean_km" in edge_data
        expected_euclid_km = ((100.0**2 + 100.0**2) ** 0.5) / 1000.0
        assert abs(edge_data["euclidean_km"] - expected_euclid_km) < 1e-3
        assert "detour_ratio" in edge_data
        # With length_km=141.4 and euclidean_km≈0.1414, detour ratio ≈ 1000
        assert 900 < edge_data["detour_ratio"] < 1100

    def test_corridor_graph_preserves_all_distances(self):
        full_graph = nx.Graph()

        metro1_coords = (0.0, 0.0)
        metro2_coords = (100.0, 100.0)

        full_graph.add_node(
            metro1_coords,
            node_type="metro",
            name="metro1",
            metro_id="001",
            name_orig="metro1",
        )
        full_graph.add_node(
            metro2_coords,
            node_type="metro",
            name="metro2",
            metro_id="002",
            name_orig="metro2",
        )

        full_graph.add_edge(
            (10.0, 10.0),
            (20.0, 20.0),
            corridor=[
                {
                    "metro_a": "001",
                    "metro_b": "002",
                    "path_index": 0,
                    "distance_km": 150.0,
                }
            ],  # Longer path
        )
        full_graph.add_edge(
            (30.0, 30.0),
            (40.0, 40.0),
            corridor=[
                {
                    "metro_a": "001",
                    "metro_b": "002",
                    "path_index": 1,
                    "distance_km": 120.0,
                }
            ],  # Shorter path
        )

        metros = [
            MetroCluster("001", "metro1", "Metro 1", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster("002", "metro2", "Metro 2", "002", 100.0, 100.0, 100.0, 25.0),
        ]

        reg = {}
        pid_long = ("001", "002", 0)
        pid_short = ("001", "002", 1)
        reg[pid_long] = CorridorPath(
            edges=[((10.0, 10.0), (20.0, 20.0))],
            segment_ids=[],
            length_km=150.0,
            geometry=[metro1_coords, metro2_coords],
        )
        reg[pid_short] = CorridorPath(
            edges=[((30.0, 30.0), (40.0, 40.0))],
            segment_ids=[],
            length_km=120.0,
            geometry=[metro1_coords, metro2_coords],
        )
        full_graph.graph["corridor_paths"] = reg

        corridor_graph = extract_corridor_graph(full_graph, metros)

        edge_data = corridor_graph[metro1_coords][metro2_coords][0]
        assert edge_data["length_km"] == 150.0
        assert corridor_graph[metro1_coords][metro2_coords][1]["length_km"] == 120.0

    def test_corridor_graph_preserves_risk_groups(self):
        full_graph = nx.Graph()

        metro1_coords = (0.0, 0.0)
        metro2_coords = (100.0, 100.0)

        full_graph.add_node(
            metro1_coords,
            node_type="metro",
            name="metro1",
            metro_id="001",
            name_orig="metro1",
        )
        full_graph.add_node(
            metro2_coords,
            node_type="metro",
            name="metro2",
            metro_id="002",
            name_orig="metro2",
        )

        full_graph.add_edge(
            (10.0, 10.0),
            (20.0, 20.0),
            corridor=[
                {
                    "metro_a": "001",
                    "metro_b": "002",
                    "path_index": 0,
                    "distance_km": 100.0,
                }
            ],
            risk_groups=["corridor_risk_metro1_metro2", "corridor_risk_metro1_metro3"],
        )
        full_graph.add_edge(
            (30.0, 30.0),
            (40.0, 40.0),
            corridor=[
                {
                    "metro_a": "001",
                    "metro_b": "002",
                    "path_index": 0,
                    "distance_km": 100.0,
                }
            ],
            risk_groups=["corridor_risk_metro1_metro2", "corridor_risk_metro2_metro4"],
        )

        metros = [
            MetroCluster("001", "metro1", "Metro 1", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster("002", "metro2", "Metro 2", "002", 100.0, 100.0, 100.0, 25.0),
        ]

        pid = ("001", "002", 0)
        full_graph.graph["corridor_paths"] = {
            pid: CorridorPath(
                edges=[((10.0, 10.0), (20.0, 20.0)), ((30.0, 30.0), (40.0, 40.0))],
                segment_ids=[],
                length_km=100.0,
                geometry=[metro1_coords, metro2_coords],
            )
        }

        corridor_graph = extract_corridor_graph(full_graph, metros)

        edge_data = corridor_graph[metro1_coords][metro2_coords][0]
        risk_groups = set(edge_data["risk_groups"])
        expected_risk_groups = {
            "corridor_risk_metro1_metro2",
            "corridor_risk_metro1_metro3",
            "corridor_risk_metro2_metro4",
        }
        assert risk_groups == expected_risk_groups
