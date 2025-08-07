"""Functional tests for corridor discovery logic."""

import networkx as nx
import pytest

from topogen.config import CorridorsConfig
from topogen.integrated_graph import add_corridors, extract_corridor_graph
from topogen.metro_clusters import MetroCluster


class TestCorridorDiscovery:
    """Test corridor discovery functionality."""

    def test_basic_corridor_discovery(self):
        """Test basic corridor discovery between metros."""
        # Create simple highway network
        graph = nx.Graph()

        # Highway path: A --- B --- C
        graph.add_edge((0.0, 0.0), (500.0, 0.0), length_km=500.0)
        graph.add_edge((500.0, 0.0), (1000.0, 0.0), length_km=500.0)

        # Create metros at endpoints
        metros = [
            MetroCluster("metro1", "metro-a", "Metro A", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster(
                "metro2", "metro-b", "Metro B", "002", 100.0, 1000.0, 0.0, 25.0
            ),
        ]

        # Anchors at endpoints
        anchors = {
            "metro1": (0.0, 0.0),
            "metro2": (1000.0, 0.0),
        }

        # Configure corridors
        config = CorridorsConfig()
        config.k_paths = 1
        config.k_nearest = 5
        config.max_edge_km = 2000.0
        config.max_corridor_distance_km = 2000.0

        # Run corridor discovery
        add_corridors(graph, anchors, metros, config)

        # Verify corridor tags were added
        corridor_edges = 0
        for _u, _v, data in graph.edges(data=True):
            if "corridor" in data and data["corridor"]:
                corridor_edges += 1
                # Check corridor info structure
                corridor_info = data["corridor"][0]
                assert "metro_a" in corridor_info
                assert "metro_b" in corridor_info
                assert "path_index" in corridor_info
                assert "distance_km" in corridor_info

        assert corridor_edges == 2  # Both highway edges should be tagged

    def test_corridor_discovery_no_path(self):
        """Test corridor discovery when no path exists between metros."""
        # Create disconnected graph
        graph = nx.Graph()
        graph.add_edge((0.0, 0.0), (100.0, 0.0), length_km=100.0)
        graph.add_edge((2000.0, 0.0), (2100.0, 0.0), length_km=100.0)  # Disconnected

        metros = [
            MetroCluster("metro1", "metro-a", "Metro A", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster(
                "metro2", "metro-b", "Metro B", "002", 100.0, 2000.0, 0.0, 25.0
            ),
        ]

        anchors = {
            "metro1": (0.0, 0.0),
            "metro2": (2000.0, 0.0),
        }

        config = CorridorsConfig()
        config.k_paths = 1
        config.k_nearest = 5
        config.max_edge_km = 3000.0
        config.max_corridor_distance_km = 3000.0

        # Should raise exception when no paths exist between metros
        with pytest.raises(
            ValueError, match="No corridors found - corridor discovery failed"
        ):
            add_corridors(graph, anchors, metros, config)

    def test_corridor_distance_limiting(self):
        """Test that corridors beyond max distance are skipped."""
        # Create long highway network
        graph = nx.Graph()

        # Very long path: 5000km total
        for i in range(10):
            start = (i * 500000.0, 0.0)  # 500km intervals in EPSG:5070 meters
            end = ((i + 1) * 500000.0, 0.0)
            graph.add_edge(start, end, length_km=500.0)

        metros = [
            MetroCluster("metro1", "metro-a", "Metro A", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster(
                "metro2", "metro-b", "Metro B", "002", 100.0, 5000000.0, 0.0, 25.0
            ),
        ]

        anchors = {
            "metro1": (0.0, 0.0),
            "metro2": (5000000.0, 0.0),
        }

        config = CorridorsConfig()
        config.k_paths = 1
        config.max_corridor_distance_km = 3000.0  # Shorter than actual distance

        # Should raise exception when metros are beyond max_edge_km distance limit
        with pytest.raises(
            ValueError, match="No adjacent metro pairs found for corridor discovery"
        ):
            add_corridors(graph, anchors, metros, config)

    def test_multiple_paths_discovery(self):
        """Test discovery of multiple paths between metro pairs."""
        # Create diamond network with two paths
        graph = nx.Graph()

        # Path 1: A --- B --- D
        graph.add_edge((0.0, 0.0), (500.0, 100.0), length_km=510.0)
        graph.add_edge((500.0, 100.0), (1000.0, 0.0), length_km=510.0)

        # Path 2: A --- C --- D (shorter)
        graph.add_edge((0.0, 0.0), (500.0, -100.0), length_km=510.0)
        graph.add_edge((500.0, -100.0), (1000.0, 0.0), length_km=510.0)

        metros = [
            MetroCluster("metro1", "metro-a", "Metro A", "001", 100.0, 0.0, 0.0, 25.0),
            MetroCluster(
                "metro2", "metro-d", "Metro D", "002", 100.0, 1000.0, 0.0, 25.0
            ),
        ]

        anchors = {
            "metro1": (0.0, 0.0),
            "metro2": (1000.0, 0.0),
        }

        config = CorridorsConfig()
        config.k_paths = 2  # Request 2 paths
        config.k_nearest = 5
        config.max_edge_km = 2000.0
        config.max_corridor_distance_km = 2000.0

        add_corridors(graph, anchors, metros, config)

        # Check that multiple paths were found
        path_indices = set()
        for _u, _v, data in graph.edges(data=True):
            if "corridor" in data and data["corridor"]:
                for corridor_info in data["corridor"]:
                    path_indices.add(corridor_info["path_index"])

        # Should have found both path 0 and path 1
        assert 0 in path_indices
        assert 1 in path_indices


class TestCorridorGraphExtraction:
    """Test corridor graph extraction functionality."""

    def test_basic_corridor_graph_extraction(self):
        """Test extraction of corridor-level graph from highway graph."""
        # Create highway graph with corridor tags
        full_graph = nx.Graph()

        # Add metro nodes
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
        )
        full_graph.add_node(
            metro2_coords,
            node_type="metro",
            name="metro-b",
            metro_id="002",
            x=200.0,
            y=300.0,
            radius_km=25.0,
        )

        # Add highway edges with corridor tags
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
        )

        metros = [
            MetroCluster("001", "metro-a", "Metro A", "001", 100.0, 100.0, 200.0, 25.0),
            MetroCluster("002", "metro-b", "Metro B", "002", 100.0, 200.0, 300.0, 25.0),
        ]

        # Extract corridor graph
        corridor_graph = extract_corridor_graph(full_graph, metros)

        # Should have 2 metro nodes
        assert len(corridor_graph.nodes) == 2
        assert metro1_coords in corridor_graph.nodes
        assert metro2_coords in corridor_graph.nodes

        # Should have 1 corridor edge
        assert len(corridor_graph.edges) == 1

        # Check edge data
        edge_data = corridor_graph[metro1_coords][metro2_coords]
        assert edge_data["edge_type"] == "corridor"
        assert edge_data["length_km"] == 141.4
        assert edge_data["metro_a"] == "001"
        assert edge_data["metro_b"] == "002"
        assert "corridor_risk_metro-a_metro-b" in edge_data["risk_groups"]

    def test_corridor_graph_aggregates_shortest_distance(self):
        """Test that corridor graph uses shortest distance between metro pairs."""
        full_graph = nx.Graph()

        metro1_coords = (0.0, 0.0)
        metro2_coords = (100.0, 100.0)

        full_graph.add_node(
            metro1_coords, node_type="metro", name="metro1", metro_id="001"
        )
        full_graph.add_node(
            metro2_coords, node_type="metro", name="metro2", metro_id="002"
        )

        # Add multiple highway edges with different distances for same metro pair
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

        corridor_graph = extract_corridor_graph(full_graph, metros)

        # Should use shortest distance
        edge_data = corridor_graph[metro1_coords][metro2_coords]
        assert edge_data["length_km"] == 120.0  # Shorter distance

    def test_corridor_graph_preserves_risk_groups(self):
        """Test that risk groups are preserved in corridor graph extraction."""
        full_graph = nx.Graph()

        metro1_coords = (0.0, 0.0)
        metro2_coords = (100.0, 100.0)

        full_graph.add_node(
            metro1_coords, node_type="metro", name="metro1", metro_id="001"
        )
        full_graph.add_node(
            metro2_coords, node_type="metro", name="metro2", metro_id="002"
        )

        # Add edges with different risk groups
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

        corridor_graph = extract_corridor_graph(full_graph, metros)

        # Should collect all unique risk groups
        edge_data = corridor_graph[metro1_coords][metro2_coords]
        risk_groups = set(edge_data["risk_groups"])
        expected_risk_groups = {
            "corridor_risk_metro1_metro2",
            "corridor_risk_metro1_metro3",
            "corridor_risk_metro2_metro4",
        }
        assert risk_groups == expected_risk_groups
