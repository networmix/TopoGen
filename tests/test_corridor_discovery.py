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

        # Add metro nodes and anchor edges (metro-to-anchor Euclidean edges)
        for metro in metros:
            # Add/merge metro node
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
            )

        # Configure corridors
        config = CorridorsConfig()
        config.k_paths = 1
        config.k_nearest = 5
        config.max_edge_km = 2000.0
        config.max_corridor_distance_km = 2000.0

        # Run corridor discovery (metro-to-metro paths via anchors/highways)
        add_corridors(graph, metros, config)

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

        # Add metro nodes and anchor edges
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
            )

        config = CorridorsConfig()
        config.k_paths = 1
        config.k_nearest = 5
        config.max_edge_km = 3000.0
        config.max_corridor_distance_km = 3000.0

        # Should raise exception when no paths exist between metros
        with pytest.raises(
            ValueError, match="No corridors found - corridor discovery failed"
        ):
            add_corridors(graph, metros, config)

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

        # Add metro nodes and anchor edges
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
            )

        config = CorridorsConfig()
        config.k_paths = 1
        config.max_corridor_distance_km = 3000.0  # Shorter than actual distance

        # Should raise exception when metros are beyond max_edge_km distance limit
        with pytest.raises(
            ValueError, match="No adjacent metro pairs found for corridor discovery"
        ):
            add_corridors(graph, metros, config)

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

        # Add metro nodes and anchor edges
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
            )

        config = CorridorsConfig()
        config.k_paths = 2  # Request 2 paths
        config.k_nearest = 5
        config.max_edge_km = 2000.0
        config.max_corridor_distance_km = 2000.0

        add_corridors(graph, metros, config)

        # Check that multiple paths were found
        path_indices = set()
        for _u, _v, data in graph.edges(data=True):
            if "corridor" in data and data["corridor"]:
                for corridor_info in data["corridor"]:
                    path_indices.add(corridor_info["path_index"])

        # Should have found both path 0 and path 1
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

        graph.add_edge(A, B, length_km=100.0)
        graph.add_edge(B, C, length_km=100.0)

        metros = [
            MetroCluster(
                "metroA", "metro-a", "Metro A", "001", 100.0, A[0], A[1], 25.0
            ),
            MetroCluster(
                "metroC", "metro-c", "Metro C", "002", 100.0, C[0], C[1], 25.0
            ),
        ]

        # Add metro nodes and anchor edges
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
            )

        config = CorridorsConfig()
        config.k_paths = 1
        config.k_nearest = 1
        config.max_edge_km = 1000.0
        config.max_corridor_distance_km = 1000.0

        # Run corridor discovery
        add_corridors(graph, metros, config)

        # Compute expected distances
        euclidean_km = 100.0  # straight-line between A and C (100km)
        # Sum of edge lengths along the shortest path (A->B->C)
        path = nx.shortest_path(graph, A, C, weight="length_km")
        path_km = sum(
            graph[path[i]][path[i + 1]]["length_km"] for i in range(len(path) - 1)
        )

        assert path_km == 200.0  # sanity check for constructed graph

        # Verify corridor tags recorded the path length, not the euclidean distance
        tagged = 0
        for _u, _v, data in graph.edges(data=True):
            if "corridor" in data and data["corridor"]:
                tagged += 1
                info = data["corridor"][0]
                assert info["distance_km"] == path_km
                assert info["distance_km"] != euclidean_km

        # Both highway edges should be tagged
        assert tagged == 2

    def test_path_length_filtering_over_euclidean(self):
        """Filter by path length even when Euclidean separation is below threshold."""
        graph = nx.Graph()

        # Construct a 2-edge path totaling 1200 km between endpoints 800 km apart
        A = (0.0, 0.0)
        B = (400000.0, 300000.0)  # arbitrary detour coordinate (meters)
        C = (800000.0, 0.0)

        graph.add_edge(A, B, length_km=600.0)
        graph.add_edge(B, C, length_km=600.0)

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

        # Populate corridor path registry and tag edges with path id
        path_id = ("001", "002", 0)
        full_graph.graph["corridor_paths"] = {
            path_id: CorridorPath(
                metros=("001", "002"),
                path_index=0,
                nodes=[metro1_coords, metro2_coords],
                edges=[(metro1_coords, metro2_coords)],
                segment_ids=[],
                length_km=141.4,
                geometry=[metro1_coords, metro2_coords],
            )
        }
        # Mark the single edge as part of the chosen path
        e = full_graph[(150.0, 220.0)][(180.0, 280.0)]
        e["corridor_path_ids"] = {path_id}

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

        # Check euclidean and detour ratio are computed
        # Coordinates are in meters; euclidean distance in km ~ 0.1414
        assert "euclidean_km" in edge_data
        expected_euclid_km = ((100.0**2 + 100.0**2) ** 0.5) / 1000.0
        assert abs(edge_data["euclidean_km"] - expected_euclid_km) < 1e-3
        assert "detour_ratio" in edge_data
        # With length_km=141.4 and euclidean_km≈0.1414, detour ratio ≈ 1000
        assert 900 < edge_data["detour_ratio"] < 1100

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

        # Build two alternative paths and tag edges accordingly
        reg = {}
        pid_long = ("001", "002", 0)
        pid_short = ("001", "002", 1)
        reg[pid_long] = CorridorPath(
            metros=("001", "002"),
            path_index=0,
            nodes=[metro1_coords, metro2_coords],
            edges=[((10.0, 10.0), (20.0, 20.0))],
            segment_ids=[],
            length_km=150.0,
            geometry=[metro1_coords, metro2_coords],
        )
        reg[pid_short] = CorridorPath(
            metros=("001", "002"),
            path_index=1,
            nodes=[metro1_coords, metro2_coords],
            edges=[((30.0, 30.0), (40.0, 40.0))],
            segment_ids=[],
            length_km=120.0,
            geometry=[metro1_coords, metro2_coords],
        )
        full_graph.graph["corridor_paths"] = reg
        full_graph[(10.0, 10.0)][(20.0, 20.0)]["corridor_path_ids"] = {pid_long}
        full_graph[(30.0, 30.0)][(40.0, 40.0)]["corridor_path_ids"] = {pid_short}

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

        # Registry and edge tagging for the path
        pid = ("001", "002", 0)
        full_graph.graph["corridor_paths"] = {
            pid: CorridorPath(
                metros=("001", "002"),
                path_index=0,
                nodes=[metro1_coords, metro2_coords],
                edges=[((10.0, 10.0), (20.0, 20.0)), ((30.0, 30.0), (40.0, 40.0))],
                segment_ids=[],
                length_km=100.0,
                geometry=[metro1_coords, metro2_coords],
            )
        }
        full_graph[(10.0, 10.0)][(20.0, 20.0)]["corridor_path_ids"] = {pid}
        full_graph[(30.0, 30.0)][(40.0, 40.0)]["corridor_path_ids"] = {pid}

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
