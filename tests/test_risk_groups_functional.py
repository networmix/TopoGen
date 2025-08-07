"""Functional tests for risk group assignment logic."""

import networkx as nx

from topogen.config import CorridorsConfig, RiskGroupsConfig
from topogen.corridors import assign_risk_groups as assign_risk_groups_to_corridors
from topogen.metro_clusters import MetroCluster


class TestRiskGroupAssignment:
    """Test risk group assignment functionality."""

    def test_basic_risk_group_assignment(self):
        """Test basic risk group assignment to corridor edges."""
        # Create test metros
        metros = [
            MetroCluster(
                metro_id="01171",
                name="albuquerque",
                name_orig="Albuquerque, NM",
                uac_code="01171",
                land_area_km2=681.0,
                centroid_x=1000.0,
                centroid_y=2000.0,
                radius_km=30.0,
            ),
            MetroCluster(
                metro_id="23527",
                name="denver-aurora",
                name_orig="Denver--Aurora, CO",
                uac_code="23527",
                land_area_km2=1669.0,
                centroid_x=2000.0,
                centroid_y=3000.0,
                radius_km=35.0,
            ),
        ]

        # Create test graph with corridor edge
        graph = nx.Graph()
        graph.add_edge(
            (1500.0, 2500.0),
            (1600.0, 2600.0),
            corridor=[
                {
                    "metro_a": "01171",
                    "metro_b": "23527",
                    "path_index": 0,
                    "distance_km": 450.0,
                }
            ],
        )

        # Configure risk groups
        config = CorridorsConfig()
        config.risk_groups = RiskGroupsConfig(
            enabled=True,
            group_prefix="corridor_risk",
            exclude_metro_radius_shared=False,  # Don't exclude for this test
        )

        # Assign risk groups
        assign_risk_groups_to_corridors(graph, metros, config)

        # Verify risk group was assigned
        edge_data = graph[(1500.0, 2500.0)][(1600.0, 2600.0)]
        assert "risk_groups" in edge_data
        assert len(edge_data["risk_groups"]) == 1
        assert edge_data["risk_groups"][0] == "corridor_risk_albuquerque_denver-aurora"

    def test_risk_group_naming_consistency(self):
        """Test that risk group names are generated consistently."""
        metros = [
            MetroCluster(
                "metro1", "zzz-metro", "ZZZ Metro", "001", 100.0, 0.0, 0.0, 25.0
            ),
            MetroCluster(
                "metro2", "aaa-metro", "AAA Metro", "002", 100.0, 100.0, 100.0, 25.0
            ),
        ]

        # Create edges with metros in different orders
        graph = nx.Graph()
        graph.add_edge(
            (50.0, 50.0),
            (150.0, 150.0),
            corridor=[
                {
                    "metro_a": "metro1",
                    "metro_b": "metro2",
                    "path_index": 0,
                    "distance_km": 100.0,
                }
            ],
        )
        graph.add_edge(
            (200.0, 200.0),
            (250.0, 250.0),
            corridor=[
                {
                    "metro_a": "metro2",
                    "metro_b": "metro1",
                    "path_index": 0,
                    "distance_km": 100.0,
                }
            ],
        )

        config = CorridorsConfig()
        config.risk_groups = RiskGroupsConfig(
            enabled=True, exclude_metro_radius_shared=False
        )

        assign_risk_groups_to_corridors(graph, metros, config)

        # Both edges should get the same risk group name (alphabetically sorted)
        edge1_risks = graph[(50.0, 50.0)][(150.0, 150.0)]["risk_groups"]
        edge2_risks = graph[(200.0, 200.0)][(250.0, 250.0)]["risk_groups"]

        assert len(edge1_risks) == 1
        assert len(edge2_risks) == 1
        assert edge1_risks[0] == edge2_risks[0]
        assert (
            edge1_risks[0] == "corridor_risk_aaa-metro_zzz-metro"
        )  # Alphabetically sorted

    def test_metro_radius_exclusion(self):
        """Test that edges within metro radius are excluded from risk groups."""
        # Use coordinates in meters to match the actual implementation
        metros = [
            MetroCluster(
                "metro1",
                "test-metro",
                "Test Metro",
                "001",
                100.0,
                1000000.0,
                1000000.0,
                50.0,
            ),  # 50km radius
            MetroCluster(
                "metro2",
                "far-metro",
                "Far Metro",
                "002",
                100.0,
                2000000.0,
                2000000.0,
                25.0,
            ),  # Far away
        ]

        graph = nx.Graph()

        # Edge close to metro1 (within 50km radius)
        close_edge = (
            (980000.0, 1000000.0),
            (1020000.0, 1000000.0),
        )  # 20km from metro1 center
        graph.add_edge(
            *close_edge,
            corridor=[
                {
                    "metro_a": "metro1",
                    "metro_b": "metro2",
                    "path_index": 0,
                    "distance_km": 100.0,
                }
            ],
        )

        # Edge far from both metros (outside radius)
        far_edge = (
            (1500000.0, 1500000.0),
            (1600000.0, 1600000.0),
        )  # >50km from both metros
        graph.add_edge(
            *far_edge,
            corridor=[
                {
                    "metro_a": "metro1",
                    "metro_b": "metro2",
                    "path_index": 0,
                    "distance_km": 100.0,
                }
            ],
        )

        config = CorridorsConfig()
        config.risk_groups = RiskGroupsConfig(
            enabled=True,
            exclude_metro_radius_shared=True,  # Enable exclusion
        )

        assign_risk_groups_to_corridors(graph, metros, config)

        # Close edge should NOT have risk groups (excluded)
        close_edge_data = graph[close_edge[0]][close_edge[1]]
        assert (
            "risk_groups" not in close_edge_data
            or len(close_edge_data.get("risk_groups", [])) == 0
        )

        # Far edge should have risk groups
        far_edge_data = graph[far_edge[0]][far_edge[1]]
        assert "risk_groups" in far_edge_data
        assert len(far_edge_data["risk_groups"]) > 0

    def test_multiple_corridors_same_edge(self):
        """Test edge with multiple corridor assignments gets multiple risk groups."""
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
                100.0,
                100.0,
                25.0,
            ),
            MetroCluster(
                "43912",
                "kansas-city",
                "Kansas City, MO--KS",
                "43912",
                100.0,
                200.0,
                200.0,
                25.0,
            ),
        ]

        # Create edge that's shared by multiple corridors
        graph = nx.Graph()
        graph.add_edge(
            (1500.0, 1500.0),
            (1600.0, 1600.0),
            corridor=[
                {
                    "metro_a": "01171",
                    "metro_b": "23527",
                    "path_index": 0,
                    "distance_km": 150.0,
                },
                {
                    "metro_a": "23527",
                    "metro_b": "43912",
                    "path_index": 0,
                    "distance_km": 200.0,
                },
            ],
        )

        config = CorridorsConfig()
        config.risk_groups = RiskGroupsConfig(
            enabled=True, exclude_metro_radius_shared=False
        )

        assign_risk_groups_to_corridors(graph, metros, config)

        # Should have risk groups for both corridors
        edge_data = graph[(1500.0, 1500.0)][(1600.0, 1600.0)]
        risk_groups = edge_data["risk_groups"]

        assert len(risk_groups) == 2
        assert "corridor_risk_albuquerque_denver-aurora" in risk_groups
        assert "corridor_risk_denver-aurora_kansas-city" in risk_groups

    def test_risk_groups_disabled(self):
        """Test that no risk groups are assigned when disabled."""
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
                100.0,
                100.0,
                25.0,
            ),
        ]

        graph = nx.Graph()
        graph.add_edge(
            (500.0, 500.0),
            (600.0, 600.0),
            corridor=[
                {
                    "metro_a": "01171",
                    "metro_b": "23527",
                    "path_index": 0,
                    "distance_km": 100.0,
                }
            ],
        )

        config = CorridorsConfig()
        config.risk_groups = RiskGroupsConfig(enabled=False)  # Disabled

        assign_risk_groups_to_corridors(graph, metros, config)

        # Should not add any risk groups
        edge_data = graph[(500.0, 500.0)][(600.0, 600.0)]
        assert "risk_groups" not in edge_data

    def test_multi_path_risk_groups(self):
        """Test risk group naming for multiple paths between same metro pair."""
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
                100.0,
                100.0,
                25.0,
            ),
        ]

        graph = nx.Graph()

        # Path 0 (primary path)
        graph.add_edge(
            (100.0, 100.0),
            (200.0, 200.0),
            corridor=[
                {
                    "metro_a": "01171",
                    "metro_b": "23527",
                    "path_index": 0,
                    "distance_km": 100.0,
                }
            ],
        )

        # Path 1 (alternate path)
        graph.add_edge(
            (300.0, 300.0),
            (400.0, 400.0),
            corridor=[
                {
                    "metro_a": "01171",
                    "metro_b": "23527",
                    "path_index": 1,
                    "distance_km": 120.0,
                }
            ],
        )

        config = CorridorsConfig()
        config.risk_groups = RiskGroupsConfig(
            enabled=True, exclude_metro_radius_shared=False
        )

        assign_risk_groups_to_corridors(graph, metros, config)

        # Path 0 should have base risk group name
        edge0_risks = graph[(100.0, 100.0)][(200.0, 200.0)]["risk_groups"]
        assert len(edge0_risks) == 1
        assert edge0_risks[0] == "corridor_risk_albuquerque_denver-aurora"

        # Path 1 should have path suffix
        edge1_risks = graph[(300.0, 300.0)][(400.0, 400.0)]["risk_groups"]
        assert len(edge1_risks) == 1
        assert edge1_risks[0] == "corridor_risk_albuquerque_denver-aurora_path1"


class TestMetroNameSanitization:
    """Test metro name sanitization functionality."""

    def test_metro_name_sanitization(self):
        """Test that metro names are properly sanitized."""
        test_cases = [
            ("New York--Jersey City--Newark, NY--NJ", "new-york-jersey-city-newark"),
            ("Dallas--Fort Worth--Arlington, TX", "dallas-fort-worth-arlington"),
            ("Washington--Arlington, DC--VA--MD", "washington-arlington"),
            ("Atlanta, GA", "atlanta"),
            ("Kansas City, MO--KS", "kansas-city"),
            ("Salt Lake City, UT", "salt-lake-city"),
            ("Las Vegas--Henderson--Paradise, NV", "las-vegas-henderson-paradise"),
        ]

        for original, expected in test_cases:
            sanitized = MetroCluster._sanitize_metro_name(original)
            assert sanitized == expected, (
                f"Failed for {original}: got {sanitized}, expected {expected}"
            )

    def test_sanitization_edge_cases(self):
        """Test sanitization handles edge cases correctly."""
        # Special characters
        assert MetroCluster._sanitize_metro_name("Test & City, CA") == "test-city"

        # Multiple spaces/dashes
        assert (
            MetroCluster._sanitize_metro_name("Multi  --  Space,  TX") == "multi-space"
        )

        # Length limiting
        very_long_name = (
            "Very Long Metro Name That Exceeds Thirty Characters, State--Extra--Parts"
        )
        sanitized = MetroCluster._sanitize_metro_name(very_long_name)
        assert len(sanitized) <= 30
        assert sanitized == "very-long-metro-name-that-exce"  # Truncated at 30 chars

    def test_metro_cluster_uses_sanitized_names(self):
        """Test that MetroCluster stores both sanitized and original names."""
        original_name = "Denver--Aurora, CO"
        metro = MetroCluster(
            metro_id="23527",
            name=MetroCluster._sanitize_metro_name(original_name),
            name_orig=original_name,
            uac_code="23527",
            land_area_km2=1669.0,
            centroid_x=100.0,
            centroid_y=200.0,
            radius_km=35.0,
        )

        assert metro.name == "denver-aurora"  # Sanitized
        assert metro.name_orig == "Denver--Aurora, CO"  # Original
