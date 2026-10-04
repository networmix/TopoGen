"""Tests for integrated graph construction and chain contraction."""

import networkx as nx

from topogen.config import ValidationConfig
from topogen.integrated_graph import (
    _contract_degree2_chains,
    _keep_largest_component,
    _remove_slivers,
)


class TestDegree2Contraction:
    def test_contract_simple_chain(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (50.0, 0.0), length_km=50.0)
        G.add_edge((50.0, 0.0), (100.0, 0.0), length_km=50.0)
        G.add_edge((100.0, 0.0), (150.0, 0.0), length_km=50.0)

        contracted = _contract_degree2_chains(G)

        assert len(contracted.nodes) == 2
        assert len(contracted.edges) == 1

        edge_data = list(contracted.edges(data=True))[0][2]
        assert "length_km" in edge_data
        assert edge_data["length_km"] == 150.0  # Sum of original lengths

        assert "geometry" in edge_data
        geometry = edge_data["geometry"]
        assert (
            len(geometry) == 4
        )  # All points in the path: start, middle1, middle2, end
        assert geometry == [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (150.0, 0.0)]

    def test_contract_isolated_cycle(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (100.0, 0.0), length_km=100.0)
        G.add_edge((100.0, 0.0), (100.0, 100.0), length_km=100.0)
        G.add_edge((100.0, 100.0), (0.0, 100.0), length_km=100.0)
        G.add_edge((0.0, 100.0), (0.0, 0.0), length_km=100.0)

        contracted = _contract_degree2_chains(G)

        assert nx.is_isomorphic(contracted, G)
        assert sum(d["length_km"] for _, _, d in contracted.edges(data=True)) == 400.0
        assert (
            nx.shortest_path_length(
                contracted, (0.0, 0.0), (100.0, 100.0), weight="length_km"
            )
            == 200.0
        )

    def test_contract_mixed_topology(self):
        G = nx.Graph()
        # Junction with 3 branches
        G.add_edge((0.0, 0.0), (50.0, 0.0), length_km=50.0)  # Branch 1 (part of chain)
        G.add_edge((0.0, 0.0), (0.0, 50.0), length_km=50.0)  # Branch 2 (stub)
        G.add_edge((0.0, 0.0), (0.0, -50.0), length_km=50.0)  # Branch 3 (stub)

        G.add_edge((50.0, 0.0), (100.0, 0.0), length_km=50.0)
        G.add_edge((100.0, 0.0), (150.0, 0.0), length_km=50.0)

        contracted = _contract_degree2_chains(G)

        junction = (0.0, 0.0)
        assert junction in contracted.nodes
        assert len(list(contracted.neighbors(junction))) == 3

        edge_lengths = [data["length_km"] for _, _, data in contracted.edges(data=True)]
        assert 150.0 in edge_lengths  # Contracted chain
        assert edge_lengths.count(50.0) == 2  # Two stubs

    def test_geometry_path_correctness(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (50.0, 0.0), length_km=50.0)
        G.add_edge((50.0, 0.0), (100.0, 0.0), length_km=50.0)

        contracted = _contract_degree2_chains(G)

        edge_data = list(contracted.edges(data=True))[0][2]
        geometry = edge_data["geometry"]

        assert len(geometry) == len(set(geometry))
        assert geometry == [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)]


class TestSliverRemoval:
    def test_remove_short_edges(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (1.0, 0.0), length_km=0.005)  # Very short edge
        G.add_edge((1.0, 0.0), (100.0, 0.0), length_km=99.0)  # Long edge

        validation_config = ValidationConfig()
        cleaned = _remove_slivers(
            G, min_length_km=0.01, validation_config=validation_config
        )

        assert len(cleaned.edges) == 1
        assert len(cleaned.nodes) == 2

    def test_preserve_long_edges(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (1.0, 0.0), length_km=0.1)  # Above threshold
        G.add_edge((1.0, 0.0), (100.0, 0.0), length_km=99.0)  # Long edge

        validation_config = ValidationConfig()
        cleaned = _remove_slivers(
            G, min_length_km=0.01, validation_config=validation_config
        )

        assert len(cleaned.edges) == 2
        assert len(cleaned.nodes) == 3


class TestComponentFiltering:
    def test_keep_largest_component(self):
        G = nx.Graph()

        # Large component
        G.add_edge((0.0, 0.0), (1.0, 0.0), length_km=1.0)
        G.add_edge((1.0, 0.0), (2.0, 0.0), length_km=1.0)
        G.add_edge((2.0, 0.0), (3.0, 0.0), length_km=1.0)

        # Small isolated component
        G.add_edge((100.0, 100.0), (101.0, 100.0), length_km=1.0)

        largest = _keep_largest_component(G)

        assert len(largest.nodes) == 4
        assert len(largest.edges) == 3
        assert (100.0, 100.0) not in largest.nodes
        assert (101.0, 100.0) not in largest.nodes

    def test_already_connected_graph(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (1.0, 0.0), length_km=1.0)
        G.add_edge((1.0, 0.0), (2.0, 0.0), length_km=1.0)

        result = _keep_largest_component(G)

        assert len(result.nodes) == len(G.nodes)
        assert len(result.edges) == len(G.edges)


class TestMetroMergeAttributes:
    """Regression test for merging metro attributes onto existing highway node."""

    def test_metro_highway_collision_sets_full_metro_attrs(self, monkeypatch):
        # Build a tiny highway graph with nodes at both metro centroids
        from topogen import integrated_graph as ig
        from topogen.config import TopologyConfig
        from topogen.integrated_graph import build_integrated_graph

        def fake_build_highway_graph(*_args, **_kwargs):
            G = nx.Graph()
            metro_a = (1000.0, 2000.0)
            metro_b = (3000.0, 4000.0)
            G.add_node(metro_a, node_type="highway")
            G.add_node(metro_b, node_type="highway")
            # Connect metros so corridor discovery works
            G.add_edge(metro_a, metro_b, length_km=10.0)
            return G

        def fake_load_metro_clusters(*_args, **_kwargs):
            from topogen.metro_clusters import MetroCluster

            return [
                MetroCluster(
                    metro_id="001",
                    name="metro-a",
                    name_orig="Metro A",
                    uac_code="001",
                    land_area_km2=123.0,
                    centroid_x=1000.0,
                    centroid_y=2000.0,
                    radius_km=25.0,
                ),
                MetroCluster(
                    metro_id="002",
                    name="metro-b",
                    name_orig="Metro B",
                    uac_code="002",
                    land_area_km2=456.0,
                    centroid_x=3000.0,
                    centroid_y=4000.0,
                    radius_km=30.0,
                ),
            ]

        captured_full_graphs: list[nx.Graph] = []

        def fake_extract_corridor_graph(full_graph: nx.Graph, metros):
            captured_full_graphs.append(full_graph.copy())
            # Return minimal valid corridor graph
            cg = nx.Graph()
            a = (1000.0, 2000.0)
            b = (3000.0, 4000.0)
            cg.add_node(
                a,
                node_type="metro",
                metro_id="001",
                name="metro-a",
                radius_km=10.0,
                name_orig="metro-a",
            )
            cg.add_node(
                b,
                node_type="metro",
                metro_id="002",
                name="metro-b",
                radius_km=20.0,
                name_orig="metro-b",
            )
            cg.add_edge(a, b, edge_type="corridor", length_km=10.0)
            return cg

        # Monkeypatch heavy I/O and extraction capture
        monkeypatch.setattr(ig, "build_highway_graph", fake_build_highway_graph)
        monkeypatch.setattr(ig, "load_metro_clusters", fake_load_metro_clusters)
        monkeypatch.setattr(ig, "extract_corridor_graph", fake_extract_corridor_graph)

        # Use default config; relax validation to avoid noise
        config = TopologyConfig()
        config.highway_processing.filter_largest_component = False
        config.validation.require_connected = False
        config.corridors.k_paths = 1
        config.corridors.k_nearest = 1
        config.corridors.max_edge_km = 10000.0
        config.corridors.max_corridor_distance_km = 10000.0
        config.corridors.risk_groups.enabled = False
        config.clustering.export_integrated_graph = False

        # Build integrated graph (returns corridor graph, but capture full graph)
        _ = build_integrated_graph(config)

        assert captured_full_graphs, "Did not capture integrated graph"
        full_graph = captured_full_graphs[-1]

        key_a = (1000.0, 2000.0)
        key_b = (3000.0, 4000.0)
        assert key_a in full_graph.nodes
        assert key_b in full_graph.nodes

        node_a = full_graph.nodes[key_a]
        node_b = full_graph.nodes[key_b]

        assert node_a.get("node_type") in {"metro", "metro+highway"}
        assert node_a["name"] == "metro-a"
        assert node_a["name_orig"] == "Metro A"
        assert node_a["metro_id"] == "001"
        assert node_a["x"] == 1000.0
        assert node_a["y"] == 2000.0
        assert node_a["uac_code"] == "001"
        assert node_a["land_area_km2"] == 123.0

        assert node_b.get("node_type") in {"metro", "metro+highway"}
        assert node_b["name"] == "metro-b"
        assert node_b["name_orig"] == "Metro B"
        assert node_b["metro_id"] == "002"
        assert node_b["x"] == 3000.0
        assert node_b["y"] == 4000.0
        assert node_b["uac_code"] == "002"
        assert node_b["land_area_km2"] == 456.0
