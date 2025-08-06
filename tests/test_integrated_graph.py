"""Tests for integrated graph construction and chain contraction."""

import networkx as nx

from topogen.config import ValidationConfig
from topogen.integrated_graph import (
    _contract_degree2_chains,
    _keep_largest_component,
    _remove_slivers,
)


class TestDegree2Contraction:
    """Test degree-2 chain contraction functionality."""

    def test_contract_simple_chain(self):
        """Test contracting a simple degree-2 chain."""
        # Create a graph with a degree-2 chain
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (50.0, 0.0), length_km=50.0)
        G.add_edge((50.0, 0.0), (100.0, 0.0), length_km=50.0)
        G.add_edge((100.0, 0.0), (150.0, 0.0), length_km=50.0)

        contracted = _contract_degree2_chains(G)

        # Should contract middle node
        assert len(contracted.nodes) == 2  # Only endpoints remain
        assert len(contracted.edges) == 1

        # Check that length is preserved
        edge_data = list(contracted.edges(data=True))[0][2]
        assert "length_km" in edge_data
        assert edge_data["length_km"] == 150.0  # Sum of original lengths

        # Check geometry is preserved
        assert "geometry" in edge_data
        geometry = edge_data["geometry"]
        assert (
            len(geometry) == 4
        )  # All points in the path: start, middle1, middle2, end
        assert geometry == [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (150.0, 0.0)]

    def test_contract_isolated_cycle(self):
        """Test contracting isolated degree-2 cycles."""
        # Create a simple cycle where all nodes have degree 2
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (100.0, 0.0), length_km=100.0)
        G.add_edge((100.0, 0.0), (100.0, 100.0), length_km=100.0)
        G.add_edge((100.0, 100.0), (0.0, 100.0), length_km=100.0)
        G.add_edge((0.0, 100.0), (0.0, 0.0), length_km=100.0)

        contracted = _contract_degree2_chains(G)

        # Should contract cycle into a single edge
        assert len(contracted.nodes) == 2
        assert len(contracted.edges) == 1

        # Total length should be preserved
        edge_data = list(contracted.edges(data=True))[0][2]
        assert edge_data["length_km"] == 400.0  # Sum of cycle

    def test_contract_mixed_topology(self):
        """Test contraction with mixed junctions and chains."""
        # Create graph with junction connected to a chain
        G = nx.Graph()
        # Junction with 3 branches
        G.add_edge((0.0, 0.0), (50.0, 0.0), length_km=50.0)  # Branch 1 (part of chain)
        G.add_edge((0.0, 0.0), (0.0, 50.0), length_km=50.0)  # Branch 2 (stub)
        G.add_edge((0.0, 0.0), (0.0, -50.0), length_km=50.0)  # Branch 3 (stub)

        # Extend branch 1 into a chain
        G.add_edge((50.0, 0.0), (100.0, 0.0), length_km=50.0)
        G.add_edge((100.0, 0.0), (150.0, 0.0), length_km=50.0)

        contracted = _contract_degree2_chains(G)

        # Junction should remain with 3 edges: 2 stubs + 1 contracted chain
        junction = (0.0, 0.0)
        assert junction in contracted.nodes
        assert len(list(contracted.neighbors(junction))) == 3

        # One edge should be contracted (length 150km)
        edge_lengths = [data["length_km"] for _, _, data in contracted.edges(data=True)]
        assert 150.0 in edge_lengths  # Contracted chain
        assert edge_lengths.count(50.0) == 2  # Two stubs

    def test_geometry_path_correctness(self):
        """Test that geometry paths are built correctly without duplication."""
        # Create a simple chain
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (50.0, 0.0), length_km=50.0)
        G.add_edge((50.0, 0.0), (100.0, 0.0), length_km=50.0)

        contracted = _contract_degree2_chains(G)

        # Check geometry path
        edge_data = list(contracted.edges(data=True))[0][2]
        geometry = edge_data["geometry"]

        # Should have no duplicated vertices
        assert len(geometry) == len(set(geometry))
        assert geometry == [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)]


class TestSliverRemoval:
    """Test sliver removal functionality."""

    def test_remove_short_edges(self):
        """Test removal of edges shorter than threshold."""
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (1.0, 0.0), length_km=0.005)  # Very short edge
        G.add_edge((1.0, 0.0), (100.0, 0.0), length_km=99.0)  # Long edge

        validation_config = ValidationConfig()
        cleaned = _remove_slivers(
            G, min_length_km=0.01, validation_config=validation_config
        )

        # Short edge should be removed, but node (1.0, 0.0) remains as it connects to the long edge
        assert len(cleaned.edges) == 1  # Only the long edge remains
        assert len(cleaned.nodes) == 2  # Nodes (1.0, 0.0) and (100.0, 0.0) remain

    def test_preserve_long_edges(self):
        """Test that long edges are preserved."""
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (1.0, 0.0), length_km=0.1)  # Above threshold
        G.add_edge((1.0, 0.0), (100.0, 0.0), length_km=99.0)  # Long edge

        validation_config = ValidationConfig()
        cleaned = _remove_slivers(
            G, min_length_km=0.01, validation_config=validation_config
        )

        # Both edges should be preserved
        assert len(cleaned.edges) == 2
        assert len(cleaned.nodes) == 3


class TestComponentFiltering:
    """Test largest component filtering functionality."""

    def test_keep_largest_component(self):
        """Test keeping only the largest connected component."""
        G = nx.Graph()

        # Large component
        G.add_edge((0.0, 0.0), (1.0, 0.0), length_km=1.0)
        G.add_edge((1.0, 0.0), (2.0, 0.0), length_km=1.0)
        G.add_edge((2.0, 0.0), (3.0, 0.0), length_km=1.0)

        # Small isolated component
        G.add_edge((100.0, 100.0), (101.0, 100.0), length_km=1.0)

        validation_config = ValidationConfig()
        largest = _keep_largest_component(G, validation_config)

        # Should keep only the larger component
        assert len(largest.nodes) == 4
        assert len(largest.edges) == 3
        assert (100.0, 100.0) not in largest.nodes
        assert (101.0, 100.0) not in largest.nodes

    def test_already_connected_graph(self):
        """Test handling of already connected graphs."""
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (1.0, 0.0), length_km=1.0)
        G.add_edge((1.0, 0.0), (2.0, 0.0), length_km=1.0)

        validation_config = ValidationConfig()
        result = _keep_largest_component(G, validation_config)

        # Should return the same graph
        assert len(result.nodes) == len(G.nodes)
        assert len(result.edges) == len(G.edges)
