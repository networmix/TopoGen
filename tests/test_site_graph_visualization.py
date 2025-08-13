"""Tests for site-level graph visualization export."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from topogen.visualization import export_site_graph_map


def test_export_site_graph_map(tmp_path: Path) -> None:
    """Export a tiny site graph and verify JPEG is created and non-trivial."""
    G = nx.MultiGraph()

    # Two sites in one metro with radius 10 km (10000 m)
    G.add_node(
        "metro1/pop1",
        metro_idx=1,
        site_kind="pop",
        site_ordinal=1,
        pos_x=1000.0,
        pos_y=2000.0,
        center_x=1000.0,
        center_y=2000.0,
        radius_m=10000.0,
    )
    G.add_node(
        "metro1/dc1",
        metro_idx=1,
        site_kind="dc",
        site_ordinal=1,
        pos_x=1020.0,
        pos_y=2100.0,
        center_x=1000.0,
        center_y=2000.0,
        radius_m=10000.0,
    )

    # One adjacency with a labeled capacity
    G.add_edge(
        "metro1/pop1",
        "metro1/dc1",
        key="dc_to_pop:1-1",
        link_type="dc_to_pop",
        base_capacity=400.0,
        target_capacity=400.0,
        cost=10,
    )

    out = tmp_path / "site_graph.jpg"
    export_site_graph_map(G, out)
    assert out.exists()
    assert out.stat().st_size > 1000
