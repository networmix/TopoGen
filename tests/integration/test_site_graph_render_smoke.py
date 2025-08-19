from __future__ import annotations

from pathlib import Path

import networkx as nx

from topogen.visualization import export_site_graph_map


def test_export_site_graph_map_smoke(tmp_path: Path) -> None:
    G = nx.MultiGraph()
    # Two sites in one metro with required attributes
    G.add_node(
        "metro1/dc1/A1",
        pos_x=0.0,
        pos_y=0.0,
        center_x=0.0,
        center_y=0.0,
        radius_m=10_000.0,
        metro_idx=1,
        metro_name="M1",
    )
    G.add_node(
        "metro1/dc1/B1",
        pos_x=1.0,
        pos_y=0.0,
        center_x=0.0,
        center_y=0.0,
        radius_m=10_000.0,
        metro_idx=1,
        metro_name="M1",
    )
    G.add_edge(
        "metro1/dc1/A1",
        "metro1/dc1/B1",
        key=0,
        target_capacity=100.0,
        link_type="intra_metro",
    )
    # One inter-metro corridor label
    G.add_edge(
        "metro1/dc1/A1",
        "metro1/dc1/B1",
        key=1,
        target_capacity=200.0,
        link_type="inter_metro_corridor",
        source_metro="M1",
        target_metro="M2",
    )
    out = tmp_path / "site.jpg"
    export_site_graph_map(G, out, figure_size=(6, 4), metro_scale=1.0, dpi=120)
    assert out.exists() and out.stat().st_size > 1000
