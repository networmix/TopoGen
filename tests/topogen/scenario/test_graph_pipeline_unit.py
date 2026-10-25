from __future__ import annotations

from pathlib import Path

import networkx as nx

import topogen.scenario.graph_pipeline as gp
import topogen.scenario.network as nw
from topogen.config import (
    BuildConfig,
    BuildTmSizingConfig,
    TopologyConfig,
    TrafficConfig,
)
from topogen.library import load_library, thaw
from topogen.naming import site_id


def _cfg() -> TopologyConfig:
    # Minimal config object with nested fields used by to_network_sections
    return TopologyConfig(
        traffic=TrafficConfig(enabled=True, mw_per_dc_region=10.0, gbps_per_mw=100.0),
        build=BuildConfig(tm_sizing=BuildTmSizingConfig(enabled=False)),
    )


def test_metro_index_and_node_id_helpers() -> None:
    metros = [
        {"name": "A", "node_key": (0.0, 0.0)},
        {"name": "B", "node_key": (1.0, 1.0)},
    ]
    idx_map, by_node = gp._metro_index_maps(metros)
    assert idx_map["A"] == 1 and idx_map["B"] == 2
    assert by_node[(0.0, 0.0)]["name"] == "A"
    assert site_id(3, "pop", 2) == "metro3/pop2"


def test_assign_site_positions_inside_radius() -> None:
    G = nx.MultiGraph()
    metros = [
        {
            "name": "A",
            "x": 100.0,
            "y": 200.0,
            "radius_km": 10.0,
            "node_key": (100.0, 200.0),
        }
    ]
    idx_map = {"A": 1}
    for p in range(1, 4):
        G.add_node(site_id(1, "pop", p), metro_idx=1, site_kind="pop", site_ordinal=p)
    G.add_node(site_id(1, "dc", 1), metro_idx=1, site_kind="dc", site_ordinal=1)
    gp._assign_site_positions(G, metros, idx_map)
    for _n, data in G.nodes(data=True):
        assert "pos_x" in data and "pos_y" in data
        assert data["radius_m"] == 10000.0


def test_add_intra_metro_edges_cost_arc() -> None:
    G = nx.MultiGraph()
    metros = [
        {"name": "A", "x": 0.0, "y": 0.0, "radius_km": 10.0, "node_key": (0.0, 0.0)}
    ]
    idx_map = {"A": 1}
    for p in range(1, 4):
        G.add_node(site_id(1, "pop", p), metro_idx=1, site_kind="pop", site_ordinal=p)
    metro_settings = {
        "A": {
            "pop_per_metro": 3,
            "dc_regions_per_metro": 0,
            "intra_metro_link": {"capacity": 3200, "cost": 1},
        }
    }
    gp._add_intra_metro_edges(G, metros, metro_settings, idx_map)
    assert G.number_of_edges() == 3
    # Three equally spaced PoPs have the same shortest ring-arc distance.
    data = G.get_edge_data("metro1/pop1", "metro1/pop3")
    any_key = next(iter(data))
    assert data[any_key]["cost"] >= 1


def test_add_inter_metro_edges_one_to_one() -> None:
    G = nx.MultiGraph()
    metros = [
        {"name": "A", "x": 0.0, "y": 0.0, "radius_km": 10.0, "node_key": (0.0, 0.0)},
        {
            "name": "B",
            "x": 100.0,
            "y": 0.0,
            "radius_km": 10.0,
            "node_key": (100.0, 0.0),
        },
    ]
    idx_map = {"A": 1, "B": 2}
    for p in range(1, 3):
        G.add_node(site_id(1, "pop", p), site_kind="pop")
        G.add_node(site_id(2, "pop", p), site_kind="pop")

    corridors = nx.MultiGraph()
    corridors.add_edge(
        (0.0, 0.0),
        (100.0, 0.0),
        key=0,
        length_km=500.0,
        edge_type="corridor",
        risk_groups=[],
    )

    metro_settings = {
        "A": {
            "pop_per_metro": 2,
            "dc_regions_per_metro": 0,
            "inter_metro_link": {
                "capacity": 3200,
                "cost": 500,
                "mode": "one_to_one",
                "role_pairs": ["core|core"],
            },
        },
        "B": {"pop_per_metro": 2, "dc_regions_per_metro": 0},
    }
    gp._add_inter_metro_edges(
        G,
        metro_settings,
        corridors,
        idx_map,
        {(0.0, 0.0): metros[0], (100.0, 0.0): metros[1]},
        gp.StripePlanner(thaw(load_library(Path("lib")).blueprints), set()),
    )
    assert G.number_of_edges() == 2
    for _u, _v, d in G.edges(data=True):
        assert d["link_type"] == "inter_metro_corridor" and d["cost"] == 500


def test_to_network_sections_serializes_groups_and_adjacency() -> None:
    G = nx.MultiGraph()
    metros = [
        {"name": "A", "x": 0.0, "y": 0.0, "radius_km": 10.0, "node_key": (0.0, 0.0)},
        {
            "name": "B",
            "x": 100.0,
            "y": 0.0,
            "radius_km": 10.0,
            "node_key": (100.0, 0.0),
        },
    ]
    settings = {
        "A": {
            "pop_per_metro": 1,
            "dc_regions_per_metro": 1,
            "site_blueprint": "SingleRouter",
            "dc_region_blueprint": "DCRegion",
        },
        "B": {
            "pop_per_metro": 1,
            "dc_regions_per_metro": 0,
            "site_blueprint": "SingleRouter",
            "dc_region_blueprint": "DCRegion",
        },
    }
    G.add_node("metro1/pop1", site_blueprint="SingleRouter", site_kind="pop")
    G.add_node("metro1/dc1", site_blueprint="DCRegion", site_kind="dc")
    G.add_node("metro2/pop1", site_blueprint="SingleRouter", site_kind="pop")
    G.add_edge(
        "metro1/pop1",
        "metro2/pop1",
        key="k",
        link_type="inter_metro_corridor",
        base_capacity=3200,
        cost=500,
        source_metro="A",
        target_metro="B",
    )
    for index, metro in enumerate(metros):
        metro["name_orig"] = metro["name"]
        metro["metro_id"] = str(index)
    groups, adjacency = nw.to_network_sections(G, metros, settings, _cfg())
    assert any(path.endswith("/pop[1-1]") for path in groups)
    assert any(path.endswith("/dc[1-1]") for path in groups)
    assert any(
        isinstance(a.get("attrs", {}).get("target_capacity"), float) for a in adjacency
    )
