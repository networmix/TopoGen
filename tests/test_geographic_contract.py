"""Geographic paths retain identity, length, geometry, and shared risks."""

import networkx as nx
import pytest

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.config import (
    CorridorsConfig,
    FormattingConfig,
    RiskGroupsConfig,
    TopologyConfig,
)
from topogen.corridors import add_corridors, assign_risk_groups, extract_corridor_graph
from topogen.integrated_graph import (
    _contract_degree2_chains,
    load_from_json,
    save_to_json,
)
from topogen.metro_clusters import MetroCluster
from topogen.scenario.config import _determine_metro_settings
from topogen.scenario.graph_pipeline import build_site_graph
from topogen.scenario.network import _extract_metros_from_graph
from topogen.scenario.risk import _build_risk_groups_section


def metro(identifier, x):
    return MetroCluster(
        identifier, identifier, identifier, identifier, 1.0, x, 0.0, 0.0
    )


def add_edge(graph, a, b, length):
    graph.add_edge(a, b, length_km=length, geometry=[a, b], segment_id=f"{a}:{b}")


def test_nearest_neighbor_union_does_not_depend_on_id_order():
    metros = [metro("a", 0), metro("b", 1000), metro("z", 4000)]
    graph = nx.Graph()
    add_edge(graph, metros[0].node_key, metros[1].node_key, 1)
    add_edge(graph, metros[1].node_key, metros[2].node_key, 3)
    add_corridors(graph, metros, CorridorsConfig(k_nearest=1))
    assert set(graph.graph["corridor_paths"]) == {("a", "b", 0), ("b", "z", 0)}


def test_contraction_preserves_parallel_routes():
    graph = nx.Graph()
    a, b = (0.0, 0.0), (4000.0, 0.0)
    for y, length in [(1000, 2), (2000, 3), (3000, 4)]:
        mid = (2000.0, float(y))
        add_edge(graph, a, mid, length / 2)
        add_edge(graph, mid, b, length / 2)
    actual = _contract_degree2_chains(graph, {a, b})
    paths = list(nx.shortest_simple_paths(actual, a, b, weight="length_km"))
    assert [
        sum(actual[u][v]["length_km"] for u, v in zip(p, p[1:], strict=False))
        for p in paths
    ] == [2, 3, 4]
    for u, v, data in actual.edges(data=True):
        assert {tuple(data["geometry"][0]), tuple(data["geometry"][-1])} == {u, v}


def test_parallel_corridors_survive_json_and_site_expansion(tmp_path):
    metros = [metro("a", 0), metro("b", 4000)]
    a, b = (m.node_key for m in metros)
    graph = nx.Graph()
    for y, length in [(1000, 5), (2000, 6)]:
        mid = (2000.0, float(y))
        add_edge(graph, a, mid, length / 2)
        add_edge(graph, mid, b, length / 2)
    corridors = CorridorsConfig(
        k_paths=2,
        k_nearest=1,
        risk_groups=RiskGroupsConfig(exclude_metro_radius_shared=False),
    )
    add_corridors(graph, metros, corridors)
    assign_risk_groups(graph, metros, corridors)
    extracted = extract_corridor_graph(graph, metros)
    assert extracted.is_multigraph()
    assert extracted.number_of_edges(a, b) == 2
    assert sorted(d["risk_groups"] for _, _, d in extracted.edges(data=True)) == [
        ["corridor_risk_a_b"],
        ["corridor_risk_a_b_path1"],
    ]
    path = tmp_path / "corridors.json"
    save_to_json(extracted, path, "EPSG:5070", FormattingConfig())
    loaded, crs = load_from_json(path)
    assert crs == "EPSG:5070"
    assert dict(loaded.edges) == dict(extracted.edges)
    config = TopologyConfig()
    config.build.build_defaults.pop_per_metro = 1
    config.build.build_defaults.dc_regions_per_metro = 0
    selected = _extract_metros_from_graph(loaded)
    sites = build_site_graph(
        selected,
        _determine_metro_settings(selected, config),
        loaded,
        get_builtin_blueprints(),
    )
    wan = [
        d
        for _, _, d in sites.edges(data=True)
        if d["link_type"] == "inter_metro_corridor"
    ]
    assert len(wan) == 2
    assert {d["cost"] for d in wan} == {5, 6}

    assert [
        risk["attrs"]["distance_km"]
        for risk in _build_risk_groups_section(loaded, config)
    ] == [5, 6]


def test_component_filter_reports_removed_metro_anchor(monkeypatch):
    from topogen import integrated_graph as pipeline

    metros = [metro("main", 0), metro("island", 100_000)]
    roads = nx.Graph()
    for left, right in ((0, 1000), (1000, 2000), (2000, 0), (100_000, 101_000)):
        add_edge(roads, (float(left), 0.0), (float(right), 0.0), 1.0)
    monkeypatch.setattr(pipeline, "load_metro_clusters", lambda **_kwargs: metros)
    monkeypatch.setattr(pipeline, "build_highway_graph", lambda **_kwargs: roads)
    config = TopologyConfig()
    config.highway_processing.min_edge_length_km = 0
    with pytest.raises(ValueError, match="Highway filtering removed.*island"):
        pipeline.build_integrated_graph(config)
