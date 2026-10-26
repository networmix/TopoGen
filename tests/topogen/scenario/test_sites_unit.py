"""Site order, ring geometry and corridor metadata have independent witnesses."""

import math
from dataclasses import replace

import networkx as nx
import pytest

from topogen.scenario.records import StripeGroups
from topogen.scenario.sites import plan_sites, read_corridors, resolve_metro_settings


def test_read_preserves_graph_order_parallel_keys_and_owns_risks(corridor_graph):
    graph = nx.relabel_nodes(corridor_graph, {0: (9, 0), 1: (1, 0), 2: (5, 0)})
    graph.add_edge(
        (9, 0),
        (1, 0),
        key="second",
        length_km=450.0,
        euclidean_km=300.0,
        detour_ratio=1.5,
        risk_groups=["shared"],
    )
    corridors = read_corridors(graph)
    assert [(m.index, m.name) for m in corridors.metros] == [
        (1, "alpha"),
        (2, "beta"),
        (3, "gamma"),
    ]
    assert [
        (c.source.name, c.target.name, c.path, c.length_km) for c in corridors.corridors
    ] == [
        ("alpha", "beta", 0, 300.0),
        ("alpha", "beta", "second", 450.0),
        ("alpha", "gamma", 0, 300.0),
        ("beta", "gamma", 0, 300.0),
    ]
    second = corridors.corridors[1]
    assert (second.euclidean_km, second.detour_ratio, second.risk_groups) == (
        300.0,
        1.5,
        ("shared",),
    )
    assert second.source is corridors.metros[0] and second.target is corridors.metros[1]
    graph[(9, 0)][(1, 0)]["second"]["risk_groups"].append("later")
    graph.nodes[(9, 0)]["name"] = "changed"
    assert second.risk_groups == ("shared",) and second.source.name == "alpha"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("name", "", "nonempty"),
        ("metro_id", "", "nonempty"),
        ("x", True, "finite numeric"),
        ("y", float("nan"), "finite numeric"),
        ("radius_km", float("inf"), "finite numeric"),
        ("radius_km", -1, "negative radius"),
    ],
)
def test_bad_metro_fields_refuse(corridor_graph, field, value, message):
    corridor_graph.nodes[0][field] = value
    with pytest.raises(ValueError, match=message):
        read_corridors(corridor_graph)


def test_missing_duplicate_wrong_graph_and_empty(corridor_graph):
    for graph in (nx.Graph(), nx.MultiDiGraph()):
        with pytest.raises(TypeError, match="undirected corridor MultiGraph"):
            read_corridors(graph)
    assert read_corridors(nx.MultiGraph()).metros == ()
    corridor_graph.nodes[1]["metro_id"] = "0"
    with pytest.raises(ValueError, match="Duplicate metro metro_id: 0"):
        read_corridors(corridor_graph)
    corridor_graph.nodes[1]["metro_id"] = "1"
    del corridor_graph.nodes[0]["x"]
    with pytest.raises(ValueError, match="missing required attributes.*x"):
        read_corridors(corridor_graph)
    corridor_graph.nodes[0]["x"] = 0
    corridor_graph.nodes[0]["node_type"] = "other"
    with pytest.raises(ValueError, match="Expected metro node"):
        read_corridors(corridor_graph)


def test_site_order_ring_arcs_and_link_orientation(corridor_graph, scenario_config):
    cfg = scenario_config
    cfg.build.build_defaults.pop_per_metro = 3
    cfg.build.build_defaults.dc_regions_per_metro = 2
    cfg.build.build_defaults.inter_metro_link.mode = "one_to_one"
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    plan = plan_sites(corridors, settings, StripeGroups({}))
    assert [s.id for s in plan.sites] == [
        f"metro{i}/{kind}{n}"
        for i in range(1, 4)
        for kind, count in (("pop", 3), ("dc", 2))
        for n in range(1, count + 1)
    ]
    kinds = [link.kind for link in plan.links]
    assert (
        kinds == ["intra_metro"] * 9 + ["dc_to_pop"] * 18 + ["inter_metro_corridor"] * 9
    )
    order = {site.id: i for i, site in enumerate(plan.sites)}
    assert all(order[link.source] < order[link.target] for link in plan.links)
    assert [link.cost for link in plan.links[:3]] == [math.ceil(40 * math.pi / 3)] * 3
    assert plan.sites[0].position == (14000.0, 0.0)
    # DC2 lies halfway between PoPs 2 and 3, so each is a unit-cost exit.
    assert [
        link.distance_km
        for link in plan.links
        if link.kind == "dc_to_pop" and link.source.startswith("metro1/")
    ] == [1, 42, 42, 42, 1, 1]
    dc = plan.sites[3]
    assert dc.position == pytest.approx((8000 / math.sqrt(2), 8000 / math.sqrt(2)))
    assert all(
        site.metro is corridors.metros[site.metro.index - 1] for site in plan.sites
    )
    wan = [link for link in plan.links if link.kind == "inter_metro_corridor"]
    assert all(link.source.split("/")[1] == link.target.split("/")[1] for link in wan)
    assert wan[0].id == "metro1/pop1|metro2/pop1|inter_metro:1-2:path0:1-1"
    assert all(link.distance_km == 300 and link.cost == 300 for link in wan)


def test_policy_from_first_graph_metro_and_parallel_order(
    corridor_graph, scenario_config
):
    cfg = scenario_config
    cfg.build.build_overrides = {
        "alpha": {"inter_metro_link": {"cost": 700, "attrs": {"policy": "alpha"}}},
        "beta": {"inter_metro_link": {"cost": 900, "attrs": {"policy": "beta"}}},
    }
    corridor_graph.add_edge(0, 1, key=9, length_km=450.0)
    corridors = read_corridors(corridor_graph)
    plan = plan_sites(
        corridors,
        resolve_metro_settings(corridors, cfg.build, cfg.traffic),
        StripeGroups({}),
    )
    wan = [link for link in plan.links if link.kind == "inter_metro_corridor"]
    assert [link.cost for link in wan] == [700, 700, 700, 900]
    assert [link.attrs["policy"] for link in wan] == ["alpha", "alpha", "alpha", "beta"]
    assert [link.distance_km for link in wan] == [300, 450, 300, 300]
    assert "path0" in wan[0].id and "path9" in wan[1].id


def test_zero_radius_boundary_and_ring_cost_floor(corridor_graph, scenario_config):
    corridor_graph.nodes[0]["radius_km"] = 0
    cfg = scenario_config
    corridors = read_corridors(corridor_graph)
    settings = resolve_metro_settings(corridors, cfg.build, cfg.traffic)
    with pytest.raises(ValueError, match="expected > 0 for ring-based adjacency"):
        plan_sites(corridors, settings, StripeGroups({}))
    settings = (replace(settings[0], dcs=0, dc_power_mw=()), *settings[1:])
    plan = plan_sites(corridors, settings, StripeGroups({}))
    assert plan.sites[0].position == (0, 0)
    assert all(link.cost >= 1 for link in plan.links)


def test_ring_positions_use_each_metros_center_radius_and_both_axes(
    corridor_graph, scenario_config
):
    corridor_graph.nodes[1].update(x=300000.0, y=0.0, radius_km=40.0)
    corridor_graph.nodes[2].update(x=300000.0, y=400000.0, radius_km=20.0)
    cfg = scenario_config
    cfg.build.build_defaults.pop_per_metro = (
        cfg.build.build_defaults.dc_regions_per_metro
    ) = 2
    corridors = read_corridors(corridor_graph)
    plan = plan_sites(
        corridors,
        resolve_metro_settings(corridors, cfg.build, cfg.traffic),
        StripeGroups({}),
    )
    diagonal = 4000 * math.sqrt(2)
    expected = [
        (14000.0, 0.0),
        (-14000.0, 0.0),
        (diagonal, diagonal),
        (-diagonal, -diagonal),
        (328000.0, 0.0),
        (272000.0, 0.0),
        (300000 + 2 * diagonal, 2 * diagonal),
        (300000 - 2 * diagonal, -2 * diagonal),
        (314000.0, 400000.0),
        (286000.0, 400000.0),
        (300000 + diagonal, 400000 + diagonal),
        (300000 - diagonal, 400000 - diagonal),
    ]
    assert len(plan.sites) == len(expected) == 12
    for site, position in zip(plan.sites, expected, strict=True):
        assert site.position == pytest.approx(position, rel=0, abs=1e-8)
        assert all(type(value) is float for value in site.position)


def test_multi_digit_site_ordinals_keep_numeric_orientation(
    corridor_graph, scenario_config
):
    cfg = scenario_config
    cfg.build.build_defaults.pop_per_metro = 10
    cfg.build.build_defaults.dc_regions_per_metro = 0
    corridors = read_corridors(corridor_graph.subgraph([0]).copy())
    plan = plan_sites(
        corridors,
        resolve_metro_settings(corridors, cfg.build, cfg.traffic),
        StripeGroups({}),
    )
    assert [site.id for site in plan.sites] == [f"metro1/pop{i}" for i in range(1, 11)]
    assert len(plan.links) == 45
    link = next(
        link
        for link in plan.links
        if {link.source, link.target} == {"metro1/pop2", "metro1/pop10"}
    )
    assert (link.source, link.target, link.id) == (
        "metro1/pop2",
        "metro1/pop10",
        "metro1/pop2|metro1/pop10|intra_metro:alpha:2-10",
    )
