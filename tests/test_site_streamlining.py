"""Adjacency modes and stripe identities survive actual blueprint expansion."""

import networkx as nx
import pytest
import yaml
from ngraph.dsl.blueprints.expand import expand_network_dsl

from topogen.config import TopologyConfig
from topogen.scenario import build_scenario


def build_case(tmp_path, monkeypatch, blueprint):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/blueprints.yml").write_text(yaml.safe_dump({"Test": blueprint}))
    graph = nx.MultiGraph()
    for i, name in enumerate(("a", "b")):
        graph.add_node(
            (i * 1000.0, 0.0),
            node_type="metro",
            name=name,
            name_orig=name,
            metro_id=name,
            x=i * 1000.0,
            y=0.0,
            radius_km=1.0,
        )
    graph.add_edge((0.0, 0.0), (1000.0, 0.0), length_km=1.0, edge_type="corridor")
    config = TopologyConfig()
    config.traffic.enabled = False
    config.build.build_defaults.dc_regions_per_metro = 0
    config.build.build_defaults.site_blueprint = "Test"
    return graph, config


def test_striping_respects_one_to_one_site_mode(tmp_path, monkeypatch):
    blueprint = {
        "nodes": {"r": {"count": 1, "template": "r{n}", "attrs": {"role": "core"}}}
    }
    graph, config = build_case(tmp_path, monkeypatch, blueprint)
    config.build.build_defaults.inter_metro_link.mode = "one_to_one"
    config.build.build_defaults.inter_metro_link.striping = {
        "mode": "width",
        "width": 1,
    }
    network = expand_network_dsl(yaml.safe_load(build_scenario(graph, config)))
    wan = [
        link
        for link in network.links.values()
        if link.attrs["link_type"] == "inter_metro_corridor"
    ]
    assert len(wan) == 2
    assert all(link.source.split("/")[1] == link.target.split("/")[1] for link in wan)


@pytest.mark.parametrize("template", ["node", "node+{n}"])
def test_stripe_members_keep_full_paths_and_literal_names(
    tmp_path, monkeypatch, template
):
    blueprint = {
        "nodes": {
            "left": {
                "count": 1,
                "template": template,
                "attrs": {"role": "core", "stripe": "a"},
            },
            "right": {
                "count": 1,
                "template": template,
                "attrs": {"role": "core", "stripe": "b"},
            },
        }
    }
    graph, config = build_case(tmp_path, monkeypatch, blueprint)
    config.build.build_defaults.inter_metro_link.striping = {
        "mode": "by_attr",
        "attribute": "stripe",
    }
    network = expand_network_dsl(yaml.safe_load(build_scenario(graph, config)))
    wan = [
        link
        for link in network.links.values()
        if link.attrs["link_type"] == "inter_metro_corridor"
    ]
    assert len(wan) == 4
    assert {network.nodes[link.source].attrs["stripe"] for link in wan} == {"a", "b"}
    assert all(
        network.nodes[link.source].attrs["stripe"]
        == network.nodes[link.target].attrs["stripe"]
        for link in wan
    )


def test_striping_preserves_custom_endpoint_match(tmp_path, monkeypatch):
    blueprint = {
        "nodes": {
            "left": {
                "count": 1,
                "template": "r{n}",
                "attrs": {"role": "core", "chosen": True},
            },
            "right": {
                "count": 1,
                "template": "r{n}",
                "attrs": {"role": "core", "chosen": False},
            },
        }
    }
    graph, config = build_case(tmp_path, monkeypatch, blueprint)
    link = config.build.build_defaults.inter_metro_link
    link.striping = {"mode": "width", "width": 1}
    link.match = {"conditions": [{"attr": "chosen", "op": "==", "value": True}]}
    network = expand_network_dsl(yaml.safe_load(build_scenario(graph, config)))
    wan = [
        link
        for link in network.links.values()
        if link.attrs["link_type"] == "inter_metro_corridor"
    ]
    assert wan and all(
        network.nodes[link.source].attrs["chosen"]
        and network.nodes[link.target].attrs["chosen"]
        for link in wan
    )


def test_disabled_risks_are_not_emitted_from_saved_graph(tmp_path, monkeypatch):
    blueprint = {
        "nodes": {"r": {"count": 1, "template": "r{n}", "attrs": {"role": "core"}}}
    }
    graph, config = build_case(tmp_path, monkeypatch, blueprint)
    for _, _, attrs in graph.edges(data=True):
        attrs["risk_groups"] = ["corridor_risk_a_b"]
    config.corridors.risk_groups.enabled = False
    scenario = yaml.safe_load(build_scenario(graph, config))
    assert "risk_groups" not in scenario
    assert all("risk_groups" not in link for link in scenario["network"]["links"])


def test_configured_cost_is_a_floor_without_changing_distance(tmp_path, monkeypatch):
    graph, config = build_case(
        tmp_path, monkeypatch, {"nodes": {"r": {"count": 1, "attrs": {"role": "core"}}}}
    )
    config.build.build_defaults.inter_metro_link.cost = 500
    scenario = yaml.safe_load(build_scenario(graph, config))
    links = [
        link
        for link in scenario["network"]["links"]
        if link["attrs"]["link_type"] == "inter_metro_corridor"
    ]
    assert all(link["cost"] == 500 for link in links)
    assert all(link["attrs"]["distance_km"] < 500 for link in links)


def test_nested_blueprint_dependencies_are_included(tmp_path, monkeypatch):
    graph, config = build_case(
        tmp_path,
        monkeypatch,
        {"nodes": {"inner": {"blueprint": "SingleRouter", "attrs": {"role": "core"}}}},
    )
    scenario = yaml.safe_load(build_scenario(graph, config))
    assert set(scenario["blueprints"]) == {"Test", "SingleRouter"}
    assert len(expand_network_dsl(scenario).nodes) == 4
