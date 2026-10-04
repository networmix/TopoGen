"""Behavioral contracts shared by sizing, device expansion and hardware."""

import math

import networkx as nx
import yaml
from ngraph.dsl.blueprints.expand import expand_network_dsl

from topogen.config import TopologyConfig
from topogen.scenario import build_scenario


def test_optics_follow_each_site_edge_capacity():
    """Different PoP egress budgets in one metro need different optic counts."""
    graph = nx.MultiGraph()
    for i, name in enumerate(("alpha", "beta")):
        graph.add_node(
            (i * 100_000.0, 0.0),
            node_type="metro",
            name=name,
            metro_id=name,
            x=i * 100_000.0,
            y=0.0,
            radius_km=10.0,
            name_orig=name,
        )
    graph.add_edge((0.0, 0.0), (100_000.0, 0.0), edge_type="corridor", length_km=100.0)
    cfg = TopologyConfig()
    defaults = cfg.build.build_defaults
    defaults.pop_per_metro = 3
    defaults.dc_regions_per_metro = 1
    defaults.inter_metro_link.mode = "one_to_one"
    defaults.inter_metro_link.attrs = {"owner": "fixture"}
    defaults.intra_metro_link.role_pairs = ["core|core"]
    defaults.inter_metro_link.role_pairs = ["core|core"]
    defaults.dc_to_pop_link.role_pairs = ["dc|core"]
    cfg.build.build_overrides = {"beta": {"pop_per_metro": 2}}
    cfg.build.tm_sizing.enabled = True
    cfg.traffic.enabled = True
    cfg.traffic.model = "uniform"
    cfg.traffic.mw_per_dc_region = 100.0
    cfg.traffic.gbps_per_mw = 200.0
    cfg.components.hw_component = {"core": "CoreRouter", "dc": ""}
    cfg.components.optics = {
        "core->core": "800G-ZR+",
        "dc->core": "800G-ZR+",
        "core->dc": "800G-ZR+",
    }

    scenario = yaml.safe_load(build_scenario(graph, cfg))
    network = expand_network_dsl(scenario)
    wan = [
        link
        for link in network.links.values()
        if link.attrs["link_type"] == "inter_metro_corridor"
    ]
    assert wan and all(link.attrs["owner"] == "fixture" for link in wan)
    capacities = {link.capacity for link in network.links.values()}
    assert len(capacities) > 1
    for link in network.links.values():
        for endpoint in ("source", "target"):
            hardware = link.attrs["hardware"][endpoint]
            component = scenario["components"][hardware["component"]]
            expected = math.ceil(link.capacity / component["capacity"])
            assert hardware["count"] == expected, (
                link.source,
                link.target,
                link.capacity,
                hardware,
            )


def test_directional_optics_are_resolved_for_each_endpoint():
    import pytest

    from topogen.naming import site_edge_id
    from topogen.scenario.expansion import resolve_network

    graph = nx.MultiGraph()
    graph.add_edge("a", "b", key="wan", base_capacity=2400)
    edge_id = site_edge_id("a", "b", "wan")
    scenario = {
        "components": {"small": {"capacity": 800}, "large": {"capacity": 1600}},
        "network": {
            "nodes": {
                "a": {"attrs": {"role": "leaf"}},
                "b": {"attrs": {"role": "spine"}},
            },
            "links": [
                {
                    "source": "a",
                    "target": "b",
                    "capacity": 1,
                    "attrs": {"site_edge": edge_id},
                }
            ],
        },
    }
    network = resolve_network(
        graph, scenario, {"leaf->spine": "small", "spine->leaf": "large"}
    )
    expected = {
        "source": {"component": "small", "count": 3},
        "target": {"component": "large", "count": 2},
    }
    assert graph["a"]["b"]["wan"]["hardware"] == expected
    assert scenario["network"]["links"][0]["attrs"]["hardware"] == expected
    assert next(iter(network.links.values())).attrs["hardware"] == expected
    with pytest.raises(ValueError, match="no target-end optic mapping"):
        resolve_network(graph, scenario, {"leaf->spine": "small"})
