"""Small explicit corridor and site inventories for pipeline contracts."""

import networkx as nx
import pytest

from topogen.config import TopologyConfig


@pytest.fixture
def corridor_graph():
    graph = nx.MultiGraph()
    for i, name in enumerate(("alpha", "beta", "gamma")):
        graph.add_node(
            i,
            node_type="metro",
            name=name,
            name_orig=name,
            metro_id=str(i),
            x=i * 300000.0,
            y=0.0,
            radius_km=20.0,
        )
    for a, b in ((0, 1), (1, 2), (0, 2)):
        graph.add_edge(a, b, edge_type="corridor", length_km=300.0)
    return graph


@pytest.fixture
def scenario_config():
    config = TopologyConfig()
    config.build.build_defaults.pop_per_metro = 1
    config.build.build_defaults.dc_regions_per_metro = 1
    config.traffic.gbps_per_mw = 1
    config.traffic.mw_per_dc_region = 10
    config.traffic.priority_ratios = {0: 1}
    return config


@pytest.fixture
def make_plan(corridor_graph, scenario_config):
    from copy import deepcopy

    from topogen.scenario.records import StripeGroups
    from topogen.scenario.sites import (
        plan_sites,
        read_corridors,
        resolve_metro_settings,
    )

    def make(traffic=None, counts=None):
        config = deepcopy(scenario_config)
        if traffic is not None:
            config.traffic = traffic
        if counts is not None:
            config.build.build_overrides = {
                name: {"dc_regions_per_metro": count}
                for name, count in zip(("alpha", "beta", "gamma"), counts, strict=True)
            }
        corridors = read_corridors(corridor_graph)
        return plan_sites(
            corridors,
            resolve_metro_settings(corridors, config.build, config.traffic),
            StripeGroups({}),
        )

    return make


@pytest.fixture
def site_plan(make_plan):
    return make_plan()
