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
def site_plan():
    graph = nx.MultiGraph()
    for i, name in enumerate(("alpha", "beta", "gamma"), 1):
        graph.add_node(
            f"metro{i}/dc1",
            site_kind="dc",
            site_ordinal=1,
            metro_name=name,
            center_x=(i - 1) * 300000.0,
            center_y=0.0,
        )
        graph.add_node(
            f"metro{i}/pop1",
            site_kind="pop",
            site_ordinal=1,
            metro_name=name,
            center_x=(i - 1) * 300000.0,
            center_y=0.0,
        )
    return graph
