"""Traffic sizing must preserve directional load and NetGraph demand semantics."""

import math
from unittest.mock import patch

import networkx as nx
import pytest
from ngraph import Network, Node
from ngraph.analysis.demand import expand_demands
from ngraph.model.demand.builder import build_demand_set

from topogen.config import TopologyConfig
from topogen.scenario.graph_pipeline import tm_based_size_capacities
from topogen.scenario.traffic import _build_traffic_matrix_section


def _inputs(base_capacity=1.0):
    config = TopologyConfig()
    config.traffic.model = "uniform"
    config.traffic.gbps_per_mw = 10.0
    config.traffic.mw_per_dc_region = 1.0
    config.traffic.priority_ratios = {0: 1.0}
    config.build.tm_sizing.enabled = True
    config.build.tm_sizing.quantum_gbps = 1.0
    config.build.tm_sizing.headroom = 1.0
    config.build.tm_sizing.respect_min_base_capacity = False
    metros = [{"name": "A"}, {"name": "B"}]
    settings = {
        "A": {"dc_regions_per_metro": 2},
        "B": {"dc_regions_per_metro": 1},
    }
    graph = nx.MultiGraph()
    graph.add_edge(
        "metro1/pop1",
        "metro2/pop1",
        key="corridor",
        link_type="inter_metro_corridor",
        base_capacity=base_capacity,
        target_capacity=base_capacity,
        cost=1,
        source_metro="A",
        target_metro="B",
    )
    return graph, metros, settings, config


@pytest.mark.parametrize(
    "respect_min,base,expected",
    [(False, 2000, 1000), (True, 50, 1000), (True, 2000, 2000)],
)
def test_capacity_covers_both_directions(respect_min, base, expected):
    graph, metros, settings, config = _inputs(base)
    config.build.tm_sizing.respect_min_base_capacity = respect_min
    demands = [
        {"source_path": "^metro1/dc1", "sink_path": "^metro2/dc1", "demand": 1000.0},
        {"source_path": "^metro2/dc1", "sink_path": "^metro1/dc1", "demand": 100.0},
    ]
    with patch(
        "topogen.traffic_matrix.generate_traffic_matrix",
        return_value={"default": demands},
    ):
        tm_based_size_capacities(graph, metros, settings, config)
    edge = graph["metro1/pop1"]["metro2/pop1"]["corridor"]
    assert edge["base_capacity"] == expected
    assert edge["target_capacity"] == expected


@pytest.mark.parametrize("dc_counts", [(2, 1), (1, 1), (3, 2), (2, 2)])
def test_uniform_sizing_matches_netgraph_pairwise_expansion(dc_counts):
    graph, metros, settings, config = _inputs()
    settings["A"]["dc_regions_per_metro"] = dc_counts[0]
    settings["B"]["dc_regions_per_metro"] = dc_counts[1]
    config.traffic.priority_ratios = {0: 0.75, 1: 0.25}
    network = Network()
    for metro, count in enumerate(dc_counts, 1):
        for dc in range(1, count + 1):
            network.add_node(Node(f"metro{metro}/dc{dc}/dc"))
    demand_set = build_demand_set(
        _build_traffic_matrix_section(metros, settings, config)
    )
    expanded = expand_demands(network, demand_set.get_set("default"))
    # NetGraph divides the volume among all ordered, distinct DC pairs;
    # same-metro pairs consume part of the volume but no inter-metro capacity.
    expected = sum(
        d.volume
        for d in expanded.demands
        if d.src_name.startswith("metro1/") and d.dst_name.startswith("metro2/")
    )
    assert expected > 0
    tm_based_size_capacities(graph, metros, settings, config)
    assert graph["metro1/pop1"]["metro2/pop1"]["corridor"][
        "base_capacity"
    ] == math.ceil(expected)


def test_unknown_sizing_selector_is_not_silently_dropped():
    graph, metros, settings, config = _inputs()
    demands = [{"source_path": "unknown", "sink_path": "^metro2/dc1", "demand": 100.0}]
    with patch(
        "topogen.traffic_matrix.generate_traffic_matrix",
        return_value={"default": demands},
    ):
        with pytest.raises(ValueError, match="unsupported.*endpoint"):
            tm_based_size_capacities(graph, metros, settings, config)


def test_sizing_preserves_corridors_between_distinct_pop_pairs():
    graph, metros, settings, config = _inputs()
    # NetworkX edge keys are unique only within a node pair. Collapsing two
    # different PoP pairs to one metro pair must not overwrite either corridor.
    graph.add_edge(
        "metro1/pop2",
        "metro2/pop2",
        key="corridor",
        **graph["metro1/pop1"]["metro2/pop1"]["corridor"],
    )
    tm_based_size_capacities(graph, metros, settings, config)
    assert [data["base_capacity"] for _, _, data in graph.edges(data=True)] == [5, 5]
