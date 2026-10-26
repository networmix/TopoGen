"""Directed rational loads and capacity boundaries have independent witnesses."""

import io
import json
import random
from dataclasses import replace
from fractions import Fraction as F

import networkx as nx
import pytest
import yaml
from ngraph import Scenario, prepare, run, write_json

from topogen.config import TopologyConfig
from topogen.naming import site_edge_id
from topogen.scenario.records import (
    Metro,
    Site,
    SiteDemand,
    SiteLink,
    SitePlan,
    TrafficMatrix,
)
from topogen.scenario.sizing import _split_loads, size_links

A = [(1, 2, 1), (1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 4, 1), (3, 4, 1), (3, 4, 1)]


def make_link(source, target, kind, cost=1, base=1, key="k"):
    return SiteLink(
        site_edge_id(source, target, key),
        kind,
        source,
        target,
        cost,
        cost,
        base,
        {},
        {},
        key,
        role_pairs=(),
        risk_groups=(),
        euclidean_km=None,
        detour_ratio=None,
    )


def inputs(edges=A, volume=120, target=4):
    metros = {
        node: Metro(node, str(node), str(node), str(node), node * 1000.0, 0.0, 1.0)
        for node in sorted({n for u, v, _ in edges for n in (u, v)})
    }
    sites = tuple(
        Site(
            f"metro{node}/{kind}1",
            metro,
            kind,
            1,
            "SingleRouter",
            10 if kind == "dc" else None,
            (metro.x, metro.y),
        )
        for node, metro in metros.items()
        for kind in ("pop", "dc")
    )
    links = tuple(
        make_link(
            f"metro{u}/pop1", f"metro{v}/pop1", "inter_metro_corridor", cost, key=str(i)
        )
        for i, (u, v, cost) in enumerate(edges)
    )
    plan = SitePlan(sites, links, node_rules=())
    config = TopologyConfig()
    config.build.tm_sizing.enabled = True
    config.build.tm_sizing.headroom = 1.5
    config.build.tm_sizing.quantum_gbps = 10
    config.build.tm_sizing.respect_min_base_capacity = False
    traffic = (
        TrafficMatrix(
            "default",
            (SiteDemand("metro1/dc1", f"metro{target}/dc1", 0, F(str(volume))),),
        ),
    )
    return plan, config, traffic


def metro_graph(edges):
    graph = nx.MultiDiGraph()
    for index, (u, v, cost) in enumerate(edges):
        graph.add_edge(str(u), str(v), key=str(index), cost=cost)
        graph.add_edge(str(v), str(u), key=str(index), cost=cost)
    return graph


def test_exact_local_parallel_load_vector():
    loads = _split_loads(metro_graph(A), (("1", "4", F(120)),))
    assert [loads[str(u), str(v), str(i)] for i, (u, v, _) in enumerate(A)] == [
        40,
        40,
        40,
        80,
        F(40, 3),
        F(40, 3),
        F(40, 3),
    ]
    assert all(loads[str(v), str(u), str(i)] == 0 for i, (u, v, _) in enumerate(A))
    graph, config, traffic = inputs()
    result = size_links(graph, traffic, config)
    assert sorted(result.capacity.values()) == [20, 20, 20, 60, 60, 60, 120]


@pytest.mark.parametrize(
    "volume,stepped,exact",
    [
        ("119.999", 20, F(119999, 6000)),
        ("120", 20, F(20)),
        ("120.001", 30, F(120001, 6000)),
    ],
)
def test_step_neighbors_and_no_quantum(volume, stepped, exact):
    graph, config, traffic = inputs(volume=volume)
    refs = [
        link.id
        for link in graph.links
        if {link.source, link.target} == {"metro3/pop1", "metro4/pop1"}
    ]
    assert [size_links(graph, traffic, config).capacity[e] for e in refs] == [
        stepped
    ] * 3
    config.build.tm_sizing.quantum_gbps = 0
    assert [size_links(graph, traffic, config).capacity[e] for e in refs] == [exact] * 3


def test_equal_cost_different_hops_and_accumulation():
    edges = [(1, 2, 2), (1, 3, 1), (3, 2, 1)]
    graph = metro_graph(edges)
    demands = (("1", "2", F(60)), ("1", "2", F(60)), ("2", "1", F(20)))
    loads = _split_loads(graph, demands)
    assert [loads[str(u), str(v), str(i)] for i, (u, v, _) in enumerate(edges)] == [
        60,
        60,
        60,
    ]
    assert [loads[str(v), str(u), str(i)] for i, (u, v, _) in enumerate(edges)] == [
        10,
        10,
        10,
    ]


@pytest.mark.parametrize(
    "volume,h,q,expected",
    [(F(1, 10), 10, 1, 1), (F(5000), 1.3, 100, 6500), (F(2**63), 1, 0, 2**63)],
)
def test_decimal_and_wide_totals(volume, h, q, expected):
    graph, config, traffic = inputs([(1, 2, 1)], volume=volume, target=2)
    config.build.tm_sizing.headroom, config.build.tm_sizing.quantum_gbps = h, q
    assert list(size_links(graph, traffic, config).capacity.values()) == [expected]


def test_peak_minimum_unused_and_derived_budgets():
    graph, config, _ = inputs([(1, 2, 1), (1, 3, 1)], target=2)
    extra_site = Site(
        "metro1/pop2", graph.sites[0].metro, "pop", 2, "SingleRouter", None, (0.0, 0.0)
    )
    graph = replace(
        graph,
        sites=(*graph.sites, extra_site),
        links=(
            replace(graph.links[0], base_capacity=100),
            replace(graph.links[1], base_capacity=200),
            make_link("metro1/pop2", "metro2/pop1", "inter_metro_corridor", base=80),
            make_link("metro1/dc1", "metro1/pop1", "dc_to_pop"),
            make_link("metro2/dc1", "metro1/pop2", "dc_to_pop"),
            make_link("metro1/pop1", "metro1/pop2", "intra_metro"),
        ),
    )
    zero = (
        TrafficMatrix("default", (SiteDemand("metro1/dc1", "metro2/dc1", 0, F(0)),)),
    )
    result = size_links(graph, zero, config)
    for link in graph.links:
        if link.kind == "dc_to_pop":
            assert result.capacity[link.id] == (
                360 if "metro1/pop1" in (link.source, link.target) else 100
            )
        if link.kind == "intra_metro":
            assert result.capacity[link.id] == 70
    assert graph.links[0].base_capacity == 100


def test_unreachable_demand_refuses_without_mutation():
    graph, config, traffic = inputs([(1, 2, 1), (3, 4, 1)])
    before = graph
    with pytest.raises(ValueError, match="no corridor route"):
        size_links(graph, traffic, config)
    assert graph == before
    config.build.tm_sizing.enabled = False
    assert list(size_links(graph, traffic, config).capacity.values()) == [1, 1]


def test_random_loads_against_yaml_ecmp():
    rng = random.Random(90210)
    cases = [A] + [
        [(i, i + 1, 1) for i in range(1, 5)]
        + [(rng.randint(1, 4), 5, rng.randint(1, 4)) for _ in range(6)]
        for _ in range(40)
    ]
    saw_parallel = saw_split = False
    for edges in cases:
        graph = metro_graph(edges)
        target = str(max(graph.nodes, key=int))
        expected = _split_loads(graph, (("1", target, F(120)),))
        doc = {
            "units": {"resolution": 2**-10},
            "network": {
                "nodes": {n: {} for n in graph},
                "links": [
                    {
                        "source": str(u),
                        "target": str(v),
                        "capacity": 100000,
                        "cost": cost,
                        "attrs": {"index": str(i)},
                    }
                    for i, (u, v, cost) in enumerate(edges)
                ],
            },
            "demands": {
                "tm": [
                    {
                        "source": "^1$",
                        "target": f"^{target}$",
                        "volume": 120,
                        "preset": "SHORTEST_PATHS_ECMP",
                    }
                ]
            },
            "workflow": [
                {
                    "type": "TrafficMatrixPlacement",
                    "name": "p",
                    "demand_set": "tm",
                    "workers": 1,
                    "outputs": {"edge_flows": True},
                }
            ],
        }
        prep = prepare(Scenario.from_yaml(yaml.safe_dump(doc)), demand_sets=("tm",))
        stream = io.StringIO()
        write_json(run(prep), stream)
        payload = json.loads(stream.getvalue())
        row = payload["steps"][0]["result"]["cases"][0]["states"]["converged"]["rows"][
            0
        ]
        observed = {edge: 0 for edge in expected}
        for flow in row["result"]["edge_flows"]:
            link = prep.scenario.network.links[flow["edge"]["link"]]
            u, v = link.source, link.target
            if flow["edge"]["direction"] == "reverse":
                u, v = v, u
            observed[u, v, link.attrs["index"]] = flow["units"] * prep.topology.q
        assert all(
            abs(observed[e] - expected[e]) <= len(graph) * prep.topology.q
            for e in expected
        )
        saw_split |= len([v for (u, _, _), v in expected.items() if u == "1" and v]) > 1
        saw_parallel |= len(edges) > len({tuple(sorted((u, v))) for u, v, _ in edges})
    assert len(cases) == 41 and saw_split and saw_parallel


@pytest.mark.parametrize(
    "respect,base,expected", [(False, 2000, 1000), (True, 50, 1000), (True, 2000, 2000)]
)
def test_directional_peak_and_minimum(respect, base, expected):
    graph, config, _ = inputs([(1, 2, 1)], target=2)
    graph = replace(graph, links=(replace(graph.links[0], base_capacity=base),))
    config.build.tm_sizing.headroom = 1
    config.build.tm_sizing.quantum_gbps = 1
    config.build.tm_sizing.respect_min_base_capacity = respect
    demands = (
        SiteDemand("metro1/dc1", "metro2/dc1", 0, F(1000)),
        SiteDemand("metro2/dc1", "metro1/dc1", 0, F(100)),
    )
    result = size_links(graph, (TrafficMatrix("default", demands),), config)
    assert list(result.load.values()) == [1000]
    assert list(result.capacity.values()) == [expected]
    assert graph.links[0].base_capacity == base


@pytest.mark.parametrize(
    "bad,error",
    [
        ("disabled_traffic", "traffic generation is disabled"),
        ("no_dc", "no DC"),
        ("missing_matrix", "available:"),
        ("empty_matrix", "is empty"),
        ("no_corridor", "no inter-metro"),
        ("bad_link_endpoint", "unsupported site-link endpoint"),
        ("bad_cost", "positive integer cost"),
        ("no_pop", "does not connect to a PoP endpoint"),
        ("negative_volume", "finite non-negative"),
        ("bad_endpoint", "unsupported demand endpoint"),
    ],
)
def test_sizing_refusal_boundaries(bad, error):
    graph, config, traffic = inputs([(1, 2, 1)], target=2)
    assert size_links(graph, traffic, config).capacity
    if bad == "disabled_traffic":
        config.traffic.enabled = False
    elif bad == "no_dc":
        graph = replace(
            graph, sites=tuple(site for site in graph.sites if site.kind != "dc")
        )
    elif bad == "missing_matrix":
        config.build.tm_sizing.matrix_name = "missing"
    elif bad == "empty_matrix":
        traffic = (TrafficMatrix("default", ()),)
    elif bad == "no_corridor":
        graph = replace(graph, links=())
    elif bad == "bad_link_endpoint":
        graph = replace(graph, links=(replace(graph.links[0], source="missing"),))
    elif bad == "bad_cost":
        graph = replace(graph, links=(replace(graph.links[0], cost=0),))
    elif bad == "no_pop":
        graph = replace(
            graph,
            links=(*graph.links, make_link("metro1/dc1", "metro2/dc1", "dc_to_pop")),
        )
    elif bad == "negative_volume":
        traffic = (
            TrafficMatrix(
                "default", (SiteDemand("metro1/dc1", "metro2/dc1", 0, F(-1)),)
            ),
        )
    elif bad == "bad_endpoint":
        traffic = (
            TrafficMatrix("default", (SiteDemand("missing", "metro2/dc1", 0, F(120)),)),
        )
    with pytest.raises(ValueError, match=error):
        size_links(graph, traffic, config)


def test_explicit_sizing_matrix_overrides_traffic_default():
    graph, config, traffic = inputs([(1, 2, 1)], volume=10, target=2)
    selected = TrafficMatrix(
        "selected", (SiteDemand("metro1/dc1", "metro2/dc1", 0, F(100)),)
    )
    assert list(size_links(graph, traffic + (selected,), config).capacity.values()) == [
        20
    ]
    config.build.tm_sizing.matrix_name = "selected"
    assert list(size_links(graph, traffic + (selected,), config).capacity.values()) == [
        150
    ]


def test_minimum_applies_after_quantum_when_base_exceeds_the_step():
    graph, config, traffic = inputs([(1, 2, 1)], volume=100, target=2)
    graph = replace(graph, links=(replace(graph.links[0], base_capacity=2000),))
    config.build.tm_sizing.headroom = 1
    config.build.tm_sizing.quantum_gbps = 1600
    config.build.tm_sizing.respect_min_base_capacity = True
    assert list(size_links(graph, traffic, config).capacity.values()) == [2000]
    config.build.tm_sizing.respect_min_base_capacity = False
    assert list(size_links(graph, traffic, config).capacity.values()) == [1600]
