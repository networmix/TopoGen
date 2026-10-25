"""A stage's published graph and numeric maps cannot borrow mutable inputs."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict

import networkx as nx
import pytest

from topogen.library import load_library
from topogen.scenario import build_scenario


def test_build_owns_a_readonly_plan(corridor_graph, scenario_config):
    before = deepcopy(dict(corridor_graph.nodes(data=True)))
    config_before = asdict(scenario_config)
    build = build_scenario(corridor_graph, scenario_config, load_library(None))
    assert dict(corridor_graph.nodes(data=True)) == before
    assert asdict(scenario_config) == config_before
    with pytest.raises(FrozenInstanceError):
        build.yaml = "changed"
    with pytest.raises(nx.NetworkXError):
        build.site_plan.add_node("extra")
    with pytest.raises(TypeError):
        build.site_plan.nodes["metro1/dc1"]["metro_name"] = "changed"
    with pytest.raises(TypeError):
        build.sizing.capacity["extra"] = 1
    corridor_graph.nodes[0]["name"] = "changed"
    assert build.site_plan.nodes["metro1/dc1"]["metro_name"] == "alpha"


def test_readonly_graph_owns_every_nested_attribute_level():
    from topogen.scenario.records import readonly_graph

    graph = nx.MultiGraph()
    graph.graph["nested"] = {"items": [1]}
    graph.add_node("a", nested={"items": [2]})
    graph.add_edge("a", "b", nested={"items": [3]})
    owned = readonly_graph(graph)
    for original, frozen in (
        (graph.graph, owned.graph),
        (graph.nodes["a"], owned.nodes["a"]),
        (graph["a"]["b"][0], owned["a"]["b"][0]),
    ):
        original["nested"]["items"].append(99)
        assert frozen["nested"]["items"] == tuple(original["nested"]["items"][:-1])
        with pytest.raises(TypeError):
            frozen["nested"]["new"] = True
    assert owned["a"]["b"][0] is owned["b"]["a"][0]


def test_failed_build_and_independent_concurrent_builds_keep_inputs(
    corridor_graph, scenario_config
):
    from concurrent.futures import ThreadPoolExecutor

    from topogen.library import Library, thaw

    library = load_library(None)
    before_graph = deepcopy(nx.node_link_data(corridor_graph, edges="edges"))
    before_config = asdict(scenario_config)
    before_library = thaw(library.blueprints)
    reference = build_scenario(corridor_graph, scenario_config, library)
    broken = thaw(library.blueprints)
    broken["SingleRouter"]["links"] = [
        {"source": "missing", "target": "core", "cost": 1, "capacity": 1}
    ]
    bad_library = Library(
        broken, library.components, library.failure_policies, library.workflows
    )
    with pytest.raises(ValueError, match="expands to no links"):
        build_scenario(corridor_graph, scenario_config, bad_library)
    assert nx.node_link_data(corridor_graph, edges="edges") == before_graph
    assert asdict(scenario_config) == before_config
    assert thaw(library.blueprints) == before_library
    assert thaw(bad_library.blueprints) == broken
    with ThreadPoolExecutor(max_workers=2) as pool:
        builds = list(
            pool.map(
                lambda _: build_scenario(corridor_graph, scenario_config, library),
                range(2),
            )
        )
    assert all(b.yaml == reference.yaml for b in builds)
    assert builds[0].site_plan is not builds[1].site_plan
    assert builds[0].hardware is not builds[1].hardware
