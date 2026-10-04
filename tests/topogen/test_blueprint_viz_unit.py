from typing import Any

from ngraph.dsl.blueprints.expand import expand_network_dsl

from topogen.blueprint_viz import build_abstract_view, collect_concrete_site
from topogen.blueprints_lib import get_builtin_blueprints


def test_builtin_dragonfly_abstract_view_expands_placeholders():
    network = expand_network_dsl(
        {
            "blueprints": get_builtin_blueprints(),
            "network": {"nodes": {"metro1/pop1": {"blueprint": "Dragonfly_A3H2G7"}}},
        }
    )
    view = build_abstract_view(network, "metro1/pop1")
    groups = {f"G{i}" for i in range(1, 8)}
    assert set(view.graph.nodes) == groups
    assert set(view.graph.edges()) == {
        (f"G{u}", f"G{v}") for u in range(1, 8) for v in range(u + 1, 8)
    }
    assert {g for g, _ in view.self_loops} == groups
    assert all("N=3" in label for label in view.node_labels.values())
    assert set(view.edge_labels.values()) == {"800 Gbps"}
    assert set(label for _, label in view.self_loops) == {"2,400 Gbps"}


def test_dict_selectors_do_not_create_phantom_groups():
    network = expand_network_dsl(
        {
            "network": {
                "nodes": {
                    "metro1/pop1/a": {"count": 2, "template": "r{n}"},
                    "metro1/pop1/b": {"count": 1, "template": "r{n}"},
                },
                "links": [
                    {
                        "source": {"path": "metro1/pop1/a"},
                        "target": {"path": "metro1/pop1/b"},
                        "pattern": "mesh",
                        "capacity": 10,
                    }
                ],
            }
        }
    )
    view = build_abstract_view(network, "metro1/pop1")
    assert set(view.graph) == {"a", "b"}
    assert list(view.graph.edges()) == [("a", "b")]
    assert list(view.edge_labels.values()) == ["20 Gbps"]
    assert (
        build_abstract_view(network, "metro1/pop1", include_self_loops=False).self_loops
        == []
    )


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


class _Link:
    def __init__(self, src: Any, dst: Any, cap: float) -> None:
        self.source = src
        self.target = dst
        self.capacity = cap


def test_collect_concrete_site_filters_and_positions() -> None:
    # Build a tiny stub network object
    class _Net:
        pass

    net = _Net()
    net.nodes = {
        1: _Node("metro1/dc1/A1"),
        2: _Node("metro1/dc1/B1"),
        3: _Node("metro2/dc3/X"),
    }
    net.links = {
        1: _Link(net.nodes[1].name, net.nodes[2].name, 10.0),
        2: _Link(net.nodes[1].name, net.nodes[3].name, 20.0),
    }

    ns, pos, links = collect_concrete_site(net, "metro1/dc1")
    assert set(ns) == {"metro1/dc1/A1", "metro1/dc1/B1"}
    assert set((s, t) for s, t, _ in links) == {("metro1/dc1/A1", "metro1/dc1/B1")}
    for n in ns:
        assert (
            n in pos and isinstance(pos[n][0], float) and isinstance(pos[n][1], float)
        )
