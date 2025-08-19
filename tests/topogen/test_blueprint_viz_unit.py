from __future__ import annotations

from typing import Any

from topogen.blueprint_viz import (
    AbstractView,
    _extract_vars,
    _first_path_component,
    _iter_assignments,
    _subst,
    build_abstract_view,
    collect_concrete_site,
)


def test_helper_functions() -> None:
    assert _first_path_component("/G{g}/G{g}_r{r}") == "G{g}"
    assert _first_path_component("G1/G1_r1") == "G1"
    assert _extract_vars("G{a}_x{b}") == ["a", "b"]

    # zip alignment (truncate to min length)
    assigns = _iter_assignments({"a": [1, 2], "b": [3]}, ["a", "b"], "zip")
    assert assigns == [{"a": 1, "b": 3}]
    # product
    assigns = _iter_assignments({"a": [1, 2], "b": [3, 4]}, ["a", "b"], "product")
    assert {tuple(sorted(x.items())) for x in assigns} == {
        (("a", 1), ("b", 3)),
        (("a", 1), ("b", 4)),
        (("a", 2), ("b", 3)),
        (("a", 2), ("b", 4)),
    }
    # missing vars → single empty assignment
    assert _iter_assignments({"a": []}, ["a"], "zip") == [{}]

    # substitution leaves unknowns intact
    assert _subst("G{a}_x{b}", {"a": 1}) == "G1_x{b}"


def test_build_abstract_view_and_self_loops() -> None:
    bp = {
        "groups": {
            "G1": {"node_count": 2, "attrs": {"role": "agg"}},
            "G2": {"node_count": 1},
            "G1_r1": {"node_count": 2},
            "G1_r2": {"node_count": 2},
        },
        "adjacency": [
            {
                "source": "G{g}",
                "target": "G{g}_r{r}",
                "pattern": "uplink",
                "expand_vars": {"g": [1, 2], "r": [1, 1]},
                "expansion_mode": "zip",
                "link_params": {"attrs": {"target_capacity": 100}},
            },
            {
                "source": "G1_r{r}",
                "target": "G1_r{r}",
                "pattern": "mesh",
                "expand_vars": {"r": [1, 2]},
                "expansion_mode": "zip",
                "link_params": {"attrs": {"target_capacity": 50}},
            },
        ],
    }
    av: AbstractView = build_abstract_view(bp)
    # Nodes created and labeled
    assert set(av.graph.nodes) >= {"G1", "G2", "G1_r1", "G1_r2"}
    assert "role=agg" in av.node_labels["G1"]
    # Inter-group edges deduplicated by unordered pair and labeled
    edges = list(av.edge_labels.keys())
    assert edges, "expected at least one inter-group edge"
    any_label = next(iter(av.edge_labels.values()))
    assert "uplink" in any_label and "100" in any_label
    # Self-loops reflected as notes
    loop_groups = {g for (g, _lbl) in av.self_loops}
    assert {"G1_r1", "G1_r2"} & loop_groups


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
        1: _Link(net.nodes[1], net.nodes[2], 10.0),
        2: _Link(net.nodes[1], net.nodes[3], 20.0),
    }

    ns, pos, links = collect_concrete_site(net, "metro1/dc1")
    assert set(ns) == {"metro1/dc1/A1", "metro1/dc1/B1"}
    assert set((s, t) for s, t, _ in links) == {("metro1/dc1/A1", "metro1/dc1/B1")}
    # Position entries exist for internal nodes
    for n in ns:
        assert (
            n in pos and isinstance(pos[n][0], float) and isinstance(pos[n][1], float)
        )
