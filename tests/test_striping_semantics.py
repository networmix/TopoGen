from __future__ import annotations

import contextlib
from typing import Any

import networkx as nx
import yaml

from topogen.config import TopologyConfig
from topogen.scenario_builder import build_scenario
from topogen.workflows_lib import get_builtin_workflows


def _expand_yaml(yaml_text: str):
    from ngraph.dsl.blueprints.expand import expand_network_dsl

    data = yaml.safe_load(yaml_text)
    return expand_network_dsl(data)


def _site_of(node_name: str) -> str:
    parts = node_name.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else node_name


def _counts_per_pair(expanded_net, link_type: str) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for link in expanded_net.links.values():
        if link.attrs.get("link_type") != link_type:
            continue
        u = _site_of(str(link.source))
        v = _site_of(str(link.target))
        a, b = tuple(sorted([u, v]))
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _make_sample_graph() -> nx.Graph:
    g = nx.Graph()
    # three metros with ring radii
    a = (10.0, 10.0)
    b = (20.0, 20.0)
    c = (30.0, 25.0)
    g.add_node(
        a,
        node_type="metro",
        name="Alpha",
        metro_id="alpha",
        x=10.0,
        y=10.0,
        radius_km=40.0,
    )
    g.add_node(
        b,
        node_type="metro",
        name="Beta",
        metro_id="beta",
        x=20.0,
        y=20.0,
        radius_km=35.0,
    )
    g.add_node(
        c,
        node_type="metro",
        name="Gamma",
        metro_id="gamma",
        x=30.0,
        y=25.0,
        radius_km=30.0,
    )
    g.add_edge(a, b, length_km=400.0, capacity=3200, edge_type="corridor")
    g.add_edge(b, c, length_km=600.0, capacity=3200, edge_type="corridor")
    g.add_edge(a, c, length_km=800.0, capacity=3200, edge_type="corridor")
    return g


def _base_cfg() -> TopologyConfig:
    cfg = TopologyConfig()
    cfg.build.build_defaults.pop_per_metro = 2
    cfg.build.build_defaults.site_blueprint = "SingleRouter"
    cfg.build.build_defaults.dc_regions_per_metro = 1
    cfg.build.build_defaults.dc_region_blueprint = "DCRegion"
    cfg.build.tm_sizing.enabled = False
    cfg.components.optics = {}
    # pick a valid built-in workflow
    wf = next(iter(get_builtin_workflows().keys()))
    cfg.workflows.assignments.default = wf
    return cfg


@contextlib.contextmanager
def _patch_blueprints(new_entries: dict[str, dict[str, Any]]):
    import topogen.blueprints_lib as BL

    removed: list[str] = []
    try:
        for name, defn in new_entries.items():
            if name not in BL._BUILTIN_BLUEPRINTS:  # type: ignore[attr-defined]
                removed.append(name)
            BL._BUILTIN_BLUEPRINTS[name] = defn  # type: ignore[attr-defined]
        yield
    finally:
        # remove only those we added
        for name in removed:
            with contextlib.suppress(Exception):
                del BL._BUILTIN_BLUEPRINTS[name]  # type: ignore[attr-defined]


def test_equivalence_one_device_each_end_width1():
    graph = _make_sample_graph()
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b.build.build_defaults.inter_metro_link.striping = {"mode": "width", "width": 1}
    cfg_b.build.build_defaults.dc_to_pop_link.striping = {"mode": "width", "width": 1}

    yml_a = build_scenario(graph, cfg_a)
    yml_b = build_scenario(graph, cfg_b)
    net_a = _expand_yaml(yml_a)
    net_b = _expand_yaml(yml_b)

    for lt in ("inter_metro_corridor", "dc_to_pop"):
        ca = _counts_per_pair(net_a, lt)
        cb = _counts_per_pair(net_b, lt)
        assert ca == cb


def test_dc_pop_one_vs_many_width1_reduces_links():
    # POP has 4 core nodes; DC has 1 -> baseline produces 4 per DC-POP pair, width=1 produces 1
    graph = _make_sample_graph()
    core4 = {
        "groups": {
            "core": {
                "node_count": 4,
                "name_template": "core{node_num}",
                "attrs": {"role": "core"},
            }
        },
        "adjacency": [],
    }
    with _patch_blueprints({"Core4": core4}):
        cfg_a = _base_cfg()
        cfg_a.build.build_defaults.site_blueprint = "Core4"
        cfg_b = _base_cfg()
        cfg_b.build.build_defaults.site_blueprint = "Core4"
        cfg_b.build.build_defaults.dc_to_pop_link.striping = {
            "mode": "width",
            "width": 1,
        }

        yml_a = build_scenario(graph, cfg_a)
        yml_b = build_scenario(graph, cfg_b)
        net_a = _expand_yaml(yml_a)
        net_b = _expand_yaml(yml_b)

        ca = _counts_per_pair(net_a, "dc_to_pop")
        cb = _counts_per_pair(net_b, "dc_to_pop")
        assert ca  # non-empty
        # Every DC-POP pair should reduce from 4 to 1
        for k, v in ca.items():
            assert v == 4
            assert cb.get(k, 0) == 1


def test_dc_pop_many_vs_many_width2_groups_links():
    # POP 4 core, DC 4 dc -> baseline 4; width=2 -> 2 per DC-POP pair
    graph = _make_sample_graph()
    core4 = {
        "groups": {
            "core": {
                "node_count": 4,
                "name_template": "core{node_num}",
                "attrs": {"role": "core"},
            }
        },
        "adjacency": [],
    }
    dc4 = {
        "groups": {
            "dc": {
                "node_count": 4,
                "name_template": "dc{node_num}",
                "attrs": {"role": "dc"},
            }
        },
        "adjacency": [],
    }
    with _patch_blueprints({"Core4": core4, "DCRegion4": dc4}):
        cfg_a = _base_cfg()
        cfg_a.build.build_defaults.site_blueprint = "Core4"
        cfg_a.build.build_defaults.dc_region_blueprint = "DCRegion4"
        cfg_b = _base_cfg()
        cfg_b.build.build_defaults.site_blueprint = "Core4"
        cfg_b.build.build_defaults.dc_region_blueprint = "DCRegion4"
        cfg_b.build.build_defaults.dc_to_pop_link.striping = {
            "mode": "width",
            "width": 2,
        }

        yml_a = build_scenario(graph, cfg_a)
        yml_b = build_scenario(graph, cfg_b)
        net_a = _expand_yaml(yml_a)
        net_b = _expand_yaml(yml_b)

        ca = _counts_per_pair(net_a, "dc_to_pop")
        cb = _counts_per_pair(net_b, "dc_to_pop")
        assert ca
        for k, v in ca.items():
            assert v == 4
            assert cb.get(k, 0) == 2
