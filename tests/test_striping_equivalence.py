from __future__ import annotations

import networkx as nx
import yaml

from topogen.config import TopologyConfig
from topogen.scenario_builder import build_scenario
from topogen.workflows_lib import get_builtin_workflows


def _make_sample_integrated_graph() -> nx.Graph:
    g = nx.Graph()

    # Three metros with radius for ring adjacency
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

    # Corridor edges between metros (symmetric undirected)
    g.add_edge(a, b, length_km=400.0, capacity=3200, edge_type="corridor")
    g.add_edge(b, c, length_km=600.0, capacity=3200, edge_type="corridor")
    g.add_edge(a, c, length_km=800.0, capacity=3200, edge_type="corridor")
    return g


def _baseline_config() -> TopologyConfig:
    cfg = TopologyConfig()
    # Minimal build settings
    cfg.build.build_defaults.pop_per_metro = 2
    cfg.build.build_defaults.site_blueprint = "SingleRouter"
    cfg.build.build_defaults.dc_regions_per_metro = 1
    cfg.build.build_defaults.dc_region_blueprint = "DCRegion"
    # Disable TM sizing and leave optics empty to avoid late-HW dependencies
    cfg.build.tm_sizing.enabled = False
    cfg.components.optics = {}
    # Pick a valid built-in workflow to satisfy assembly
    wf = next(iter(get_builtin_workflows().keys()))
    cfg.workflows.assignments.default = wf
    return cfg


def _striped_width1_config() -> TopologyConfig:
    cfg = _baseline_config()
    # width=1 on both inter_metro and dc_to_pop
    cfg.build.build_defaults.inter_metro_link.striping = {"mode": "width", "width": 1}
    cfg.build.build_defaults.dc_to_pop_link.striping = {"mode": "width", "width": 1}
    return cfg


def _expand_yaml(yaml_text: str):
    from ngraph.dsl.blueprints.expand import expand_network_dsl

    data = yaml.safe_load(yaml_text)
    return expand_network_dsl(data)


def _site_of(node_name: str) -> str:
    parts = node_name.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else node_name


def test_width1_equivalence_inter_and_dc_to_pop():
    graph = _make_sample_integrated_graph()

    # Baseline
    cfg_base = _baseline_config()
    yaml_base = build_scenario(graph, cfg_base)
    net_base = _expand_yaml(yaml_base)

    # Striped width=1
    cfg_striped = _striped_width1_config()
    yaml_striped = build_scenario(graph, cfg_striped)
    net_striped = _expand_yaml(yaml_striped)

    # Collect endpoint site pairs for each link_type
    def pairs(net, link_type: str) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for link in net.links.values():
            if link.attrs.get("link_type") != link_type:
                continue
            u = _site_of(str(link.source))
            v = _site_of(str(link.target))
            a, b = sorted([u, v])
            out.add((a, b))
        return out

    # Inter-metro equivalence
    base_inter = pairs(net_base, "inter_metro_corridor")
    striped_inter = pairs(net_striped, "inter_metro_corridor")
    assert base_inter == striped_inter

    # DC-to-PoP equivalence
    base_dc = pairs(net_base, "dc_to_pop")
    striped_dc = pairs(net_striped, "dc_to_pop")
    assert base_dc == striped_dc

    # Count equality as a stronger check
    assert len(base_inter) > 0
    assert len(base_dc) > 0
    assert len(base_inter) == len(striped_inter)
    assert len(base_dc) == len(striped_dc)
