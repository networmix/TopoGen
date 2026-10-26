"""Pin emitted gravity and hose rows to the released implementation."""

import json
import re
import sys
from pathlib import Path

import networkx as nx
import pytest
import yaml

from topogen.config import TopologyConfig
from topogen.library import load_library
from topogen.scenario import build_scenario

# Python 3.12 changed float summation; each interpreter keeps the release's rows.
REFERENCE = (
    "traffic_052_py311.json" if sys.version_info[:2] == (3, 11) else "traffic_052.json"
)
DATA = json.loads((Path(__file__).parent / "fixtures" / REFERENCE).read_text())
assert DATA["baseline_commit"] == "3fa6c2995ea0e883da276e49f8cef0a1b0d275e5"
assert len(DATA["cases"]) == 10
assert sum("rows" in case for case in DATA["cases"]) == 9


def traffic_graph():
    graph = nx.MultiGraph()
    for i, (name, x, y, radius) in enumerate(
        [
            ("Alpha City", 0.0, 0.0, 20.0),
            ("Beta", 300000.0, 0.0, 40.0),
            ("Gamma", 300000.0, 400000.0, 20.0),
            ("Delta", 900000.0, 100000.0, 15.0),
        ]
    ):
        graph.add_node(
            i,
            node_type="metro",
            name=name,
            name_orig=name,
            metro_id=str(i),
            x=x,
            y=y,
            radius_km=radius,
        )
    for source, target, distance in [
        (0, 1, 300),
        (0, 1, 450),
        (1, 2, 400),
        (0, 2, 700),
        (2, 3, 650),
        (1, 3, 610),
    ]:
        graph.add_edge(source, target, edge_type="corridor", length_km=float(distance))
    return graph


def traffic_config(case):
    config = TopologyConfig()
    config.build.build_defaults.pop_per_metro = 1
    config.build.build_defaults.dc_regions_per_metro = 1
    config.build.build_overrides = {
        "beta": {"dc_regions_per_metro": 2},
        "delta": {"dc_regions_per_metro": 3, "pop_per_metro": 2},
    }
    config.build.tm_sizing.enabled = False
    params = case["parameters"]
    config.output.scenario_seed = params.get("seed", 7)
    traffic = config.traffic
    traffic.model = case["model"]
    traffic.gbps_per_mw = params.get("gbps_per_mw", 3.7)
    traffic.mw_per_dc_region = params.get("mw", 25.3)
    traffic.priority_ratios = {
        int(key): value
        for key, value in params.get("ratios", {0: 0.25, 1: 0.75}).items()
    }
    traffic.flow_policy_config = {
        int(key): value
        for key, value in params.get("policies", {0: "SHORTEST_PATHS_ECMP"}).items()
    }
    traffic.samples = params.get("samples", 1)
    for section in ("gravity", "hose"):
        for key, value in params.get(section, {}).items():
            setattr(getattr(traffic, section), key, value)
    return config


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda case: case["name"])
def test_emitted_rows_match_released_traffic(case):
    if "error" in case:
        with pytest.raises(ValueError) as error:
            build_scenario(traffic_graph(), traffic_config(case), load_library(None))
        assert f"ValueError: {error.value}" == case["error"]
        return
    text = build_scenario(
        traffic_graph(), traffic_config(case), load_library(None)
    ).yaml
    rows = []
    for name, entries in yaml.safe_load(text)["demands"].items():
        for entry in entries:
            source = re.fullmatch(r"\^(metro\d+/dc\d+)/\.\*", entry["source"])
            target = re.fullmatch(r"\^(metro\d+/dc\d+)/\.\*", entry["target"])
            assert source and target
            rows.append(
                [
                    name,
                    source[1],
                    target[1],
                    entry["priority"],
                    repr(entry["volume"]),
                    entry.get("preset"),
                    entry.get("attrs", {}).get("euclidean_km"),
                ]
            )
    assert rows and len(rows) == len(case["rows"])
    for index, (actual, expected) in enumerate(zip(rows, case["rows"], strict=True)):
        assert actual == expected, (case["name"], index, actual, expected)
        assert [type(value) for value in actual] == [type(value) for value in expected]
