"""Behavioral witnesses found during the post-cleanup review."""

from collections import defaultdict
from copy import deepcopy

import networkx as nx
import pytest
import yaml

from topogen.config import TopologyConfig, _normalize_int
from topogen.scenario import build_scenario
from topogen.traffic_matrix import generate_traffic_matrix
from topogen.validation import validate_scenario_yaml


def dc_scenario():
    return {
        "network": {
            "nodes": {
                "metro1/dc1/r": {"attrs": {"role": "dc"}},
                "metro2/dc1/r": {"attrs": {"role": "dc"}},
                "core": {"attrs": {"role": "core"}},
            },
            "links": [
                {"source": "metro1/dc1/r", "target": "core", "capacity": 10},
                {"source": "metro2/dc1/r", "target": "core", "capacity": 10},
            ],
        },
        "demands": {
            "first": [
                {
                    "source": "^metro1/dc1/.*",
                    "target": "^metro2/dc1/.*",
                    "volume": 8,
                    "mode": "pairwise",
                }
            ]
        },
    }


def test_independent_demand_sets_do_not_sum_capacity():
    data = dc_scenario()
    data["demands"]["second"] = deepcopy(data["demands"]["first"])
    assert not validate_scenario_yaml(yaml.safe_dump(data))
    data["demands"]["second"][0]["volume"] = 11
    assert any(
        "dc capacity" in issue and "second" in issue
        for issue in validate_scenario_yaml(yaml.safe_dump(data))
    )


def test_uniform_selectors_participate_in_dc_capacity_audit():
    data = dc_scenario()
    data["demands"]["first"] = [
        {
            "source": "(metro[0-9]+/dc[0-9]+)",
            "target": "(metro[0-9]+/dc[0-9]+)",
            "volume": 30,
            "mode": "pairwise",
        }
    ]
    assert any(
        "dc capacity" in issue for issue in validate_scenario_yaml(yaml.safe_dump(data))
    )


def test_malformed_network_is_an_issue_not_an_exception():
    assert validate_scenario_yaml("network: []")


def test_integer_string_preserves_exact_value():
    assert _normalize_int("9007199254740993", "capacity") == 9007199254740993
    with pytest.raises(ValueError):
        _normalize_int("9007199254740992.5", "capacity")


def hose_case(powers):
    config = TopologyConfig()
    config.traffic.model = "hose"
    config.traffic.gbps_per_mw = 1.0
    config.traffic.priority_ratios = {0: 1.0}
    config.traffic.gravity.mw_per_dc_region_overrides = powers
    metros = [
        {"name": name, "x": i * 1000.0, "y": 0.0} for i, name in enumerate(powers)
    ]
    settings = {name: {"dc_regions_per_metro": 1} for name in powers}
    return config, metros, settings


def test_infeasible_hose_is_rejected():
    config, metros, settings = hose_case({"a": 10.0, "b": 100.0})
    with pytest.raises(ValueError, match="hose.*(infeasible|converge)"):
        generate_traffic_matrix(metros, settings, config)


@pytest.mark.parametrize("model", ["hose", "gravity"])
def test_disabled_rounding_preserves_offered_traffic(model):
    config, metros, settings = hose_case({"a": 0.0002, "b": 0.0002, "c": 0.0002})
    config.traffic.model = model
    matrices = generate_traffic_matrix(metros, settings, config)
    demands = matrices[config.traffic.matrix_name]
    assert sum(d["demand"] for d in demands) == pytest.approx(0.0006, abs=1e-9)
    if model == "hose":
        totals = defaultdict(float)
        for demand in demands:
            totals[demand["source_path"]] += demand["demand"]
        assert all(abs(value - 0.0002) < 1e-9 for value in totals.values())


def test_duplicate_metro_names_are_rejected_before_site_merge():
    graph = nx.MultiGraph()
    for i in range(2):
        graph.add_node(
            (float(i), 0.0),
            node_type="metro",
            name="same",
            name_orig="Same",
            metro_id=str(i),
            x=float(i),
            y=0.0,
            radius_km=1.0,
        )
    config = TopologyConfig()
    config.traffic.enabled = False
    with pytest.raises(ValueError, match="[Dd]uplicate.*name"):
        build_scenario(graph, config)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_config_values_are_rejected(value):
    from pathlib import Path

    raw = yaml.safe_load(Path("examples/small_baseline.yml").read_text())
    raw["traffic"]["gravity"]["min_distance_km"] = value
    with pytest.raises(ValueError, match="finite"):
        TopologyConfig._from_dict(raw)


def test_internal_dc_links_do_not_inflate_external_capacity():
    data = dc_scenario()
    data["network"]["nodes"]["metro1/dc1/extra"] = {"attrs": {"role": "dc"}}
    data["network"]["links"].append(
        {"source": "metro1/dc1/r", "target": "metro1/dc1/extra", "capacity": 1000}
    )
    data["demands"]["first"][0]["volume"] = 15
    assert any(
        "dc capacity" in issue and "metro1/dc1" in issue
        for issue in validate_scenario_yaml(yaml.safe_dump(data))
    )


def test_unmatched_demand_cannot_hide_behind_another_valid_demand():
    data = dc_scenario()
    bad = deepcopy(data["demands"]["first"][0])
    bad["source"] = "missing-source"
    data["demands"]["first"].append(bad)
    assert any(
        "entry 1" in issue and "expanded" in issue
        for issue in validate_scenario_yaml(yaml.safe_dump(data))
    )


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:2263"])
def test_geographic_or_feet_crs_is_rejected(crs):
    from topogen.config import ProjectionConfig

    with pytest.raises(ValueError, match="projected.*metre"):
        ProjectionConfig(target_crs=crs)


def test_pop_coordinates_are_checked_without_dc_groups():
    from topogen.validation import validate_scenario_dict

    data = {
        "network": {
            "nodes": {
                "metro1/pop[1]": {
                    "attrs": {"metro_name": "a", "location_x": 1, "location_y": 2}
                }
            },
            "links": [{"source": "a", "target": "b"}],
        }
    }
    assert any(
        "location differs" in issue
        for issue in validate_scenario_dict(data, {"a": (5, 6)})
    )


def test_component_inventory_uses_final_blueprint_assignments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    library = tmp_path / "lib"
    library.mkdir()
    (library / "blueprints.yml").write_text(
        yaml.safe_dump(
            {
                "SingleRouter": {
                    "nodes": {
                        "core": {
                            "count": 1,
                            "attrs": {
                                "role": "core",
                                "hardware": {
                                    "component": "ReplacedPlatform",
                                    "count": 1,
                                },
                            },
                        }
                    },
                    "links": [],
                }
            }
        )
    )
    graph = nx.MultiGraph()
    graph.add_node(
        (0.0, 0.0),
        node_type="metro",
        name="a",
        name_orig="A",
        metro_id="a",
        x=0.0,
        y=0.0,
        radius_km=1.0,
    )
    config = TopologyConfig()
    config.traffic.enabled = False
    config.components.hw_component = {"core": "CoreRouter"}
    scenario = yaml.safe_load(build_scenario(graph, config))
    assert "CoreRouter" in scenario["components"]
    assert "ReplacedPlatform" not in scenario["components"]
