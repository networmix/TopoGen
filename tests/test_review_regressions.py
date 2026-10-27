"""Behavioral witnesses found during the post-cleanup review."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import networkx as nx
import pytest
import yaml

from netlab_topogen.config import ComponentsConfig, TopologyConfig, _normalize_int
from netlab_topogen.library import load_library
from netlab_topogen.scenario import build_scenario
from netlab_topogen.scenario.traffic import generate_traffic
from netlab_topogen.validation import validate_scenario


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
    assert not validate_scenario(yaml.safe_dump(data), ComponentsConfig()).issues
    data["demands"]["second"][0]["volume"] = 11
    assert any(
        "dc capacity" in issue and "second" in issue
        for issue in validate_scenario(yaml.safe_dump(data), ComponentsConfig()).issues
    )


def test_explicit_uniform_pairs_participate_in_dc_audit():
    data = dc_scenario()
    data["demands"]["first"] = [
        {
            "source": f"^metro{a}/dc1/.*",
            "target": f"^metro{b}/dc1/.*",
            "volume": 15,
            "mode": "pairwise",
        }
        for a, b in ((1, 2), (2, 1))
    ]
    assert any(
        "dc capacity" in issue
        for issue in validate_scenario(yaml.safe_dump(data), ComponentsConfig()).issues
    )


def test_malformed_network_is_an_issue_not_an_exception():
    assert validate_scenario("network: []", ComponentsConfig()).issues


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
    from netlab_topogen.scenario.records import Metro, Site, SitePlan

    plan = SitePlan(
        tuple(
            Site(
                f"metro{i}/dc1",
                Metro(i, name, name, name, i * 1000.0, 0.0, 1.0),
                "dc",
                1,
                "DCRegion",
                mass,
                (i * 1000.0, 0.0),
            )
            for i, (name, mass) in enumerate(powers.items(), 1)
        ),
        (),
        node_rules=(),
    )
    return config, plan


def test_infeasible_hose_is_rejected():
    config, plan = hose_case({"a": 10.0, "b": 100.0})
    with pytest.raises(ValueError, match="hose.*(infeasible|converge)"):
        generate_traffic(plan, config.traffic, config.output.scenario_seed)


@pytest.mark.parametrize("model", ["hose", "gravity"])
def test_disabled_rounding_preserves_offered_traffic(model):
    config, plan = hose_case({"a": 0.0002, "b": 0.0002, "c": 0.0002})
    config.traffic.model = model
    matrices = generate_traffic(plan, config.traffic, config.output.scenario_seed)
    demands = matrices[0].demands
    assert sum(d.volume for d in demands) == pytest.approx(0.0006, abs=1e-9)
    if model == "hose":
        totals = defaultdict(float)
        for demand in demands:
            totals[demand.source] += float(demand.volume)
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
        build_scenario(graph, config, load_library(Path("lib")))


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
        for issue in validate_scenario(yaml.safe_dump(data), ComponentsConfig()).issues
    )


def test_unmatched_demand_cannot_hide_behind_another_valid_demand():
    data = dc_scenario()
    bad = deepcopy(data["demands"]["first"][0])
    bad["source"] = "missing-source"
    data["demands"]["first"].append(bad)
    assert any(
        "first|1" in issue and "discarded units" in issue
        for issue in validate_scenario(yaml.safe_dump(data), ComponentsConfig()).issues
    )


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:2263"])
def test_geographic_or_feet_crs_is_rejected(crs):
    from netlab_topogen.config import ProjectionConfig

    with pytest.raises(ValueError, match="projected.*metre"):
        ProjectionConfig(target_crs=crs)


def test_pop_coordinates_without_dc_groups(corridor_graph, scenario_config):
    scenario_config.build.build_defaults.dc_regions_per_metro = 0
    scenario_config.traffic.enabled = False
    document = yaml.safe_load(
        build_scenario(corridor_graph, scenario_config, load_library(None)).yaml
    )
    for i, (_, metro) in enumerate(corridor_graph.nodes(data=True), 1):
        attrs = document["network"]["nodes"][f"metro{i}/pop1"]["attrs"]
        assert (attrs["location_x"], attrs["location_y"]) == (metro["x"], metro["y"])


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
    scenario = yaml.safe_load(
        build_scenario(graph, config, load_library(Path("lib"))).yaml
    )
    assert "CoreRouter" in scenario["components"]
    assert "ReplacedPlatform" not in scenario["components"]
