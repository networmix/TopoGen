"""Current contracts reject silent substitutions and preserve published output."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import yaml

from topogen import RunContext, TopologyConfig
from topogen.cli import _run_pipeline
from topogen.config import _normalize_int
from topogen.integrated_graph import load_from_json
from topogen.library import load_library
from topogen.scenario import build_scenario


def test_failed_validation_preserves_previous_scenario(tmp_path, monkeypatch):
    import topogen
    import topogen.scenario
    import topogen.validation

    context = RunContext(tmp_path, "input")
    context.path("integrated_graph.json").write_text("{}")
    target = tmp_path / "scenario.yml"
    target.write_text("previous valid scenario")
    monkeypatch.setattr(
        topogen, "load_from_json", lambda path: (nx.MultiGraph(), "EPSG:5070")
    )
    monkeypatch.setattr(
        topogen.scenario,
        "build_scenario",
        lambda *args, **kwargs: SimpleNamespace(yaml="invalid scenario"),
    )
    monkeypatch.setattr(
        topogen.validation,
        "validate_scenario",
        lambda *args, **kwargs: topogen.validation.Validation(("bad optics",), None),
    )
    with pytest.raises(ValueError, match="bad optics"):
        _run_pipeline(TopologyConfig(), target, context=context)
    assert target.read_text() == "previous valid scenario"
    assert not list(tmp_path.glob(".scenario.yml.*"))


def test_library_build_has_no_implicit_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = TopologyConfig()
    config.traffic.enabled = False
    graph = nx.MultiGraph()
    graph.add_node(
        (0.0, 0.0),
        node_type="metro",
        name="a",
        name_orig="A",
        metro_id="001",
        x=0.0,
        y=0.0,
        radius_km=10.0,
    )
    before = asdict(config)
    result = build_scenario(graph, config, load_library(Path("lib"))).yaml
    assert yaml.safe_load(result)["network"]["nodes"]
    assert asdict(config) == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "section, key",
    [
        ("highway_processing", "min_cycle_nodes"),
        ("highway_processing", "validation_sample_size"),
        ("output", "scenario_metadata"),
        ("components", "assignments"),
        ("components", "library"),
    ],
)
def test_removed_settings_are_rejected(section, key):
    raw = yaml.safe_load(Path("examples/small_baseline.yml").read_text())
    raw[section][key] = {}
    with pytest.raises(ValueError, match=key):
        TopologyConfig._from_dict(raw)


def test_config_parser_does_not_mutate_input():
    raw = yaml.safe_load(Path("examples/small_baseline.yml").read_text())
    raw["clustering"].pop("override_metro_clusters")
    before = deepcopy(raw)
    TopologyConfig._from_dict(raw)
    assert raw == before


@pytest.mark.parametrize("value", [True, 1.5, "1.5", float("nan"), float("inf")])
def test_integer_quantities_are_not_coerced(value):
    with pytest.raises(ValueError, match="Invalid integer"):
        _normalize_int(value, "cost")


def test_old_graph_format_is_rejected(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"nodes": [], "edges": [], "target_crs": "EPSG:5070"}')
    with pytest.raises(ValueError, match="regenerate"):
        load_from_json(path)


def test_hardware_validation_uses_only_scenario_definitions():
    from topogen.config import ComponentsConfig
    from topogen.validation import validate_scenario

    data = {
        "network": {
            "nodes": {
                "a": {
                    "attrs": {
                        "role": "core",
                        "hardware": {"component": "CoreRouter", "count": 1},
                    }
                },
                "b": {"attrs": {"role": "core"}},
            },
            "links": [{"source": "a", "target": "b", "capacity": 100, "cost": 1}],
        }
    }
    issues = validate_scenario(yaml.safe_dump(data), ComponentsConfig()).issues
    assert any("unknown component 'CoreRouter'" in issue for issue in issues)


def test_public_stage_signatures_and_exports():
    import inspect

    import topogen.scenario as scenario
    import topogen.validation as validation
    from topogen.library import load_library

    assert scenario.__all__ == [
        "Build",
        "build_scenario",
        "write_scenario",
        "write_artifacts",
    ]
    assert validation.__all__ == ["Validation", "validate_scenario"]
    assert list(inspect.signature(scenario.build_scenario).parameters) == [
        "graph",
        "config",
        "library",
    ]
    assert list(inspect.signature(validation.validate_scenario).parameters) == [
        "text",
        "components",
    ]
    assert list(inspect.signature(scenario.write_artifacts).parameters) == [
        "build",
        "validation",
        "config",
        "context",
    ]
    assert list(inspect.signature(scenario.write_scenario).parameters) == [
        "path",
        "text",
    ]
    assert list(inspect.signature(load_library).parameters) == ["directory"]
    assert all(
        p.default is inspect.Parameter.empty
        for p in inspect.signature(scenario.build_scenario).parameters.values()
    )


def test_site_identifiers_and_selectors_have_the_documented_boundaries():
    from topogen.naming import site_edge_id, site_id, site_of, site_selector

    assert site_id(3, "dc", 2) == "metro3/dc2"
    assert site_of("metro3/dc2/nested/device") == "metro3/dc2"
    assert site_selector("metro3/dc2") == "^metro3/dc2/.*"
    assert site_edge_id("metro3/pop1", "metro4/pop2", "parallel-2") == (
        "metro3/pop1|metro4/pop2|parallel-2"
    )
