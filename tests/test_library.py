"""Library provenance and ownership are observable build inputs."""

from dataclasses import FrozenInstanceError

import pytest
import yaml

from topogen.library import Library, WorkflowDefinition, load_library, thaw
from topogen.scenario import build_scenario
from topogen.scenario.policies import _build_workflow_section


def test_builtin_and_identical_user_workflow_have_distinct_binding(
    tmp_path, scenario_config
):
    builtin = load_library(None)
    name = "design_analysis_brief"
    (tmp_path / "workflows.yml").write_text(
        yaml.safe_dump({name: thaw(builtin.workflows[name].steps)})
    )
    user = load_library(tmp_path)
    scenario_config.traffic.matrix_name = "custom_tm"
    assert builtin.workflows[name].follows_configured_matrix
    assert not user.workflows[name].follows_configured_matrix
    assert [
        s["demand_set"]
        for s in _build_workflow_section(scenario_config, builtin)
        if "demand_set" in s
    ] == ["custom_tm", "custom_tm"]
    assert [
        s["demand_set"]
        for s in _build_workflow_section(scenario_config, user)
        if "demand_set" in s
    ] == ["baseline_traffic_matrix", "baseline_traffic_matrix"]


def test_library_owns_nested_definitions():
    raw = {"A": {"nodes": {"n": {"attrs": {"list": [1]}}}}}
    steps = [{"type": "NetworkStats", "name": "stats"}]
    definition = WorkflowDefinition(tuple(steps))
    library = Library(raw, {}, {}, {"w": definition})
    raw["A"]["nodes"]["n"]["attrs"]["list"].append(2)
    steps[0]["name"] = "changed"
    assert library.blueprints["A"]["nodes"]["n"]["attrs"]["list"] == (1,)
    assert library.workflows["w"].steps[0]["name"] == "stats"
    assert not definition.follows_configured_matrix
    with pytest.raises(TypeError):
        library.blueprints["A"]["nodes"]["n"] = {}
    with pytest.raises(FrozenInstanceError):
        definition.follows_configured_matrix = True


def test_build_ignores_cwd_and_reads_no_library(
    corridor_graph, scenario_config, tmp_path, monkeypatch
):
    library = load_library(None)
    first = build_scenario(corridor_graph, scenario_config, library)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/blueprints.yml").write_text("DCRegion: {nodes: {}, links: []}")
    monkeypatch.chdir(tmp_path)
    second = build_scenario(corridor_graph, scenario_config, library)
    assert first.yaml == second.yaml
    assert not list(tmp_path.glob("*_scenario.yml"))
    with pytest.raises(ValueError, match="no devices"):
        build_scenario(corridor_graph, scenario_config, load_library(tmp_path / "lib"))


@pytest.mark.parametrize("content", ["- not-a-mapping", "blueprints: [invalid: :"])
def test_bad_library_is_refused_by_name(tmp_path, content):
    (tmp_path / "blueprints.yml").write_text(content)
    with pytest.raises(ValueError, match="blueprints.yml"):
        load_library(tmp_path)


def test_unused_workflow_still_checked(tmp_path):
    (tmp_path / "workflows.yml").write_text(
        "unused: [{type: NetworkStats, unknown: true}]"
    )
    with pytest.raises(ValueError, match="Workflow 'unused'"):
        load_library(tmp_path)


def test_none_library_does_not_read_hostile_cwd(tmp_path, monkeypatch):
    directory = tmp_path / "lib"
    directory.mkdir()
    (directory / "blueprints.yml").write_text("DCRegion: {nodes: {}, links: []}")
    monkeypatch.chdir(tmp_path)
    assert load_library(None).blueprints["DCRegion"]["nodes"]["dc"]["count"] == 1
    assert not load_library(directory).blueprints["DCRegion"]["nodes"]


def test_library_snapshots_preserve_set_values_for_netgraph_refusal(
    tmp_path, corridor_graph, scenario_config
):
    from ngraph import Scenario

    builtin = load_library(None)
    blueprints = thaw(builtin.blueprints)
    blueprints["SingleRouter"]["nodes"]["core"]["attrs"]["bad"] = {"red", "blue"}
    text = yaml.safe_dump({"SingleRouter": blueprints["SingleRouter"]})
    (tmp_path / "blueprints.yml").write_text(text)
    library = load_library(tmp_path)
    assert library.blueprints["SingleRouter"]["nodes"]["core"]["attrs"][
        "bad"
    ] == frozenset({"red", "blue"})
    assert isinstance(
        thaw(library.blueprints)["SingleRouter"]["nodes"]["core"]["attrs"]["bad"], set
    )
    doc = {
        "blueprints": {"SingleRouter": blueprints["SingleRouter"]},
        "network": {"nodes": {"a": {"blueprint": "SingleRouter"}}},
    }
    with pytest.raises(ValueError, match="finite JSON") as upstream:
        Scenario.from_yaml(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="finite JSON"):
        build_scenario(corridor_graph, scenario_config, library)
    assert "attrs" in str(upstream.value)
    blueprints["SingleRouter"]["nodes"]["core"]["attrs"]["bad"] = ["blue", "red"]
    (tmp_path / "blueprints.yml").write_text(
        yaml.safe_dump({"SingleRouter": blueprints["SingleRouter"]})
    )
    assert build_scenario(corridor_graph, scenario_config, load_library(tmp_path)).yaml
