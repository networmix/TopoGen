"""Publication is atomic per file, after validation, with explicit partial success."""

import networkx as nx
import pytest

from netlab_topogen import RunContext, save_to_json
from netlab_topogen.cli import _run_pipeline
from netlab_topogen.library import load_library
from netlab_topogen.scenario import build_scenario, write_artifacts, write_scenario
from netlab_topogen.scenario.publication import write_atomic
from netlab_topogen.validation import Validation, validate_scenario


@pytest.mark.parametrize("partial", [False, True])
def test_atomic_helper_keeps_old_destination_on_failure(tmp_path, partial):
    target = tmp_path / "out.yml"
    target.write_bytes(b"old")

    def writer(path):
        assert path.parent == target.parent and path != target
        if partial:
            path.write_bytes(b"partial")
        raise OSError("injected writer failure")

    with pytest.raises(OSError, match="injected writer failure"):
        write_atomic(target, writer)
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.yml"]
    write_scenario(target, "complete\n")
    assert target.read_text() == "complete\n"


def test_real_cli_refusal_keeps_yaml_site_json_and_artifacts(
    tmp_path, monkeypatch, corridor_graph, scenario_config
):
    monkeypatch.chdir(tmp_path)
    context = RunContext(tmp_path, "input", tmp_path / "debug")
    saved_graph = nx.relabel_nodes(
        corridor_graph,
        {
            node: (data["x"], data["y"])
            for node, data in corridor_graph.nodes(data=True)
        },
    )
    for u, v, data in saved_graph.edges(data=True):
        data["geometry"] = [u, v]
    save_to_json(
        saved_graph,
        context.path("integrated_graph.json"),
        scenario_config.projection.target_crs,
        scenario_config.output.formatting,
    )
    target = tmp_path / "input_scenario.yml"
    paths = [
        target,
        context.path("network_graph.json"),
        context.path("site_graph.jpg"),
        context.debug_dir / "input_traffic_debug.json",
    ]
    for path in paths:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"old " + path.name.encode())
    before = {p: p.read_bytes() for p in paths}
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/blueprints.yml").write_text("""SingleRouter:
  nodes:
    core: {count: 2, template: "core{n}", attrs: {role: core}}
  links:
    - {source: core/core1, target: core/core2, capacity: 100, cost: 0}
""")
    with pytest.raises(ValueError, match="SingleRouter|core1"):
        _run_pipeline(scenario_config, target, context=context)
    assert {p: p.read_bytes() for p in paths} == before
    assert not [p for p in tmp_path.rglob(".*") if p.is_file()]
    (tmp_path / "lib/blueprints.yml").unlink()
    text = _run_pipeline(scenario_config, target, context=context)
    assert target.read_text() == text
    assert (
        context.path("network_graph.json").read_bytes()
        != before[context.path("network_graph.json")]
    )

    import netlab_topogen.scenario as scenario

    def fail_artifacts(*args):
        raise OSError("artifact failure after YAML")

    monkeypatch.setattr(scenario, "write_artifacts", fail_artifacts)
    target.write_bytes(b"old YAML again")
    previous_json = context.path("network_graph.json").read_bytes()
    with pytest.raises(OSError, match="artifact failure after YAML"):
        _run_pipeline(scenario_config, target, context=context)
    assert target.read_text() == text
    assert context.path("network_graph.json").read_bytes() == previous_json


def test_artifact_failure_retains_earlier_files_and_later_destinations(
    tmp_path, monkeypatch, corridor_graph, scenario_config
):
    import netlab_topogen.scenario.artifacts as artifacts

    build = build_scenario(corridor_graph, scenario_config, load_library(None))
    validation = validate_scenario(build.yaml, scenario_config.components)
    assert not validation.issues
    context = RunContext(tmp_path, "input", tmp_path / "debug")
    target = tmp_path / "scenario.yml"
    write_scenario(target, build.yaml)
    graph_path = context.path("network_graph.json")
    graph_path.write_bytes(b"previous JSON")
    later = context.path("site_graph.jpg")
    later.write_bytes(b"previous image")

    def fail_json(graph, path, **kwargs):
        path.write_bytes(b"partial JSON")
        raise OSError("JSON serialization failed")

    monkeypatch.setattr(artifacts, "save_site_graph_json", fail_json)
    with pytest.raises(OSError, match="JSON serialization failed"):
        write_artifacts(build, validation, scenario_config, context)
    assert target.read_text() == build.yaml
    assert (context.debug_dir / "input_traffic_debug.json").exists()
    assert graph_path.read_bytes() == b"previous JSON"
    assert later.read_bytes() == b"previous image"
    assert not list(tmp_path.glob(".input_network_graph.*"))


@pytest.mark.parametrize("has_prepared", [False, True])
def test_invalid_validation_writes_nothing(
    tmp_path, corridor_graph, scenario_config, has_prepared
):
    build = build_scenario(corridor_graph, scenario_config, load_library(None))
    prepared = validate_scenario(build.yaml, scenario_config.components).prepared
    assert prepared is not None
    out = tmp_path / "absent"
    with pytest.raises(ValueError, match="invalid scenario"):
        write_artifacts(
            build,
            Validation(("refused",), prepared if has_prepared else None),
            scenario_config,
            RunContext(out),
        )
    assert not out.exists()


def test_artifact_image_suffix_and_failure_order(
    tmp_path, monkeypatch, corridor_graph, scenario_config
):
    import netlab_topogen.visualization as drawing

    build = build_scenario(corridor_graph, scenario_config, load_library(None))
    validation = validate_scenario(build.yaml, scenario_config.components)
    scenario_config.visualization.export_site_graph = True
    scenario_config.visualization.export_blueprint_diagrams = True
    context = RunContext(tmp_path, "input")
    target = context.path("site_graph.jpg")
    target.write_bytes(b"old JPEG")
    called = []

    def fail_image(graph, path, **kwargs):
        assert path.suffix == ".jpg"
        called.append(path)
        path.write_bytes(b"partial JPEG")
        raise OSError("renderer failed")

    monkeypatch.setattr(drawing, "export_site_graph_map", fail_image)
    with pytest.raises(OSError, match="renderer failed"):
        write_artifacts(build, validation, scenario_config, context)
    assert called and target.read_bytes() == b"old JPEG"
    assert context.path("network_graph.json").exists()
    assert not list(tmp_path.glob("input_blueprint*"))


@pytest.mark.parametrize("mask, expected", [(0o022, 0o644), (0o077, 0o600)])
@pytest.mark.parametrize("suffix", [".yml", ".json", ".jpg"])
@pytest.mark.parametrize("existing", [False, True])
def test_atomic_publication_uses_plain_write_mode(
    tmp_path, mask, expected, suffix, existing
):
    import os
    import stat

    target = tmp_path / ("artifact" + suffix)
    if existing:
        target.write_bytes(b"old")
        target.chmod(0o600)
    previous = os.umask(mask)
    try:
        write_atomic(target, lambda path: path.write_bytes(b"complete"))
    finally:
        os.umask(previous)
    assert target.read_bytes() == b"complete"
    assert stat.S_IMODE(target.stat().st_mode) == expected


def test_temp_collision_keeps_the_unowned_file(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import netlab_topogen.scenario.publication as publication

    target = tmp_path / "out.yml"
    target.write_bytes(b"old destination")
    collision = tmp_path / ".out.collision.yml"
    collision.write_bytes(b"unowned")
    monkeypatch.setattr(publication, "uuid4", lambda: SimpleNamespace(hex="collision"))
    with pytest.raises(FileExistsError):
        write_atomic(target, lambda path: path.write_bytes(b"new"))
    assert collision.read_bytes() == b"unowned"
    assert target.read_bytes() == b"old destination"
    collision.unlink()
    write_atomic(target, lambda path: path.write_bytes(b"new"))
    assert target.read_bytes() == b"new"


def test_site_graph_json_contains_sized_budgets_and_optics(
    tmp_path, corridor_graph, scenario_config
):
    import json

    from netlab_topogen.library import Library, thaw

    corridor_graph.remove_node(2)
    cfg = scenario_config
    cfg.build.build_defaults.site_blueprint = "Core2"
    cfg.build.build_defaults.inter_metro_link.capacity = 800
    cfg.build.build_defaults.inter_metro_link.mode = "one_to_one"
    cfg.build.build_defaults.dc_to_pop_link.capacity = 800
    cfg.build.tm_sizing.enabled = True
    cfg.build.tm_sizing.headroom = 1
    cfg.build.tm_sizing.quantum_gbps = 1000
    cfg.build.tm_sizing.alpha_dc_to_pop = 1
    cfg.traffic.mw_per_dc_region = 1
    cfg.traffic.gbps_per_mw = 8000
    cfg.components.optics = {"core->core": "O1k", "core->dc": "O1k", "dc->core": "O1k"}
    cfg.visualization.export_site_graph = (
        cfg.visualization.export_blueprint_diagrams
    ) = False
    base = load_library(None)
    blueprints = thaw(base.blueprints)
    blueprints["Core2"] = {
        "nodes": {
            "core": {"count": 2, "template": "core{n}", "attrs": {"role": "core"}}
        },
        "links": [],
    }
    components = thaw(base.components)
    components["O1k"] = {"capacity": 1000, "ports": 1}
    build = build_scenario(
        corridor_graph,
        cfg,
        Library(blueprints, components, base.failure_policies, base.workflows),
    )
    validation = validate_scenario(build.yaml, cfg.components)
    assert not validation.issues
    write_artifacts(build, validation, cfg, RunContext(tmp_path, "hand"))
    rows = json.loads((tmp_path / "hand_network_graph.json").read_text())["edges"]
    assert len(rows) == 3
    assert sum(row["link_type"] == "dc_to_pop" for row in rows) == 2
    for row in rows:
        assert row["base_capacity"] == row["target_capacity"] == 8000
        assert row["capacity"] == 4000
        assert all(
            type(row[key]) is float
            for key in ("base_capacity", "target_capacity", "capacity")
        )
        assert row["hardware"] == {
            end: {"component": "O1k", "count": 4.0} for end in ("source", "target")
        }
        assert all(type(end["count"]) is float for end in row["hardware"].values())


def test_blueprint_representative_tie_keeps_the_first_site(
    tmp_path, monkeypatch, corridor_graph, scenario_config
):
    import netlab_topogen.visualization as drawing
    from netlab_topogen.library import Library, WorkflowDefinition

    corridor_graph.remove_node(2)
    cfg = scenario_config
    cfg.build.build_defaults.dc_regions_per_metro = 0
    cfg.traffic.enabled = False
    cfg.visualization.export_site_graph = False
    cfg.visualization.export_blueprint_diagrams = True
    cfg.workflows.assignments.default = "stats"
    base = load_library(None)
    library = Library(
        base.blueprints,
        base.components,
        base.failure_policies,
        {"stats": WorkflowDefinition(({"type": "NetworkStats", "name": "stats"},))},
    )
    build = build_scenario(corridor_graph, cfg, library)
    validation = validate_scenario(build.yaml, cfg.components)
    assert not validation.issues
    selected = []

    def draw(blueprint, network, site, path, **kwargs):
        selected.append((blueprint, site))
        path.write_bytes(b"complete image")

    monkeypatch.setattr(drawing, "export_blueprint_diagram", draw)
    write_artifacts(build, validation, cfg, RunContext(tmp_path, "tie"))
    assert selected == [("SingleRouter", "metro1/pop1")]
    assert (
        tmp_path / "tie_blueprint_SingleRouter.jpg"
    ).read_bytes() == b"complete image"


def test_artifact_retains_declared_role_pairs_even_when_matches_agree(
    tmp_path, corridor_graph, scenario_config
):
    import json

    snapshots = []
    cfg = scenario_config
    for index, pairs in enumerate((["core|core", "dc|dc"], ["core|dc"])):
        cfg.build.build_defaults.inter_metro_link.role_pairs = pairs
        build = build_scenario(corridor_graph, cfg, load_library(None))
        validation = validate_scenario(build.yaml, cfg.components)
        assert not validation.issues
        context = RunContext(tmp_path / str(index), "input")
        write_artifacts(build, validation, cfg, context)
        data = json.loads(context.path("network_graph.json").read_text())
        for edge in data["edges"]:
            if edge["link_type"] == "inter_metro_corridor":
                assert list(edge) == [
                    "source",
                    "target",
                    "key",
                    "attrs",
                    "link_type",
                    "base_capacity",
                    "target_capacity",
                    "cost",
                    "adjacency_id",
                    "distance_km",
                    "source_metro",
                    "target_metro",
                    "match",
                    "role_pairs",
                    "risk_groups",
                    "euclidean_km",
                    "detour_ratio",
                    "capacity",
                ]
                assert edge.pop("role_pairs") == pairs
        snapshots.append(data)
    assert snapshots[0] == snapshots[1]
