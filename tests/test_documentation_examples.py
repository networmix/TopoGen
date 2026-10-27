"""The examples of dev/PIPELINE.md run as written and give the numbers it states."""

import re
import shutil
from pathlib import Path

import networkx as nx
import pytest
import yaml

from netlab_topogen import TopologyConfig, save_to_json
from netlab_topogen.config import FormattingConfig
from netlab_topogen.library import load_library
from netlab_topogen.scenario import build_scenario
from netlab_topogen.validation import validate_scenario

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "dev/PIPELINE.md"
EXAMPLE_METROS = ("new-york-jersey-city-newark", "columbus", "washington-arlington")


def _page_text() -> str:
    return " ".join(PAGE.read_text(encoding="utf-8").split())


def _python_block() -> str:
    blocks = re.findall(r"^```python\n(.*?)^```", PAGE.read_text(), re.M | re.S)
    assert len(blocks) == 1, "dev/PIPELINE.md must hold exactly one Python example"
    return blocks[0]


def _saved_graph(path: Path, names: tuple[str, ...]) -> None:
    """Metros on a line 300 km apart, keyed by coordinates as generate saves them."""
    graph = nx.MultiGraph()
    nodes = [(index * 300000.0, 0.0) for index in range(len(names))]
    for node, name in zip(nodes, names, strict=True):
        graph.add_node(
            node,
            node_type="metro",
            name=name,
            name_orig=name,
            metro_id=name,
            x=node[0],
            y=node[1],
            radius_km=20.0,
        )
    for a, b in zip(nodes, nodes[1:], strict=False):
        graph.add_edge(a, b, edge_type="corridor", length_km=300.0, geometry=[a, b])
    save_to_json(graph, path, "EPSG:5070", FormattingConfig())


def test_pipeline_python_block_publishes_the_files_it_names(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "lib", tmp_path / "lib")
    (tmp_path / "examples").mkdir()
    raw = yaml.safe_load((ROOT / "examples/small_baseline.yml").read_text())
    # Map tiles need the network; the drawings are not what the block shows.
    raw["visualization"]["site_graph"]["export"] = False
    raw["visualization"]["blueprints"]["export"] = False
    (tmp_path / "examples/small_baseline.yml").write_text(yaml.safe_dump(raw))
    output = tmp_path / "output"
    _saved_graph(output / "small_baseline_integrated_graph.json", EXAMPLE_METROS)
    monkeypatch.chdir(tmp_path)

    namespace: dict = {}
    exec(compile(_python_block(), str(PAGE), "exec"), namespace)

    assert namespace["validation"].issues == ()
    assert sorted(path.name for path in output.iterdir()) == [
        "small_baseline_integrated_graph.json",
        "small_baseline_network_graph.json",
        "small_baseline_scenario.yml",
    ]
    assert (output / "small_baseline_scenario.yml").read_text() == namespace[
        "build"
    ].yaml


def _two_metro_build(mw: float, headroom: float, quantum: float):
    graph = nx.MultiGraph()
    for index, name in enumerate(("alpha", "beta")):
        graph.add_node(
            name,
            node_type="metro",
            name=name,
            name_orig=name,
            metro_id=name,
            x=index * 300000.0,
            y=0.0,
            radius_km=20.0,
        )
    graph.add_edge("alpha", "beta", key=0, length_km=300.0, risk_groups=[])
    config = TopologyConfig()
    config.build.build_defaults.pop_per_metro = 1
    config.build.build_defaults.dc_regions_per_metro = 1
    config.traffic.model = "uniform"
    config.traffic.mw_per_dc_region = mw
    config.traffic.gbps_per_mw = 200.0
    config.traffic.priority_ratios = {0: 1.0}
    sizing = config.build.tm_sizing
    sizing.enabled = True
    sizing.headroom = headroom
    sizing.quantum_gbps = quantum
    sizing.alpha_dc_to_pop = 1.2
    config.components.optics = {
        pair: "800G-ZR+" for pair in ("core->core", "core->dc", "dc->core")
    }
    config.failure_policies.assignments.default = "empty"
    build = build_scenario(graph, config, load_library(None))
    assert validate_scenario(build.yaml, config.components).issues == ()
    links = {link.kind: link.id for link in build.site_plan.links}
    return build, links


@pytest.mark.parametrize(
    "statement",
    [
        "Each direction carries 5,000 Gbps.",
        "the corridor needs 6,500 Gbps, which rounds up to 9,600 Gbps.",
        "each data-center link needs 11,520 Gbps and gets 12,800.",
        "needs 12 optics at each end, and each 12,800 Gbps data-center link needs 16.",
    ],
)
def test_sizing_example_numbers(statement):
    assert statement in _page_text()
    build, links = _two_metro_build(25.0, 1.3, 3200.0)
    (matrix,) = build.traffic
    assert [demand.volume for demand in matrix.demands] == [5000, 5000]
    assert build.sizing.load[links["inter_metro_corridor"]] == 5000
    assert build.sizing.capacity[links["inter_metro_corridor"]] == 9600
    assert build.sizing.capacity[links["dc_to_pop"]] == 12800
    corridor = build.hardware[links["inter_metro_corridor"]]
    dc = build.hardware[links["dc_to_pop"]]
    assert (corridor.source_count, corridor.target_count) == (12, 12)
    assert (dc.source_count, dc.target_count) == (16, 16)


def test_sizing_example_exact_step():
    assert (
        "each direction carries 200 Gbps; with headroom 1.1 and a 10 Gbps quantum "
        "the corridor needs exactly 220 Gbps and gets 220." in _page_text()
    )
    build, links = _two_metro_build(1.0, 1.1, 10.0)
    assert build.sizing.load[links["inter_metro_corridor"]] == 200
    assert build.sizing.capacity[links["inter_metro_corridor"]] == 220
