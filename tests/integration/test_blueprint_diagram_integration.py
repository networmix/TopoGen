from __future__ import annotations

from pathlib import Path

from topogen.visualization import export_blueprint_diagram


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


class _Link:
    def __init__(self, src, dst, cap: float) -> None:
        self.source = src
        self.target = dst
        self.capacity = cap


def _make_stub_net():
    class _Net:
        pass

    net = _Net()
    net.nodes = {
        1: _Node("metro1/dc1/A1"),
        2: _Node("metro1/dc1/B1"),
        3: _Node("metro2/dc3/X1"),
    }
    net.links = {
        1: _Link(net.nodes[1], net.nodes[2], 10.0),
        2: _Link(net.nodes[1], net.nodes[3], 20.0),
    }
    return net


def test_export_blueprint_diagram_smoke(tmp_path: Path) -> None:
    # Minimal blueprint with one intra-site adjacency and one external
    bp = {
        "groups": {
            "G1": {"node_count": 2},
            "G2": {"node_count": 1},
        },
        "adjacency": [
            {"source": "G1", "target": "G1", "pattern": "mesh"},
            {"source": "G1", "target": "G2", "pattern": "uplink"},
        ],
    }
    net = _make_stub_net()
    out = tmp_path / "bp.jpg"
    export_blueprint_diagram("UnitBP", bp, net, "metro1/dc1", out)
    assert out.exists() and out.stat().st_size > 1000
