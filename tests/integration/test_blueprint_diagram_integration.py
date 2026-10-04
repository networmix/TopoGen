from pathlib import Path

from ngraph import Link, Network, Node

from topogen.visualization import export_blueprint_diagram


def test_export_blueprint_diagram_smoke(tmp_path: Path) -> None:
    network = Network()
    for name in (
        "metro1/dc1/a/r1",
        "metro1/dc1/a/r2",
        "metro1/dc1/b/r1",
        "metro2/dc1/c/r1",
    ):
        network.add_node(Node(name, attrs={"role": "core"}))
    for source, target in (
        ("metro1/dc1/a/r1", "metro1/dc1/a/r2"),
        ("metro1/dc1/a/r1", "metro1/dc1/b/r1"),
        ("metro1/dc1/a/r1", "metro2/dc1/c/r1"),
    ):
        network.add_link(Link(source, target, capacity=10))
    output = tmp_path / "bp.jpg"
    export_blueprint_diagram("UnitBP", network, "metro1/dc1", output)
    assert output.exists() and output.stat().st_size > 1000
