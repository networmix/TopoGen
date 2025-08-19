from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

import topogen.geo_utils as geoutils
import topogen.visualization as viz


class _ConusStub:
    def plot(self, ax=None, **kwargs):  # type: ignore[no-untyped-def]
        return None


class _Metro:
    def __init__(
        self, metro_id: str, name: str, coords: tuple[float, float], radius_km: float
    ) -> None:
        self.metro_id = metro_id
        self.name = name
        self.coordinates = coords
        self.radius_km = radius_km


def _patch_context_and_conus(monkeypatch):  # type: ignore[no-untyped-def]
    # Avoid network calls and GIS dependencies for basemap and CONUS
    monkeypatch.setattr(viz.cx, "add_basemap", lambda *args, **kwargs: None)
    # export_* functions import create_conus_mask from topogen.geo_utils at call time
    monkeypatch.setattr(
        geoutils, "create_conus_mask", lambda *args, **kwargs: _ConusStub()
    )


def test_export_cluster_map_success(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _patch_context_and_conus(monkeypatch)
    out = tmp_path / "clusters.jpg"
    # Create a dummy file so path.exists() passes
    conus = tmp_path / "conus.boundary"
    conus.write_text("stub")
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    viz.export_cluster_map(centroids, out, conus, "EPSG:5070")
    assert out.exists() and out.stat().st_size > 1000


def test_export_integrated_graph_map_corridor_graph(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    _patch_context_and_conus(monkeypatch)
    out = tmp_path / "integrated_corridor.jpg"
    conus = tmp_path / "conus.boundary"
    conus.write_text("stub")

    metros = [
        _Metro("A", "A_name", (0.0, 0.0), 10.0),
        _Metro("B", "B_name", (2.0, 0.0), 8.0),
    ]
    G = nx.Graph()
    # corridor edges with metadata specifying metro ids (straight line mode)
    G.add_edge("A_node", "B_node", edge_type="corridor", metro_a="A", metro_b="B")

    viz.export_integrated_graph_map(
        metros, G, out, conus, "EPSG:5070", use_real_geometry=False, dpi=120
    )
    assert out.exists() and out.stat().st_size > 1000


def test_export_integrated_graph_map_real_geometry(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _patch_context_and_conus(monkeypatch)
    out = tmp_path / "integrated_geom.jpg"
    conus = tmp_path / "conus.boundary"
    conus.write_text("stub")

    metros = [
        _Metro("A", "A_name", (0.0, 0.0), 10.0),
        _Metro("B", "B_name", (2.0, 0.0), 8.0),
    ]
    G = nx.Graph()
    # corridor edges providing explicit geometry
    G.add_edge(
        "A_node",
        "B_node",
        edge_type="corridor",
        geometry=[(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)],
    )

    viz.export_integrated_graph_map(
        metros, G, out, conus, "EPSG:5070", use_real_geometry=True, dpi=120
    )
    assert out.exists() and out.stat().st_size > 1000


def test_export_integrated_graph_map_full_graph_from_tags(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    _patch_context_and_conus(monkeypatch)
    out = tmp_path / "integrated_full.jpg"
    conus = tmp_path / "conus.boundary"
    conus.write_text("stub")

    metros = [
        _Metro("A", "A_name", (0.0, 0.0), 10.0),
        _Metro("B", "B_name", (2.0, 0.0), 8.0),
    ]
    G = nx.Graph()
    G.add_node("mA")
    G.add_node("mB")
    # Full highway graph style: corridor tag on edges with list of pairs
    G.add_edge("mA", "mB", corridor=[{"metro_a": "A", "metro_b": "B"}])

    viz.export_integrated_graph_map(
        metros, G, out, conus, "EPSG:5070", use_real_geometry=False, dpi=120
    )
    assert out.exists() and out.stat().st_size > 1000
