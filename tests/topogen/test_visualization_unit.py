from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from topogen.visualization import (
    export_cluster_map,
    export_integrated_graph_map,
    export_site_graph_map,
)


def test_export_cluster_map_validations(tmp_path: Path) -> None:
    out = tmp_path / "clusters.jpg"
    conus = tmp_path / "conus.shp"
    # Missing boundary file should produce ValueError before plotting
    with pytest.raises(ValueError, match="no centroids provided"):
        export_cluster_map(np.array([]).reshape(0, 2), out, conus, "EPSG:5070")
    assert not out.exists()


def test_export_integrated_graph_map_validations(tmp_path: Path) -> None:
    out = tmp_path / "integrated.jpg"
    conus = tmp_path / "conus.shp"
    G = nx.Graph()
    with pytest.raises(ValueError, match="no metros"):
        export_integrated_graph_map([], G, out, conus, "EPSG:5070")
    assert not out.exists()


def test_export_site_graph_map_validations(tmp_path: Path) -> None:
    out = tmp_path / "site.jpg"
    G = nx.MultiGraph()
    with pytest.raises(ValueError, match="graph is empty"):
        export_site_graph_map(G, out)
    assert not out.exists()
