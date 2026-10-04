"""Tests for highway graph construction with grid-snap approach."""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from topogen.config import HighwayProcessingConfig, ValidationConfig
from topogen.highway_graph import (
    _build_intersection_graph,
    _filter_highway_classes,
    _fix_geometries,
    _iter_snapped_edges,
    _validate_final_graph,
    build_highway_graph,
)


class TestGridSnapApproach:
    def test_iter_snapped_edges_basic(self):
        lines = [
            LineString([(0.0, 0.0), (100.0, 0.0)]),  # Horizontal line
            LineString([(100.0, 0.0), (200.0, 0.0)]),  # Connected horizontal line
            LineString([(100.0, 0.0), (100.0, 100.0)]),  # Vertical branch
        ]

        edges = list(_iter_snapped_edges(lines, snap_m=10.0))

        assert len(edges) >= 3

        for start, end, length in edges:
            assert isinstance(start, tuple)
            assert isinstance(end, tuple)
            assert isinstance(length, float)
            assert length > 0

    def test_grid_snapping_precision(self):
        lines = [
            LineString(
                [(5.0, 5.0), (15.0, 5.0)]
            ),  # Should snap to (0.0, 0.0), (20.0, 0.0)
            LineString(
                [(4.0, 6.0), (16.0, 4.0)]
            ),  # Should snap to (0.0, 10.0), (20.0, 0.0)
        ]

        edges = list(_iter_snapped_edges(lines, snap_m=10.0))

        for start, end, _ in edges:
            assert start[0] % 10.0 == 0.0
            assert start[1] % 10.0 == 0.0
            assert end[0] % 10.0 == 0.0
            assert end[1] % 10.0 == 0.0

    def test_degenerate_edge_filtering(self):
        # Line that becomes degenerate after snapping
        lines = [
            LineString([(1.0, 1.0), (2.0, 2.0)]),  # Snaps to (0.0, 0.0), (0.0, 0.0)
        ]

        edges = list(_iter_snapped_edges(lines, snap_m=10.0))

        assert len(edges) == 0

    def test_build_intersection_graph(self):
        lines = [
            LineString([(0.0, 0.0), (100.0, 0.0)]),  # Horizontal
            LineString([(100.0, 0.0), (200.0, 0.0)]),  # Connected horizontal
            LineString([(100.0, 0.0), (100.0, 100.0)]),  # Vertical branch
        ]

        G = _build_intersection_graph(lines, snap_precision_m=10.0)

        assert isinstance(G, nx.Graph)
        assert len(G.nodes) > 0
        assert len(G.edges) > 0

        for _, _, data in G.edges(data=True):
            assert "length_km" in data
            assert data["length_km"] > 0
            assert not np.isnan(data["length_km"])

        # Node at (100, 0) should have degree 3 (junction)
        junction_node = (100.0, 0.0)
        assert G.degree[junction_node] == 3

    def test_segment_length_calculation_horizontal_vertical(self):
        """Segments should have accurate length_km for axis-aligned geometries."""
        # 1000 meters horizontally => 1.0 km
        # 2000 meters vertically => 2.0 km
        lines = [
            LineString([(0.0, 0.0), (1000.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 2000.0)]),
        ]

        edges = list(
            _iter_snapped_edges(lines, snap_m=1.0)
        )  # 1m snap preserves coordinates

        # Normalize to set for order independence
        got = {
            ((sx, sy), (ex, ey), round(length_km, 6))
            for (sx, sy), (ex, ey), length_km in edges
        }
        expected = {
            ((0.0, 0.0), (1000.0, 0.0), 1.0),
            ((0.0, 0.0), (0.0, 2000.0), 2.0),
        }
        assert got == expected

    def test_segment_length_calculation_diagonal(self):
        """Segments should have accurate length_km for diagonal geometries."""
        # 3-4-5 triangle: 3000m x, 4000m y => 5000m => 5.0 km
        lines = [LineString([(0.0, 0.0), (3000.0, 4000.0)])]
        edges = list(_iter_snapped_edges(lines, snap_m=1.0))
        assert len(edges) == 1
        (_s, _e, length_km) = edges[0]
        assert abs(length_km - 5.0) < 1e-6

    def test_path_length_equals_sum_of_segments(self):
        """Shortest path length should equal the sum of constituent segment lengths."""
        # Build an L-shaped path: (0,0)->(3000,0)->(3000,4000)
        # Expected path length = 3km + 4km = 7km
        lines = [
            LineString([(0.0, 0.0), (3000.0, 0.0)]),
            LineString([(3000.0, 0.0), (3000.0, 4000.0)]),
        ]
        G = _build_intersection_graph(lines, snap_precision_m=1.0)

        start = (0.0, 0.0)
        end = (3000.0, 4000.0)
        assert start in G.nodes and end in G.nodes

        path = nx.shortest_path(G, start, end, weight="length_km")
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total += G[u][v]["length_km"]

        assert abs(total - 7.0) < 1e-9


class TestDataValidation:
    def _create_test_gdf(self, geometries, mtfcc_codes=None):
        """Helper to create test GeoDataFrame."""
        if mtfcc_codes is None:
            mtfcc_codes = ["S1100"] * len(geometries)

        gdf = gpd.GeoDataFrame(
            {
                "MTFCC": mtfcc_codes,
                "geometry": geometries,
            }
        )
        gdf.crs = "EPSG:4326"
        return gdf

    def test_filter_highway_classes(self):
        geometries = [
            LineString([(0.0, 0.0), (100.0, 0.0)]),
            LineString([(100.0, 0.0), (200.0, 0.0)]),
            LineString([(200.0, 0.0), (300.0, 0.0)]),
        ]

        mtfcc_codes = ["S1100", "S1200", "S1630"]  # Interstate, US Highway, Ramp
        gdf = self._create_test_gdf(geometries, mtfcc_codes)

        filtered = _filter_highway_classes(gdf, ["S1100", "S1200"])

        assert len(filtered) == 2
        assert set(filtered.MTFCC) == {"S1100", "S1200"}

    def test_filter_highway_classes_empty_result(self):
        geometries = [LineString([(0.0, 0.0), (100.0, 0.0)])]
        mtfcc_codes = ["S1630"]  # Only ramps
        gdf = self._create_test_gdf(geometries, mtfcc_codes)

        with pytest.raises(ValueError, match="No backbone highway segments found"):
            _filter_highway_classes(gdf, ["S1100", "S1200"])

    def test_fix_geometries(self):
        from shapely.geometry import MultiLineString

        geometries = [
            LineString([(0.0, 0.0), (100.0, 0.0)]),  # Valid
            MultiLineString(
                [  # MultiLineString (should be exploded)
                    LineString([(100.0, 0.0), (200.0, 0.0)]),
                    LineString([(200.0, 0.0), (300.0, 0.0)]),
                ]
            ),
            LineString([]),  # Empty (should be removed)
        ]

        gdf = self._create_test_gdf(geometries)
        fixed = _fix_geometries(gdf)

        assert len(fixed) == 3  # 1 valid + 2 from exploded MultiLineString

        for geom in fixed.geometry:
            assert geom.geom_type == "LineString"
            assert geom.is_valid
            assert not geom.is_empty

    def test_validate_final_graph(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (100.0, 0.0), length_km=100.0)
        G.add_edge((100.0, 0.0), (200.0, 0.0), length_km=50.0)

        validation_config = ValidationConfig()
        _validate_final_graph(G, validation_config)

    def test_validate_final_graph_disconnected(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (100.0, 0.0), length_km=100.0)
        G.add_edge((1000.0, 0.0), (1100.0, 0.0), length_km=100.0)  # Isolated

        validation_config = ValidationConfig()
        _validate_final_graph(G, validation_config)

    def test_validate_final_graph_invalid_length(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (100.0, 0.0), length_km=-50.0)  # Negative length

        validation_config = ValidationConfig()
        with pytest.raises(ValueError, match="has invalid length_km"):
            _validate_final_graph(G, validation_config)

    def test_validate_final_graph_high_degree(self):
        G = nx.Graph()
        center = (0.0, 0.0)

        for i in range(1500):
            G.add_edge(center, (float(i), float(i)), length_km=1.0)

        validation_config = ValidationConfig()
        with pytest.raises(ValueError, match="accidental mass snapping bug"):
            _validate_final_graph(G, validation_config)


class TestIntegratedGraphConstruction:
    def _create_mock_tiger_zip(
        self, temp_dir: Path, geometries, mtfcc_codes=None
    ) -> Path:
        if mtfcc_codes is None:
            mtfcc_codes = ["S1100"] * len(geometries)

        gdf = gpd.GeoDataFrame(
            {
                "MTFCC": mtfcc_codes,
                "geometry": geometries,
            }
        )
        gdf.crs = "EPSG:4326"

        shp_path = temp_dir / "test_roads.shp"
        gdf.to_file(shp_path)

        import zipfile

        zip_path = temp_dir / "test_tiger.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for ext in [".shp", ".shx", ".dbf", ".prj"]:
                file_path = shp_path.with_suffix(ext)
                if file_path.exists():
                    zf.write(file_path, file_path.name)

        return zip_path

    def test_build_highway_graph_complete_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            geometries = [
                LineString([(0.0, 0.0), (1.0, 0.0)]),  # Main east-west road
                LineString([(1.0, 0.0), (2.0, 0.0)]),  # Extension
                LineString([(1.0, 0.0), (1.0, 1.0)]),  # North-south branch
                LineString([(0.5, 0.1), (1.5, 0.1)]),  # Parallel road (should snap)
            ]

            tiger_zip = self._create_mock_tiger_zip(temp_path, geometries)

            highway_config = HighwayProcessingConfig(min_edge_length_km=0.001)
            validation_config = ValidationConfig()
            G = build_highway_graph(
                tiger_zip=tiger_zip,
                target_crs="EPSG:5070",
                highway_config=highway_config,
                validation_config=validation_config,
            )

            assert isinstance(G, nx.Graph)
            assert len(G.nodes) > 0
            assert len(G.edges) > 0

            for _, _, data in G.edges(data=True):
                assert "length_km" in data
                assert data["length_km"] > 0
                assert not np.isnan(data["length_km"])

            total_coords = sum(len(list(geom.coords)) for geom in geometries)
            assert len(G.nodes) < total_coords

    def test_build_highway_graph_empty_after_filtering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create data with no backbone highways
            geometries = [LineString([(0.0, 0.0), (1.0, 0.0)])]
            mtfcc_codes = ["S1630"]  # Ramp (not backbone)

            tiger_zip = self._create_mock_tiger_zip(temp_path, geometries, mtfcc_codes)

            highway_config = HighwayProcessingConfig(min_edge_length_km=0.001)
            validation_config = ValidationConfig()
            with pytest.raises(ValueError, match="No backbone highway segments found"):
                build_highway_graph(
                    tiger_zip=tiger_zip,
                    target_crs="EPSG:5070",
                    highway_config=highway_config,
                    validation_config=validation_config,
                )

    def test_snap_precision_configuration(self):
        snap_precision_m = 10.0

        lines = [LineString([(5.0, 5.0), (15.0, 15.0)])]
        edges = list(_iter_snapped_edges(lines, snap_m=snap_precision_m))

        for start, end, _ in edges:
            assert start[0] % snap_precision_m == 0.0
            assert start[1] % snap_precision_m == 0.0
            assert end[0] % snap_precision_m == 0.0
            assert end[1] % snap_precision_m == 0.0


class TestPerformanceCharacteristics:
    def test_linear_time_complexity(self):
        sizes = [10, 20, 40]
        processing_times = []

        import time

        for size in sizes:
            lines = [
                LineString([(i * 100.0, 0.0), (i * 100.0 + 50.0, 0.0)])
                for i in range(size)
            ]

            start_time = time.time()
            edges = list(_iter_snapped_edges(lines, snap_m=10.0))
            end_time = time.time()

            processing_times.append(end_time - start_time)
            assert len(edges) == size

        assert processing_times[-1] < processing_times[0] * 10

    def test_no_quadratic_behavior(self):
        # Crossing lines exercise vertex snapping without geometric intersection splitting.
        lines = []
        for i in range(20):
            # Horizontal lines
            lines.append(LineString([(0.0, i * 10.0), (200.0, i * 10.0)]))
            # Vertical lines
            lines.append(LineString([(i * 10.0, 0.0), (i * 10.0, 200.0)]))

        import time

        start_time = time.time()
        G = _build_intersection_graph(lines, snap_precision_m=10.0)
        end_time = time.time()

        assert end_time - start_time < 1.0
        assert len(G.nodes) > 0
        assert len(G.edges) > 0
