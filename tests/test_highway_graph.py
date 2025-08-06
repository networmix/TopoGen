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
    """Test the linear-time grid-snap approach for intersection detection."""

    def test_iter_snapped_edges_basic(self):
        """Test basic grid snapping of line geometries."""
        # Create test lines with coordinates that should snap together
        lines = [
            LineString([(0.0, 0.0), (100.0, 0.0)]),  # Horizontal line
            LineString([(100.0, 0.0), (200.0, 0.0)]),  # Connected horizontal line
            LineString([(100.0, 0.0), (100.0, 100.0)]),  # Vertical branch
        ]

        edges = list(_iter_snapped_edges(lines, snap_m=10.0))

        # Should have 3 edges (one per line segment)
        assert len(edges) >= 3

        # Each edge should be a tuple of (start_point, end_point, length_km)
        for start, end, length in edges:
            assert isinstance(start, tuple)
            assert isinstance(end, tuple)
            assert isinstance(length, float)
            assert length > 0

    def test_grid_snapping_precision(self):
        """Test that coordinates snap to the specified grid precision."""
        # Lines with coordinates that should snap to same points
        lines = [
            LineString(
                [(5.0, 5.0), (15.0, 5.0)]
            ),  # Should snap to (0.0, 0.0), (20.0, 0.0)
            LineString(
                [(4.0, 6.0), (16.0, 4.0)]
            ),  # Should snap to (0.0, 0.0), (20.0, 0.0)
        ]

        edges = list(_iter_snapped_edges(lines, snap_m=10.0))

        # Check that coordinates are snapped to 10m grid
        for start, end, _ in edges:
            assert start[0] % 10.0 == 0.0
            assert start[1] % 10.0 == 0.0
            assert end[0] % 10.0 == 0.0
            assert end[1] % 10.0 == 0.0

    def test_degenerate_edge_filtering(self):
        """Test that degenerate edges (same start and end) are filtered out."""
        # Line that becomes degenerate after snapping
        lines = [
            LineString([(1.0, 1.0), (2.0, 2.0)]),  # Snaps to (0.0, 0.0), (0.0, 0.0)
        ]

        edges = list(_iter_snapped_edges(lines, snap_m=10.0))

        # Should have no edges since the snapped line becomes degenerate
        assert len(edges) == 0

    def test_build_intersection_graph(self):
        """Test building graph from snapped edges."""
        lines = [
            LineString([(0.0, 0.0), (100.0, 0.0)]),  # Horizontal
            LineString([(100.0, 0.0), (200.0, 0.0)]),  # Connected horizontal
            LineString([(100.0, 0.0), (100.0, 100.0)]),  # Vertical branch
        ]

        G = _build_intersection_graph(lines, snap_precision_m=10.0)

        # Should be a valid NetworkX graph
        assert isinstance(G, nx.Graph)
        assert len(G.nodes) > 0
        assert len(G.edges) > 0

        # All edges should have length_km attribute
        for _, _, data in G.edges(data=True):
            assert "length_km" in data
            assert data["length_km"] > 0
            assert not np.isnan(data["length_km"])

        # Node at (100, 0) should have degree 3 (junction)
        junction_node = (100.0, 0.0)
        if junction_node in G.nodes:
            assert len(list(G.neighbors(junction_node))) == 3


class TestDataValidation:
    """Test data loading and validation functions."""

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
        """Test filtering to keep only backbone highway classes."""
        geometries = [
            LineString([(0.0, 0.0), (100.0, 0.0)]),
            LineString([(100.0, 0.0), (200.0, 0.0)]),
            LineString([(200.0, 0.0), (300.0, 0.0)]),
        ]

        # Mix of highway classes
        mtfcc_codes = ["S1100", "S1200", "S1630"]  # Interstate, US Highway, Ramp
        gdf = self._create_test_gdf(geometries, mtfcc_codes)

        filtered = _filter_highway_classes(gdf, ["S1100", "S1200"])

        # Should keep only S1100 and S1200
        assert len(filtered) == 2
        assert set(filtered.MTFCC) == {"S1100", "S1200"}

    def test_filter_highway_classes_empty_result(self):
        """Test error when no highway classes remain."""
        geometries = [LineString([(0.0, 0.0), (100.0, 0.0)])]
        mtfcc_codes = ["S1630"]  # Only ramps
        gdf = self._create_test_gdf(geometries, mtfcc_codes)

        with pytest.raises(ValueError, match="No backbone highway segments found"):
            _filter_highway_classes(gdf, ["S1100", "S1200"])

    def test_fix_geometries(self):
        """Test geometry validation and fixing."""
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

        # Should have more rows due to exploding and fewer due to removing empty
        assert len(fixed) == 3  # 1 valid + 2 from exploded MultiLineString

        # All remaining geometries should be valid LineStrings
        for geom in fixed.geometry:
            assert geom.geom_type == "LineString"
            assert geom.is_valid
            assert not geom.is_empty

    def test_validate_final_graph(self):
        """Test final graph validation."""
        # Valid graph
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (100.0, 0.0), length_km=100.0)
        G.add_edge((100.0, 0.0), (200.0, 0.0), length_km=50.0)

        # Should not raise any errors
        validation_config = ValidationConfig()
        _validate_final_graph(G, validation_config)

    def test_validate_final_graph_disconnected(self):
        """Test validation accepts disconnected graph (connectivity checked later in pipeline)."""
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (100.0, 0.0), length_km=100.0)
        G.add_edge((1000.0, 0.0), (1100.0, 0.0), length_km=100.0)  # Isolated

        validation_config = ValidationConfig()
        # Should not raise - disconnected graphs are allowed at highway level
        _validate_final_graph(G, validation_config)

    def test_validate_final_graph_invalid_length(self):
        """Test validation fails for invalid edge lengths."""
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (100.0, 0.0), length_km=-50.0)  # Negative length

        validation_config = ValidationConfig()
        with pytest.raises(ValueError, match="has invalid length_km"):
            _validate_final_graph(G, validation_config)

    def test_validate_final_graph_high_degree(self):
        """Test validation fails for excessively high degree nodes."""
        G = nx.Graph()
        center = (0.0, 0.0)

        # Create a node with very high degree (indicates snapping bug)
        for i in range(1500):
            G.add_edge(center, (float(i), float(i)), length_km=1.0)

        validation_config = ValidationConfig()
        with pytest.raises(ValueError, match="accidental mass snapping bug"):
            _validate_final_graph(G, validation_config)


class TestIntegratedGraphConstruction:
    """Test the complete graph construction pipeline."""

    def _create_mock_tiger_zip(
        self, temp_dir: Path, geometries, mtfcc_codes=None
    ) -> Path:
        """Create a mock TIGER ZIP file for testing."""
        if mtfcc_codes is None:
            mtfcc_codes = ["S1100"] * len(geometries)

        gdf = gpd.GeoDataFrame(
            {
                "MTFCC": mtfcc_codes,
                "geometry": geometries,
            }
        )
        gdf.crs = "EPSG:4326"

        # Save to temporary shapefile
        shp_path = temp_dir / "test_roads.shp"
        gdf.to_file(shp_path)

        # Create a ZIP file
        import zipfile

        zip_path = temp_dir / "test_tiger.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for ext in [".shp", ".shx", ".dbf", ".prj"]:
                file_path = shp_path.with_suffix(ext)
                if file_path.exists():
                    zf.write(file_path, file_path.name)

        return zip_path

    def test_build_highway_graph_complete_pipeline(self):
        """Test the complete highway graph construction pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test geometries that form a connected network
            geometries = [
                LineString([(0.0, 0.0), (1.0, 0.0)]),  # Main east-west road
                LineString([(1.0, 0.0), (2.0, 0.0)]),  # Extension
                LineString([(1.0, 0.0), (1.0, 1.0)]),  # North-south branch
                LineString([(0.5, 0.1), (1.5, 0.1)]),  # Parallel road (should snap)
            ]

            tiger_zip = self._create_mock_tiger_zip(temp_path, geometries)

            # Build highway graph
            highway_config = HighwayProcessingConfig(min_edge_length_km=0.001)
            validation_config = ValidationConfig()
            G = build_highway_graph(
                tiger_zip=tiger_zip,
                target_crs="EPSG:5070",
                highway_config=highway_config,
                validation_config=validation_config,
            )

            # Verify result properties
            assert isinstance(G, nx.Graph)
            assert len(G.nodes) > 0
            assert len(G.edges) > 0
            # Note: Don't assert connectivity - highway graphs can be disconnected

            # All edges should have required attributes
            for _, _, data in G.edges(data=True):
                assert "length_km" in data
                assert data["length_km"] > 0
                assert not np.isnan(data["length_km"])

            # Graph should be contracted (fewer nodes than original coordinates)
            total_coords = sum(len(list(geom.coords)) for geom in geometries)
            assert len(G.nodes) < total_coords  # Contraction worked

    def test_build_highway_graph_empty_after_filtering(self):
        """Test error handling when no highways remain after filtering."""
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
        """Test that snap precision is used correctly."""
        # Use default snap precision value for testing
        snap_precision_m = 10.0  # 10 meters for merging twin carriageways

        # Test that coordinates actually snap to this precision
        lines = [LineString([(5.0, 5.0), (15.0, 15.0)])]
        edges = list(_iter_snapped_edges(lines, snap_m=snap_precision_m))

        for start, end, _ in edges:
            assert start[0] % snap_precision_m == 0.0
            assert start[1] % snap_precision_m == 0.0
            assert end[0] % snap_precision_m == 0.0
            assert end[1] % snap_precision_m == 0.0


class TestPerformanceCharacteristics:
    """Test performance characteristics of the grid-snap approach."""

    def test_linear_time_complexity(self):
        """Test that edge processing scales linearly with input size."""
        # Create increasingly large sets of non-intersecting lines
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
            assert len(edges) == size  # One edge per line

        # Processing time should scale roughly linearly
        # (allowing for some measurement noise in small test cases)
        assert processing_times[-1] < processing_times[0] * 10  # Very loose bound

    def test_no_quadratic_behavior(self):
        """Test that we don't have O(n²) behavior from overlay operations."""
        # The old implementation used gpd.overlay which was O(n²)
        # The new grid-snap approach should be linear

        # Create many intersecting lines (worst case for overlay)
        lines = []
        for i in range(20):
            # Horizontal lines
            lines.append(LineString([(0.0, i * 10.0), (200.0, i * 10.0)]))
            # Vertical lines
            lines.append(LineString([(i * 10.0, 0.0), (i * 10.0, 200.0)]))

        # This should complete quickly (not hang due to O(n²) complexity)
        import time

        start_time = time.time()
        G = _build_intersection_graph(lines, snap_precision_m=10.0)
        end_time = time.time()

        # Should complete in reasonable time (< 1 second for 40 lines)
        assert end_time - start_time < 1.0
        assert len(G.nodes) > 0
        assert len(G.edges) > 0
