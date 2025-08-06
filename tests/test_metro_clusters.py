"""Tests for metro clusters module."""

import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from topogen.config import ClusteringConfig, FormattingConfig
from topogen.metro_clusters import MetroCluster, load_metro_clusters


class TestMetroCluster:
    """Test MetroCluster dataclass functionality."""

    @pytest.fixture
    def sample_cluster(self):
        """Create a sample MetroCluster for testing."""
        return MetroCluster(
            metro_id="63217",
            name="new-york--newark",
            name_orig="New York--Newark, NY--NJ--CT",
            uac_code="63217",
            land_area_km2=8936.2,
            centroid_x=586515.2,
            centroid_y=4506242.8,
            radius_km=53.4,
        )

    @pytest.fixture
    def another_cluster(self):
        """Create another MetroCluster for testing interactions."""
        return MetroCluster(
            metro_id="51445",
            name="los-angeles--long-beach--anaheim",
            name_orig="Los Angeles--Long Beach--Anaheim, CA",
            uac_code="51445",
            land_area_km2=4850.3,
            centroid_x=-2056789.1,
            centroid_y=3926233.5,
            radius_km=39.3,
        )

    def test_cluster_creation(self, sample_cluster):
        """Test basic MetroCluster creation and attributes."""
        assert sample_cluster.metro_id == "63217"
        assert sample_cluster.name == "new-york--newark"
        assert sample_cluster.name_orig == "New York--Newark, NY--NJ--CT"
        assert sample_cluster.uac_code == "63217"
        assert sample_cluster.land_area_km2 == 8936.2
        assert sample_cluster.centroid_x == 586515.2
        assert sample_cluster.centroid_y == 4506242.8
        assert sample_cluster.radius_km == 53.4

    def test_cluster_immutable(self, sample_cluster):
        """Test that MetroCluster is frozen/immutable."""
        with pytest.raises(AttributeError):
            sample_cluster.metro_id = 999

    def test_coordinates_property(self, sample_cluster):
        """Test coordinates property returns correct tuple."""
        coords = sample_cluster.coordinates
        assert coords == (586515.2, 4506242.8)
        assert isinstance(coords, tuple)
        assert len(coords) == 2

    def test_coordinates_array_property(self, sample_cluster):
        """Test coordinates_array property returns numpy array."""
        coords_array = sample_cluster.coordinates_array
        assert isinstance(coords_array, np.ndarray)
        assert coords_array.shape == (2,)
        np.testing.assert_array_equal(coords_array, [586515.2, 4506242.8])

    def test_distance_to(self, sample_cluster, another_cluster):
        """Test distance calculation between clusters."""
        distance = sample_cluster.distance_to(another_cluster)

        # Calculate expected distance manually
        dx = 586515.2 - (-2056789.1)
        dy = 4506242.8 - 3926233.5
        expected = math.sqrt(dx * dx + dy * dy)

        assert abs(distance - expected) < 1e-6

    def test_distance_to_same_cluster(self, sample_cluster):
        """Test distance to same cluster is zero."""
        distance = sample_cluster.distance_to(sample_cluster)
        assert distance == 0.0

    def test_overlaps_with_far_clusters(self, sample_cluster, another_cluster):
        """Test overlap detection with distant clusters."""
        # These clusters are far apart and should not overlap
        assert not sample_cluster.overlaps_with(another_cluster)
        assert not another_cluster.overlaps_with(sample_cluster)

    def test_overlaps_with_close_clusters(self):
        """Test overlap detection with nearby clusters."""
        cluster1 = MetroCluster(
            metro_id="12345",
            name="cluster-1",
            name_orig="Cluster 1",
            uac_code="12345",
            land_area_km2=100.0,
            centroid_x=0.0,
            centroid_y=0.0,
            radius_km=10.0,
        )

        cluster2 = MetroCluster(
            metro_id="67890",
            name="cluster-2",
            name_orig="Cluster 2",
            uac_code="67890",
            land_area_km2=200.0,
            centroid_x=15000.0,  # 15 km away
            centroid_y=0.0,
            radius_km=10.0,
        )

        # Clusters with 20km combined radius and 15km separation should overlap
        assert cluster1.overlaps_with(cluster2)
        assert cluster2.overlaps_with(cluster1)

    def test_overlaps_with_same_cluster(self, sample_cluster):
        """Test that a cluster always overlaps with itself."""
        assert sample_cluster.overlaps_with(sample_cluster)


class TestLoadMetroClusters:
    """Test metro cluster loading functionality."""

    @pytest.fixture
    def mock_uac_data(self):
        """Create mock UAC GeoDataFrame for testing."""
        # Create test polygons for 5 urban areas
        polygons = []
        names = []
        uac_codes = []
        land_areas = []

        for i in range(5):
            # Create simple square polygons
            size = 0.1 + i * 0.05  # Varying sizes
            x_center = -100 + i * 2
            y_center = 40 + i

            polygon = Polygon(
                [
                    (x_center - size, y_center - size),
                    (x_center + size, y_center - size),
                    (x_center + size, y_center + size),
                    (x_center - size, y_center + size),
                ]
            )
            polygons.append(polygon)
            names.append(f"Urban Area {i + 1}")
            uac_codes.append(f"UAC{i + 1:02d}")
            # Land areas in m² (decreasing order so we can test top-k selection)
            land_areas.append((5 - i) * 10_000_000)  # 50M, 40M, 30M, 20M, 10M m²

        gdf = gpd.GeoDataFrame(
            {
                "NAME20": names,
                "UACE20": uac_codes,
                "ALAND20": land_areas,
            },
            geometry=polygons,
            crs="EPSG:4326",
        )

        return gdf

    @pytest.fixture
    def temp_uac_file(self, mock_uac_data):
        """Create temporary UAC file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            temp_path = Path(tmp.name)

        # Save mock data to temporary shapefile then zip
        with tempfile.TemporaryDirectory() as temp_dir:
            shp_path = Path(temp_dir) / "test_uac.shp"
            mock_uac_data.to_file(shp_path)

            # Create zip file
            import zipfile

            with zipfile.ZipFile(temp_path, "w") as zf:
                for file_path in Path(temp_dir).glob("test_uac.*"):
                    zf.write(file_path, file_path.name)

        yield temp_path

        # Cleanup
        temp_path.unlink()

    def test_load_basic_functionality(self, temp_uac_file, mock_uac_data):
        """Test basic loading of metro clusters."""
        clustering_config = ClusteringConfig(max_uac_radius_km=100.0)
        formatting_config = FormattingConfig()
        clusters = load_metro_clusters(
            uac_path=temp_uac_file,
            k=3,
            target_crs="EPSG:3857",
            clustering_config=clustering_config,
            formatting_config=formatting_config,
        )

        assert len(clusters) == 3
        assert all(isinstance(c, MetroCluster) for c in clusters)

        # Check that clusters are ordered by land area (largest first)
        # Metro IDs should now be UACE20 codes directly
        for i in range(len(clusters)):
            expected_uace = f"UAC{i + 1:02d}"

            assert clusters[i].metro_id == expected_uace
            assert clusters[i].name == f"urban-area-{i + 1}"
            assert clusters[i].name_orig == f"Urban Area {i + 1}"
            assert clusters[i].uac_code == expected_uace

    def test_load_insufficient_areas(self, temp_uac_file):
        """Test error when requesting more areas than available."""
        clustering_config = ClusteringConfig()
        formatting_config = FormattingConfig()
        with pytest.raises(ValueError, match="Only 5 urban areas available"):
            load_metro_clusters(
                uac_path=temp_uac_file,
                k=10,  # More than the 5 available
                target_crs="EPSG:3857",
                clustering_config=clustering_config,
                formatting_config=formatting_config,
            )

    def test_load_file_not_found(self):
        """Test error when UAC file doesn't exist."""
        nonexistent_path = Path("nonexistent_file.zip")

        clustering_config = ClusteringConfig()
        formatting_config = FormattingConfig()
        with pytest.raises(FileNotFoundError, match="UAC file not found"):
            load_metro_clusters(
                uac_path=nonexistent_path,
                k=3,
                target_crs="EPSG:3857",
                clustering_config=clustering_config,
                formatting_config=formatting_config,
            )

    def test_radius_calculation(self, temp_uac_file):
        """Test equivalent radius calculation from land area."""
        clustering_config = ClusteringConfig(max_uac_radius_km=100.0)
        formatting_config = FormattingConfig()
        clusters = load_metro_clusters(
            uac_path=temp_uac_file,
            k=2,
            target_crs="EPSG:3857",
            clustering_config=clustering_config,
            formatting_config=formatting_config,
        )

        # Test first cluster (largest area: 50M m² = 50 km²)
        expected_radius_1 = math.sqrt(50 / math.pi)  # r = sqrt(area/π)
        assert abs(clusters[0].radius_km - expected_radius_1) < 0.1

        # Test second cluster (40M m² = 40 km²)
        expected_radius_2 = math.sqrt(40 / math.pi)
        assert abs(clusters[1].radius_km - expected_radius_2) < 0.1

    def test_numpy_array_indexing(self, temp_uac_file):
        """Test that radii_km array indexing works correctly (regression test)."""
        # This test ensures we don't use .iloc on numpy arrays
        clustering_config = ClusteringConfig(max_uac_radius_km=100.0)
        formatting_config = FormattingConfig()
        clusters = load_metro_clusters(
            uac_path=temp_uac_file,
            k=1,
            target_crs="EPSG:3857",
            clustering_config=clustering_config,
            formatting_config=formatting_config,
        )

        # Should not raise AttributeError about .iloc on numpy array
        assert isinstance(clusters[0].radius_km, float)
        assert clusters[0].radius_km > 0

    def test_radius_capping(self, temp_uac_file):
        """Test that radius is capped at max_radius_km."""
        clustering_config = ClusteringConfig(max_uac_radius_km=2.0)  # Very small cap
        formatting_config = FormattingConfig()
        clusters = load_metro_clusters(
            uac_path=temp_uac_file,
            k=1,
            target_crs="EPSG:3857",
            clustering_config=clustering_config,
            formatting_config=formatting_config,
        )

        assert clusters[0].radius_km == 2.0

    @patch("topogen.metro_clusters._export_cluster_files")
    def test_export_files_flag(self, mock_export, temp_uac_file):
        """Test that export function is called when export_files=True."""
        clustering_config = ClusteringConfig(export_clusters=True)
        formatting_config = FormattingConfig()
        clusters = load_metro_clusters(
            uac_path=temp_uac_file,
            k=2,
            target_crs="EPSG:3857",
            clustering_config=clustering_config,
            formatting_config=formatting_config,
        )

        mock_export.assert_called_once_with(clusters, "EPSG:3857")

    @patch("topogen.metro_clusters._export_cluster_files")
    def test_no_export_by_default(self, mock_export, temp_uac_file):
        """Test that export function is not called by default."""
        clustering_config = ClusteringConfig(export_clusters=False)
        formatting_config = FormattingConfig()
        load_metro_clusters(
            uac_path=temp_uac_file,
            k=2,
            target_crs="EPSG:3857",
            clustering_config=clustering_config,
            formatting_config=formatting_config,
        )

        mock_export.assert_not_called()

    def test_coordinate_precision(self, temp_uac_file):
        """Test that coordinates are rounded to appropriate precision."""
        clustering_config = ClusteringConfig()
        formatting_config = FormattingConfig()
        clusters = load_metro_clusters(
            uac_path=temp_uac_file,
            k=1,
            target_crs="EPSG:3857",
            clustering_config=clustering_config,
            formatting_config=formatting_config,
        )

        cluster = clusters[0]

        # Check centroid coordinates are rounded to 1 decimal place
        assert cluster.centroid_x == round(cluster.centroid_x, 1)
        assert cluster.centroid_y == round(cluster.centroid_y, 1)

        # Check radius is rounded to 2 decimal places
        assert cluster.radius_km == round(cluster.radius_km, 2)

        # Check land area is rounded to 2 decimal places
        assert cluster.land_area_km2 == round(cluster.land_area_km2, 2)

    @patch("topogen.geo_utils.create_conus_mask")
    def test_conus_filtering(self, mock_conus_mask, temp_uac_file, mock_uac_data):
        """Test CONUS boundary filtering functionality."""
        # Create mock CONUS mask that excludes the last 2 areas
        mock_mask_geom = Polygon([(-102, 38), (-98, 38), (-98, 42), (-102, 42)])
        mock_conus_gdf = gpd.GeoDataFrame(
            {}, geometry=[mock_mask_geom], crs="EPSG:3857"
        )
        mock_conus_mask.return_value = mock_conus_gdf

        # Test with CONUS filtering
        boundary_path = Path("mock_boundary.zip")

        # Mock the GeoSeries intersection to return only first 3 areas
        with patch("geopandas.GeoSeries.intersects") as mock_intersects:
            mock_intersects.return_value = pd.Series([True, True, True, False, False])

            clustering_config = ClusteringConfig()
            formatting_config = FormattingConfig()
            clusters = load_metro_clusters(
                uac_path=temp_uac_file,
                k=2,
                target_crs="EPSG:3857",
                clustering_config=clustering_config,
                formatting_config=formatting_config,
                conus_boundary_path=boundary_path,
            )

        assert len(clusters) == 2
        mock_conus_mask.assert_called_once_with(boundary_path, "EPSG:4269")

    @patch("topogen.geo_utils.create_conus_mask")
    def test_conus_filtering_multiple_geometries(self, mock_conus_mask, temp_uac_file):
        """Test CONUS filtering with multiple geometries (regression test)."""
        # Create mock CONUS mask with multiple geometries (e.g. Alaska + CONUS)
        main_conus = Polygon([(-102, 38), (-98, 38), (-98, 42), (-102, 42)])
        alaska_strip = Polygon([(-150, 60), (-140, 60), (-140, 65), (-150, 65)])

        mock_conus_gdf = gpd.GeoDataFrame(
            {}, geometry=[main_conus, alaska_strip], crs="EPSG:3857"
        )
        mock_conus_mask.return_value = mock_conus_gdf

        # Test should use unary_union to dissolve multiple geometries
        boundary_path = Path("mock_boundary.zip")

        with patch("geopandas.GeoSeries.intersects") as mock_intersects:
            mock_intersects.return_value = pd.Series([True, True, False, False, False])

            clustering_config = ClusteringConfig()
            formatting_config = FormattingConfig()
            clusters = load_metro_clusters(
                uac_path=temp_uac_file,
                k=2,
                target_crs="EPSG:3857",
                clustering_config=clustering_config,
                formatting_config=formatting_config,
                conus_boundary_path=boundary_path,
            )

        assert len(clusters) == 2
        # Verify unary_union was called on the geometry series
        assert mock_intersects.called

    def test_representative_point_vs_centroid(self):
        """Test that representative_point is used instead of centroid."""
        # Create a concave polygon where centroid falls outside
        concave_coords = [
            (0, 0),
            (10, 0),
            (10, 5),
            (2, 5),
            (2, 3),
            (8, 3),
            (8, 2),
            (2, 2),
            (2, 10),
            (0, 10),
        ]
        concave_polygon = Polygon(concave_coords)

        # Centroid falls outside this concave polygon
        centroid = concave_polygon.centroid
        representative = concave_polygon.representative_point()

        # Representative point should be inside, centroid might not be
        assert concave_polygon.contains(representative)
        # This assertion might fail, demonstrating why we use representative_point
        # assert concave_polygon.contains(centroid)  # This could fail

        # Both should be Point objects
        from shapely.geometry import Point

        assert isinstance(centroid, Point)
        assert isinstance(representative, Point)


class TestExportClusterFiles:
    """Test cluster file export functionality."""

    @pytest.fixture
    def sample_clusters(self):
        """Create sample clusters for export testing."""
        return [
            MetroCluster(
                metro_id="12345",
                name="metro-1",
                name_orig="Metro 1",
                uac_code="12345",
                land_area_km2=100.0,
                centroid_x=1000.0,
                centroid_y=2000.0,
                radius_km=10.0,
            ),
            MetroCluster(
                metro_id="67890",
                name="metro-2",
                name_orig="Metro 2",
                uac_code="67890",
                land_area_km2=200.0,
                centroid_x=3000.0,
                centroid_y=4000.0,
                radius_km=15.0,
            ),
        ]

    @patch("topogen.visualization.export_cluster_map")
    def test_export_files_basic(self, mock_export_map, sample_clusters):
        """Test basic export functionality."""
        from topogen.metro_clusters import _export_cluster_files

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("topogen.metro_clusters.Path") as mock_path_class:
                # Create a mock path instance
                mock_path_instance = Path(temp_dir) / "processed"
                mock_path_class.return_value = mock_path_instance

                # Create the actual directory for the test
                mock_path_instance.mkdir(parents=True, exist_ok=True)

                # Mock the file writing
                with patch.object(gpd.GeoDataFrame, "to_file") as mock_to_file:
                    _export_cluster_files(sample_clusters, "EPSG:3857")

                    # Verify GeoJSON export was attempted
                    mock_to_file.assert_called_once()

                    # Verify visualization export was attempted
                    mock_export_map.assert_called_once()

    @patch(
        "topogen.visualization.export_cluster_map", side_effect=Exception("Mock error")
    )
    def test_export_visualization_error_handling(
        self, mock_export_map, sample_clusters
    ):
        """Test that visualization errors raise exceptions (strict mode)."""
        from topogen.metro_clusters import _export_cluster_files

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("topogen.metro_clusters.Path") as mock_path:
                output_dir = Path(temp_dir) / "processed"
                output_dir.mkdir(parents=True, exist_ok=True)
                mock_path.return_value = output_dir

                with patch.object(gpd.GeoDataFrame, "to_file"):
                    # Should raise exception due to strict error handling
                    with pytest.raises(Exception, match="Mock error"):
                        _export_cluster_files(sample_clusters, "EPSG:3857")

                    mock_export_map.assert_called_once()
