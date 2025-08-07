"""Configuration management for topology generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from topogen.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class DataSources:
    """Data source configuration for topology generation.

    Contains paths to required geospatial datasets including Urban Area Centroids,
    TIGER/Line Primary Roads, and CONUS boundary files.
    """

    uac_polygons: Path
    tiger_roads: Path
    conus_boundary: Path

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        self.uac_polygons = Path(self.uac_polygons)
        self.tiger_roads = Path(self.tiger_roads)
        self.conus_boundary = Path(self.conus_boundary)


@dataclass
class ProjectionConfig:
    """Geographic projection configuration for coordinate reference systems.

    Defines the target coordinate reference system for spatial operations and
    transformations during topology generation.
    """

    target_crs: str = "EPSG:5070"


@dataclass
class HighwayProcessingConfig:
    """Highway data processing configuration for lightweight approach.

    Contains parameters for filtering, simplifying, and processing highway network
    data from TIGER/Line sources to create a backbone topology graph.
    """

    min_edge_length_km: float = 0.05  # Minimum edge length in kilometers
    snap_precision_m: float = 10.0  # Grid snap precision in meters
    highway_classes: list[str] = field(
        default_factory=lambda: ["S1100", "S1200"]
    )  # TIGER highway classes to keep
    min_cycle_nodes: int = 3  # Minimum nodes to contract isolated cycles
    filter_largest_component: bool = (
        True  # Keep only largest connected component if highway graph is disconnected
    )
    validation_sample_size: int = 5  # Number of edges to validate for efficiency


@dataclass
class RiskGroupsConfig:
    """Risk groups configuration for corridor edge assignment.

    Defines how risk groups are created and assigned to corridor edges,
    supporting failure scenario analysis across the network.
    """

    enabled: bool = True  # Enable risk group assignment for corridor edges
    group_prefix: str = "corridor_risk"  # Prefix for generated risk group names
    exclude_metro_radius_shared: bool = (
        True  # Exclude highway segments within metro radius from risk groups
    )


@dataclass
class CorridorsConfig:
    """Corridor discovery configuration for metro connectivity.

    Parameters for discovering and tagging corridors between metropolitan areas,
    including adjacency rules and risk group assignment settings.
    """

    k_paths: int = 1  # Maximum number of diverse paths per adjacent metro pair (reduced for performance)
    k_nearest: int = 3  # Number of nearest neighbors per metro for adjacency
    max_edge_km: float = 600.0  # Maximum distance for metro pair connections (km)
    max_corridor_distance_km: float = (
        1000.0  # Skip corridors longer than this distance (km)
    )
    risk_groups: RiskGroupsConfig = field(default_factory=RiskGroupsConfig)


@dataclass
class ValidationConfig:
    """Validation parameters for topology generation quality checks.

    Contains thresholds and requirements for validating the generated topology
    meets connectivity and structural requirements.
    """

    max_metro_highway_distance_km: float = 10.0
    require_connected: bool = True
    max_degree_threshold: int = 1000  # Maximum node degree before validation error
    high_degree_warning: int = 20  # Node degree threshold for warnings
    min_largest_component_fraction: float = 0.5  # Fails if sliver removal causes largest component to drop below 50% OR if corridor graph's largest component is below 50%


@dataclass
class LinkParams:
    """Link parameter configuration including capacity, cost, and attributes.

    Defines link properties with both functional parameters (capacity, cost)
    and metadata attributes that appear in the generated scenario.
    """

    capacity: int
    cost: int
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildDefaults:
    """Default configuration for build operations and site generation.

    Default settings for generating sites within metropolitan areas,
    including site count, blueprint assignments, and link parameters.
    """

    sites_per_metro: int = 2
    site_blueprint: str = "SingleRouter"
    intra_metro_link: LinkParams = field(
        default_factory=lambda: LinkParams(
            capacity=400, cost=1, attrs={"link_type": "intra_metro"}
        )
    )
    inter_metro_link: LinkParams = field(
        default_factory=lambda: LinkParams(
            capacity=100, cost=1, attrs={"link_type": "inter_metro_corridor"}
        )
    )


@dataclass
class BuildConfig:
    """Configuration for the build process and metro customization.

    Contains default build settings and per-metro overrides for
    customizing site generation within metropolitan areas.
    """

    build_defaults: BuildDefaults = field(default_factory=BuildDefaults)
    build_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ComponentAssignment:
    """Component assignment configuration for a network role.

    Defines hardware component and optics assignments for specific
    roles within the network hierarchy (spine, leaf, core).
    """

    hw_component: str = ""
    optics: str = ""


@dataclass
class ComponentAssignments:
    """Component assignments configuration for network roles and blueprints.

    Manages hardware component assignments across different network roles
    and blueprint-specific overrides for customized deployments.
    """

    # Default role assignments
    spine: ComponentAssignment = field(default_factory=ComponentAssignment)
    leaf: ComponentAssignment = field(default_factory=ComponentAssignment)
    core: ComponentAssignment = field(default_factory=ComponentAssignment)

    # Per-blueprint overrides
    blueprint_overrides: dict[str, dict[str, ComponentAssignment]] = field(
        default_factory=dict
    )


@dataclass
class ComponentsConfig:
    """Component library and assignment configuration for network hardware.

    Contains hardware component definitions and role-based assignment rules
    for building detailed network equipment specifications.
    """

    library: dict[str, dict[str, Any]] = field(default_factory=dict)
    assignments: ComponentAssignments = field(default_factory=ComponentAssignments)


@dataclass
class FailurePolicyAssignments:
    """Failure policy assignment configuration for scenarios.

    Defines default failure policies and scenario-specific overrides
    for network resilience testing and analysis.
    """

    default: str = "single_random_link_failure"
    scenario_overrides: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class FailurePoliciesConfig:
    """Failure policy library and assignment configuration for network analysis.

    Manages failure policy definitions and their assignment to scenarios
    for comprehensive network resilience testing.
    """

    library: dict[str, dict[str, Any]] = field(default_factory=dict)
    assignments: FailurePolicyAssignments = field(
        default_factory=FailurePolicyAssignments
    )


@dataclass
class WorkflowAssignments:
    """Workflow assignment configuration for analysis execution.

    Defines default analysis workflows and scenario-specific overrides
    for customizing network analysis procedures.
    """

    default: str = "basic_capacity_analysis"
    scenario_overrides: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class WorkflowsConfig:
    """Workflow library and assignment configuration for network analysis.

    Contains workflow step definitions and assignment rules for executing
    comprehensive network performance and resilience analysis.
    """

    library: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    assignments: WorkflowAssignments = field(default_factory=WorkflowAssignments)


@dataclass
class ScenarioMetadata:
    """NetGraph scenario metadata for generated topologies.

    Contains descriptive information about generated network scenarios
    including title, description, and version information.
    """

    title: str = "Continental US Backbone Topology"
    description: str = "Generated backbone topology based on population density and highway infrastructure"
    version: str = "1.0"


@dataclass
class ClusteringConfig:
    """Metro clustering configuration for urban area processing.

    Parameters for selecting and processing metropolitan areas from Census data,
    including clustering parameters and visualization export settings.
    """

    metro_clusters: int = 30  # Target number of metro clusters
    max_uac_radius_km: float = 100.0  # Maximum radius for UAC urban areas
    export_clusters: bool = (
        False  # Export cluster visualization files (JPEG + simplified GeoJSON)
    )
    export_integrated_graph: bool = (
        False  # Export integrated graph visualization (metro clusters + corridors)
    )
    override_metro_clusters: list[str] = field(
        default_factory=list
    )  # Metro names/patterns to force include regardless of size ranking
    coordinate_precision: int = 1  # Decimal places for coordinate rounding
    area_precision: int = 2  # Decimal places for area/radius rounding


@dataclass
class FormattingConfig:
    """Data formatting and precision configuration for output generation.

    Controls output formatting and precision settings for consistent
    data representation across all generated files.
    """

    json_indent: int = 2  # JSON output indentation


@dataclass
class OutputConfig:
    """Output configuration for scenario generation and formatting.

    Combines scenario metadata and formatting settings to control
    how generated topologies are structured and presented.
    """

    scenario_metadata: ScenarioMetadata = field(default_factory=ScenarioMetadata)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)


@dataclass
class TopologyConfig:
    """Complete topology generator configuration for backbone generation.

    Main configuration class that aggregates all subsystem configurations
    for comprehensive topology generation from raw data to NetGraph scenarios.
    """

    # Configuration sections
    data_sources: DataSources = field(
        default_factory=lambda: DataSources(
            uac_polygons=Path("data/tl_2020_us_uac20.zip"),
            tiger_roads=Path("data/tl_2024_us_primaryroads.zip"),
            conus_boundary=Path("data/cb_2024_us_state_500k.zip"),
        )
    )
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    highway_processing: HighwayProcessingConfig = field(
        default_factory=HighwayProcessingConfig
    )
    corridors: CorridorsConfig = field(default_factory=CorridorsConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    components: ComponentsConfig = field(default_factory=ComponentsConfig)
    failure_policies: FailurePoliciesConfig = field(
        default_factory=FailurePoliciesConfig
    )
    workflows: WorkflowsConfig = field(default_factory=WorkflowsConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> TopologyConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Parsed configuration object.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML is invalid.
            ValueError: If configuration is invalid.
        """
        logger.info(f"Loading configuration from: {config_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f)

        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration: {e}")
            raise

        return cls._from_dict(raw_config)

    @classmethod
    def _from_dict(cls, config_dict: dict[str, Any]) -> TopologyConfig:
        """Create configuration from dictionary.

        Args:
            config_dict: Raw configuration dictionary.

        Returns:
            Parsed configuration object.
        """
        # Extract nested configurations with strict validation
        if "data_sources" not in config_dict:
            raise ValueError("Missing required 'data_sources' configuration section")
        if "projection" not in config_dict:
            raise ValueError("Missing required 'projection' configuration section")
        if "clustering" not in config_dict:
            raise ValueError("Missing required 'clustering' configuration section")
        if "highway_processing" not in config_dict:
            raise ValueError(
                "Missing required 'highway_processing' configuration section"
            )
        if "corridors" not in config_dict:
            raise ValueError("Missing required 'corridors' configuration section")
        if "validation" not in config_dict:
            raise ValueError("Missing required 'validation' configuration section")
        if "output" not in config_dict:
            raise ValueError("Missing required 'output' configuration section")

        data_sources_dict = config_dict["data_sources"]
        projection_dict = config_dict["projection"]
        clustering_dict = config_dict["clustering"]

        # Handle override_metro_clusters as optional list
        if "override_metro_clusters" not in clustering_dict:
            clustering_dict["override_metro_clusters"] = []
        highway_dict = config_dict["highway_processing"]
        corridors_dict = config_dict["corridors"]
        validation_dict = config_dict["validation"]

        # Handle risk_groups configuration within corridors
        if "risk_groups" in corridors_dict:
            risk_groups_dict = corridors_dict["risk_groups"]
            risk_groups_config = RiskGroupsConfig(**risk_groups_dict)
            corridors_dict = corridors_dict.copy()
            corridors_dict["risk_groups"] = risk_groups_config
        output_dict = config_dict["output"]

        # Create nested objects with validation
        data_sources = DataSources(**data_sources_dict)
        projection = ProjectionConfig(**projection_dict)
        clustering = ClusteringConfig(**clustering_dict)
        highway_processing = HighwayProcessingConfig(**highway_dict)
        corridors = CorridorsConfig(**corridors_dict)
        validation = ValidationConfig(**validation_dict)

        # Handle output configuration with strict validation
        if "scenario_metadata" not in output_dict:
            raise ValueError(
                "Missing required 'scenario_metadata' in output configuration"
            )
        if "formatting" not in output_dict:
            raise ValueError("Missing required 'formatting' in output configuration")

        scenario_metadata_dict = output_dict["scenario_metadata"]
        formatting_dict = output_dict["formatting"]

        scenario_metadata = ScenarioMetadata(**scenario_metadata_dict)
        formatting = FormattingConfig(**formatting_dict)
        output = OutputConfig(
            scenario_metadata=scenario_metadata,
            formatting=formatting,
        )

        # Handle optional build configuration
        build_dict = config_dict.get("build", {})
        if not isinstance(build_dict, dict):
            raise ValueError("'build' configuration section must be a dictionary")

        build_defaults_dict = build_dict.get("build_defaults", {})
        build_overrides_dict = build_dict.get("build_overrides", {})

        if not isinstance(build_defaults_dict, dict):
            raise ValueError("'build_defaults' must be a dictionary")
        if not isinstance(build_overrides_dict, dict):
            raise ValueError("'build_overrides' must be a dictionary")

        # Parse link parameter configurations
        intra_metro_link_dict = build_defaults_dict.get("intra_metro_link", {})
        inter_metro_link_dict = build_defaults_dict.get("inter_metro_link", {})

        if not isinstance(intra_metro_link_dict, dict):
            raise ValueError("'build_defaults.intra_metro_link' must be a dictionary")
        if not isinstance(inter_metro_link_dict, dict):
            raise ValueError("'build_defaults.inter_metro_link' must be a dictionary")

        # Create link parameter objects with defaults
        intra_metro_link = LinkParams(
            capacity=intra_metro_link_dict.get("capacity", 400),
            cost=intra_metro_link_dict.get("cost", 1),
            attrs={
                **{"link_type": "intra_metro"},
                **intra_metro_link_dict.get("attrs", {}),
            },
        )
        inter_metro_link = LinkParams(
            capacity=inter_metro_link_dict.get("capacity", 100),
            cost=inter_metro_link_dict.get("cost", 1),
            attrs={
                **{"link_type": "inter_metro_corridor"},
                **inter_metro_link_dict.get("attrs", {}),
            },
        )

        # Create BuildDefaults with explicit parameters
        build_defaults = BuildDefaults(
            sites_per_metro=build_defaults_dict.get("sites_per_metro", 2),
            site_blueprint=build_defaults_dict.get("site_blueprint", "SingleRouter"),
            intra_metro_link=intra_metro_link,
            inter_metro_link=inter_metro_link,
        )
        build = BuildConfig(
            build_defaults=build_defaults,
            build_overrides=build_overrides_dict,
        )

        # Handle optional components configuration
        components_dict = config_dict.get("components", {})
        if not isinstance(components_dict, dict):
            raise ValueError("'components' configuration section must be a dictionary")

        # Parse component library
        library_dict = components_dict.get("library", {})
        if library_dict is None:
            library_dict = {}
        if not isinstance(library_dict, dict):
            raise ValueError("'components.library' must be a dictionary")

        # Parse component assignments
        assignments_dict = components_dict.get("assignments", {})
        if not isinstance(assignments_dict, dict):
            raise ValueError("'components.assignments' must be a dictionary")

        # Parse role assignments
        spine_assignment = ComponentAssignment(**assignments_dict.get("spine", {}))
        leaf_assignment = ComponentAssignment(**assignments_dict.get("leaf", {}))
        core_assignment = ComponentAssignment(**assignments_dict.get("core", {}))

        # Parse blueprint overrides
        blueprint_overrides_dict = assignments_dict.get("blueprint_overrides", {})
        if not isinstance(blueprint_overrides_dict, dict):
            raise ValueError("'blueprint_overrides' must be a dictionary")

        blueprint_overrides = {}
        for blueprint_name, roles_dict in blueprint_overrides_dict.items():
            if not isinstance(roles_dict, dict):
                raise ValueError(
                    f"Blueprint override '{blueprint_name}' must be a dictionary"
                )

            blueprint_overrides[blueprint_name] = {}
            for role_name, assignment_dict in roles_dict.items():
                if not isinstance(assignment_dict, dict):
                    raise ValueError(
                        f"Assignment for '{blueprint_name}.{role_name}' must be a dictionary"
                    )
                blueprint_overrides[blueprint_name][role_name] = ComponentAssignment(
                    **assignment_dict
                )

        assignments = ComponentAssignments(
            spine=spine_assignment,
            leaf=leaf_assignment,
            core=core_assignment,
            blueprint_overrides=blueprint_overrides,
        )

        components = ComponentsConfig(
            library=library_dict,
            assignments=assignments,
        )

        # Handle optional failure_policies configuration
        failure_policies_dict = config_dict.get("failure_policies", {})
        if not isinstance(failure_policies_dict, dict):
            raise ValueError(
                "'failure_policies' configuration section must be a dictionary"
            )

        # Parse failure policy library
        fp_library_dict = failure_policies_dict.get("library", {})
        if fp_library_dict is None:
            fp_library_dict = {}
        if not isinstance(fp_library_dict, dict):
            raise ValueError("'failure_policies.library' must be a dictionary")

        # Parse failure policy assignments
        fp_assignments_dict = failure_policies_dict.get("assignments", {})
        if fp_assignments_dict is None:
            fp_assignments_dict = {}
        if not isinstance(fp_assignments_dict, dict):
            raise ValueError("'failure_policies.assignments' must be a dictionary")

        fp_default = fp_assignments_dict.get("default", "single_random_link_failure")
        fp_scenario_overrides = fp_assignments_dict.get("scenario_overrides", {})
        if fp_scenario_overrides is None:
            fp_scenario_overrides = {}

        fp_assignments = FailurePolicyAssignments(
            default=fp_default,
            scenario_overrides=fp_scenario_overrides,
        )

        failure_policies = FailurePoliciesConfig(
            library=fp_library_dict,
            assignments=fp_assignments,
        )

        # Handle optional workflows configuration
        workflows_dict = config_dict.get("workflows", {})
        if not isinstance(workflows_dict, dict):
            raise ValueError("'workflows' configuration section must be a dictionary")

        # Parse workflow library
        wf_library_dict = workflows_dict.get("library", {})
        if wf_library_dict is None:
            wf_library_dict = {}
        if not isinstance(wf_library_dict, dict):
            raise ValueError("'workflows.library' must be a dictionary")

        # Parse workflow assignments
        wf_assignments_dict = workflows_dict.get("assignments", {})
        if wf_assignments_dict is None:
            wf_assignments_dict = {}
        if not isinstance(wf_assignments_dict, dict):
            raise ValueError("'workflows.assignments' must be a dictionary")

        wf_default = wf_assignments_dict.get("default", "basic_capacity_analysis")
        wf_scenario_overrides = wf_assignments_dict.get("scenario_overrides", {})
        if wf_scenario_overrides is None:
            wf_scenario_overrides = {}

        wf_assignments = WorkflowAssignments(
            default=wf_default,
            scenario_overrides=wf_scenario_overrides,
        )

        workflows = WorkflowsConfig(
            library=wf_library_dict,
            assignments=wf_assignments,
        )

        # Create main configuration
        return cls(
            data_sources=data_sources,
            projection=projection,
            clustering=clustering,
            highway_processing=highway_processing,
            corridors=corridors,
            validation=validation,
            output=output,
            build=build,
            components=components,
            failure_policies=failure_policies,
            workflows=workflows,
        )

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid.
        """
        logger.info("Validating configuration")

        # Validate numeric parameters first
        if self.clustering.metro_clusters <= 0:
            raise ValueError("metro_clusters must be positive")

        # Check data sources exist (only if doing full validation)
        if not self.data_sources.uac_polygons.exists():
            raise ValueError(
                f"UAC polygons file not found: {self.data_sources.uac_polygons}"
            )

        if not self.data_sources.tiger_roads.exists():
            raise ValueError(
                f"TIGER roads file not found: {self.data_sources.tiger_roads}"
            )

        if not self.data_sources.conus_boundary.exists():
            raise ValueError(
                f"CONUS boundary file not found: {self.data_sources.conus_boundary}"
            )

        logger.info("Configuration validation passed")

    def summary(self) -> str:
        """Generate configuration summary string.

        Returns:
            Human-readable configuration summary.
        """
        lines = [
            "TOPOLOGY GENERATOR CONFIGURATION",
            "=" * 60,
            "",
            "CLUSTERING PARAMETERS",
            "-" * 30,
            f"   Metro Clusters: ~{self.clustering.metro_clusters}",
            f"   Max UAC Radius: {self.clustering.max_uac_radius_km}km",
            f"   Export Clusters: {self.clustering.export_clusters}",
            "",
            "DATA SOURCES",
            "-" * 30,
            f"   UAC Polygons: {self.data_sources.uac_polygons}",
            f"   TIGER Roads: {self.data_sources.tiger_roads}",
            f"   CONUS Boundary: {self.data_sources.conus_boundary}",
            f"   Target CRS: {self.projection.target_crs}",
            "",
            "HIGHWAY PROCESSING (Lightweight)",
            "-" * 30,
            f"   Min Edge Length: {self.highway_processing.min_edge_length_km}km",
            "",
            "CORRIDOR DISCOVERY",
            "-" * 30,
            f"   K-Paths per Metro Pair: {self.corridors.k_paths}",
            "",
            "VALIDATION",
            "-" * 30,
            f"   Max Metro-Highway Distance: {self.validation.max_metro_highway_distance_km}km",
            f"   Require Connected: {self.validation.require_connected}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)
