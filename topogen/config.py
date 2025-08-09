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
    DC Regions are single-node groups connected to all local PoPs.
    """

    pop_per_metro: int = 2
    site_blueprint: str = "SingleRouter"
    dc_regions_per_metro: int = 2
    dc_region_blueprint: str = "DCRegion"
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
    dc_to_pop_link: LinkParams = field(
        default_factory=lambda: LinkParams(
            capacity=400, cost=1, attrs={"link_type": "dc_to_pop"}
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
    # Capacity allocation policy
    # Hardware-aware capacity allocation is disabled by default for
    # strict backward compatibility.
    capacity_allocation: "BuildCapacityAllocationConfig" = field(  # type: ignore[name-defined]
        default_factory=lambda: BuildCapacityAllocationConfig()
    )


@dataclass
class BuildCapacityAllocationConfig:
    """Hardware-aware capacity allocation settings.

    When enabled, capacity allocation for dc_to_pop, intra_metro, and
    inter_metro links becomes aware of platform capacities. Base capacities
    from configuration are treated as minimums. Remaining capacity is
    allocated to inter-metro links in discrete increments.

    Attributes:
        enabled: Turn on HW-aware allocation. Default False.
    """

    enabled: bool = False


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
    dc: ComponentAssignment = field(default_factory=ComponentAssignment)

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
class TrafficGravityConfig:
    """Gravity model parameters for DC-to-DC traffic generation.

    Attributes:
        alpha: Exponent applied to DC mass (power) terms.
        beta: Exponent applied to distance term in km.
        min_distance_km: Minimum effective distance to avoid singularities.
        exclude_same_metro: If True, skip intra-metro DC pairs. Default includes them.
        distance_metric: One of {"euclidean_km", "corridor_length", "auto"}.
        emission: One of {"explicit_pairs", "macro_pairwise"} for output format.
        max_partners_per_dc: If set, keeps top-K partners per DC by weight.
        jitter_stddev: Lognormal sigma for multiplicative noise (0 disables jitter).
        rounding_gbps: If > 0, round per-pair demands to this step size and conserve totals.
        mw_per_dc_region_overrides: Optional overrides by metro name or full DC path
            (e.g., "metro3/dc2"). Overrides apply after defaults.
    """

    alpha: float = 1.0
    beta: float = 1.0
    min_distance_km: float = 1.0
    exclude_same_metro: bool = False
    distance_metric: str = "euclidean_km"
    emission: str = "explicit_pairs"
    max_partners_per_dc: int | None = None
    jitter_stddev: float = 0.0
    rounding_gbps: float = 0.0
    mw_per_dc_region_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class TrafficConfig:
    """Traffic generation configuration for scenario build.

    Defines parameters for generating DC-to-DC traffic matrices.

    Attributes:
        enabled: Whether to generate and include a traffic matrix.
        gbps_per_mw: Offered traffic per MW of DC power (Gbps/MW).
        mw_per_dc_region: Power per DC region (MW).
        priority_ratios: Mapping from priority class to ratio. Values must sum to 1.0.
        matrix_name: Name of the traffic matrix in the scenario.
        model: "uniform_pairwise" (default) or "gravity".
        gravity: Parameters for gravity model when model == "gravity".
    """

    enabled: bool = True
    gbps_per_mw: float = 1000.0
    mw_per_dc_region: float = 150.0
    priority_ratios: dict[int, float] = field(
        default_factory=lambda: {0: 0.3, 1: 0.3, 2: 0.4}
    )
    matrix_name: str = "default"
    model: str = "uniform_pairwise"
    gravity: TrafficGravityConfig = field(default_factory=TrafficGravityConfig)


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
    yaml_anchors: bool = True  # Emit YAML anchors/aliases when dumping scenario YAML


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
    traffic: TrafficConfig = field(default_factory=TrafficConfig)

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
        capacity_alloc_dict = build_dict.get("capacity_allocation", {})

        if not isinstance(build_defaults_dict, dict):
            raise ValueError("'build_defaults' must be a dictionary")
        if not isinstance(build_overrides_dict, dict):
            raise ValueError("'build_overrides' must be a dictionary")
        if capacity_alloc_dict is None:
            capacity_alloc_dict = {}
        if not isinstance(capacity_alloc_dict, dict):
            raise ValueError("'capacity_allocation' must be a dictionary if provided")

        # Parse link parameter configurations
        intra_metro_link_dict = build_defaults_dict.get("intra_metro_link", {})
        inter_metro_link_dict = build_defaults_dict.get("inter_metro_link", {})
        dc_to_pop_link_dict = build_defaults_dict.get("dc_to_pop_link", {})

        if not isinstance(intra_metro_link_dict, dict):
            raise ValueError("'build_defaults.intra_metro_link' must be a dictionary")
        if not isinstance(inter_metro_link_dict, dict):
            raise ValueError("'build_defaults.inter_metro_link' must be a dictionary")
        if not isinstance(dc_to_pop_link_dict, dict):
            raise ValueError("'build_defaults.dc_to_pop_link' must be a dictionary")

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
        dc_to_pop_link = LinkParams(
            capacity=dc_to_pop_link_dict.get("capacity", 400),
            cost=dc_to_pop_link_dict.get("cost", 1),
            attrs={
                **{"link_type": "dc_to_pop"},
                **dc_to_pop_link_dict.get("attrs", {}),
            },
        )

        # Create BuildDefaults with explicit parameters
        build_defaults = BuildDefaults(
            pop_per_metro=build_defaults_dict.get("pop_per_metro", 2),
            site_blueprint=build_defaults_dict.get("site_blueprint", "SingleRouter"),
            dc_regions_per_metro=build_defaults_dict.get("dc_regions_per_metro", 2),
            dc_region_blueprint=build_defaults_dict.get(
                "dc_region_blueprint", "DCRegion"
            ),
            intra_metro_link=intra_metro_link,
            inter_metro_link=inter_metro_link,
            dc_to_pop_link=dc_to_pop_link,
        )
        # Capacity allocation configuration
        # Strictly allow only the supported key 'enabled'
        _allowed_ca_keys = {"enabled"}
        _extra_keys = set(capacity_alloc_dict.keys()) - _allowed_ca_keys
        if _extra_keys:
            raise ValueError(
                f"Unknown keys in 'build.capacity_allocation': {_extra_keys}. Allowed keys: {_allowed_ca_keys}"
            )
        capacity_allocation = BuildCapacityAllocationConfig(
            enabled=bool(capacity_alloc_dict.get("enabled", False))
        )

        build = BuildConfig(
            build_defaults=build_defaults,
            build_overrides=build_overrides_dict,
            capacity_allocation=capacity_allocation,
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
        dc_assignment = ComponentAssignment(**assignments_dict.get("dc", {}))

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
            dc=dc_assignment,
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

        # Handle optional traffic configuration
        traffic_dict = config_dict.get("traffic", {})
        if traffic_dict is None:
            traffic_dict = {}
        if not isinstance(traffic_dict, dict):
            raise ValueError("'traffic' configuration section must be a dictionary")

        # Normalize ratios to int keys if strings were provided by YAML loader
        ratios_input = traffic_dict.get("priority_ratios")
        if ratios_input is not None:
            if not isinstance(ratios_input, dict):
                raise ValueError("'traffic.priority_ratios' must be a dictionary")
            normalized_ratios: dict[int, float] = {}
            for k, v in ratios_input.items():
                try:
                    key_int = int(k)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "'traffic.priority_ratios' keys must be integers"
                    ) from exc
                normalized_ratios[key_int] = float(v)
            traffic_dict = {**traffic_dict, "priority_ratios": normalized_ratios}

        # Normalize nested gravity config if present
        gravity_dict = traffic_dict.get("gravity", None)
        if gravity_dict is None:
            gravity_cfg = TrafficGravityConfig()
        else:
            if not isinstance(gravity_dict, dict):
                raise ValueError("'traffic.gravity' must be a dictionary")
            gravity_cfg = TrafficGravityConfig(**gravity_dict)

        # Compose TrafficConfig
        traffic = TrafficConfig(
            enabled=bool(traffic_dict.get("enabled", True)),
            gbps_per_mw=float(traffic_dict.get("gbps_per_mw", 1000.0)),
            mw_per_dc_region=float(traffic_dict.get("mw_per_dc_region", 150.0)),
            priority_ratios=traffic_dict.get(
                "priority_ratios", {0: 0.3, 1: 0.3, 2: 0.4}
            ),
            matrix_name=str(traffic_dict.get("matrix_name", "default")),
            model=str(traffic_dict.get("model", "uniform_pairwise")),
            gravity=gravity_cfg,
        )

        # Validate traffic configuration
        if traffic.enabled:
            if traffic.gbps_per_mw < 0:
                raise ValueError("traffic.gbps_per_mw must be non-negative")
            if traffic.mw_per_dc_region < 0:
                raise ValueError("traffic.mw_per_dc_region must be non-negative")
            if not traffic.priority_ratios:
                raise ValueError("traffic.priority_ratios must not be empty")
            # Require contiguous classes from 0..N-1
            classes = sorted(traffic.priority_ratios.keys())
            expected = list(range(len(classes)))
            if classes != expected:
                raise ValueError(
                    "traffic.priority_ratios must have contiguous integer keys from 0..N-1"
                )
            total_ratio = sum(traffic.priority_ratios.values())
            if abs(total_ratio - 1.0) > 1e-9:
                raise ValueError("traffic.priority_ratios values must sum to 1.0")

            # Validate traffic model
            if traffic.model not in {"uniform_pairwise", "gravity"}:
                raise ValueError(
                    "traffic.model must be 'uniform_pairwise' or 'gravity'"
                )

            # Validate gravity sub-config
            g = traffic.gravity
            if g.alpha <= 0.0:
                raise ValueError("traffic.gravity.alpha must be positive")
            if g.beta <= 0.0:
                raise ValueError("traffic.gravity.beta must be positive")
            if g.min_distance_km <= 0.0:
                raise ValueError("traffic.gravity.min_distance_km must be positive")
            if g.distance_metric not in {"euclidean_km", "corridor_length", "auto"}:
                raise ValueError(
                    "traffic.gravity.distance_metric must be one of 'euclidean_km', 'corridor_length', 'auto'"
                )
            if g.emission not in {"explicit_pairs", "macro_pairwise"}:
                raise ValueError(
                    "traffic.gravity.emission must be 'explicit_pairs' or 'macro_pairwise'"
                )
            if g.max_partners_per_dc is not None and g.max_partners_per_dc <= 0:
                raise ValueError(
                    "traffic.gravity.max_partners_per_dc must be positive when set"
                )
            if g.jitter_stddev < 0.0:
                raise ValueError("traffic.gravity.jitter_stddev must be non-negative")
            if g.rounding_gbps < 0.0:
                raise ValueError("traffic.gravity.rounding_gbps must be non-negative")

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
            traffic=traffic,
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
