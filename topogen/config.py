"""Configuration management for topology generator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from topogen.log_config import get_logger
from topogen.naming import metro_slug

logger = get_logger(__name__)


def _validate_flow_policy_names(policies: dict[int, str]) -> None:
    """Validate priority-to-preset mappings."""
    from ngraph.model.flow import FlowPolicyPreset

    if not isinstance(policies, dict):
        raise ValueError(
            "traffic.flow_policy_config must map priorities to preset names"
        )
    for value in policies.values():
        if (
            not isinstance(value, str)
            or value.strip().upper() not in FlowPolicyPreset.__members__
        ):
            raise ValueError(
                f"traffic.flow_policy_config values must be preset names, got {value!r}"
            )


def _normalize_int(value: Any, field: str) -> int:
    """Return an int from possibly underscored or float-like input.

    Accepts ints, floats, or strings like "12_800" or "3200.0". Raises a
    ValueError with field context on failure.
    """
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            s = value.strip().replace("_", "")
            if not s:
                raise ValueError("empty string")
            # Support simple decimal strings by truncation to int
            if "." in s:
                return int(float(s))
            return int(s)
        return int(value)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid integer for {field}: {value!r}") from exc


@dataclass
class DataSources:
    """Paths to Census urban areas, TIGER/Line roads, and state boundaries."""

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
    """Target coordinate reference system for spatial operations."""

    target_crs: str = "EPSG:5070"


@dataclass
class HighwayProcessingConfig:
    """Highway class filtering, coordinate snapping, and graph cleanup settings.

    The pipeline does not use ``min_cycle_nodes`` or ``validation_sample_size``.
    """

    min_edge_length_km: float = 0.05
    snap_precision_m: float = 10.0
    highway_classes: list[str] = field(
        default_factory=lambda: ["S1100", "S1200"]
    )  # TIGER highway classes to keep
    min_cycle_nodes: int = 3
    filter_largest_component: bool = (
        True  # Keep only largest connected component if highway graph is disconnected
    )
    validation_sample_size: int = 5


@dataclass
class RiskGroupsConfig:
    """Corridor risk-group naming and metro-radius exclusion settings."""

    enabled: bool = True
    group_prefix: str = "corridor_risk"
    exclude_metro_radius_shared: bool = (
        True  # Exclude highway segments within metro radius from risk groups
    )


@dataclass
class CorridorsConfig:
    """Metro adjacency, path discovery, and corridor risk-group settings."""

    k_paths: int = 1  # Maximum number of shortest simple paths per adjacent metro pair
    k_nearest: int = 3  # Number of nearest neighbors per metro for adjacency
    # Euclidean threshold (km) for k-NN adjacency between metros.
    # Computed from centroid separation in target CRS; does not limit path length.
    max_edge_km: float = 600.0
    # Path-length threshold (km) along the discovered corridor over the highway graph.
    # Skip a metro pair when all candidate paths exceed this length.
    max_corridor_distance_km: float = 1000.0
    risk_groups: RiskGroupsConfig = field(default_factory=RiskGroupsConfig)


@dataclass
class ValidationConfig:
    """Distance, degree, and connectivity thresholds for graph validation."""

    max_metro_highway_distance_km: float = 10.0
    require_connected: bool = True
    max_degree_threshold: int = 1000
    high_degree_warning: int = 20
    min_largest_component_fraction: float = (
        0.5  # Minimum fraction of nodes in the largest component
    )


@dataclass
class LinkParams:
    """Capacity budget, cost, and attributes for a site-to-site adjacency.

    ``match`` is applied to both endpoints during NetGraph expansion.
    """

    capacity: int
    cost: int
    attrs: dict[str, Any] = field(default_factory=dict)
    match: dict[str, Any] = field(default_factory=dict)
    # Unordered role-pairs allowed for this link type. Examples:
    #   ["core|core", "core|leaf"] or [["core", "core"], ["core", "leaf"]]
    # The pipeline will auto-render a symmetric match from the union of roles.
    role_pairs: list[Any] = field(default_factory=list)
    # Optional striping configuration for controlled device-group partitioning
    # during adjacency creation. Example: {"width": 4}
    striping: dict[str, Any] = field(default_factory=dict)
    # Optional adjacency formation mode (used by inter-metro links).
    # "mesh" connects all PoP pairs between metros; "one_to_one" connects
    # only corresponding indices: pop1-pop1, pop2-pop2, ... up to min counts.
    mode: str = "mesh"


@dataclass
class BuildDefaults:
    """Site counts, blueprints, and link settings used unless a metro overrides them."""

    pop_per_metro: int = 2
    site_blueprint: str = "SingleRouter"
    dc_regions_per_metro: int = 2
    dc_region_blueprint: str = "DCRegion"
    intra_metro_link: LinkParams = field(
        default_factory=lambda: LinkParams(
            capacity=400,
            cost=1,
            attrs={"link_type": "intra_metro"},
            match={},
            striping={},
        )
    )
    inter_metro_link: LinkParams = field(
        default_factory=lambda: LinkParams(
            capacity=100,
            cost=1,
            attrs={"link_type": "inter_metro_corridor"},
            match={},
            striping={},
        )
    )
    dc_to_pop_link: LinkParams = field(
        default_factory=lambda: LinkParams(
            capacity=400,
            cost=1,
            attrs={"link_type": "dc_to_pop"},
            match={},
            striping={},
        )
    )


@dataclass
class BuildConfig:
    """Scenario build defaults, per-metro overrides, and traffic sizing settings."""

    build_defaults: BuildDefaults = field(default_factory=BuildDefaults)
    build_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    tm_sizing: "BuildTmSizingConfig" = field(  # type: ignore[name-defined]
        default_factory=lambda: BuildTmSizingConfig()
    )


@dataclass
class BuildTmSizingConfig:
    """Capacity sizing from traffic routed on a collapsed metro graph.

    Corridors are sized for peak directional load, with headroom and rounding
    up to capacity increments. Local link capacities derive from PoP egress.

    Attributes:
        matrix_name: Demand set to size for; defaults to ``traffic.matrix_name``.
        quantum_gbps: Capacity increment in Gb/s.
        headroom: Multiplier applied to corridor loads before rounding up.
        alpha_dc_to_pop: DC-to-PoP capacity multiplier relative to PoP egress.
        beta_intra_pop: Intra-metro capacity multiplier relative to the smaller
            egress of the two PoPs.
        flow_placement: ``EQUAL_BALANCED`` or ``PROPORTIONAL`` splitting.
        edge_select: Stored setting; sizing uses all minimum-cost edges
            regardless of this value.
        respect_min_base_capacity: Keep capacities at least at their configured values.
    """

    enabled: bool = False
    matrix_name: str | None = None
    quantum_gbps: float = 3200.0
    headroom: float = 1.3
    alpha_dc_to_pop: float = 1.2
    beta_intra_pop: float = 0.8
    flow_placement: str = "EQUAL_BALANCED"
    edge_select: str = "ALL_MIN_COST"
    respect_min_base_capacity: bool = True


@dataclass
class ComponentAssignment:
    """Platform and optic names assigned to a network role."""

    hw_component: str = ""
    optics: str = ""


@dataclass
class ComponentAssignments:
    """Component assignments for network roles."""

    spine: ComponentAssignment = field(default_factory=ComponentAssignment)
    leaf: ComponentAssignment = field(default_factory=ComponentAssignment)
    core: ComponentAssignment = field(default_factory=ComponentAssignment)
    dc: ComponentAssignment = field(default_factory=ComponentAssignment)


@dataclass
class ComponentsConfig:
    """Role-to-platform and role-pair-to-optic assignments.

    Definitions come from the built-ins and ``cwd/lib/components.yml``.
    """

    assignments: ComponentAssignments = field(default_factory=ComponentAssignments)
    # hw_component: role -> platform component name
    # optics: "source_role|target_role" -> optic component name
    hw_component: dict[str, str] = field(default_factory=dict)
    optics: dict[str, str] = field(default_factory=dict)


@dataclass
class FailurePolicyAssignments:
    """Failure-policy selection.

    Scenario assembly uses ``default``; ``scenario_overrides`` is stored but not applied.
    """

    default: str = "single_random_link_failure"
    scenario_overrides: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class FailurePoliciesConfig:
    """Select policies from the built-ins and ``cwd/lib/failure_policies.yml``."""

    assignments: FailurePolicyAssignments = field(
        default_factory=FailurePolicyAssignments
    )


@dataclass
class WorkflowAssignments:
    """Workflow selection.

    Scenario assembly uses ``default``; ``scenario_overrides`` is stored but not applied.
    """

    default: str = "design_analysis_brief"
    scenario_overrides: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class WorkflowsConfig:
    """Select workflows from the built-ins and ``cwd/lib/workflows.yml``."""

    assignments: WorkflowAssignments = field(default_factory=WorkflowAssignments)


@dataclass
class TrafficGravityConfig:
    """Gravity model parameters for DC-to-DC traffic generation.

    Attributes:
        alpha: Exponent applied to DC mass (power) terms.
        beta: Exponent applied to distance term in km.
        min_distance_km: Minimum effective distance to avoid singularities.
        exclude_same_metro: If True, skip intra-metro DC pairs. Default includes them.
        max_partners_per_dc: If set, keeps top-K partners per DC by weight.
        jitter_stddev: Lognormal sigma for multiplicative noise (0 disables jitter).
        rounding_gbps: If > 0, quantize undirected per-pair totals to this step size.
        rounding_policy: Quantization policy for undirected totals. One of
            {"nearest", "ceil", "floor"}. Positive residual is distributed by
            largest remainder; the final total may differ from offered traffic.
        mw_per_dc_region_overrides: Optional overrides by metro name or full DC path
            (e.g., "salt-lake-city/dc2"). Overrides apply after defaults.
    """

    alpha: float = 1.0
    beta: float = 1.0
    min_distance_km: float = 1.0
    exclude_same_metro: bool = False
    max_partners_per_dc: int | None = None
    jitter_stddev: float = 0.0
    rounding_gbps: float = 0.0
    rounding_policy: str = "nearest"
    mw_per_dc_region_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class TrafficHoseConfig:
    """Hose model parameters for DC-to-DC traffic generation.

    Attributes:
        tilt_exponent: Non-negative exponent controlling gravity tilt strength.
            0.0 disables distance weighting; higher values bias initialization toward
            shorter-distance pairs via a distance kernel before IPF.
        beta: Distance exponent in the tilt kernel (km-based).
        min_distance_km: Minimum effective distance to avoid singularities.
        exclude_same_metro: If True, skip intra-metro DC pairs in the tilt kernel.
    """

    tilt_exponent: float = 0.0
    beta: float = 1.0
    min_distance_km: float = 1.0
    exclude_same_metro: bool = False
    # Retain the union of each DC's top-K partners by fitted traffic volume.
    carve_top_k: int | None = None


@dataclass
class TrafficConfig:
    """Traffic generation configuration for scenario build.

    Attributes:
        enabled: Whether to generate and include a traffic matrix.
        gbps_per_mw: Offered traffic per MW of DC power (Gbps/MW).
        mw_per_dc_region: Power per DC region (MW).
        priority_ratios: Mapping from priority class to ratio. Values must sum to 1.0.
        flow_policy_config: Optional mapping from priority class to NetGraph flow
            policy preset name, emitted as ``flow_policy`` in the scenario.
            Keys are integers matching priority classes.
        matrix_name: Name of the demand set in the emitted scenario.
        model: "uniform" (default), "gravity", or "hose".
        gravity: Parameters for gravity model when model == "gravity".
    """

    enabled: bool = True
    gbps_per_mw: float = 1000.0
    mw_per_dc_region: float = 150.0
    priority_ratios: dict[int, float] = field(
        default_factory=lambda: {0: 0.3, 1: 0.3, 2: 0.4}
    )
    flow_policy_config: dict[int, str] = field(default_factory=dict)
    matrix_name: str = "default"
    model: str = "uniform"
    # Number of samples for models that support stochastic emission (e.g., hose)
    samples: int = 1
    gravity: TrafficGravityConfig = field(default_factory=TrafficGravityConfig)
    hose: TrafficHoseConfig = field(default_factory=TrafficHoseConfig)

    def __post_init__(self) -> None:
        _validate_flow_policy_names(self.flow_policy_config)


@dataclass
class ScenarioMetadata:
    """Title, description, and version for generated scenarios."""

    title: str = "Continental US Backbone Topology"
    description: str = (
        "Generated backbone topology from Census urban areas and highway corridors"
    )
    version: str = "1.0"


@dataclass
class ClusteringConfig:
    """Urban-area selection, radius limits, precision, and map export settings."""

    metro_clusters: int = 30  # Target number of metro clusters
    max_uac_radius_km: float = 100.0  # Maximum radius for UAC urban areas
    export_clusters: bool = False  # Export metro points as GeoJSON and a JPEG map
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
    """JSON indentation and YAML anchor settings."""

    json_indent: int = 2  # JSON output indentation
    yaml_anchors: bool = True  # Emit YAML anchors/aliases when dumping scenario YAML


@dataclass
class OutputConfig:
    """Scenario metadata, formatting, and random seed."""

    scenario_metadata: ScenarioMetadata = field(default_factory=ScenarioMetadata)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    # Top-level scenario random seed to emit in generated scenario YAML
    scenario_seed: int = 42


@dataclass
class TopologyConfig:
    """Settings for geographic graph generation and NetGraph scenario assembly."""

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
    _use_real_corridor_geometry: bool = False
    _export_site_graph: bool = False
    _visualization_dpi: int = 300
    _source_path: Path | None = None
    # Export per-blueprint abstract+concrete diagrams
    _export_blueprint_diagrams: bool = False
    # Optional instrumentation fields for debugging/export
    _debug_dir: Path | None = None
    _source_stem: str | None = None

    @classmethod
    def from_yaml(cls, config_path: Path) -> TopologyConfig:
        """Load YAML, validate it against the packaged schema, and parse settings.

        Raises FileNotFoundError for a missing file, yaml.YAMLError for invalid
        YAML, and ValueError for invalid settings.
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

        # Unconditional JSON Schema validation for TopoGen config
        try:
            import jsonschema  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "jsonschema is required for TopoGen configuration validation. Install dev extras or add 'jsonschema' to dependencies."
            ) from exc

        # Load packaged schema
        try:
            with (
                resources.files("topogen.schemas")
                .joinpath("topogen_config.json")
                .open("r", encoding="utf-8")
            ) as f:
                schema_data = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                "Failed to locate packaged TopoGen config schema 'topogen/schemas/topogen_config.json'."
            ) from exc

        try:
            jsonschema.validate(raw_config, schema_data)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(
                f"TopoGen config JSON Schema validation failed: {exc}"
            ) from exc

        cfg = cls._from_dict(raw_config)
        # Attach the config file source path for downstream naming of artefacts
        cfg._source_path = Path(config_path)
        return cfg

    @classmethod
    def _from_dict(cls, config_dict: dict[str, Any]) -> TopologyConfig:
        """Parse configuration sections and validate their types and values."""
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

        if "override_metro_clusters" not in clustering_dict:
            clustering_dict["override_metro_clusters"] = []
        highway_dict = config_dict["highway_processing"]
        corridors_dict = config_dict["corridors"]
        validation_dict = config_dict["validation"]

        if "risk_groups" in corridors_dict:
            risk_groups_dict = corridors_dict["risk_groups"]
            risk_groups_config = RiskGroupsConfig(**risk_groups_dict)
            corridors_dict = corridors_dict.copy()
            corridors_dict["risk_groups"] = risk_groups_config
        output_dict = config_dict["output"]

        data_sources = DataSources(**data_sources_dict)
        projection = ProjectionConfig(**projection_dict)
        clustering = ClusteringConfig(**clustering_dict)
        highway_processing = HighwayProcessingConfig(**highway_dict)
        corridors = CorridorsConfig(**corridors_dict)
        validation = ValidationConfig(**validation_dict)

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
            scenario_seed=int(output_dict.get("scenario_seed", 42)),
        )

        build_dict = config_dict.get("build", {})
        if not isinstance(build_dict, dict):
            raise ValueError("'build' configuration section must be a dictionary")

        build_defaults_dict = build_dict.get("build_defaults", {})
        build_overrides_list = build_dict.get("build_overrides", [])
        tm_sizing_dict = build_dict.get("tm_sizing", {})

        if not isinstance(build_defaults_dict, dict):
            raise ValueError("'build_defaults' must be a dictionary")
        if not isinstance(build_overrides_list, list):
            raise ValueError("'build_overrides' must be a list of override entries")
        if tm_sizing_dict is None:
            tm_sizing_dict = {}
        if not isinstance(tm_sizing_dict, dict):
            raise ValueError("'tm_sizing' must be a dictionary if provided")

        intra_metro_link_dict = build_defaults_dict.get("intra_metro_link", {})
        inter_metro_link_dict = build_defaults_dict.get("inter_metro_link", {})
        dc_to_pop_link_dict = build_defaults_dict.get("dc_to_pop_link", {})

        if not isinstance(intra_metro_link_dict, dict):
            raise ValueError("'build_defaults.intra_metro_link' must be a dictionary")
        if not isinstance(inter_metro_link_dict, dict):
            raise ValueError("'build_defaults.inter_metro_link' must be a dictionary")
        if not isinstance(dc_to_pop_link_dict, dict):
            raise ValueError("'build_defaults.dc_to_pop_link' must be a dictionary")

        intra_metro_link = LinkParams(
            capacity=_normalize_int(
                intra_metro_link_dict.get("capacity", 400),
                "build_defaults.intra_metro_link.capacity",
            ),
            cost=_normalize_int(
                intra_metro_link_dict.get("cost", 1),
                "build_defaults.intra_metro_link.cost",
            ),
            attrs={
                **{"link_type": "intra_metro"},
                **intra_metro_link_dict.get("attrs", {}),
            },
            match=intra_metro_link_dict.get("match", {}),
            role_pairs=intra_metro_link_dict.get("role_pairs", []) or [],
            striping=intra_metro_link_dict.get("striping", {}) or {},
            mode=str(intra_metro_link_dict.get("mode", "mesh")),
        )
        inter_metro_link = LinkParams(
            capacity=_normalize_int(
                inter_metro_link_dict.get("capacity", 100),
                "build_defaults.inter_metro_link.capacity",
            ),
            cost=_normalize_int(
                inter_metro_link_dict.get("cost", 1),
                "build_defaults.inter_metro_link.cost",
            ),
            attrs={
                **{"link_type": "inter_metro_corridor"},
                **inter_metro_link_dict.get("attrs", {}),
            },
            match=inter_metro_link_dict.get("match", {}),
            role_pairs=inter_metro_link_dict.get("role_pairs", []) or [],
            striping=inter_metro_link_dict.get("striping", {}) or {},
            mode=str(inter_metro_link_dict.get("mode", "mesh")),
        )
        dc_to_pop_link = LinkParams(
            capacity=_normalize_int(
                dc_to_pop_link_dict.get("capacity", 400),
                "build_defaults.dc_to_pop_link.capacity",
            ),
            cost=_normalize_int(
                dc_to_pop_link_dict.get("cost", 1), "build_defaults.dc_to_pop_link.cost"
            ),
            attrs={
                **{"link_type": "dc_to_pop"},
                **dc_to_pop_link_dict.get("attrs", {}),
            },
            match=dc_to_pop_link_dict.get("match", {}),
            role_pairs=dc_to_pop_link_dict.get("role_pairs", []) or [],
            striping=dc_to_pop_link_dict.get("striping", {}) or {},
            mode=str(dc_to_pop_link_dict.get("mode", "mesh")),
        )

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
        _allowed_tm_keys = {
            "enabled",
            "matrix_name",
            "quantum_gbps",
            "headroom",
            "alpha_dc_to_pop",
            "beta_intra_pop",
            "flow_placement",
            "edge_select",
            "respect_min_base_capacity",
        }
        _tm_extra = set(tm_sizing_dict.keys()) - _allowed_tm_keys
        if _tm_extra:
            raise ValueError(
                f"Unknown keys in 'build.tm_sizing': {sorted(_tm_extra)}. Allowed keys: {sorted(_allowed_tm_keys)}"
            )
        tm_sizing = BuildTmSizingConfig(
            enabled=bool(tm_sizing_dict.get("enabled", False)),
            matrix_name=(
                str(tm_sizing_dict.get("matrix_name"))
                if tm_sizing_dict.get("matrix_name") is not None
                else None
            ),
            quantum_gbps=float(tm_sizing_dict.get("quantum_gbps", 3200.0)),
            headroom=float(tm_sizing_dict.get("headroom", 1.3)),
            alpha_dc_to_pop=float(tm_sizing_dict.get("alpha_dc_to_pop", 1.2)),
            beta_intra_pop=float(tm_sizing_dict.get("beta_intra_pop", 0.8)),
            flow_placement=str(tm_sizing_dict.get("flow_placement", "EQUAL_BALANCED")),
            edge_select=str(tm_sizing_dict.get("edge_select", "ALL_MIN_COST")),
            respect_min_base_capacity=bool(
                tm_sizing_dict.get("respect_min_base_capacity", True)
            ),
        )

        # Normalize list of overrides into slug->override mapping.
        # Schema per entry: { metros: [str, ...], <override_fields> }
        allowed_override_keys = {
            "pop_per_metro",
            "site_blueprint",
            "dc_regions_per_metro",
            "dc_region_blueprint",
            "intra_metro_link",
            "inter_metro_link",
            "dc_to_pop_link",
        }

        normalized_overrides: dict[str, dict[str, Any]] = {}
        for idx, entry in enumerate(build_overrides_list):
            if not isinstance(entry, dict):
                raise ValueError("Each item in 'build_overrides' must be a dictionary")
            if "metros" not in entry:
                raise ValueError("Each build override must include 'metros' list")
            metros_field = entry.get("metros")
            if isinstance(metros_field, str):
                metro_names: list[str] = [metros_field]
            elif isinstance(metros_field, list):
                if not all(isinstance(m, str) for m in metros_field):
                    raise ValueError("'metros' must be a list of strings")
                metro_names = list(metros_field)
            else:
                raise ValueError("'metros' must be a string or a list of strings")

            override_body = {k: v for k, v in entry.items() if k != "metros"}

            extra_keys = set(override_body.keys()) - allowed_override_keys
            if extra_keys:
                raise ValueError(
                    f"Unknown keys in build override entry {idx}: {sorted(extra_keys)}. "
                    f"Allowed keys: {sorted(allowed_override_keys)}"
                )

            for raw_name in metro_names:
                slug = metro_slug(raw_name)
                # Last one wins for duplicates
                normalized_overrides[slug] = override_body

        build = BuildConfig(
            build_defaults=build_defaults,
            build_overrides=normalized_overrides,
            tm_sizing=tm_sizing,
        )

        components_dict = config_dict.get("components", {})
        if not isinstance(components_dict, dict):
            raise ValueError("'components' configuration section must be a dictionary")

        assignments_dict = components_dict.get("assignments", {})
        if not isinstance(assignments_dict, dict):
            raise ValueError("'components.assignments' must be a dictionary")

        spine_assignment = ComponentAssignment(**assignments_dict.get("spine", {}))
        leaf_assignment = ComponentAssignment(**assignments_dict.get("leaf", {}))
        core_assignment = ComponentAssignment(**assignments_dict.get("core", {}))
        dc_assignment = ComponentAssignment(**assignments_dict.get("dc", {}))

        assignments = ComponentAssignments(
            spine=spine_assignment,
            leaf=leaf_assignment,
            core=core_assignment,
            dc=dc_assignment,
        )

        hw_component_map = components_dict.get("hw_component", {}) or {}
        optics_map = components_dict.get("optics", {}) or {}
        if hw_component_map is not None and not isinstance(hw_component_map, dict):
            raise ValueError(
                "'components.hw_component' must be a mapping when provided"
            )
        if optics_map is not None and not isinstance(optics_map, dict):
            raise ValueError("'components.optics' must be a mapping when provided")

        components = ComponentsConfig(
            assignments=assignments,
            hw_component=hw_component_map if isinstance(hw_component_map, dict) else {},
            optics=optics_map if isinstance(optics_map, dict) else {},
        )

        failure_policies_dict = config_dict.get("failure_policies", {})
        if not isinstance(failure_policies_dict, dict):
            raise ValueError(
                "'failure_policies' configuration section must be a dictionary"
            )

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

        failure_policies = FailurePoliciesConfig(assignments=fp_assignments)

        workflows_dict = config_dict.get("workflows", {})
        if not isinstance(workflows_dict, dict):
            raise ValueError("'workflows' configuration section must be a dictionary")

        wf_assignments_dict = workflows_dict.get("assignments", {})
        if wf_assignments_dict is None:
            wf_assignments_dict = {}
        if not isinstance(wf_assignments_dict, dict):
            raise ValueError("'workflows.assignments' must be a dictionary")

        wf_default = wf_assignments_dict.get("default", WorkflowAssignments().default)
        wf_scenario_overrides = wf_assignments_dict.get("scenario_overrides", {})
        if wf_scenario_overrides is None:
            wf_scenario_overrides = {}

        wf_assignments = WorkflowAssignments(
            default=wf_default,
            scenario_overrides=wf_scenario_overrides,
        )

        workflows = WorkflowsConfig(assignments=wf_assignments)

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

        gravity_dict = traffic_dict.get("gravity", None)
        if gravity_dict is None:
            gravity_cfg = TrafficGravityConfig()
        else:
            if not isinstance(gravity_dict, dict):
                raise ValueError("'traffic.gravity' must be a dictionary")
            gravity_cfg = TrafficGravityConfig(**gravity_dict)

        hose_dict = traffic_dict.get("hose", None)
        if hose_dict is None:
            hose_cfg = TrafficHoseConfig()
        else:
            if not isinstance(hose_dict, dict):
                raise ValueError("'traffic.hose' must be a dictionary")
            hose_cfg = TrafficHoseConfig(**hose_dict)

        # Normalize optional flow_policy_config mapping to int keys
        fpc_input = traffic_dict.get("flow_policy_config")
        if fpc_input is None:
            flow_policy_cfg_map: dict[int, str] = {}
        else:
            if not isinstance(fpc_input, dict):
                raise ValueError(
                    "'traffic.flow_policy_config' must be a dictionary mapping "
                    "priority class to policy name"
                )
            flow_policy_cfg_map = {}
            for k, v in fpc_input.items():
                try:
                    key_int = int(k)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "'traffic.flow_policy_config' keys must be integers"
                    ) from exc
                flow_policy_cfg_map[key_int] = v

        traffic = TrafficConfig(
            enabled=bool(traffic_dict.get("enabled", True)),
            gbps_per_mw=float(traffic_dict.get("gbps_per_mw", 1000.0)),
            mw_per_dc_region=float(traffic_dict.get("mw_per_dc_region", 150.0)),
            priority_ratios=traffic_dict.get(
                "priority_ratios", {0: 0.3, 1: 0.3, 2: 0.4}
            ),
            flow_policy_config=flow_policy_cfg_map,
            matrix_name=str(traffic_dict.get("matrix_name", "default")),
            model=str(traffic_dict.get("model", "uniform")).strip(),
            samples=int(traffic_dict.get("samples", 1)),
            gravity=gravity_cfg,
            hose=hose_cfg,
        )

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

            if traffic.flow_policy_config:
                # Keys must be subset of defined priority classes
                extra_keys = [
                    k for k in traffic.flow_policy_config.keys() if k not in classes
                ]
                if extra_keys:
                    raise ValueError(
                        "traffic.flow_policy_config contains unknown priority classes; "
                        f"allowed: {classes}, got: {sorted(extra_keys)}"
                    )

            if traffic.model not in {"uniform", "gravity", "hose"}:
                raise ValueError(
                    "traffic.model must be 'uniform', 'gravity', or 'hose'"
                )
            if traffic.model == "hose":
                if not isinstance(traffic.samples, int) or traffic.samples <= 0:
                    raise ValueError(
                        "traffic.samples must be a positive integer for hose model"
                    )
                h = getattr(traffic, "hose", TrafficHoseConfig())
                if h.tilt_exponent < 0.0:
                    raise ValueError("traffic.hose.tilt_exponent must be non-negative")
                if h.beta <= 0.0:
                    raise ValueError("traffic.hose.beta must be positive")
                if h.min_distance_km <= 0.0:
                    raise ValueError("traffic.hose.min_distance_km must be positive")
                if h.carve_top_k is not None and int(h.carve_top_k) <= 0:
                    raise ValueError(
                        "traffic.hose.carve_top_k must be positive when set"
                    )

            g = traffic.gravity
            if g.alpha <= 0.0:
                raise ValueError("traffic.gravity.alpha must be positive")
            if g.beta <= 0.0:
                raise ValueError("traffic.gravity.beta must be positive")
            if g.min_distance_km <= 0.0:
                raise ValueError("traffic.gravity.min_distance_km must be positive")
            if g.max_partners_per_dc is not None and g.max_partners_per_dc <= 0:
                raise ValueError(
                    "traffic.gravity.max_partners_per_dc must be positive when set"
                )
            if g.jitter_stddev < 0.0:
                raise ValueError("traffic.gravity.jitter_stddev must be non-negative")
            if g.rounding_gbps < 0.0:
                raise ValueError("traffic.gravity.rounding_gbps must be non-negative")

        vis = config_dict.get("visualization", {}) or {}
        if not isinstance(vis, dict):
            raise ValueError(
                "'visualization' configuration section must be a dictionary if provided"
            )
        vis_corridors = vis.get("corridors", {}) or {}
        if not isinstance(vis_corridors, dict):
            raise ValueError(
                "'visualization.corridors' must be a dictionary if provided"
            )
        use_real_geometry = bool(vis_corridors.get("use_real_geometry", False))
        vis_site = vis.get("site_graph", {}) or {}
        if not isinstance(vis_site, dict):
            raise ValueError(
                "'visualization.site_graph' must be a dictionary if provided"
            )
        export_site_graph = bool(vis_site.get("export", False))
        vis_blueprints = vis.get("blueprints", {}) or {}
        if not isinstance(vis_blueprints, dict):
            raise ValueError(
                "'visualization.blueprints' must be a dictionary if provided"
            )
        export_blueprints = bool(vis_blueprints.get("export", False))
        dpi_val = vis.get("dpi", 300)
        try:
            visualization_dpi = int(dpi_val)
            if visualization_dpi <= 0:
                raise ValueError
        except Exception as exc:
            raise ValueError("'visualization.dpi' must be a positive integer") from exc

        cfg = cls(
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
        cfg._use_real_corridor_geometry = use_real_geometry
        cfg._export_site_graph = export_site_graph
        cfg._visualization_dpi = visualization_dpi
        cfg._export_blueprint_diagrams = export_blueprints
        return cfg

    def validate(self) -> None:
        """Check the metro count and required data files, raising ValueError on failure."""
        logger.info("Validating configuration")

        if self.clustering.metro_clusters <= 0:
            raise ValueError("metro_clusters must be positive")

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
        """Return a human-readable configuration summary."""
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
