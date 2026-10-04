"""Configuration management for topology generator."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pyproj import CRS

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


def _check_finite_numbers(value: Any, path: str = "config") -> None:
    """YAML accepts NaN/infinity, while JSON Schema bounds do not reject NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite_numbers(child, f"{path}[{index}]")


def _normalize_int(value: Any, field: str) -> int:
    """Parse exact, finite integer quantities; never truncate or accept booleans."""
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not an integer quantity")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            exact = Decimal(value.replace("_", ""))
            if not exact.is_finite() or exact != exact.to_integral_value():
                raise ValueError("expected a finite integer")
            return int(exact)
        parsed = float(value)
        if not math.isfinite(parsed) or not parsed.is_integer():
            raise ValueError("expected a finite integer")
        return int(parsed)
    except (TypeError, ValueError, OverflowError, InvalidOperation) as exc:
        raise ValueError(f"Invalid integer for {field}: {value!r}") from exc


@dataclass(slots=True)
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


@lru_cache(maxsize=16)
def _validate_projection(target_crs: str) -> None:
    crs = CRS.from_user_input(target_crs)
    if (
        not crs.is_projected
        or len(crs.axis_info) != 2
        or any(axis.unit_conversion_factor != 1.0 for axis in crs.axis_info)
    ):
        raise ValueError(
            "projection.target_crs must be a projected CRS with metre units"
        )


@dataclass(slots=True)
class ProjectionConfig:
    """Target coordinate reference system for spatial operations."""

    target_crs: str = "EPSG:5070"

    def __post_init__(self) -> None:
        _validate_projection(self.target_crs)


@dataclass(slots=True)
class HighwayProcessingConfig:
    """Highway filtering, coordinate snapping, and graph cleanup settings."""

    min_edge_length_km: float = 0.05
    snap_precision_m: float = 10.0
    highway_classes: list[str] = field(
        default_factory=lambda: ["S1100", "S1200"]
    )  # TIGER highway classes to keep
    filter_largest_component: bool = (
        True  # Keep only largest connected component if highway graph is disconnected
    )


@dataclass(slots=True)
class RiskGroupsConfig:
    """Corridor risk-group naming and metro-radius exclusion settings."""

    enabled: bool = True
    group_prefix: str = "corridor_risk"
    exclude_metro_radius_shared: bool = (
        True  # Exclude highway segments within metro radius from risk groups
    )


@dataclass(slots=True)
class CorridorsConfig:
    """Metro adjacency, path discovery, and corridor risk-group settings."""

    k_paths: int = 1  # Maximum number of shortest simple paths per adjacent metro pair
    k_nearest: int = 3  # Number of nearest neighbors per metro for adjacency
    # Euclidean threshold (km) for k-NN adjacency between metros.
    # Computed between representative points in target CRS; does not limit path length.
    max_edge_km: float = 600.0
    # Path-length threshold (km) along the discovered corridor over the highway graph.
    # Skip a metro pair when all candidate paths exceed this length.
    max_corridor_distance_km: float = 1000.0
    risk_groups: RiskGroupsConfig = field(default_factory=RiskGroupsConfig)


@dataclass(slots=True)
class ValidationConfig:
    """Distance, degree, and connectivity thresholds for graph validation."""

    max_metro_highway_distance_km: float = 10.0
    require_connected: bool = True
    max_degree_threshold: int = 1000
    high_degree_warning: int = 20
    min_largest_component_fraction: float = (
        0.5  # Minimum fraction of nodes in the largest component
    )


@dataclass(slots=True)
class LinkParams:
    """Capacity budget, minimum routing cost, and attributes for an adjacency.

    ``match`` is applied to both endpoints during NetGraph expansion.
    """

    capacity: int
    cost: int
    attrs: dict[str, Any] = field(default_factory=dict)
    match: dict[str, Any] = field(default_factory=dict)
    # Unordered role-pairs allowed for this link type. Examples:
    #   ["core|core", "core|leaf"]
    # The pipeline will auto-render a symmetric match from the union of roles.
    role_pairs: list[str] = field(default_factory=list)
    # Optional striping configuration for controlled device-group partitioning
    # during adjacency creation. Example: {"width": 4}
    striping: dict[str, Any] = field(default_factory=dict)
    # Optional adjacency formation mode (used by inter-metro links).
    # "mesh" connects all PoP pairs between metros; "one_to_one" connects
    # only corresponding indices: pop1-pop1, pop2-pop2, ... up to min counts.
    mode: str = "mesh"


@dataclass(slots=True)
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


@dataclass(slots=True)
class BuildConfig:
    """Scenario build defaults, per-metro overrides, and traffic sizing settings."""

    build_defaults: BuildDefaults = field(default_factory=BuildDefaults)
    build_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    tm_sizing: "BuildTmSizingConfig" = field(
        default_factory=lambda: BuildTmSizingConfig()
    )


@dataclass(slots=True)
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
        respect_min_base_capacity: Keep capacities at least at their configured values.
    """

    enabled: bool = False
    matrix_name: str | None = None
    quantum_gbps: float = 3200.0
    headroom: float = 1.3
    alpha_dc_to_pop: float = 1.2
    beta_intra_pop: float = 0.8
    flow_placement: str = "EQUAL_BALANCED"
    respect_min_base_capacity: bool = True


@dataclass(slots=True)
class ComponentsConfig:
    """Role-to-platform and role-pair-to-optic assignments.

    Definitions come from the built-ins and ``cwd/lib/components.yml``.
    """

    # hw_component: role -> platform component name
    # optics: "local_role->remote_role" -> local optic component name
    hw_component: dict[str, str] = field(default_factory=dict)
    optics: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class FailurePolicyAssignments:
    """Failure-policy selection."""

    default: str = "single_random_link_failure"


@dataclass(slots=True)
class FailurePoliciesConfig:
    """Select policies from the built-ins and ``cwd/lib/failure_policies.yml``."""

    assignments: FailurePolicyAssignments = field(
        default_factory=FailurePolicyAssignments
    )


@dataclass(slots=True)
class WorkflowAssignments:
    """Workflow selection."""

    default: str = "design_analysis_brief"


@dataclass(slots=True)
class WorkflowsConfig:
    """Select workflows from the built-ins and ``cwd/lib/workflows.yml``."""

    assignments: WorkflowAssignments = field(default_factory=WorkflowAssignments)


@dataclass(slots=True)
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


@dataclass(slots=True)
class TrafficHoseConfig:
    """Hose model parameters for DC-to-DC traffic generation.

    Attributes:
        tilt_exponent: Non-negative exponent controlling gravity tilt strength.
            0.0 disables distance weighting; higher values bias initialization toward
            shorter-distance pairs via a distance kernel before IPF.
        beta: Distance exponent in the tilt kernel (km-based).
        min_distance_km: Minimum effective distance to avoid singularities.
        exclude_same_metro: With nonzero tilt, downweight same-metro pairs while
            retaining positive support for marginal fitting; does not ban traffic.
    """

    tilt_exponent: float = 0.0
    beta: float = 1.0
    min_distance_km: float = 1.0
    exclude_same_metro: bool = False
    # Retain the union of each DC's top-K partners by fitted traffic volume.
    carve_top_k: int | None = None


@dataclass(slots=True)
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
    # Number of hose samples; uniform and gravity emit one matrix.
    samples: int = 1
    gravity: TrafficGravityConfig = field(default_factory=TrafficGravityConfig)
    hose: TrafficHoseConfig = field(default_factory=TrafficHoseConfig)

    def __post_init__(self) -> None:
        _validate_flow_policy_names(self.flow_policy_config)


@dataclass(slots=True)
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


@dataclass(slots=True)
class FormattingConfig:
    """JSON indentation and YAML anchor settings."""

    json_indent: int = 2  # JSON output indentation
    yaml_anchors: bool = True  # Emit YAML anchors/aliases when dumping scenario YAML


@dataclass(slots=True)
class OutputConfig:
    """Formatting and scenario random seed."""

    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    # Top-level scenario random seed to emit in generated scenario YAML
    scenario_seed: int = 42


@dataclass(slots=True)
class VisualizationConfig:
    """Rendering settings, independent of artifact destinations."""

    use_real_corridor_geometry: bool = False
    export_site_graph: bool = False
    export_blueprint_diagrams: bool = False
    dpi: int = 300


@dataclass(slots=True)
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
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

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

        return cls._from_dict(raw_config)

    @classmethod
    def _from_dict(cls, config_dict: dict[str, Any]) -> TopologyConfig:
        """Validate and parse settings without mutating the input mapping."""
        _check_finite_numbers(config_dict)
        try:
            _config_validator().validate(config_dict)
        except jsonschema.ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path) or "config"
            raise ValueError(f"TopoGen config {path}: {exc.message}") from exc

        corridors_dict = dict(config_dict["corridors"])
        corridors_dict["risk_groups"] = RiskGroupsConfig(
            **corridors_dict.get("risk_groups", {})
        )
        output_dict = dict(config_dict["output"])
        output_dict["formatting"] = FormattingConfig(**output_dict["formatting"])

        build_dict = config_dict.get("build", {})
        defaults = BuildDefaults()
        defaults_dict = dict(build_dict.get("build_defaults", {}))
        for name in LINK_TYPES:
            defaults_dict[name] = parse_link_params(
                defaults_dict.get(name, {}), getattr(defaults, name), name
            )
        normalized_overrides: dict[str, dict[str, Any]] = {}
        for entry in build_dict.get("build_overrides", []):
            names = entry["metros"]
            if isinstance(names, str):
                names = [names]
            body = {key: value for key, value in entry.items() if key != "metros"}
            for name in names:
                slug = metro_slug(name)
                if slug in normalized_overrides:
                    raise ValueError(f"Duplicate metro override: {slug}")
                normalized_overrides[slug] = body
        build = BuildConfig(
            build_defaults=BuildDefaults(**defaults_dict),
            build_overrides=normalized_overrides,
            tm_sizing=BuildTmSizingConfig(**build_dict.get("tm_sizing", {})),
        )
        components = ComponentsConfig(**config_dict.get("components", {}))
        failure_policies = FailurePoliciesConfig(
            assignments=FailurePolicyAssignments(
                **config_dict.get("failure_policies", {}).get("assignments", {})
            )
        )
        workflows = WorkflowsConfig(
            assignments=WorkflowAssignments(
                **config_dict.get("workflows", {}).get("assignments", {})
            )
        )

        traffic_dict = dict(config_dict.get("traffic", {}))
        for field_name in ("priority_ratios", "flow_policy_config"):
            if field_name not in traffic_dict:
                continue
            normalized = {}
            for key, value in traffic_dict[field_name].items():
                priority = _normalize_int(key, f"traffic.{field_name} key")
                if priority in normalized:
                    raise ValueError(
                        f"Duplicate traffic.{field_name} priority: {priority}"
                    )
                normalized[priority] = value
            traffic_dict[field_name] = normalized
        traffic_dict["gravity"] = TrafficGravityConfig(
            **traffic_dict.get("gravity", {})
        )
        traffic_dict["hose"] = TrafficHoseConfig(**traffic_dict.get("hose", {}))
        traffic = TrafficConfig(**traffic_dict)
        classes = sorted(traffic.priority_ratios)
        if classes != list(range(len(classes))) or not classes:
            raise ValueError(
                "traffic.priority_ratios must have contiguous integer keys from 0..N-1"
            )
        if not math.isclose(
            sum(traffic.priority_ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("traffic.priority_ratios values must sum to 1.0")
        if extra := traffic.flow_policy_config.keys() - traffic.priority_ratios.keys():
            raise ValueError(
                f"traffic.flow_policy_config contains unknown priority classes: {sorted(extra)}"
            )

        vis = config_dict.get("visualization", {})
        return cls(
            data_sources=DataSources(**config_dict["data_sources"]),
            projection=ProjectionConfig(**config_dict["projection"]),
            clustering=ClusteringConfig(**config_dict["clustering"]),
            highway_processing=HighwayProcessingConfig(
                **config_dict["highway_processing"]
            ),
            corridors=CorridorsConfig(**corridors_dict),
            validation=ValidationConfig(**config_dict["validation"]),
            output=OutputConfig(**output_dict),
            build=build,
            components=components,
            failure_policies=failure_policies,
            workflows=workflows,
            traffic=traffic,
            visualization=VisualizationConfig(
                use_real_corridor_geometry=vis.get("corridors", {}).get(
                    "use_real_geometry", False
                ),
                export_site_graph=vis.get("site_graph", {}).get("export", False),
                export_blueprint_diagrams=vis.get("blueprints", {}).get(
                    "export", False
                ),
                dpi=vis.get("dpi", 300),
            ),
        )

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


LINK_TYPES = ("intra_metro_link", "inter_metro_link", "dc_to_pop_link")


def parse_link_params(
    values: dict[str, Any], defaults: LinkParams, name: str
) -> LinkParams:
    """Merge attributes, replace selectors, and normalize numeric link settings."""
    merged = asdict(defaults)
    merged.update(values)
    merged["attrs"] = {**defaults.attrs, **values.get("attrs", {})}
    for key in ("capacity", "cost"):
        merged[key] = _normalize_int(merged[key], f"{name}.{key}")
    return LinkParams(**merged)


@lru_cache(maxsize=1)
def _config_validator() -> jsonschema.Draft7Validator:
    schema = json.loads(
        resources.files("topogen.schemas").joinpath("topogen_config.json").read_text()
    )
    jsonschema.Draft7Validator.check_schema(schema)
    return jsonschema.Draft7Validator(schema)
