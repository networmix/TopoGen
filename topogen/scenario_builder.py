"""NetGraph scenario builder for topology generation.

Transforms integrated metro-highway graphs into complete NetGraph YAML scenarios
by expanding metro nodes into detailed site hierarchies using blueprint templates.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import networkx as nx
import yaml

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.components_lib import get_builtin_components
from topogen.failure_policies_lib import get_builtin_failure_policies
from topogen.log_config import get_logger
from topogen.naming import metro_slug
from topogen.traffic_matrix import generate_traffic_matrix
from topogen.workflows_lib import get_builtin_workflows

if TYPE_CHECKING:
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def _count_nodes_with_role(
    blueprints: dict[str, dict[str, Any]], blueprint_name: str, role: str
) -> int:
    """Return the number of nodes with a given role in a blueprint.

    Args:
        blueprints: Mapping of blueprint names to definitions.
        blueprint_name: Name of the blueprint to inspect.
        role: Role attribute to match (e.g., "core", "dc").

    Returns:
        Number of nodes across all groups whose attrs.role equals the given role.
        Returns 0 when the blueprint or role is not found.
    """
    try:
        bp = blueprints.get(blueprint_name, {})
        groups = bp.get("groups", {}) or {}
        total = 0
        for _gname, gdef in groups.items():
            attrs = gdef.get("attrs", {}) or {}
            if attrs.get("role") == role:
                try:
                    total += int(gdef.get("node_count", 0))
                except Exception:
                    continue
        return int(total)
    except Exception:
        return 0


# Note: metro-to-path slug logic now lives in topogen.traffic_matrix for the
# traffic generation algorithms.


def build_scenario(
    graph: nx.Graph,
    config: TopologyConfig,
) -> str:
    """Build a NetGraph scenario YAML from an integrated metro-highway graph.

    Transforms each metro node into a hierarchical site structure using
    blueprint templates, preserving corridor connectivity between metros.

    Args:
        graph: Integrated graph. Inter-metro adjacency is derived from
            corridor edges between metro nodes; highway nodes are ignored here.
        config: Topology configuration including build settings.

    Returns:
        Complete NetGraph scenario as YAML string.

    Raises:
        ValueError: If configuration is invalid or metro names don't match.
    """
    logger.info("Building NetGraph scenario from integrated graph")

    # Extract metros from the graph
    metros = _extract_metros_from_graph(graph)
    logger.info(f"Found {len(metros)} metro nodes")

    # Determine per-metro settings
    metro_settings = _determine_metro_settings(metros, config)
    max_sites = max(
        (settings["pop_per_metro"] for settings in metro_settings.values()), default=1
    )
    max_dc_regions = max(
        (settings["dc_regions_per_metro"] for settings in metro_settings.values()),
        default=0,
    )
    logger.info(f"Maximum sites per metro: {max_sites}")
    logger.info(f"Maximum DC regions per metro: {max_dc_regions}")

    # Collect used blueprints
    used_blueprints = set(
        settings["site_blueprint"] for settings in metro_settings.values()
    )
    used_blueprints.update(
        settings["dc_region_blueprint"]
        for settings in metro_settings.values()
        if settings["dc_regions_per_metro"] > 0
    )
    builtin_blueprints = get_builtin_blueprints()

    # Validate all referenced blueprints exist
    for bp_name in used_blueprints:
        if bp_name not in builtin_blueprints:
            available = ", ".join(sorted(builtin_blueprints.keys()))
            raise ValueError(f"Unknown blueprint '{bp_name}'. Available: {available}")

    # Build scenario sections
    scenario = {}

    # 0. Top-level seed (configurable; default 42). Ensure this appears first.
    try:
        scenario_seed = int(getattr(config.output, "scenario_seed", 42))
    except Exception:
        scenario_seed = 42
    # Insert as earliest key to influence YAML ordering (sort_keys=False below)
    scenario["seed"] = scenario_seed

    # 1. Blueprints section
    scenario["blueprints"] = _build_blueprints_section(used_blueprints, config)

    # 2. Components section
    scenario["components"] = _build_components_section(config, used_blueprints)

    # 3. Network section
    scenario["network"] = _build_network_section(
        metros, metro_settings, max_sites, max_dc_regions, graph, config
    )

    # 3b. Hardware-aware capacity allocation (optional)
    try:
        ca_cfg = getattr(config.build, "capacity_allocation", None)
    except Exception:
        ca_cfg = None
    if ca_cfg and getattr(ca_cfg, "enabled", False):
        _apply_hw_capacity_allocation(
            scenario,
            metros,
            metro_settings,
            graph,
            config,
        )

    # 4. Risk groups section (if risk groups are present)
    risk_groups = _build_risk_groups_section(graph, config)
    if risk_groups:
        scenario["risk_groups"] = risk_groups

    # 5. Failure policy set section
    scenario["failure_policy_set"] = _build_failure_policy_set_section(config)

    # 6. Traffic matrix section (optional)
    traffic_section = _build_traffic_matrix_section(metros, metro_settings, config)
    if traffic_section:
        scenario["traffic_matrix_set"] = traffic_section

    # 7. Workflow section
    scenario["workflow"] = _build_workflow_section(config)

    # Convert to YAML; optionally disable anchors/aliases per formatting config
    try:
        emit_anchors = bool(config.output.formatting.yaml_anchors)
    except Exception:
        emit_anchors = True

    if emit_anchors:
        yaml_output = yaml.safe_dump(
            scenario, sort_keys=False, default_flow_style=False
        )
    else:

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):  # type: ignore[override]
                return True

        yaml_output = yaml.dump(
            scenario,
            Dumper=NoAliasDumper,
            sort_keys=False,
            default_flow_style=False,
        )

    # Add section comments to adjacency section
    yaml_output = _add_adjacency_comments(yaml_output)

    logger.info("Generated NetGraph scenario YAML")

    return yaml_output


def _add_adjacency_comments(yaml_content: str) -> str:
    """Add section comments to the adjacency section of the YAML.

    Args:
        yaml_content: Original YAML content string.

    Returns:
        YAML content with section comments added.
    """
    lines = yaml_content.split("\n")
    result_lines = []
    in_adjacency = False
    intra_metro_added = False
    inter_metro_added = False

    for i, line in enumerate(lines):
        # Check if we're entering the adjacency section
        if line.strip() == "adjacency:" and not in_adjacency:
            in_adjacency = True
            result_lines.append(line)
            continue

        # Check if we're leaving the adjacency section
        if (
            in_adjacency
            and line
            and not line.startswith(" ")
            and not line.startswith("-")
        ):
            in_adjacency = False

        if in_adjacency and line.strip().startswith("- source:"):
            # Look ahead to see if this is intra-metro or inter-metro
            link_type = None
            for j in range(i, min(i + 15, len(lines))):
                if "link_type: intra_metro" in lines[j]:
                    link_type = "intra_metro"
                    break
                elif "link_type: inter_metro_corridor" in lines[j]:
                    link_type = "inter_metro"
                    break

            # Add appropriate comment before first occurrence
            if link_type == "intra_metro" and not intra_metro_added:
                result_lines.append(
                    "  # Intra-metro adjacency (connectivity within each metro's sites)"
                )
                intra_metro_added = True
            elif link_type == "inter_metro" and not inter_metro_added:
                result_lines.append(
                    "  # Inter-metro corridor connectivity (backbone links between metros)"
                )
                inter_metro_added = True

        result_lines.append(line)

    return "\n".join(result_lines)


def _extract_metros_from_graph(graph: nx.Graph) -> list[dict[str, Any]]:
    """Extract metro node information from the integrated graph.

    Args:
        graph: Integrated graph containing metro and highway nodes.

    Returns:
        List of metro node dictionaries with required attributes.

    Raises:
        ValueError: If metro nodes are missing required attributes.
    """
    metros = []
    for node, data in graph.nodes(data=True):
        if data.get("node_type") in ["metro", "metro+highway"]:
            # Validate required attributes
            required_attrs = ["name", "metro_id", "radius_km"]
            for attr in required_attrs:
                if attr not in data:
                    raise ValueError(
                        f"Metro node {node} missing required attribute '{attr}'"
                    )

            metros.append(
                {
                    "node_key": node,
                    "name": data["name"],
                    "name_orig": data.get(
                        "name_orig", data["name"]
                    ),  # Include original name
                    "metro_id": data["metro_id"],
                    "x": data.get("x", 0.0),
                    "y": data.get("y", 0.0),
                    "radius_km": data["radius_km"],
                }
            )

    return metros


def _determine_metro_settings(
    metros: list[dict[str, Any]], config: TopologyConfig
) -> dict[str, dict[str, Any]]:
    """Determine per-metro configuration settings from config and overrides.

    Args:
        metros: List of metro dictionaries.
        config: Topology configuration with build settings.

    Returns:
        Dictionary mapping metro names to their resolved settings.

    Raises:
        ValueError: If override references unknown metro name.
    """
    build_config = config.build
    defaults = build_config.build_defaults
    overrides = build_config.build_overrides

    # One-way matching model: overrides are keyed by slugs.
    # A slug is derived from a metro display name using a consistent rule:
    # lowercase, collapse whitespace to hyphens, collapse repeated hyphens.

    # Validate override keys strictly against available metro slugs
    if metros:
        available_slugs = set()
        for m in metros:
            available_slugs.add(metro_slug(m.get("name", "")))
            available_slugs.add(metro_slug(m.get("name_orig", m.get("name", ""))))
        for override_key in overrides.keys():
            if override_key not in available_slugs:
                available_list = ", ".join(sorted(available_slugs))
                raise ValueError(
                    f"Build override references unknown metro '{override_key}'. "
                    f"Available metro slugs: {available_list}"
                )

    # Build settings for each metro
    settings = {}
    for metro in metros:
        metro_name = metro["name"]

        # Start with defaults
        metro_settings = {
            "pop_per_metro": defaults.pop_per_metro,
            "site_blueprint": defaults.site_blueprint,
            "dc_regions_per_metro": defaults.dc_regions_per_metro,
            "dc_region_blueprint": defaults.dc_region_blueprint,
            "intra_metro_link": {
                "capacity": defaults.intra_metro_link.capacity,
                "cost": defaults.intra_metro_link.cost,
                "attrs": defaults.intra_metro_link.attrs.copy(),
            },
            "inter_metro_link": {
                "capacity": defaults.inter_metro_link.capacity,
                "cost": defaults.inter_metro_link.cost,
                "attrs": defaults.inter_metro_link.attrs.copy(),
            },
            "dc_to_pop_link": {
                "capacity": defaults.dc_to_pop_link.capacity,
                "cost": defaults.dc_to_pop_link.cost,
                "attrs": defaults.dc_to_pop_link.attrs.copy(),
            },
        }

        # Apply overrides if present (exact slug match only)
        metro_name_orig = metro.get("name_orig", metro_name)
        override = None
        slug_sanitized = metro_slug(metro_name)
        slug_original = metro_slug(metro_name_orig)
        if slug_sanitized in overrides:
            override = overrides[slug_sanitized]
        elif slug_original in overrides:
            override = overrides[slug_original]

        if override:
            if "pop_per_metro" in override:
                metro_settings["pop_per_metro"] = override["pop_per_metro"]
            if "site_blueprint" in override:
                metro_settings["site_blueprint"] = override["site_blueprint"]
            if "dc_regions_per_metro" in override:
                metro_settings["dc_regions_per_metro"] = override[
                    "dc_regions_per_metro"
                ]
            if "dc_region_blueprint" in override:
                metro_settings["dc_region_blueprint"] = override["dc_region_blueprint"]
            if "intra_metro_link" in override:
                # Merge override with defaults (override takes precedence)
                override_intra = override["intra_metro_link"]
                # Handle attrs merging first to preserve defaults
                if "attrs" in override_intra:
                    metro_settings["intra_metro_link"]["attrs"].update(
                        override_intra["attrs"]
                    )
                # Update other fields (excluding attrs to avoid overwriting merged attrs)
                for key, value in override_intra.items():
                    if key != "attrs":
                        metro_settings["intra_metro_link"][key] = value
            if "inter_metro_link" in override:
                # Merge override with defaults (override takes precedence)
                override_inter = override["inter_metro_link"]
                # Handle attrs merging first to preserve defaults
                if "attrs" in override_inter:
                    metro_settings["inter_metro_link"]["attrs"].update(
                        override_inter["attrs"]
                    )
                # Update other fields (excluding attrs to avoid overwriting merged attrs)
                for key, value in override_inter.items():
                    if key != "attrs":
                        metro_settings["inter_metro_link"][key] = value
            if "dc_to_pop_link" in override:
                # Merge override with defaults (override takes precedence)
                override_dc_pop = override["dc_to_pop_link"]
                # Handle attrs merging first to preserve defaults
                if "attrs" in override_dc_pop:
                    metro_settings["dc_to_pop_link"]["attrs"].update(
                        override_dc_pop["attrs"]
                    )
                # Update other fields (excluding attrs to avoid overwriting merged attrs)
                for key, value in override_dc_pop.items():
                    if key != "attrs":
                        metro_settings["dc_to_pop_link"][key] = value

        # Validate settings
        if metro_settings["pop_per_metro"] < 1:
            raise ValueError(
                f"Metro '{metro_name}' has invalid pop_per_metro: {metro_settings['pop_per_metro']}"
            )
        if metro_settings["dc_regions_per_metro"] < 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid dc_regions_per_metro: {metro_settings['dc_regions_per_metro']}"
            )

        # Validate link parameters
        intra_link = metro_settings["intra_metro_link"]
        if intra_link["capacity"] <= 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid intra_metro_link capacity: {intra_link['capacity']}"
            )
        if intra_link["cost"] <= 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid intra_metro_link cost: {intra_link['cost']}"
            )

        inter_link = metro_settings["inter_metro_link"]
        if inter_link["capacity"] <= 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid inter_metro_link capacity: {inter_link['capacity']}"
            )
        if inter_link["cost"] < 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid inter_metro_link cost: {inter_link['cost']}"
            )

        dc_pop_link = metro_settings["dc_to_pop_link"]
        if dc_pop_link["capacity"] <= 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid dc_to_pop_link capacity: {dc_pop_link['capacity']}"
            )
        if dc_pop_link["cost"] < 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid dc_to_pop_link cost: {dc_pop_link['cost']}"
            )

        settings[metro_name] = metro_settings

    return settings


def _build_network_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    max_sites: int,
    max_dc_regions: int,
    graph: nx.Graph,
    config: "TopologyConfig",
) -> dict[str, Any]:
    """Build the network section of the NetGraph scenario.

    Args:
        metros: List of metro dictionaries.
        metro_settings: Per-metro configuration settings.
        max_sites: Maximum number of sites across all metros.
        max_dc_regions: Maximum number of DC regions across all metros.
        graph: Original integrated graph for corridor extraction.

    Returns:
        Network section dictionary for the scenario.
    """
    network = {}

    # Build groups section
    network["groups"] = _build_groups_section(
        metros, metro_settings, max_sites, max_dc_regions, config
    )

    # Build adjacency section
    network["adjacency"] = _build_adjacency_section(
        metros, metro_settings, graph, config
    )

    # Build link_overrides with circle-based intra-metro costs
    overrides = _build_intra_metro_link_overrides(metros, metro_settings, config)
    if overrides:
        network["link_overrides"] = overrides

    return network


def _build_groups_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    max_sites: int,
    max_dc_regions: int,
    config: "TopologyConfig",
) -> dict[str, Any]:
    """Build the groups section defining site hierarchies.

    Args:
        metros: List of metro dictionaries.
        metro_settings: Per-metro configuration settings.
        max_sites: Maximum number of sites to support.
        max_dc_regions: Maximum number of DC regions to support.
        config: Full topology configuration (reads traffic parameters for DC attrs).

    Returns:
        Groups section dictionary.
    """
    groups = {}

    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        blueprint_name = settings["site_blueprint"]

        # Create metro group with bracket expansion for sites
        group_name = f"metro{idx}/pop[1-{max_sites}]"
        groups[group_name] = {
            "use_blueprint": blueprint_name,
            "attrs": {
                "metro_name": metro_name,  # Use sanitized name as primary
                "metro_name_orig": metro.get(
                    "name_orig", metro_name
                ),  # Original name for display
                "metro_id": metro["metro_id"],
                "location_x": metro["x"],
                "location_y": metro["y"],
                "radius_km": metro["radius_km"],
                "node_type": "pop",
            },
        }

        # For metros with fewer sites than max, sites will be disabled later
        # This is handled by NetGraph's bracket expansion system

        # Create DC region groups if enabled
        if max_dc_regions > 0:
            dc_blueprint_name = settings["dc_region_blueprint"]
            dc_group_name = f"metro{idx}/dc[1-{max_dc_regions}]"
            groups[dc_group_name] = {
                "use_blueprint": dc_blueprint_name,
                "attrs": {
                    "metro_name": metro_name,  # Use sanitized name as primary
                    "metro_name_orig": metro.get(
                        "name_orig", metro_name
                    ),  # Original name for display
                    "metro_id": metro["metro_id"],
                    "location_x": metro["x"],
                    "location_y": metro["y"],
                    "radius_km": metro["radius_km"],
                    "node_type": "dc_region",
                    # Include traffic-related parameters on DC nodes for downstream consumers
                    "mw_per_dc_region": float(
                        getattr(
                            getattr(config, "traffic", object()),
                            "mw_per_dc_region",
                            0.0,
                        )
                    ),
                    "gbps_per_mw": float(
                        getattr(
                            getattr(config, "traffic", object()), "gbps_per_mw", 0.0
                        )
                    ),
                },
            }

    return groups


def _build_adjacency_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    graph: nx.Graph,
    config: "TopologyConfig",
) -> list[dict[str, Any]]:
    """Build the adjacency section defining inter-metro connectivity.

    Args:
        metros: List of metro dictionaries.
        metro_settings: Per-metro configuration settings.
        graph: Original integrated graph for corridor extraction.

    Returns:
        List of adjacency rules with section comments.
    """
    adjacency = []
    # Use built-in blueprints to infer per-site fan-out by role
    blueprint_defs = get_builtin_blueprints()

    # Build metro index mapping
    metro_by_node = {metro["node_key"]: metro for metro in metros}
    metro_idx_map = {metro["name"]: idx for idx, metro in enumerate(metros, 1)}

    # Add intra-metro adjacency (full mesh within each metro's sites)
    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        sites_count = settings["pop_per_metro"]

        if sites_count > 1:
            # Create mesh connectivity between sites within metro
            # Compute circle-based cost estimate along shortest arc.
            radius_km = float(metro.get("radius_km", 0.0))
            circle_frac = 1.0
            # Fallback to 0 if radius missing
            ring_radius_km = max(0.0, radius_km * circle_frac)

            # Precompute base cost info for attrs
            # Compute per-link capacity by dividing by number of matched nodes per site
            site_bp = settings["site_blueprint"]
            core_per_site = max(
                1, _count_nodes_with_role(blueprint_defs, site_bp, "core")
            )

            adjacency.append(
                {
                    "source": {
                        "path": f"/metro{idx}/pop[1-{sites_count}]",
                        "match": {
                            "conditions": [
                                {"attr": "role", "operator": "==", "value": "core"}
                            ]
                        },
                    },
                    "target": {
                        "path": f"/metro{idx}/pop[1-{sites_count}]",
                        "match": {
                            "conditions": [
                                {"attr": "role", "operator": "==", "value": "core"}
                            ]
                        },
                    },
                    "pattern": "one_to_one",
                    "link_params": {
                        # Divide adjacency budget across leaf-level links created by blueprint fan-out
                        "capacity": int(
                            max(
                                1,
                                int(settings["intra_metro_link"]["capacity"])
                                // int(core_per_site),
                            )
                        ),
                        # Use average nearest-neighbor arc as representative base cost.
                        # Nearest neighbor steps = 1 => 2*pi*R/n
                        "cost": int(
                            math.ceil(
                                ((2.0 * math.pi * ring_radius_km) / float(sites_count))
                                if ring_radius_km > 0.0 and sites_count > 0
                                else settings["intra_metro_link"]["cost"]
                            )
                        ),
                        "attrs": {
                            **settings["intra_metro_link"][
                                "attrs"
                            ],  # Start with configured attrs
                            "metro_name": metro_name,  # Add metro-specific attrs
                            "metro_name_orig": metro.get("name_orig", metro_name),
                            "distance_model": "metro_circle_arc",
                            "circle_radius_km": ring_radius_km,
                            "pop_count": int(sites_count),
                            # No fraction parameter; using full metro radius
                        },
                    },
                }
            )

    # Add DC-to-PoP connectivity (DC regions connect to all local PoPs)
    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        sites_count = settings["pop_per_metro"]
        dc_regions_count = settings["dc_regions_per_metro"]

        if dc_regions_count > 0 and sites_count > 0:
            # Connect all DC regions to all PoPs in the same metro
            # Compute per-link capacity for DC-to-PoP mesh
            site_bp = settings["site_blueprint"]
            dc_bp = settings["dc_region_blueprint"]
            core_per_site = max(
                1, _count_nodes_with_role(blueprint_defs, site_bp, "core")
            )
            dc_nodes = max(1, _count_nodes_with_role(blueprint_defs, dc_bp, "dc"))
            dp_divisor = max(1, core_per_site * dc_nodes)

            adjacency.append(
                {
                    "source": {
                        "path": f"/metro{idx}/dc[1-{dc_regions_count}]",
                        "match": {
                            "conditions": [
                                {
                                    "attr": "role",
                                    "operator": "==",
                                    "value": "dc",
                                }
                            ]
                        },
                    },
                    "target": {
                        "path": f"/metro{idx}/pop[1-{sites_count}]",
                        "match": {
                            "conditions": [
                                {
                                    "attr": "role",
                                    "operator": "==",
                                    "value": "core",
                                }
                            ]
                        },
                    },
                    "pattern": "mesh",
                    "link_params": {
                        # Divide adjacency budget across MxN links (dc nodes × core nodes per PoP)
                        "capacity": int(
                            max(
                                1,
                                int(settings["dc_to_pop_link"]["capacity"])
                                // int(dp_divisor),
                            )
                        ),
                        "cost": settings["dc_to_pop_link"]["cost"],
                        "attrs": {
                            **settings["dc_to_pop_link"][
                                "attrs"
                            ],  # Start with configured attrs
                            "metro_name": metro_name,  # Add metro-specific attrs
                            "metro_name_orig": metro.get("name_orig", metro_name),
                        },
                    },
                }
            )

    # Add inter-metro corridor connectivity
    corridor_edges = _extract_corridor_edges(graph)
    logger.info(f"Found {len(corridor_edges)} corridor connections")

    for edge in corridor_edges:
        source_metro = metro_by_node.get(edge["source"])
        target_metro = metro_by_node.get(edge["target"])

        if not source_metro or not target_metro:
            logger.warning(f"Skipping corridor edge with unknown metro: {edge}")
            continue

        source_idx = metro_idx_map[source_metro["name"]]
        target_idx = metro_idx_map[target_metro["name"]]

        source_sites = metro_settings[source_metro["name"]]["pop_per_metro"]
        target_sites = metro_settings[target_metro["name"]]["pop_per_metro"]

        # Build link_params with risk groups if present
        source_settings = metro_settings[source_metro["name"]]

        # Use source metro's inter_metro_link settings for defaults
        default_capacity = source_settings["inter_metro_link"]["capacity"]
        base_cost = source_settings["inter_metro_link"]["cost"]

        # Divide inter-metro per-pair adjacency budget across leaf-level links
        # created by blueprint fan-out (one_to_one across core nodes per site).
        src_bp = source_settings["site_blueprint"]
        tgt_bp = metro_settings[target_metro["name"]]["site_blueprint"]
        src_core = max(1, _count_nodes_with_role(blueprint_defs, src_bp, "core"))
        tgt_core = max(1, _count_nodes_with_role(blueprint_defs, tgt_bp, "core"))
        # one_to_one yields min(src_core, tgt_core) links per PoP pair
        inter_divisor = max(1, min(src_core, tgt_core))

        link_params = {
            "capacity": int(
                max(
                    1, int(edge.get("capacity", default_capacity)) // int(inter_divisor)
                )
            ),
            "cost": math.ceil(edge.get("length_km", base_cost)),
            "attrs": {
                **source_settings["inter_metro_link"][
                    "attrs"
                ],  # Start with configured attrs
                "distance_km": math.ceil(edge.get("length_km", 0.0)),
                "source_metro": source_metro["name"],  # Use sanitized name as primary
                "source_metro_orig": source_metro.get(
                    "name_orig", source_metro["name"]
                ),
                "target_metro": target_metro["name"],  # Use sanitized name as primary
                "target_metro_orig": target_metro.get(
                    "name_orig", target_metro["name"]
                ),
            },
        }

        # Add risk groups to link_params if present
        edge_risk_groups = edge.get("risk_groups", [])
        if edge_risk_groups:
            link_params["risk_groups"] = edge_risk_groups

        # Connect all core sites between metros using expand_vars
        adjacency.append(
            {
                "source": {
                    "path": f"metro{source_idx}/pop{{src_idx}}",
                    "match": {
                        "conditions": [
                            {"attr": "role", "operator": "==", "value": "core"}
                        ]
                    },
                },
                "target": {
                    "path": f"metro{target_idx}/pop{{tgt_idx}}",
                    "match": {
                        "conditions": [
                            {"attr": "role", "operator": "==", "value": "core"}
                        ]
                    },
                },
                "expand_vars": {
                    "src_idx": list(range(1, source_sites + 1)),
                    "tgt_idx": list(range(1, target_sites + 1)),
                },
                "expansion_mode": "cartesian",
                "pattern": "one_to_one",
                "link_params": link_params,
            }
        )

    return adjacency


def _build_intra_metro_link_overrides(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: "TopologyConfig",
) -> list[dict[str, Any]]:
    """Create link_overrides assigning circle-arc distances as per-pair costs.

    For each metro, place PoPs and DC regions on a circle of radius
    metro.radius_km and set the per-pair cost to the shortest circle-arc
    distance in kilometers (ceil).

    Args:
        metros: List of metro dicts.
        metro_settings: Per-metro settings.
        config: Topology configuration for circle fraction.

    Returns:
        List of link_overrides entries.
    """
    overrides: list[dict[str, Any]] = []
    circle_frac = 1.0

    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        sites_count = int(settings["pop_per_metro"])
        dc_count = int(settings["dc_regions_per_metro"])
        if sites_count <= 0:
            continue

        ring_radius_km = max(0.0, float(metro.get("radius_km", 0.0)) * circle_frac)
        if ring_radius_km <= 0.0:
            continue

        def arc_ceil(
            n: int, i: int, j: int, *, _radius_km: float = ring_radius_km
        ) -> int:
            if n <= 1:
                return 0
            delta = abs(i - j) % n
            steps = min(delta, n - delta)
            arc = steps * ((2.0 * math.pi * _radius_km) / float(n))
            return int(math.ceil(arc))

        # PoP-PoP overrides
        if sites_count > 1:
            for i in range(1, sites_count + 1):
                for j in range(i + 1, sites_count + 1):
                    cost_ij = arc_ceil(sites_count, i, j)
                    if cost_ij <= 0:
                        continue
                    overrides.append(
                        {
                            "source": f"metro{idx}/pop{i}",
                            "target": f"metro{idx}/pop{j}",
                            "any_direction": True,
                            "link_params": {
                                "cost": cost_ij,
                                "attrs": {
                                    "distance_km": cost_ij,
                                    "distance_model": "metro_circle_arc",
                                    "circle_radius_km": ring_radius_km,
                                    # No fraction parameter; using full metro radius
                                },
                            },
                        }
                    )

        # DC-PoP overrides
        if dc_count > 0 and sites_count > 0:
            # Place DCs interleaved across same circle
            for d in range(1, dc_count + 1):
                for p in range(1, sites_count + 1):
                    # Model DCs as evenly spaced too using same n for arc steps,
                    # aligning DC index to pop index space by uniform spread.
                    # Effective step distance is min over two nearest pop slots.
                    # Simpler: treat DC positions as at indices spaced by n/dc_count,
                    # compute arc relative to pop index p.
                    # Map d to fractional index and compute integer distance.
                    n = sites_count
                    frac_index = (d - 1) * (n / max(dc_count, 1))
                    # Evaluate arc to nearest integer pop positions around frac_index
                    cand_idxs = {
                        int(math.floor(frac_index)) % n + 1,
                        int(math.ceil(frac_index)) % n + 1,
                    }
                    cost_dp = min(arc_ceil(n, p, c) for c in cand_idxs)
                    if cost_dp <= 0:
                        continue
                    overrides.append(
                        {
                            "source": f"metro{idx}/dc{d}",
                            "target": f"metro{idx}/pop{p}",
                            "any_direction": True,
                            "link_params": {
                                "cost": cost_dp,
                                "attrs": {
                                    "distance_km": cost_dp,
                                    "distance_model": "metro_circle_arc",
                                    "circle_radius_km": ring_radius_km,
                                    # No fraction parameter; using full metro radius
                                },
                            },
                        }
                    )

    return overrides


def _apply_hw_capacity_allocation(
    scenario: dict[str, Any],
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    graph: nx.Graph,
    config: "TopologyConfig",
) -> None:
    """Apply hardware-aware capacity allocation by updating inter-metro capacities.

    Keeps configured capacities as minimums and distributes remaining platform
    capacity to inter-metro POP-to-POP links using a round-robin strategy.

    Raises:
        ValueError: If base reservations exceed platform capacity.
    """
    # Build component lookup from scenario components (already merged built-ins)
    components: dict[str, dict[str, Any]] = scenario.get("components", {})
    blueprints: dict[str, dict[str, Any]] = scenario.get("blueprints", {})

    # Map metro to index (1-based) consistent with adjacency build
    metro_idx_map = {metro["name"]: idx for idx, metro in enumerate(metros, 1)}

    # Determine platform capacity per POP and per DC node
    pop_capacity: dict[tuple[int, int], float] = {}
    dc_capacity: dict[tuple[int, int], float] = {}
    pop_constrained: dict[tuple[int, int], bool] = {}
    dc_constrained: dict[tuple[int, int], bool] = {}

    for idx, metro in enumerate(metros, 1):
        settings = metro_settings[metro["name"]]
        sites_count = int(settings["pop_per_metro"])
        dc_count = int(settings["dc_regions_per_metro"])

        # POP platform capacity from blueprint core group hw_component
        pop_bp_name = settings["site_blueprint"]
        pop_bp = blueprints.get(pop_bp_name, {})
        pop_groups = pop_bp.get("groups", {})
        core_group = pop_groups.get("core", {})
        core_attrs = core_group.get("attrs", {})
        core_hw = core_attrs.get("hw_component")
        comp_entry = components.get(core_hw, {}) if core_hw else {}
        core_cap_val = comp_entry.get("capacity")
        core_cap = float(core_cap_val) if core_cap_val is not None else float("inf")

        for p in range(1, sites_count + 1):
            key = (idx, p)
            pop_capacity[key] = core_cap
            pop_constrained[key] = core_cap != float("inf")

        # DC platform capacity from blueprint dc group hw_component
        if dc_count > 0:
            dc_bp_name = settings["dc_region_blueprint"]
            dc_bp = blueprints.get(dc_bp_name, {})
            dc_groups = dc_bp.get("groups", {})
            dc_group = dc_groups.get("dc", {})
            dc_attrs = dc_group.get("attrs", {})
            dc_hw = dc_attrs.get("hw_component")
            comp_entry = components.get(dc_hw, {}) if dc_hw else {}
            dc_cap_val = comp_entry.get("capacity")
            dc_cap = float(dc_cap_val) if dc_cap_val is not None else float("inf")
            for d in range(1, dc_count + 1):
                key = (idx, d)
                dc_capacity[key] = dc_cap
                dc_constrained[key] = dc_cap != float("inf")

    # Reserve base capacities against platform budgets
    # Initialize budgets as remaining capacity
    pop_budget: dict[tuple[int, int], float] = {k: v for k, v in pop_capacity.items()}
    dc_budget: dict[tuple[int, int], float] = {k: v for k, v in dc_capacity.items()}

    # Intra-metro reservations (PoP-PoP mesh)
    for idx, metro in enumerate(metros, 1):
        settings = metro_settings[metro["name"]]
        s = int(settings["pop_per_metro"])
        if s <= 1:
            continue
        c_intra = float(settings["intra_metro_link"]["capacity"])
        reserve = max(0, s - 1) * c_intra
        for p in range(1, s + 1):
            key = (idx, p)
            if key in pop_budget:
                pop_budget[key] = pop_budget.get(key, float("inf")) - reserve

    # DC-to-PoP reservations (mesh)
    for idx, metro in enumerate(metros, 1):
        settings = metro_settings[metro["name"]]
        s = int(settings["pop_per_metro"])
        d = int(settings["dc_regions_per_metro"])
        if s <= 0 or d <= 0:
            continue
        c_dp = float(settings["dc_to_pop_link"]["capacity"])
        # PoP reservations: d links per PoP
        for p in range(1, s + 1):
            key = (idx, p)
            if key in pop_budget:
                pop_budget[key] = pop_budget.get(key, float("inf")) - d * c_dp
        # DC reservations: s links per DC
        for dc in range(1, d + 1):
            key = (idx, dc)
            if key in dc_budget:
                dc_budget[key] = dc_budget.get(key, float("inf")) - s * c_dp

    # Inter-metro reservations per corridor
    corridor_edges = _extract_corridor_edges(graph)
    # Build quick lookup of site counts per metro
    sites_per_metro: dict[int, int] = {
        metro_idx_map[m["name"]]: int(metro_settings[m["name"]]["pop_per_metro"])  # type: ignore[index]
        for m in metros
    }

    for edge in corridor_edges:
        source = edge["source"]
        target = edge["target"]
        src_metro = next(m for m in metros if m["node_key"] == source)
        tgt_metro = next(m for m in metros if m["node_key"] == target)
        s_idx = metro_idx_map[src_metro["name"]]
        t_idx = metro_idx_map[tgt_metro["name"]]
        s_sites = sites_per_metro[s_idx]
        t_sites = sites_per_metro[t_idx]

        # Base capacity per POP pair: use explicit corridor capacity if set
        src_settings = metro_settings[src_metro["name"]]
        default_c = float(src_settings["inter_metro_link"]["capacity"])
        base_c = float(edge.get("capacity", default_c))

        # Reserve for source POPs
        for p in range(1, s_sites + 1):
            key = (s_idx, p)
            if key in pop_budget:
                pop_budget[key] = pop_budget.get(key, float("inf")) - t_sites * base_c
        # Reserve for target POPs
        for q in range(1, t_sites + 1):
            key = (t_idx, q)
            if key in pop_budget:
                pop_budget[key] = pop_budget.get(key, float("inf")) - s_sites * base_c

    # Validate budgets (always raise on overcommit)
    for (idx, p), remaining in pop_budget.items():
        if remaining < -1e-9:  # allow tiny epsilon
            metro_name = next(
                m["name"] for m in metros if metro_idx_map[m["name"]] == idx
            )
            raise ValueError(
                f"Base capacities exceed platform at metro {metro_name} pop{p}: remaining {remaining:.0f} Gbps < 0"
            )
    for (idx, d), remaining in dc_budget.items():
        if remaining < -1e-9:
            metro_name = next(
                m["name"] for m in metros if metro_idx_map[m["name"]] == idx
            )
            raise ValueError(
                f"Base capacities exceed platform at metro {metro_name} dc{d}: remaining {remaining:.0f} Gbps < 0"
            )

    # Distribute remaining POP budget to inter-metro links in round-robin
    # Build candidate pairs with base capacity and step size
    candidates: list[
        tuple[int, int, int, int, float]
    ] = []  # (s_idx, p, t_idx, q, step)

    for edge in corridor_edges:
        src_metro = next(m for m in metros if m["node_key"] == edge["source"])
        tgt_metro = next(m for m in metros if m["node_key"] == edge["target"])
        s_idx = metro_idx_map[src_metro["name"]]
        t_idx = metro_idx_map[tgt_metro["name"]]
        s_sites = sites_per_metro[s_idx]
        t_sites = sites_per_metro[t_idx]

        src_settings = metro_settings[src_metro["name"]]
        default_c = float(src_settings["inter_metro_link"]["capacity"])
        base_c = float(edge.get("capacity", default_c))
        step = base_c

        for p in range(1, s_sites + 1):
            for q in range(1, t_sites + 1):
                # Only consider pairs where at least one endpoint is constrained.
                if pop_constrained.get((s_idx, p), False) or pop_constrained.get(
                    (t_idx, q), False
                ):
                    candidates.append((s_idx, p, t_idx, q, step))

    # Track increments per pair
    increments: dict[tuple[int, int, int, int], int] = {}
    if candidates:
        progressed = True
        while progressed:
            progressed = False
            for s_idx, p, t_idx, q, step in candidates:
                key_s = (s_idx, p)
                key_t = (t_idx, q)
                # If an endpoint is unconstrained, skip its budget check.
                need_s = pop_constrained.get(key_s, False)
                need_t = pop_constrained.get(key_t, False)
                b_s = pop_budget.get(key_s, float("inf"))
                b_t = pop_budget.get(key_t, float("inf"))
                ok_s = (not need_s) or (b_s >= step)
                ok_t = (not need_t) or (b_t >= step)
                if ok_s and ok_t:
                    if need_s:
                        pop_budget[key_s] = b_s - step
                    if need_t:
                        pop_budget[key_t] = b_t - step
                    pair_key = (s_idx, p, t_idx, q)
                    increments[pair_key] = increments.get(pair_key, 0) + 1
                    progressed = True

    # Compute final capacities for all POP pairs covered by corridor adjacencies
    final_capacity: dict[tuple[int, int, int, int], int] = {}
    for edge in corridor_edges:
        src_metro = next(m for m in metros if m["node_key"] == edge["source"])
        tgt_metro = next(m for m in metros if m["node_key"] == edge["target"])
        s_idx = metro_idx_map[src_metro["name"]]
        t_idx = metro_idx_map[tgt_metro["name"]]
        s_sites = sites_per_metro[s_idx]
        t_sites = sites_per_metro[t_idx]
        src_settings = metro_settings[src_metro["name"]]
        default_c = float(src_settings["inter_metro_link"]["capacity"])
        base_c = float(edge.get("capacity", default_c))
        for p in range(1, s_sites + 1):
            for q in range(1, t_sites + 1):
                k = increments.get((s_idx, p, t_idx, q), 0)
                final_capacity[(s_idx, p, t_idx, q)] = int(base_c + k * base_c)

    # Rewrite adjacency: replace inter-metro cartesian rule with per-pair entries
    network = scenario.setdefault("network", {})
    old_adj: list[dict[str, Any]] = network.get("adjacency", [])  # type: ignore[assignment]
    new_adj: list[dict[str, Any]] = []

    # Keep non inter-metro entries
    for adj in old_adj:
        try:
            attrs = adj.get("link_params", {}).get("attrs", {})
            if attrs.get("link_type") == "inter_metro_corridor":
                # Skip; will be replaced with per-pair adjacencies
                continue
        except Exception:
            pass
        new_adj.append(adj)

    # Append per-pair inter-metro adjacency entries with final capacities
    for edge in corridor_edges:
        src_metro = next(m for m in metros if m["node_key"] == edge["source"])
        tgt_metro = next(m for m in metros if m["node_key"] == edge["target"])
        s_idx = metro_idx_map[src_metro["name"]]
        t_idx = metro_idx_map[tgt_metro["name"]]
        s_sites = sites_per_metro[s_idx]
        t_sites = sites_per_metro[t_idx]
        src_settings = metro_settings[src_metro["name"]]
        inter_attrs = dict(src_settings["inter_metro_link"]["attrs"])  # copy
        base_cost = math.ceil(
            edge.get("length_km", src_settings["inter_metro_link"]["cost"])
        )
        edge_risk_groups = edge.get("risk_groups", [])

        # Determine per-link divisor from blueprint fan-out for role 'core'
        # Use the scenario's blueprints (already merged) to count nodes
        # to remain consistent with any user overrides.
        bp_defs: dict[str, Any] = scenario.get("blueprints", {})  # type: ignore[assignment]

        for p in range(1, s_sites + 1):
            for q in range(1, t_sites + 1):
                cap_total = final_capacity[(s_idx, p, t_idx, q)]
                # Compute divisor using src/tgt site blueprints
                src_bp = src_settings["site_blueprint"]
                tgt_bp = metro_settings[tgt_metro["name"]]["site_blueprint"]
                src_core = max(1, _count_nodes_with_role(bp_defs, src_bp, "core"))
                tgt_core = max(1, _count_nodes_with_role(bp_defs, tgt_bp, "core"))
                inter_divisor = max(1, min(src_core, tgt_core))
                cap_each = int(max(1, int(cap_total) // int(inter_divisor)))
                # Decide whether endpoint role filtering is required.
                # If a site's blueprint contains only 'core' role nodes, we can
                # keep shorthand string endpoints to preserve simple scenarios
                # expected by tests. Otherwise, include an explicit role match
                # to ensure only core devices are selected (e.g., in CLOS).
                src_bp_name = src_settings["site_blueprint"]
                tgt_bp_name = metro_settings[tgt_metro["name"]]["site_blueprint"]
                _src_bp = bp_defs.get(src_bp_name, {})
                _tgt_bp = bp_defs.get(tgt_bp_name, {})
                _src_groups = _src_bp.get("groups", {})
                _tgt_groups = _tgt_bp.get("groups", {})
                src_total = int(
                    sum(int(g.get("node_count", 0)) for g in _src_groups.values())
                )
                tgt_total = int(
                    sum(int(g.get("node_count", 0)) for g in _tgt_groups.values())
                )
                src_core = max(1, _count_nodes_with_role(bp_defs, src_bp_name, "core"))
                tgt_core = max(1, _count_nodes_with_role(bp_defs, tgt_bp_name, "core"))
                needs_match = (src_core < src_total) or (tgt_core < tgt_total)

                if needs_match:
                    entry = {
                        "source": {
                            "path": f"metro{s_idx}/pop{p}",
                            "match": {
                                "conditions": [
                                    {
                                        "attr": "role",
                                        "operator": "==",
                                        "value": "core",
                                    }
                                ]
                            },
                        },
                        "target": {
                            "path": f"metro{t_idx}/pop{q}",
                            "match": {
                                "conditions": [
                                    {
                                        "attr": "role",
                                        "operator": "==",
                                        "value": "core",
                                    }
                                ]
                            },
                        },
                        "pattern": "one_to_one",
                        "link_params": {
                            # Divide per-pair capacity across fan-out links implied by blueprints
                            "capacity": cap_each,
                            "cost": base_cost,
                            "attrs": {
                                **inter_attrs,
                                "distance_km": base_cost,
                                "source_metro": src_metro["name"],
                                "source_metro_orig": src_metro.get(
                                    "name_orig", src_metro["name"]
                                ),
                                "target_metro": tgt_metro["name"],
                                "target_metro_orig": tgt_metro.get(
                                    "name_orig", tgt_metro["name"]
                                ),
                            },
                        },
                    }
                else:
                    entry = {
                        "source": f"metro{s_idx}/pop{p}",
                        "target": f"metro{t_idx}/pop{q}",
                        "pattern": "one_to_one",
                        "link_params": {
                            # Divide per-pair capacity across fan-out links implied by blueprints
                            "capacity": cap_each,
                            "cost": base_cost,
                            "attrs": {
                                **inter_attrs,
                                "distance_km": base_cost,
                                "source_metro": src_metro["name"],
                                "source_metro_orig": src_metro.get(
                                    "name_orig", src_metro["name"]
                                ),
                                "target_metro": tgt_metro["name"],
                                "target_metro_orig": tgt_metro.get(
                                    "name_orig", tgt_metro["name"]
                                ),
                            },
                        },
                    }
                if edge_risk_groups:
                    entry["link_params"]["risk_groups"] = edge_risk_groups
                new_adj.append(entry)

    network["adjacency"] = new_adj


def _build_traffic_matrix_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Build the traffic_matrix_set section if enabled.

    Delegates to ``topogen.traffic_matrix.generate_traffic_matrix``.

    Args:
        metros: List of extracted metro descriptors (name, x, y, radius_km).
        metro_settings: Per-metro settings including dc_regions_per_metro.
        config: Topology configuration with traffic parameters.

    Returns:
        Mapping from matrix name to a list of demand dicts; empty dict if disabled.
    """
    return generate_traffic_matrix(metros, metro_settings, config)


def _extract_corridor_edges(graph: nx.Graph) -> list[dict[str, Any]]:
    """Extract corridor edges between metro nodes from the integrated graph.

    Args:
        graph: Integrated graph containing corridor information.

    Returns:
        List of corridor edge dictionaries.
    """
    corridor_edges = []

    for source, target, data in graph.edges(data=True):
        # Check if this edge represents a corridor between metros
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]

        # Only process edges between metro nodes
        if source_data.get("node_type") in [
            "metro",
            "metro+highway",
        ] and target_data.get("node_type") in ["metro", "metro+highway"]:
            edge_entry: dict[str, Any] = {
                "source": source,
                "target": target,
                "length_km": data.get("length_km", 0.0),
                "edge_type": data.get("edge_type", "corridor"),
                "risk_groups": data.get("risk_groups", []),
            }
            # Only include capacity if explicitly set on the input edge.
            if "capacity" in data:
                edge_entry["capacity"] = data["capacity"]
            corridor_edges.append(edge_entry)

    return corridor_edges


def _build_risk_groups_section(
    graph: nx.Graph, config: TopologyConfig
) -> list[dict[str, Any]]:
    """Build the ``risk_groups`` section of the scenario.

    This reads risk-group tags attached to corridor edges and creates scenario
    risk group entries with a canonical per-pair distance in kilometers.

    Rules:
    - Only consider risk groups on metro-to-metro corridor edges.
    - The distance for a risk group named ``<prefix>_<A>_<B>[_pathN]`` is derived
      from the corridor edge between metros A and B (ceil of ``length_km``).
      If multiple edges exist for the same pair, keep the maximum for stability.
    - Unrelated edges that share a risk-group name do not influence the distance
      for other metro pairs.

    Args:
        graph: Corridor-level or integrated graph with metro-to-metro corridor edges.
        config: Topology configuration (reads ``corridors.risk_groups.group_prefix``).

    Returns:
        List of risk group definitions; empty list if none found or disabled.
    """
    if not config.corridors.risk_groups.enabled:
        return []

    # Build canonical distance per metro pair from corridor edges
    prefix = getattr(config.corridors.risk_groups, "group_prefix", "corridor_risk")
    pair_distance_km: dict[str, int] = {}
    for source, target, data in graph.edges(data=True):
        src = graph.nodes[source]
        tgt = graph.nodes[target]
        if src.get("node_type") in ["metro", "metro+highway"] and tgt.get(
            "node_type"
        ) in ["metro", "metro+highway"]:
            src_name = str(src.get("name", "")).strip()
            tgt_name = str(tgt.get("name", "")).strip()
            a_name, b_name = (
                (src_name, tgt_name) if src_name < tgt_name else (tgt_name, src_name)
            )
            pair_rg = f"{prefix}_{a_name}_{b_name}"
            dist_km = int(math.ceil(float(data.get("length_km", 0.0))))
            prev = pair_distance_km.get(pair_rg)
            pair_distance_km[pair_rg] = dist_km if prev is None else max(prev, dist_km)

    # Collect all risk groups present on corridor edges; assign distance from the
    # canonical pair distance if available, else fall back to the current edge distance.
    risk_group_distance_km: dict[str, int] = {}
    for _source, _target, data in graph.edges(data=True):
        src = graph.nodes[_source]
        tgt = graph.nodes[_target]
        # Only consider risk groups from metro-to-metro edges
        if src.get("node_type") in ["metro", "metro+highway"] and tgt.get(
            "node_type"
        ) in ["metro", "metro+highway"]:
            edge_dist_km = int(math.ceil(float(data.get("length_km", 0.0))))
            edge_risk_groups = data.get("risk_groups", []) or []
            for rg in edge_risk_groups:
                name = str(rg)
                # Normalize base pair name by stripping optional path suffix
                base = name
                idx = base.rfind("_path")
                if idx != -1 and idx + 5 <= len(base):
                    # Ensure suffix is of the form _pathN
                    tail = base[idx + 5 :]
                    if tail.isdigit():
                        base = base[:idx]
                canonical = pair_distance_km.get(base)
                chosen = canonical if canonical is not None else edge_dist_km
                prev = risk_group_distance_km.get(name)
                risk_group_distance_km[name] = (
                    chosen if prev is None else max(prev, chosen)
                )

    if not risk_group_distance_km:
        logger.info("No risk groups found in corridor edges")
        return []

    # Create risk group definitions with distance_km attribute
    risk_groups: list[dict[str, Any]] = []
    for rg_name in sorted(risk_group_distance_km.keys()):
        risk_groups.append(
            {
                "name": rg_name,
                "attrs": {
                    "type": "corridor_risk",
                    "distance_km": int(risk_group_distance_km.get(rg_name, 0)),
                },
            }
        )

    logger.info(f"Generated {len(risk_groups)} risk group definitions")
    return risk_groups


def _build_components_section(
    config: TopologyConfig, used_blueprints: set[str]
) -> dict[str, Any]:
    """Build the components section of the NetGraph scenario.

    Uses merged component library (built-ins + lib/components.yml) and includes
    only components that are referenced by assignments and used blueprints.

    Args:
        config: Topology configuration containing component definitions.
        used_blueprints: Set of blueprint names used in the scenario.

    Returns:
        Dictionary representing the components section.
    """
    # Load merged component library (built-ins + user lib)
    components = get_builtin_components()

    # Determine which components are referenced
    referenced_components = set()

    # Add components referenced by role assignments
    assignments = config.components.assignments
    if assignments.spine.hw_component:
        referenced_components.add(assignments.spine.hw_component)
    if assignments.leaf.hw_component:
        referenced_components.add(assignments.leaf.hw_component)
    if assignments.core.hw_component:
        referenced_components.add(assignments.core.hw_component)
    if assignments.dc.hw_component:
        referenced_components.add(assignments.dc.hw_component)

    # No blueprint overrides. Role assignments only.

    # Add optics components
    if assignments.spine.optics:
        referenced_components.add(assignments.spine.optics)
    if assignments.leaf.optics:
        referenced_components.add(assignments.leaf.optics)
    if assignments.core.optics:
        referenced_components.add(assignments.core.optics)
    if assignments.dc.optics:
        referenced_components.add(assignments.dc.optics)

    # No blueprint overrides for optics either.

    # Filter to include only referenced components
    result = {}
    for comp_name in sorted(referenced_components):
        if comp_name in components:
            result[comp_name] = components[comp_name]
        else:
            logger.warning(
                f"Referenced component '{comp_name}' not found in component library"
            )

    return result


def _build_blueprints_section(
    used_blueprints: set[str], config: TopologyConfig
) -> dict[str, Any]:
    """Build the blueprints section with component assignments.

    Takes the basic blueprint definitions and enriches them with component
    assignments based on role assignments in the configuration.

    Args:
        used_blueprints: Set of blueprint names used in the scenario.
        config: Topology configuration containing component assignments.

    Returns:
        Dictionary of blueprint definitions with component assignments.
    """
    from copy import deepcopy

    builtin_blueprints = get_builtin_blueprints()
    assignments = config.components.assignments
    result = {}

    for blueprint_name in sorted(used_blueprints):
        if blueprint_name not in builtin_blueprints:
            raise ValueError(f"Unknown blueprint: {blueprint_name}")

        # Start with the basic blueprint definition
        blueprint = deepcopy(builtin_blueprints[blueprint_name])

        # Add component assignments to each group's attrs
        if "groups" in blueprint:
            for group_name, group_def in blueprint["groups"].items():
                if "attrs" not in group_def:
                    group_def["attrs"] = {}

                # Determine the role from existing attrs or group name
                role = group_def["attrs"].get("role")
                if not role:
                    # Try to infer role from group name
                    role = group_name.lower()

                # Get component assignment by role only
                assignment = getattr(assignments, role, None)

                if assignment:
                    if assignment.hw_component:
                        group_def["attrs"]["hw_component"] = assignment.hw_component
                    # Note: optics assignment will be handled at link level

        result[blueprint_name] = blueprint

    return result


def _build_failure_policy_set_section(config: TopologyConfig) -> dict[str, Any]:
    """Build the failure_policy_set section of the NetGraph scenario.

    Args:
        config: The topology configuration.

    Returns:
        Dictionary containing failure policy definitions.
    """
    # Get merged built-in + user failure policies
    builtin_policies = get_builtin_failure_policies()

    policies: dict[str, Any] = {}

    # Always include the configured default failure policy
    default_policy_name = config.failure_policies.assignments.default
    if default_policy_name in builtin_policies:
        policies[default_policy_name] = builtin_policies[default_policy_name]
    else:
        available = list(builtin_policies.keys())
        raise ValueError(
            f"Default failure policy '{default_policy_name}' not found. "
            f"Available built-in policies: {available}"
        )

    # Optionally include any policies referenced by the selected workflow steps.
    # Be tolerant when workflow configuration is not provided by the caller.
    workflows_cfg = getattr(config, "workflows", None)
    if (
        workflows_cfg is not None
        and getattr(workflows_cfg, "assignments", None) is not None
    ):
        try:
            from topogen.workflows_lib import get_builtin_workflows  # import lazily

            workflows = get_builtin_workflows()
            workflow_name = workflows_cfg.assignments.default
            steps = workflows.get(workflow_name, [])
            for step in steps:
                policy_name = step.get("failure_policy")
                if not policy_name:
                    continue
                if policy_name not in builtin_policies:
                    available = list(builtin_policies.keys())
                    raise ValueError(
                        f"Workflow '{workflow_name}' references unknown failure policy "
                        f"'{policy_name}'. Available built-in policies: {available}"
                    )
                policies[policy_name] = builtin_policies[policy_name]
        except Exception:
            # If workflow library cannot be loaded or structure unexpected, skip enrichment.
            pass

    return policies


def _build_workflow_section(config: TopologyConfig) -> list[dict[str, Any]]:
    """Build the workflow section of the NetGraph scenario.

    Args:
        config: The topology configuration.

    Returns:
        List of workflow step definitions.
    """
    # Get merged workflows (built-ins + user lib)
    builtin_workflows = get_builtin_workflows()

    # Get the default workflow name
    default_workflow_name = config.workflows.assignments.default

    # Use merged library
    if default_workflow_name in builtin_workflows:
        return builtin_workflows[default_workflow_name]

    # Workflow not found
    available_builtin = list(builtin_workflows.keys())
    raise ValueError(
        f"Default workflow '{default_workflow_name}' not found. "
        f"Available built-in workflows: {available_builtin}"
    )
