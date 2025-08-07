"""NetGraph scenario builder for topology generation.

Transforms integrated metro-highway graphs into complete NetGraph YAML scenarios
by expanding metro nodes into detailed site hierarchies using blueprint templates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import yaml

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.components_lib import get_builtin_components
from topogen.failure_policies_lib import get_builtin_failure_policies
from topogen.log_config import get_logger
from topogen.workflows_lib import get_builtin_workflows

if TYPE_CHECKING:
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def build_scenario(
    graph: nx.Graph,
    config: TopologyConfig,
) -> str:
    """Build a NetGraph scenario YAML from an integrated metro-highway graph.

    Transforms each metro node into a hierarchical site structure using
    blueprint templates, preserving corridor connectivity between metros.

    Args:
        graph: Corridor-level graph whose nodes are metros and whose edges are
            metro-to-metro corridors. If a full integrated graph (metros +
            highway + anchor edges) is provided, inter-metro adjacency will be
            empty because this function extracts corridor edges only between
            metro nodes. To build from a full graph, first call
            ``extract_corridor_graph``.
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
        (settings["sites_per_metro"] for settings in metro_settings.values()), default=1
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

    # 1. Blueprints section
    scenario["blueprints"] = _build_blueprints_section(used_blueprints, config)

    # 2. Components section
    scenario["components"] = _build_components_section(config, used_blueprints)

    # 3. Network section
    scenario["network"] = _build_network_section(
        metros, metro_settings, max_sites, max_dc_regions, graph
    )

    # 4. Risk groups section (if risk groups are present)
    risk_groups = _build_risk_groups_section(graph, config)
    if risk_groups:
        scenario["risk_groups"] = risk_groups

    # 5. Failure policy set section
    scenario["failure_policy_set"] = _build_failure_policy_set_section(config)

    # 6. Workflow section
    scenario["workflow"] = _build_workflow_section(config)

    # Convert to YAML with custom formatting for comments
    yaml_output = yaml.safe_dump(scenario, sort_keys=False, default_flow_style=False)

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
            required_attrs = ["name", "metro_id"]
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
                    "radius_km": data.get("radius_km", 50.0),
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

    # Validate override metro names with flexible matching (only if metros are present)
    if metros:  # Only validate if there are metros to check against
        metro_names = {metro["name"] for metro in metros}
        metro_names_orig = {metro.get("name_orig", metro["name"]) for metro in metros}

        for override_name in overrides:
            # Check for exact match (case-insensitive) in sanitized names
            exact_match = any(
                metro_name.lower() == override_name.lower()
                for metro_name in metro_names
            )

            # Check for exact match (case-insensitive) in original names
            exact_match_orig = any(
                metro_name.lower() == override_name.lower()
                for metro_name in metro_names_orig
            )

            # Check for substring match (case-insensitive) in sanitized names
            substring_match = any(
                override_name.lower() in metro_name.lower()
                for metro_name in metro_names
            )

            # Check for substring match (case-insensitive) in original names
            substring_match_orig = any(
                override_name.lower() in metro_name.lower()
                for metro_name in metro_names_orig
            )

            if not (
                exact_match
                or exact_match_orig
                or substring_match
                or substring_match_orig
            ):
                available_sanitized = ", ".join(sorted(metro_names))
                available_orig = ", ".join(sorted(metro_names_orig))
                raise ValueError(
                    f"Build override references unknown metro '{override_name}'. "
                    f"Available metros (sanitized): {available_sanitized}. "
                    f"Available metros (original): {available_orig}"
                )

    # Build settings for each metro
    settings = {}
    for metro in metros:
        metro_name = metro["name"]

        # Start with defaults
        metro_settings = {
            "sites_per_metro": defaults.sites_per_metro,
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

        # Apply overrides if present (flexible matching)
        metro_name_orig = metro.get("name_orig", metro_name)
        override = None

        # Find matching override with flexible matching
        for override_name, override_config in overrides.items():
            # Check for exact match (case-insensitive) in sanitized names
            if metro_name.lower() == override_name.lower():
                override = override_config
                break
            # Check for exact match (case-insensitive) in original names
            elif metro_name_orig.lower() == override_name.lower():
                override = override_config
                break
            # Check for substring match (case-insensitive) in sanitized names
            elif override_name.lower() in metro_name.lower():
                override = override_config
                break
            # Check for substring match (case-insensitive) in original names
            elif override_name.lower() in metro_name_orig.lower():
                override = override_config
                break

        if override:
            if "sites_per_metro" in override:
                metro_settings["sites_per_metro"] = override["sites_per_metro"]
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
        if metro_settings["sites_per_metro"] < 1:
            raise ValueError(
                f"Metro '{metro_name}' has invalid sites_per_metro: {metro_settings['sites_per_metro']}"
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
        metros, metro_settings, max_sites, max_dc_regions
    )

    # Build adjacency section
    network["adjacency"] = _build_adjacency_section(metros, metro_settings, graph)

    return network


def _build_groups_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    max_sites: int,
    max_dc_regions: int,
) -> dict[str, Any]:
    """Build the groups section defining site hierarchies.

    Args:
        metros: List of metro dictionaries.
        metro_settings: Per-metro configuration settings.
        max_sites: Maximum number of sites to support.
        max_dc_regions: Maximum number of DC regions to support.

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
                },
            }

    return groups


def _build_adjacency_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    graph: nx.Graph,
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

    # Build metro index mapping
    metro_by_node = {metro["node_key"]: metro for metro in metros}
    metro_idx_map = {metro["name"]: idx for idx, metro in enumerate(metros, 1)}

    # Add intra-metro adjacency (full mesh within each metro's sites)
    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        sites_count = settings["sites_per_metro"]

        if sites_count > 1:
            # Create mesh connectivity between sites within metro
            adjacency.append(
                {
                    "source": f"/metro{idx}/pop[1-{sites_count}]",
                    "target": f"/metro{idx}/pop[1-{sites_count}]",
                    "pattern": "mesh",
                    "link_params": {
                        "capacity": settings["intra_metro_link"]["capacity"],
                        "cost": settings["intra_metro_link"]["cost"],
                        "attrs": {
                            **settings["intra_metro_link"][
                                "attrs"
                            ],  # Start with configured attrs
                            "metro_name": metro_name,  # Add metro-specific attrs
                            "metro_name_orig": metro.get("name_orig", metro_name),
                        },
                    },
                }
            )

    # Add DC-to-PoP connectivity (DC regions connect to all local PoPs)
    for idx, metro in enumerate(metros, 1):
        metro_name = metro["name"]
        settings = metro_settings[metro_name]
        sites_count = settings["sites_per_metro"]
        dc_regions_count = settings["dc_regions_per_metro"]

        if dc_regions_count > 0 and sites_count > 0:
            # Connect all DC regions to all PoPs in the same metro
            adjacency.append(
                {
                    "source": f"/metro{idx}/dc[1-{dc_regions_count}]",
                    "target": f"/metro{idx}/pop[1-{sites_count}]",
                    "pattern": "mesh",
                    "link_params": {
                        "capacity": settings["dc_to_pop_link"]["capacity"],
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

        source_sites = metro_settings[source_metro["name"]]["sites_per_metro"]
        target_sites = metro_settings[target_metro["name"]]["sites_per_metro"]

        # Build link_params with risk groups if present
        source_settings = metro_settings[source_metro["name"]]

        # Use source metro's inter_metro_link settings for defaults
        default_capacity = source_settings["inter_metro_link"]["capacity"]
        base_cost = source_settings["inter_metro_link"]["cost"]

        link_params = {
            "capacity": edge.get("capacity", default_capacity),
            "cost": round(edge.get("length_km", base_cost)),
            "attrs": {
                **source_settings["inter_metro_link"][
                    "attrs"
                ],  # Start with configured attrs
                "distance_km": edge.get("length_km", 0.0),
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

        # Connect all sites between metros using expand_vars
        adjacency.append(
            {
                "source": f"metro{source_idx}/pop{{src_idx}}",
                "target": f"metro{target_idx}/pop{{tgt_idx}}",
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
            corridor_edges.append(
                {
                    "source": source,
                    "target": target,
                    "length_km": data.get("length_km", 0.0),
                    "capacity": data.get("capacity", 100),
                    "edge_type": data.get("edge_type", "corridor"),
                    "risk_groups": data.get("risk_groups", []),
                }
            )

    return corridor_edges


def _build_risk_groups_section(
    graph: nx.Graph, config: TopologyConfig
) -> list[dict[str, Any]]:
    """Build the risk_groups section of the NetGraph scenario.

    Extracts unique risk groups from corridor edges and creates risk group
    definitions for the scenario.

    Args:
        graph: Integrated graph containing corridor edges with risk groups.
        config: Topology configuration.

    Returns:
        List of risk group definitions, or empty list if none found.
    """
    if not config.corridors.risk_groups.enabled:
        return []

    # Collect all unique risk groups from corridor edges
    risk_group_names = set()

    for source, target, data in graph.edges(data=True):
        # Check if this edge represents a corridor between metros
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]

        # Only process edges between metro nodes
        if source_data.get("node_type") in [
            "metro",
            "metro+highway",
        ] and target_data.get("node_type") in ["metro", "metro+highway"]:
            edge_risk_groups = data.get("risk_groups", [])
            risk_group_names.update(edge_risk_groups)

    if not risk_group_names:
        logger.info("No risk groups found in corridor edges")
        return []

    # Create risk group definitions
    risk_groups = []
    for rg_name in sorted(risk_group_names):
        risk_groups.append(
            {
                "name": rg_name,
                "attrs": {
                    "type": "corridor_risk",
                    "auto_generated": True,
                },
            }
        )

    logger.info(f"Generated {len(risk_groups)} risk group definitions")
    return risk_groups


def _build_components_section(
    config: TopologyConfig, used_blueprints: set[str]
) -> dict[str, Any]:
    """Build the components section of the NetGraph scenario.

    Merges built-in components with custom components from configuration,
    and includes only components that are referenced by the used blueprints.

    Args:
        config: Topology configuration containing component definitions.
        used_blueprints: Set of blueprint names used in the scenario.

    Returns:
        Dictionary representing the components section.
    """
    # Start with built-in components
    builtin_components = get_builtin_components()

    # Merge with custom components from config (custom components override built-ins)
    components = builtin_components.copy()
    components.update(config.components.library)

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

    # Add components referenced by blueprint overrides
    for blueprint_name in used_blueprints:
        if blueprint_name in assignments.blueprint_overrides:
            for role_assignment in assignments.blueprint_overrides[
                blueprint_name
            ].values():
                if role_assignment.hw_component:
                    referenced_components.add(role_assignment.hw_component)

    # Add optics components
    if assignments.spine.optics:
        referenced_components.add(assignments.spine.optics)
    if assignments.leaf.optics:
        referenced_components.add(assignments.leaf.optics)
    if assignments.core.optics:
        referenced_components.add(assignments.core.optics)
    if assignments.dc.optics:
        referenced_components.add(assignments.dc.optics)

    for blueprint_name in used_blueprints:
        if blueprint_name in assignments.blueprint_overrides:
            for role_assignment in assignments.blueprint_overrides[
                blueprint_name
            ].values():
                if role_assignment.optics:
                    referenced_components.add(role_assignment.optics)

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
    assignments based on the configuration.

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

        # Check if this blueprint has component overrides
        if blueprint_name in assignments.blueprint_overrides:
            blueprint_overrides = assignments.blueprint_overrides[blueprint_name]
        else:
            blueprint_overrides = {}

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

                # Get component assignment (blueprint override takes precedence)
                if role in blueprint_overrides:
                    assignment = blueprint_overrides[role]
                else:
                    # Use default role assignment
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
    # Get built-in failure policies
    builtin_policies = get_builtin_failure_policies()

    # Start with custom policies from config
    policies = config.failure_policies.library.copy()

    # Get the default policy name
    default_policy_name = config.failure_policies.assignments.default

    # Add the default policy if not already in custom policies
    if default_policy_name not in policies and default_policy_name in builtin_policies:
        policies[default_policy_name] = builtin_policies[default_policy_name]

    # Validate the default policy exists
    if default_policy_name not in policies:
        available = list(builtin_policies.keys())
        raise ValueError(
            f"Default failure policy '{default_policy_name}' not found. "
            f"Available built-in policies: {available}"
        )

    return policies


def _build_workflow_section(config: TopologyConfig) -> list[dict[str, Any]]:
    """Build the workflow section of the NetGraph scenario.

    Args:
        config: The topology configuration.

    Returns:
        List of workflow step definitions.
    """
    # Get built-in workflows
    builtin_workflows = get_builtin_workflows()

    # Get the default workflow name
    default_workflow_name = config.workflows.assignments.default

    # Check for custom workflow first
    if default_workflow_name in config.workflows.library:
        return config.workflows.library[default_workflow_name]

    # Use built-in workflow
    if default_workflow_name in builtin_workflows:
        return builtin_workflows[default_workflow_name]

    # Workflow not found
    available_custom = list(config.workflows.library.keys())
    available_builtin = list(builtin_workflows.keys())
    raise ValueError(
        f"Default workflow '{default_workflow_name}' not found. "
        f"Available custom workflows: {available_custom}, "
        f"Available built-in workflows: {available_builtin}"
    )
