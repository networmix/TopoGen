"""Write network and complete scenario documents from immutable build records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import yaml

from netlab_topogen.config import ComponentsConfig, TopologyConfig
from netlab_topogen.corridors import corridor_risk_name
from netlab_topogen.library import Library, thaw
from netlab_topogen.naming import site_selector

from .records import CorridorGraph, LinkHardware, SitePlan, Sizing, TrafficMatrix


def component_names(
    components: ComponentsConfig, blueprints: Mapping[str, Any]
) -> tuple[str, ...]:
    """Collect references after role overrides, including blueprint-only hardware."""
    referenced_components = set(components.hw_component.values()) | set(
        components.optics.values()
    )

    def references(value):
        if isinstance(value, Mapping):
            if "component" in value:
                referenced_components.add(value["component"])
            for child in value.values():
                references(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                references(child)

    references(blueprints)
    referenced_components.discard("")  # Explicitly unassigned roles have no component.
    return tuple(sorted(referenced_components))


def check_components(
    components: ComponentsConfig, blueprints: Mapping[str, Any], library: Library
) -> None:
    """Refuse every unknown component together before hardware assignment."""
    missing = set(component_names(components, blueprints)) - library.components.keys()
    if missing:
        raise ValueError(f"Unknown components: {sorted(missing)}")


def _components(
    config: "TopologyConfig", blueprints: Mapping[str, Any], library: Library
) -> dict[str, Any]:
    """Components the configuration and the finalized blueprints reference, from the given library."""
    return {
        name: thaw(library.components[name])
        for name in component_names(config.components, blueprints)
    }


def _failure_policies(
    config: "TopologyConfig", workflow: list[dict[str, Any]], library: Library
) -> dict[str, Any]:
    """Build the ``failures`` section of the scenario."""
    policies = library.failure_policies
    result: dict[str, Any] = {}

    default_policy_name = config.failure_policies.assignments.default
    if default_policy_name in policies:
        result[default_policy_name] = thaw(policies[default_policy_name])
    else:
        available = list(policies.keys())
        raise ValueError(
            f"Default failure policy '{default_policy_name}' not found. Available failure policies: {available}"
        )

    workflow_name = config.workflows.assignments.default
    for step in workflow:
        policy_name = step.get("failure_policy")
        if policy_name is None:
            continue
        if policy_name not in policies:
            raise ValueError(
                f"Workflow '{workflow_name}' references unknown failure policy '{policy_name}'"
            )
        result[policy_name] = thaw(policies[policy_name])
    return result


def _workflow(config: "TopologyConfig", library: Library) -> list[dict[str, Any]]:
    name = config.workflows.assignments.default
    if name not in library.workflows:
        raise ValueError(
            f"Default workflow '{name}' not found. Available workflows: {list(library.workflows)}"
        )
    definition = library.workflows[name]
    steps = thaw(definition.steps)
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError(f"Workflow '{name}' must contain a list of step mappings")
    if definition.follows_configured_matrix:
        for step in steps:
            if "demand_set" in step:
                step["demand_set"] = config.traffic.matrix_name
    return steps


def to_demand_sets(
    matrices: tuple[TrafficMatrix, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Write selectors only at the scenario document boundary."""
    result = {}
    for matrix in matrices:
        entries = []
        for demand in matrix.demands:
            entry = {
                "source": site_selector(demand.source),
                "target": site_selector(demand.target),
                "mode": "pairwise",
                "priority": demand.priority,
                "volume": float(demand.volume),
            }
            if demand.euclidean_km is not None:
                entry["attrs"] = {"euclidean_km": demand.euclidean_km}
            if demand.preset is not None:
                entry["preset"] = demand.preset
            entries.append(entry)
        result[matrix.name] = entries
    return result


def _risk_groups(
    corridors: CorridorGraph, config: TopologyConfig
) -> list[dict[str, Any]]:
    if not config.corridors.risk_groups.enabled:
        return []
    distances: dict[str, int] = {}
    referenced: set[str] = set()
    for corridor in corridors.corridors:
        name = corridor_risk_name(
            config.corridors.risk_groups.group_prefix,
            corridor.source.name,
            corridor.target.name,
            corridor.path,
        )
        if name in distances:
            raise ValueError(f"Duplicate corridor risk identity: {name}")
        length = float(corridor.length_km)
        if not math.isfinite(length) or length <= 0:
            raise ValueError(f"Corridor '{name}' must have positive finite length")
        distances[name] = math.ceil(length)
        referenced.update(corridor.risk_groups)
    if unknown := referenced - distances.keys():
        raise ValueError(f"Risk groups have no owning corridor path: {sorted(unknown)}")
    return [
        {
            "name": name,
            "attrs": {"type": "corridor_risk", "distance_km": distances[name]},
        }
        for name in sorted(referenced)
    ]


def network_document(
    corridors: CorridorGraph,
    plan: SitePlan,
    blueprints: Mapping[str, Any],
    config: TopologyConfig,
    sizing: Sizing | None = None,
    hardware: Mapping[str, LinkHardware] | None = None,
) -> dict[str, Any]:
    """The common network definition used before and after capacity assignment."""
    groups = {}
    sites = {site.id: site for site in plan.sites}
    for site in plan.sites:
        metro = site.metro
        attrs = {
            "metro_name": metro.name,
            "metro_name_orig": metro.name_orig,
            "metro_id": metro.metro_id,
            "location_x": metro.x,
            "location_y": metro.y,
            "radius_km": metro.radius_km,
            "node_type": "pop" if site.kind == "pop" else "dc_region",
        }
        if site.kind == "dc":
            assert site.power_mw is not None
            attrs.update(
                mw_per_dc_region=float(site.power_mw),
                gbps_per_mw=float(config.traffic.gbps_per_mw),
            )
        groups[site.id] = {"blueprint": site.blueprint, "attrs": attrs}
    links = []
    for link in plan.links:
        budget = sizing.capacity[link.id] if sizing is not None else link.base_capacity
        assigned = hardware[link.id] if hardware is not None else None
        attrs = {
            **thaw(link.attrs),
            "link_type": link.kind,
            "source_metro": sites[link.source].metro.name,
            "target_metro": sites[link.target].metro.name,
            "target_capacity": float(budget),
            "site_edge": link.id,
            "adjacency_id": link.adjacency_id,
            "distance_km": int(link.distance_km),
        }
        if link.euclidean_km is not None:
            attrs["euclidean_km"] = float(link.euclidean_km)
        if link.detour_ratio is not None:
            attrs["detour_ratio"] = float(link.detour_ratio)
        if assigned is not None and assigned.source_optic is not None:
            attrs["hardware"] = {
                "source": {
                    "component": assigned.source_optic,
                    "count": float(assigned.source_count),
                },
                "target": {
                    "component": assigned.target_optic,
                    "count": float(assigned.target_count),
                },
            }
        match = thaw(link.match)
        entry = {
            "source": {"path": link.source, "match": match} if match else link.source,
            "target": {"path": link.target, "match": match} if match else link.target,
            "pattern": "one_to_one",
            "capacity": float(assigned.capacity)
            if assigned is not None
            else float(budget),
            "cost": int(link.cost),
            "attrs": attrs,
        }
        if config.corridors.risk_groups.enabled and link.risk_groups:
            entry["risk_groups"] = list(link.risk_groups)
        links.append(entry)
    network: dict[str, Any] = {"nodes": groups, "links": links}
    if plan.node_rules:
        network["node_rules"] = thaw(plan.node_rules)
    result: dict[str, Any] = {
        "seed": config.output.scenario_seed,
        "blueprints": thaw(blueprints),
        "network": network,
    }
    if risks := _risk_groups(corridors, config):
        result["risk_groups"] = risks
    return result


def scenario_document(
    corridors: CorridorGraph,
    plan: SitePlan,
    blueprints: Mapping[str, Any],
    traffic: tuple[TrafficMatrix, ...],
    sizing: Sizing,
    hardware: Mapping[str, LinkHardware],
    library: Library,
    config: TopologyConfig,
) -> dict[str, Any]:
    network = network_document(corridors, plan, blueprints, config, sizing, hardware)
    result = {
        "seed": network["seed"],
        "blueprints": network["blueprints"],
        "components": _components(config, blueprints, library),
        "network": network["network"],
    }
    if "risk_groups" in network:
        result["risk_groups"] = network["risk_groups"]
    workflow = _workflow(config, library)
    result["failures"] = _failure_policies(config, workflow, library)
    if traffic:
        result["demands"] = to_demand_sets(traffic)
    result["workflow"] = workflow
    return result


def to_yaml(document: Mapping[str, Any], *, yaml_anchors: bool) -> str:
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    return yaml.dump(
        document,
        Dumper=yaml.SafeDumper if yaml_anchors else NoAliasDumper,
        sort_keys=False,
        default_flow_style=False,
    )
