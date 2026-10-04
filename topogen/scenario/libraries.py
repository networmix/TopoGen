"""Scenario sections derived from component and blueprint libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.components_lib import get_builtin_components

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig


def _build_components_section(
    config: "TopologyConfig", blueprints: dict[str, Any]
) -> dict[str, Any]:
    """Build the components section of the NetGraph scenario.

    Uses merged component library (built-ins + lib/components.yml) and includes
    components referenced by configuration and finalized blueprint assignments.
    Scan after role overrides so replaced hardware does not remain a dependency.
    """
    components = get_builtin_components()
    referenced_components = set(config.components.hw_component.values()) | set(
        config.components.optics.values()
    )

    def references(value):
        if isinstance(value, dict):
            if "component" in value:
                referenced_components.add(value["component"])
            for child in value.values():
                references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)

    references(blueprints)
    referenced_components.discard("")  # Explicitly unassigned roles have no component.
    missing = referenced_components - components.keys()
    if missing:
        raise ValueError(f"Unknown components: {sorted(missing)}")
    result = {name: components[name] for name in sorted(referenced_components)}
    return result


def _build_blueprints_section(
    used_blueprints: set[str], config: "TopologyConfig"
) -> dict[str, Any]:
    """Build the blueprints section with component assignments."""
    from copy import deepcopy

    builtin_blueprints = get_builtin_blueprints()
    role_to_platform = config.components.hw_component
    if not isinstance(role_to_platform, dict):
        raise ValueError("components.hw_component must be a mapping")
    result: dict[str, Any] = {}
    visiting: set[str] = set()

    def include(name: str) -> None:
        if name in visiting:
            raise ValueError(f"Cyclic blueprint reference: {name}")
        if name in result:
            return
        if name not in builtin_blueprints:
            raise ValueError(f"Unknown blueprint: {name}")
        visiting.add(name)
        blueprint = deepcopy(builtin_blueprints[name])
        for group_name, group in blueprint["nodes"].items():
            if "blueprint" in group:
                include(group["blueprint"])
            attrs = group.setdefault("attrs", {})
            role = attrs.get("role")
            if (not isinstance(role, str) or not role) and "blueprint" not in group:
                raise ValueError(
                    f"Blueprint '{name}' group '{group_name}' is missing required 'role' attribute"
                )
            if hardware := role_to_platform.get(role):
                attrs["hardware"] = {"component": hardware, "count": 1}
        visiting.remove(name)
        result[name] = blueprint

    for name in sorted(used_blueprints):
        include(name)
    return result
