"""Scenario sections derived from component and blueprint libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.components_lib import get_builtin_components
from topogen.log_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def _build_components_section(
    config: "TopologyConfig", used_blueprints: set[str]
) -> dict[str, Any]:
    """Build the components section of the NetGraph scenario.

    Uses merged component library (built-ins + lib/components.yml) and includes
    only components referenced by assignments.
    """
    components = get_builtin_components()
    referenced_components: set[str] = set()
    assignments = config.components.assignments
    if assignments.spine.hw_component:
        referenced_components.add(assignments.spine.hw_component)
    if assignments.leaf.hw_component:
        referenced_components.add(assignments.leaf.hw_component)
    if assignments.core.hw_component:
        referenced_components.add(assignments.core.hw_component)
    if assignments.dc.hw_component:
        referenced_components.add(assignments.dc.hw_component)
    if assignments.spine.optics:
        referenced_components.add(assignments.spine.optics)
    if assignments.leaf.optics:
        referenced_components.add(assignments.leaf.optics)
    if assignments.core.optics:
        referenced_components.add(assignments.core.optics)
    if assignments.dc.optics:
        referenced_components.add(assignments.dc.optics)
    result: dict[str, Any] = {}
    for comp_name in sorted(referenced_components):
        if comp_name in components:
            result[comp_name] = components[comp_name]
        else:
            logger.warning(
                f"Referenced component '{comp_name}' not found in component library"
            )
    return result


def _build_blueprints_section(
    used_blueprints: set[str], config: "TopologyConfig"
) -> dict[str, Any]:
    """Build the blueprints section with component assignments."""
    from copy import deepcopy

    builtin_blueprints = get_builtin_blueprints()
    assignments = config.components.assignments
    result: dict[str, Any] = {}
    for blueprint_name in sorted(used_blueprints):
        if blueprint_name not in builtin_blueprints:
            raise ValueError(f"Unknown blueprint: {blueprint_name}")
        blueprint = deepcopy(builtin_blueprints[blueprint_name])
        if "groups" in blueprint:
            for group_name, group_def in blueprint["groups"].items():
                if "attrs" not in group_def:
                    group_def["attrs"] = {}
                role = group_def["attrs"].get("role")
                if not role:
                    role = group_name.lower()
                assignment = getattr(assignments, role, None)
                if assignment:
                    if assignment.hw_component:
                        group_def["attrs"]["hw_component"] = assignment.hw_component
        result[blueprint_name] = blueprint
    return result
