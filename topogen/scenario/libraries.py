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
    # Streamlined: include platforms from hw_component mapping and optics from role-pair mapping
    role_to_platform = getattr(config.components, "hw_component", {}) or {}
    optics_map = getattr(config.components, "optics", {}) or {}
    if isinstance(role_to_platform, dict):
        for v in role_to_platform.values():
            if isinstance(v, str) and v:
                referenced_components.add(v)
    if isinstance(optics_map, dict):
        for v in optics_map.values():
            if isinstance(v, str) and v:
                referenced_components.add(v)
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
    # Streamlined: use only role->platform mapping
    role_to_platform = getattr(config.components, "hw_component", {}) or {}
    if not isinstance(role_to_platform, dict):
        role_to_platform = {}
    result: dict[str, Any] = {}
    total_groups = 0
    groups_with_role = 0
    groups_with_hw = 0
    for blueprint_name in sorted(used_blueprints):
        if blueprint_name not in builtin_blueprints:
            raise ValueError(f"Unknown blueprint: {blueprint_name}")
        blueprint = deepcopy(builtin_blueprints[blueprint_name])
        if "groups" in blueprint:
            for group_name, group_def in blueprint["groups"].items():
                total_groups += 1
                if "attrs" not in group_def:
                    group_def["attrs"] = {}
                role = group_def["attrs"].get("role")
                if not isinstance(role, str) or not role:
                    raise ValueError(
                        f"Blueprint '{blueprint_name}' group '{group_name}' is missing required 'role' attribute"
                    )
                groups_with_role += 1
                hw_name = role_to_platform.get(role, "")
                if hw_name:
                    # Keep legacy field for tests/back-compat within TopoGen repo
                    group_def["attrs"]["hw_component"] = hw_name
                    # Emit ngraph-compatible node hardware mapping as well
                    group_def["attrs"]["hardware"] = {
                        "component": hw_name,
                        "count": 1,
                    }
                    groups_with_hw += 1
                    logger.info(
                        "HW: node blueprint=%s group=%s role=%s platform=%s",
                        blueprint_name,
                        group_name,
                        role,
                        hw_name,
                    )
        result[blueprint_name] = blueprint
    try:
        logger.info(
            "Node hardware assigned for %d of %d blueprint groups (with_role=%d)",
            groups_with_hw,
            total_groups,
            groups_with_role,
        )
    except Exception:
        pass
    return result
