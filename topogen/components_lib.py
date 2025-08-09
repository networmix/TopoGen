"""Built-in component library for TopoGen.

This module provides a minimal collection of standard hardware components
referenced by the scenario pipeline.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Built-in component library
_BUILTIN_COMPONENTS: dict[str, dict[str, Any]] = {
    # Router Chassis Components
    "CoreRouter": {
        "component_type": "chassis",
        "description": "Core router for metro backbone",
        "cost": 650_000.0,
        "power_watts": 40_000.0,
        "power_watts_max": 50_000.0,
        "capacity": 460_800.0,  # Gbps
        "ports": 576,
        "attrs": {"role": "core"},
    },
    "800G-ZR+": {
        "component_type": "optic",
        "description": "800G ZR+ pluggable optic",
        "cost": 10_000.0,
        "power_watts": 30.0,
        "power_watts_max": 30.0,
        "capacity": 800.0,  # Gbps
        "ports": 1,
        "attrs": {},
    },
}


def get_builtin_components() -> dict[str, dict[str, Any]]:
    """Get the complete built-in component library.

    Returns:
        Dictionary mapping component names to their definitions.
    """
    return deepcopy(_BUILTIN_COMPONENTS)


def get_builtin_component(name: str) -> dict[str, Any]:
    """Get a specific built-in component by name.

    Args:
        name: Name of the component to retrieve.

    Returns:
        Component definition dictionary.

    Raises:
        KeyError: If the component name is not found.
    """
    if name not in _BUILTIN_COMPONENTS:
        available = list(_BUILTIN_COMPONENTS.keys())
        raise KeyError(f"Component '{name}' not found. Available: {available}")

    return deepcopy(_BUILTIN_COMPONENTS[name])


def list_builtin_component_names() -> list[str]:
    """Get a list of all built-in component names.

    Returns:
        List of component names.
    """
    return sorted(_BUILTIN_COMPONENTS.keys())


def get_components_by_type(component_type: str) -> dict[str, dict[str, Any]]:
    """Get all components of a specific type.

    Args:
        component_type: Type of components to retrieve (e.g., "chassis", "optic").

    Returns:
        Dictionary of components matching the specified type.
    """
    return {
        name: deepcopy(comp)
        for name, comp in _BUILTIN_COMPONENTS.items()
        if comp.get("component_type") == component_type
    }


def get_components_by_role(role: str) -> dict[str, dict[str, Any]]:
    """Get all components that support a specific role.

    Args:
        role: Role to match (e.g., "spine", "leaf", "core").

    Returns:
        Dictionary of components that can fulfill the specified role.
    """
    return {
        name: deepcopy(comp)
        for name, comp in _BUILTIN_COMPONENTS.items()
        if comp.get("attrs", {}).get("role") == role
    }
