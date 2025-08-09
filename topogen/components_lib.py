"""Built-in component library for TopoGen.

This module provides a collection of standard hardware components that can be used
in network scenarios. Components include routers, switches, optics, and other
network hardware with realistic cost, power, and capacity characteristics.
"""

from typing import Any

# Built-in component library
_BUILTIN_COMPONENTS = {
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
    "EdgeRouter": {
        "component_type": "chassis",
        "description": "Edge router platform",
        "cost": 150_000.0,
        "power_watts": 6_000.0,
        "power_watts_max": 8_000.0,
        "capacity": 51_200.0,
        "ports": 128,
        "attrs": {"role": "core"},
    },
    "SpineChassis": {
        "component_type": "chassis",
        "description": "High-capacity spine router chassis",
        "cost": 65_000.0,
        "power_watts": 3_200.0,
        "power_watts_max": 4_000.0,
        "capacity": 25_600.0,
        "ports": 64,
        "attrs": {"role": "spine"},
    },
    "HighEndSpineChassis": {
        "component_type": "chassis",
        "description": "High-end spine router chassis",
        "cost": 120_000.0,
        "power_watts": 5_000.0,
        "power_watts_max": 6_500.0,
        "capacity": 51_200.0,
        "ports": 128,
        "attrs": {"role": "spine"},
    },
    "LeafChassis": {
        "component_type": "chassis",
        "description": "Leaf router chassis",
        "cost": 35_000.0,
        "power_watts": 1_600.0,
        "power_watts_max": 2_000.0,
        "capacity": 12_800.0,
        "ports": 64,
        "attrs": {"role": "leaf"},
    },
    "HighEndLeafChassis": {
        "component_type": "chassis",
        "description": "High-end leaf router chassis",
        "cost": 70_000.0,
        "power_watts": 2_400.0,
        "power_watts_max": 3_000.0,
        "capacity": 25_600.0,
        "ports": 64,
        "attrs": {"role": "leaf"},
    },
    "DCNode": {
        "component_type": "chassis",
        "description": "Data center node",
        "cost": 50_000.0,
        "power_watts": 1_000.0,
        "power_watts_max": 1_500.0,
        "capacity": 12_800.0,
        "ports": 32,
        "attrs": {"role": "dc"},
    },
    # Optical Transceivers
    "100G-LR4": {
        "component_type": "optic",
        "description": "100G LR4 optic",
        "cost": 1_000.0,
        "power_watts": 5.0,
        "power_watts_max": 6.0,
        "capacity": 100.0,
        "ports": 1,
        "attrs": {},
    },
    "400G-LR4": {
        "component_type": "optic",
        "description": "400G LR4 optic",
        "cost": 3_000.0,
        "power_watts": 12.0,
        "power_watts_max": 15.0,
        "capacity": 400.0,
        "ports": 1,
        "attrs": {},
    },
    "400G-SR8": {
        "component_type": "optic",
        "description": "400G SR8 optic",
        "cost": 2_500.0,
        "power_watts": 10.0,
        "power_watts_max": 12.0,
        "capacity": 400.0,
        "ports": 1,
        "attrs": {},
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
    return _BUILTIN_COMPONENTS.copy()


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

    return _BUILTIN_COMPONENTS[name].copy()


def list_builtin_component_names() -> list[str]:
    """Get a list of all built-in component names.

    Returns:
        List of component names.
    """
    return list(_BUILTIN_COMPONENTS.keys())


def get_components_by_type(component_type: str) -> dict[str, dict[str, Any]]:
    """Get all components of a specific type.

    Args:
        component_type: Type of components to retrieve (e.g., "chassis", "optic").

    Returns:
        Dictionary of components matching the specified type.
    """
    return {
        name: comp
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
        name: comp
        for name, comp in _BUILTIN_COMPONENTS.items()
        if comp.get("attrs", {}).get("role") == role
    }
