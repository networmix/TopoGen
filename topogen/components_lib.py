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
        "cost": 45000.0,
        "power_watts": 2200.0,
        "power_watts_max": 2500.0,
        "capacity": 25600.0,  # Gbps
        "ports": 32,
        "attrs": {
            "role": "core",
            "form_factor": "2RU",
            "vendor": "Generic",
            "model": "CR-2500",
        },
    },
    "EdgeRouter": {
        "component_type": "chassis",
        "description": "Edge router for single-site deployments",
        "cost": 15000.0,
        "power_watts": 800.0,
        "power_watts_max": 1000.0,
        "capacity": 6400.0,  # Gbps
        "ports": 16,
        "attrs": {
            "role": "edge",
            "form_factor": "1RU",
            "vendor": "Generic",
            "model": "ER-800",
        },
    },
    "SpineChassis": {
        "component_type": "chassis",
        "description": "High-capacity spine router chassis",
        "cost": 65000.0,
        "power_watts": 3200.0,
        "power_watts_max": 3800.0,
        "capacity": 51200.0,  # Gbps
        "ports": 64,
        "attrs": {
            "role": "spine",
            "form_factor": "4RU",
            "vendor": "Generic",
            "model": "SC-5000",
        },
    },
    "HighEndSpineChassis": {
        "component_type": "chassis",
        "description": "Ultra high-capacity spine chassis for large deployments",
        "cost": 120000.0,
        "power_watts": 5500.0,
        "power_watts_max": 6200.0,
        "capacity": 102400.0,  # Gbps
        "ports": 128,
        "attrs": {
            "role": "spine",
            "form_factor": "8RU",
            "vendor": "Generic",
            "model": "HSC-10000",
        },
    },
    "LeafChassis": {
        "component_type": "chassis",
        "description": "Leaf switch chassis for ToR deployments",
        "cost": 25000.0,
        "power_watts": 1500.0,
        "power_watts_max": 1800.0,
        "capacity": 25600.0,  # Gbps
        "ports": 48,
        "attrs": {
            "role": "leaf",
            "form_factor": "2RU",
            "vendor": "Generic",
            "model": "LC-2500",
        },
    },
    "HighEndLeafChassis": {
        "component_type": "chassis",
        "description": "High-capacity leaf chassis for dense deployments",
        "cost": 45000.0,
        "power_watts": 2400.0,
        "power_watts_max": 2800.0,
        "capacity": 51200.0,  # Gbps
        "ports": 64,
        "attrs": {
            "role": "leaf",
            "form_factor": "3RU",
            "vendor": "Generic",
            "model": "HLC-5000",
        },
    },
    # Optical Transceivers
    "100G-LR4": {
        "component_type": "optic",
        "description": "100G LR4 pluggable optic",
        "cost": 1200.0,
        "power_watts": 3.5,
        "power_watts_max": 4.0,
        "capacity": 100.0,  # Gbps
        "ports": 1,
        "attrs": {
            "reach": "10km",
            "wavelength": "1310nm",
            "form_factor": "QSFP28",
            "fiber_type": "SMF",
        },
    },
    "400G-LR4": {
        "component_type": "optic",
        "description": "400G LR4 pluggable optic",
        "cost": 3500.0,
        "power_watts": 12.0,
        "power_watts_max": 14.0,
        "capacity": 400.0,  # Gbps
        "ports": 1,
        "attrs": {
            "reach": "10km",
            "wavelength": "1310nm",
            "form_factor": "QSFP-DD",
            "fiber_type": "SMF",
        },
    },
    "400G-SR8": {
        "component_type": "optic",
        "description": "400G SR8 pluggable optic",
        "cost": 2200.0,
        "power_watts": 8.0,
        "power_watts_max": 10.0,
        "capacity": 400.0,  # Gbps
        "ports": 1,
        "attrs": {
            "reach": "100m",
            "wavelength": "850nm",
            "form_factor": "QSFP-DD",
            "fiber_type": "MMF",
        },
    },
    "800G-LR": {
        "component_type": "optic",
        "description": "800G LR pluggable optic",
        "cost": 8500.0,
        "power_watts": 20.0,
        "power_watts_max": 24.0,
        "capacity": 800.0,  # Gbps
        "ports": 1,
        "attrs": {
            "reach": "10km",
            "wavelength": "1310nm",
            "form_factor": "OSFP",
            "fiber_type": "SMF",
        },
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
