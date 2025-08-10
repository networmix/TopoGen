"""Built-in component library.

Provides a minimal collection of standard hardware components referenced by
the scenario pipeline and merges overrides from ``cwd/lib/components.yml``
when present. The user file must be direct mapping: name -> definition.
User entries override built-ins.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

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


def _load_user_library(file_name: str) -> dict[str, Any]:
    """Load a user library YAML mapping from ``lib/<file_name>`` if present.

    Args:
        file_name: YAML file name inside the ``lib`` directory.

    Returns:
        A dictionary parsed from the YAML file, or an empty dict when the file
        does not exist.

    Raises:
        ValueError: If the YAML exists but is invalid or not a mapping.
    """
    lib_path = Path.cwd() / "lib" / file_name
    if not lib_path.exists():
        return {}

    try:
        with lib_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001 - provide clear context
        raise ValueError(f"Failed to parse YAML: {lib_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"User library YAML must be a mapping: {lib_path}")

    return data


def get_builtin_components() -> dict[str, dict[str, Any]]:
    """Return the component library merged with user overrides.

    The result is a deep copy of built-ins, updated with entries from
    ``lib/components.yml`` in the current working directory if present.

    Returns:
        Dictionary mapping component names to their definitions.
    """
    components = deepcopy(_BUILTIN_COMPONENTS)
    user_components = _load_user_library("components.yml")
    # Support only direct mapping: name -> definition
    components.update(user_components)
    return components


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
