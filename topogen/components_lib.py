"""Built-in hardware with overrides from ``cwd/lib/components.yml``.

The YAML file maps names to definitions; each entry replaces the matching built-in.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_BUILTIN_COMPONENTS: dict[str, dict[str, Any]] = {
    # Router Chassis Components
    "CoreRouter": {
        "component_type": "chassis",
        "description": "16 slot, 32x800G ports per slot, 512 ports total",
        "capex": 650_000.0,
        "power_watts": 22_000.0,  # without optics, typical consumption
        "power_watts_max": 29_000.0,  # without optics, max consumption
        "capacity": 409_600.0,  # Gbps
        "ports": 512,
        "attrs": {"role": "core"},
    },
    "LeafRouter": {
        "component_type": "chassis",
        "description": "64x800G ports, Q4D chipset",
        "capex": 85_000.0,
        "power_watts": 2_000.0,  # without optics, typical consumption
        "power_watts_max": 3_000.0,  # without optics, max consumption
        "capacity": 51_200.0,  # Gbps
        "ports": 64,
        "attrs": {"role": "leaf"},
    },
    "SpineRouter": {
        "component_type": "chassis",
        "description": "64x1600G ports, TH6 chipset",
        "capex": 55_000.0,
        "power_watts": 2_000.0,  # without optics, typical consumption
        "power_watts_max": 3_000.0,  # without optics, max consumption
        "capacity": 102_400.0,  # Gbps
        "ports": 64,
        "attrs": {"role": "spine"},
    },
    "800G-ZR+": {
        "component_type": "optic",
        "description": "800G ZR+ pluggable optic",
        "capex": 15_000.0,
        "power_watts": 29.0,
        "power_watts_max": 30.0,
        "capacity": 800.0,  # Gbps
        "ports": 1,
        "attrs": {},
    },
    "1600G-2xDR4": {
        "component_type": "optic",
        "description": "1600G 2xDR4 pluggable optic",
        "capex": 5_500.0,
        "power_watts": 25.0,
        "power_watts_max": 28.0,
        "capacity": 1600.0,  # Gbps
        "ports": 1,
        "attrs": {},
    },
    "800G-DR4": {
        "component_type": "optic",
        "description": "800G DR4 pluggable optic",
        "capex": 3_000.0,
        "power_watts": 16.0,
        "power_watts_max": 18.0,
        "capacity": 800.0,  # Gbps
        "ports": 1,
        "attrs": {},
    },
}


def _load_user_library(file_name: str) -> dict[str, Any]:
    """Read a mapping from ``cwd/lib/<file_name>``, or return {} if absent.

    Raises ValueError for invalid YAML or a non-mapping value.
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
    """Return a copy of built-in components with user overrides applied."""
    components = deepcopy(_BUILTIN_COMPONENTS)
    user_components = _load_user_library("components.yml")
    components.update(user_components)
    return components


def get_builtin_component(name: str) -> dict[str, Any]:
    """Return a component from the merged library; raise KeyError if absent."""
    if name not in _BUILTIN_COMPONENTS:
        available = list(_BUILTIN_COMPONENTS.keys())
        raise KeyError(f"Component '{name}' not found. Available: {available}")

    return deepcopy(_BUILTIN_COMPONENTS[name])


def list_builtin_component_names() -> list[str]:
    """List component names from the merged library."""
    return sorted(_BUILTIN_COMPONENTS.keys())


def get_components_by_type(component_type: str) -> dict[str, dict[str, Any]]:
    """Filter the merged library by component type, such as chassis or optic."""
    return {
        name: deepcopy(comp)
        for name, comp in _BUILTIN_COMPONENTS.items()
        if comp.get("component_type") == component_type
    }


def get_components_by_role(role: str) -> dict[str, dict[str, Any]]:
    """Filter the merged library by supported role."""
    return {
        name: deepcopy(comp)
        for name, comp in _BUILTIN_COMPONENTS.items()
        if comp.get("attrs", {}).get("role") == role
    }
