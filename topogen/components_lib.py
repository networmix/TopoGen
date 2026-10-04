"""Built-in hardware with overrides from ``cwd/lib/components.yml``.

The YAML file maps names to definitions; each entry replaces the matching built-in.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from topogen.library_io import load_user_library

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


def get_builtin_components() -> dict[str, dict[str, Any]]:
    """Return a copy of built-in components with user overrides applied."""
    components = deepcopy(_BUILTIN_COMPONENTS)
    user_components = load_user_library("components.yml")
    components.update(user_components)
    return components
