"""Built-in blueprint library.

Provides built-in blueprints referenced by the scenario pipeline and merges
overrides from ``cwd/lib/blueprints.yml`` when present. The user file must be
direct mapping: name -> definition. User entries override built-ins.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# Built-in blueprints used by the scenario pipeline
_BUILTIN_BLUEPRINTS: dict[str, dict[str, Any]] = {
    "SingleRouter": {
        "groups": {
            "core": {
                "node_count": 1,
                "name_template": "core",
                "attrs": {"role": "core", "hw_type": "router_chassis"},
            }
        },
        "adjacency": [],
    },
    "FullMesh4": {
        "groups": {
            "core": {
                "node_count": 4,
                "name_template": "core{node_num}",
                "attrs": {"role": "core", "hw_type": "router_chassis"},
            }
        },
        "adjacency": [
            {
                "source": "/core",
                "target": "/core",
                "pattern": "mesh",
                "link_params": {
                    "capacity": 400,
                    "cost": 1,
                    "attrs": {"link_type": "internal_mesh"},
                },
            }
        ],
    },
    "Clos_16_8": {
        "groups": {
            "spine": {
                "node_count": 8,
                "name_template": "spine{node_num}",
                "attrs": {
                    "role": "spine",
                    "tier": "spine",
                    "hw_type": "spine_chassis",
                },
            },
            "leaf": {
                "node_count": 16,
                "name_template": "leaf{node_num}",
                "attrs": {"role": "leaf", "tier": "leaf", "hw_type": "router_chassis"},
            },
        },
        "adjacency": [
            {
                "source": "/leaf",
                "target": "/spine",
                "pattern": "mesh",
                "link_params": {
                    "capacity": 3_200,
                    "cost": 1,
                    "attrs": {"link_type": "leaf_spine"},
                },
            }
        ],
    },
    "DCRegion": {
        "groups": {
            "dc": {
                "node_count": 1,
                "name_template": "dc",
                "attrs": {"role": "dc", "hw_type": "dc_node"},
            }
        },
        "adjacency": [],
    },
}


def _load_user_library(file_name: str) -> dict[str, Any]:
    """Load user blueprint library from ``lib/<file_name>`` if present.

    Args:
        file_name: YAML file name inside ``lib``.

    Returns:
        Mapping parsed from YAML, or empty dict if the file is missing.
    """
    lib_path = Path.cwd() / "lib" / file_name
    if not lib_path.exists():
        return {}

    try:
        with lib_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse YAML: {lib_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"User library YAML must be a mapping: {lib_path}")

    return data


def get_builtin_blueprints() -> dict[str, dict[str, Any]]:
    """Return blueprint library merged with user overrides.

    Returns:
        Dictionary mapping blueprint names to their definitions.
    """
    blueprints = deepcopy(_BUILTIN_BLUEPRINTS)
    user_blueprints = _load_user_library("blueprints.yml")
    # Support only direct mapping: name -> definition
    blueprints.update(user_blueprints)
    return blueprints
