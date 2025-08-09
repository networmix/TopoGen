"""Built-in blueprint library for topology generation.

Provides a minimal API for retrieving built-in blueprints referenced by the
scenario pipeline.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

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
    "Clos_64_256": {
        "groups": {
            "spine": {
                "node_count": 8,
                "name_template": "spine{node_num}",
                "attrs": {"role": "spine", "hw_type": "spine_chassis", "tier": "spine"},
            },
            "leaf": {
                "node_count": 8,
                "name_template": "leaf{node_num}",
                "attrs": {"role": "leaf", "hw_type": "leaf_chassis", "tier": "leaf"},
            },
        },
        "adjacency": [
            {
                "source": "/leaf",
                "target": "/spine",
                "pattern": "mesh",
                "link_params": {
                    "capacity": 400,
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


def get_builtin_blueprints() -> dict[str, dict[str, Any]]:
    """Return deep-copied built-in blueprint definitions.

    Returns:
        Dictionary mapping blueprint names to their definitions.
    """
    return deepcopy(_BUILTIN_BLUEPRINTS)
