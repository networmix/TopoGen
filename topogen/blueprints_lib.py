"""Built-in blueprint library for topology generation.

Provides a collection of standard network blueprints for different site
architectures, from simple single-router sites to complex multi-stage
topologies like Clos fabrics.
"""

from __future__ import annotations

from typing import Any


def get_builtin_blueprint(name: str) -> dict[str, Any]:
    """Retrieve a built-in blueprint by name.

    Args:
        name: Name of the blueprint to retrieve.

    Returns:
        Blueprint definition dictionary.

    Raises:
        ValueError: If the blueprint name is not found.
    """
    blueprints = get_builtin_blueprints()
    if name not in blueprints:
        available = ", ".join(sorted(blueprints.keys()))
        raise ValueError(f"Unknown blueprint '{name}'. Available: {available}")
    return blueprints[name]


def get_builtin_blueprints() -> dict[str, dict[str, Any]]:
    """Get all built-in blueprints.

    Returns:
        Dictionary mapping blueprint names to their definitions.
    """
    return {
        "SingleRouter": _single_router_blueprint(),
        "FullMesh4": _full_mesh_4_blueprint(),
        "Clos_64_256": _clos_64_256_blueprint(),
        "DCRegion": _dc_region_blueprint(),
    }


def _single_router_blueprint() -> dict[str, Any]:
    """Single router blueprint for simple sites.

    Creates a single core router node with no internal connectivity.
    Suitable for small sites or edge locations.

    Returns:
        Blueprint definition for a single router.
    """
    return {
        "groups": {
            "core": {
                "node_count": 1,
                "name_template": "core",
                "attrs": {
                    "role": "core",
                    "hw_type": "router_chassis",
                },
            }
        },
        "adjacency": [],
    }


def _full_mesh_4_blueprint() -> dict[str, Any]:
    """Four-router full mesh blueprint.

    Creates four core routers connected in a full mesh topology.
    Provides high redundancy and multiple paths between any two points.

    Returns:
        Blueprint definition for a 4-router mesh.
    """
    return {
        "groups": {
            "core": {
                "node_count": 4,
                "name_template": "core{node_num}",
                "attrs": {
                    "role": "core",
                    "hw_type": "router_chassis",
                },
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
    }


def _clos_64_256_blueprint() -> dict[str, Any]:
    """2-stage Clos fabric with 64 leaf and 256 spine ports.

    Creates a folded Clos topology with:
    - 8 spine routers (32 ports each = 256 total spine ports)
    - 8 leaf routers (8 ports each = 64 total leaf ports)
    - Each leaf connects to all spines (8x8 = 64 spine-leaf links)

    This provides non-blocking capacity for the leaf tier with
    significant oversubscription capability.

    Returns:
        Blueprint definition for Clos fabric.
    """
    return {
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
    }


def _dc_region_blueprint() -> dict[str, Any]:
    """Data center region blueprint for single-node DC sites.

    Creates a single DC node suitable for data center regions.
    Each DC Region connects to all local PoPs in the same metro.

    Returns:
        Blueprint definition for a DC Region.
    """
    return {
        "groups": {
            "dc": {
                "node_count": 1,
                "name_template": "dc",
                "attrs": {
                    "role": "dc",
                    "hw_type": "dc_node",
                },
            }
        },
        "adjacency": [],
    }
