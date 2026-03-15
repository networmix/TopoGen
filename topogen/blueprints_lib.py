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
    "DCRegion": {
        "nodes": {
            "dc": {
                "count": 1,
                "template": "dc",
                "attrs": {"role": "dc"},
            }
        },
        "links": [],
    },
    "SingleRouter": {
        "nodes": {
            "core": {
                "count": 1,
                "template": "core",
                "attrs": {"role": "core"},
            }
        },
        "links": [],
    },
    "FullMesh4": {
        "nodes": {
            "core": {
                "count": 4,
                "template": "core{n}",
                "attrs": {"role": "core"},
            }
        },
        "links": [
            {
                "source": "/core",
                "target": "/core",
                "pattern": "mesh",
                "capacity": 12_800,
                "cost": 1,
                "attrs": {
                    "link_type": "internal_mesh",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 16.0},
                        "target": {"component": "800G-DR4", "count": 16.0},
                    },
                },
            }
        ],
    },
    "Clos_L16_S4": {
        "nodes": {
            "spine": {
                "count": 4,
                "template": "spine{n}",
                "attrs": {
                    "role": "spine",
                    "tier": "spine",
                },
            },
            "leaf": {
                "count": 16,
                "template": "leaf{n}",
                "attrs": {"role": "leaf", "tier": "leaf"},
            },
        },
        "links": [
            {
                "source": "/leaf",
                "target": "/spine",
                "pattern": "mesh",
                "capacity": 3_200,
                "cost": 1,
                "attrs": {
                    "link_type": "leaf_spine",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 4.0},
                        "target": {"component": "1600G-2xDR4", "count": 2.0},
                    },
                },
            }
        ],
    },
    "DragonFly_CustomG4": {
        "nodes": {
            "leafA": {
                "count": 4,
                "template": "leafA{n}",
                "attrs": {"role": "leaf", "tier": "leaf", "group": "A"},
            },
            "leafB": {
                "count": 4,
                "template": "leafB{n}",
                "attrs": {"role": "leaf", "tier": "leaf", "group": "B"},
            },
            "leafC": {
                "count": 4,
                "template": "leafC{n}",
                "attrs": {"role": "leaf", "tier": "leaf", "group": "C"},
            },
            "leafD": {
                "count": 4,
                "template": "leafD{n}",
                "attrs": {"role": "leaf", "tier": "leaf", "group": "D"},
            },
        },
        "links": [
            # Intra-group: dense (4×800G per pair ≈ 3.2 Tb/s)
            {
                "source": "/leafA",
                "target": "/leafA",
                "pattern": "mesh",
                "capacity": 3_200,
                "cost": 1,
                "attrs": {
                    "link_type": "intra_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 4.0},
                        "target": {"component": "800G-DR4", "count": 4.0},
                    },
                },
            },
            {
                "source": "/leafB",
                "target": "/leafB",
                "pattern": "mesh",
                "capacity": 3_200,
                "cost": 1,
                "attrs": {
                    "link_type": "intra_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 4.0},
                        "target": {"component": "800G-DR4", "count": 4.0},
                    },
                },
            },
            {
                "source": "/leafC",
                "target": "/leafC",
                "pattern": "mesh",
                "capacity": 3_200,
                "cost": 1,
                "attrs": {
                    "link_type": "intra_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 4.0},
                        "target": {"component": "800G-DR4", "count": 4.0},
                    },
                },
            },
            {
                "source": "/leafD",
                "target": "/leafD",
                "pattern": "mesh",
                "capacity": 3_200,
                "cost": 1,
                "attrs": {
                    "link_type": "intra_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 4.0},
                        "target": {"component": "800G-DR4", "count": 4.0},
                    },
                },
            },
            # Inter-group (reduced): ring-of-cliques, one_to_one, 1×800G per pair ≈ 0.8 Tb/s
            # A<->B and A<->D
            {
                "source": "leafA/leafA{idx}",
                "target": "leafB/leafB{idx}",
                "pattern": "one_to_one",
                "expand": {"vars": {"idx": [1, 2, 3, 4]}, "mode": "zip"},
                "capacity": 800,
                "cost": 1,
                "attrs": {
                    "link_type": "inter_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 1.0},
                        "target": {"component": "800G-DR4", "count": 1.0},
                    },
                },
            },
            {
                "source": "leafA/leafA{idx}",
                "target": "leafD/leafD{idx}",
                "pattern": "one_to_one",
                "expand": {"vars": {"idx": [1, 2, 3, 4]}, "mode": "zip"},
                "capacity": 800,
                "cost": 1,
                "attrs": {
                    "link_type": "inter_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 1.0},
                        "target": {"component": "800G-DR4", "count": 1.0},
                    },
                },
            },
            # B<->C and C<->D
            {
                "source": "leafB/leafB{idx}",
                "target": "leafC/leafC{idx}",
                "pattern": "one_to_one",
                "expand": {"vars": {"idx": [1, 2, 3, 4]}, "mode": "zip"},
                "capacity": 800,
                "cost": 1,
                "attrs": {
                    "link_type": "inter_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 1.0},
                        "target": {"component": "800G-DR4", "count": 1.0},
                    },
                },
            },
            {
                "source": "leafC/leafC{idx}",
                "target": "leafD/leafD{idx}",
                "pattern": "one_to_one",
                "expand": {"vars": {"idx": [1, 2, 3, 4]}, "mode": "zip"},
                "capacity": 800,
                "cost": 1,
                "attrs": {
                    "link_type": "inter_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 1.0},
                        "target": {"component": "800G-DR4", "count": 1.0},
                    },
                },
            },
        ],
    },
    "Dragonfly_A3H2G7": {
        "nodes": {
            "G1": {
                "count": 3,
                "template": "G1_r{n}",
                "attrs": {"role": "leaf", "tier": "dragonfly", "group": "G1"},
            },
            "G2": {
                "count": 3,
                "template": "G2_r{n}",
                "attrs": {"role": "leaf", "tier": "dragonfly", "group": "G2"},
            },
            "G3": {
                "count": 3,
                "template": "G3_r{n}",
                "attrs": {"role": "leaf", "tier": "dragonfly", "group": "G3"},
            },
            "G4": {
                "count": 3,
                "template": "G4_r{n}",
                "attrs": {"role": "leaf", "tier": "dragonfly", "group": "G4"},
            },
            "G5": {
                "count": 3,
                "template": "G5_r{n}",
                "attrs": {"role": "leaf", "tier": "dragonfly", "group": "G5"},
            },
            "G6": {
                "count": 3,
                "template": "G6_r{n}",
                "attrs": {"role": "leaf", "tier": "dragonfly", "group": "G6"},
            },
            "G7": {
                "count": 3,
                "template": "G7_r{n}",
                "attrs": {"role": "leaf", "tier": "dragonfly", "group": "G7"},
            },
        },
        "links": [
            # Intra-group: fully meshed clique, 1×800G per pair
            {
                "source": "/G{g}",
                "target": "/G{g}",
                "pattern": "mesh",
                "expand": {"vars": {"g": [1, 2, 3, 4, 5, 6, 7]}},
                "capacity": 800,
                "cost": 1,
                "attrs": {
                    "link_type": "intra_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 1.0},
                        "target": {"component": "800G-DR4", "count": 1.0},
                    },
                },
            },
            # Inter-group: canonical Dragonfly (a=3, h=2, g=7)
            # Exactly one 800G link per unordered group pair.
            # Mapping ensures each router has h=2 global links.
            {
                "source": "G{gu}/G{gu}_r{ru}",
                "target": "G{gv}/G{gv}_r{rv}",
                "pattern": "one_to_one",
                "expand": {
                    "vars": {
                        "gu": [
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            2,
                            2,
                            2,
                            2,
                            2,
                            3,
                            3,
                            3,
                            3,
                            4,
                            4,
                            4,
                            5,
                            5,
                            6,
                        ],
                        "ru": [
                            1,
                            2,
                            3,
                            1,
                            2,
                            3,
                            1,
                            2,
                            3,
                            1,
                            2,
                            1,
                            2,
                            3,
                            1,
                            1,
                            2,
                            3,
                            1,
                            2,
                            1,
                        ],
                        "gv": [
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            3,
                            4,
                            5,
                            6,
                            7,
                            4,
                            5,
                            6,
                            7,
                            5,
                            6,
                            7,
                            6,
                            7,
                            7,
                        ],
                        "rv": [
                            3,
                            2,
                            1,
                            3,
                            2,
                            1,
                            3,
                            2,
                            1,
                            3,
                            2,
                            3,
                            2,
                            1,
                            3,
                            3,
                            2,
                            1,
                            3,
                            2,
                            3,
                        ],
                    },
                    "mode": "zip",
                },
                "capacity": 800,
                "cost": 1,
                "attrs": {
                    "link_type": "inter_group",
                    "hardware": {
                        "source": {"component": "800G-DR4", "count": 1.0},
                        "target": {"component": "800G-DR4", "count": 1.0},
                    },
                },
            },
        ],
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
