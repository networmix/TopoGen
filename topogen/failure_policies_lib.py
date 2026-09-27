"""Built-in failure policies with overrides from ``cwd/lib/failure_policies.yml``.

The YAML file maps names to definitions; each entry replaces the matching built-in.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_BUILTIN_FAILURE_POLICIES: dict[str, dict[str, Any]] = {
    "empty": {"modes": [{"weight": 1.0, "rules": []}]},
    "single_random_link_failure": {
        "attrs": {
            "description": "Fails exactly one random link to test network resilience"
        },
        "modes": [
            {
                "weight": 1.0,
                "rules": [{"scope": "link", "mode": "choice", "count": 1}],
            }
        ],
    },
    "mc_baseline": {
        "attrs": {
            "description": "Balanced MC: SRLG + DC->PoP + node(maint) + intra-site fabric"
        },
        "modes": [
            # Corridor SRLG
            {
                "weight": 0.30,
                "rules": [
                    {
                        "scope": "risk_group",
                        "mode": "choice",
                        "count": 1,
                        "weight_by": "distance_km",
                    }
                ],
            },
            # DC->PoP outages
            {
                "weight": 0.35,
                "rules": [
                    {
                        "scope": "link",
                        "mode": "choice",
                        "count": 3,
                        "match": {
                            "conditions": [
                                {
                                    "attr": "link_type",
                                    "op": "==",
                                    "value": "dc_to_pop",
                                }
                            ],
                            "logic": "and",
                        },
                        "weight_by": "target_capacity",
                    }
                ],
            },
            # Node failure / maintenance (weighted by attached capacity)
            {
                "weight": 0.25,
                "rules": [
                    {
                        "scope": "node",
                        "mode": "choice",
                        "count": 1,
                        "match": {
                            "conditions": [
                                {
                                    "attr": "node_type",
                                    "op": "!=",
                                    "value": "dc_region",
                                }
                            ],
                            "logic": "and",
                        },
                        "weight_by": "attached_capacity_gbps",
                    }
                ],
            },
            # Intra-site fabric events across blueprints (Clos, Dragonfly, FullMesh)
            {
                "weight": 0.10,
                "rules": [
                    {
                        "scope": "link",
                        "mode": "choice",
                        "count": 4,
                        "match": {
                            "conditions": [
                                {
                                    "attr": "link_type",
                                    "op": "==",
                                    "value": "leaf_spine",
                                },
                                {
                                    "attr": "link_type",
                                    "op": "==",
                                    "value": "intra_group",
                                },
                                {
                                    "attr": "link_type",
                                    "op": "==",
                                    "value": "inter_group",
                                },
                                {
                                    "attr": "link_type",
                                    "op": "==",
                                    "value": "internal_mesh",
                                },
                            ],
                            "logic": "or",
                        },
                    }
                ],
            },
        ],
    },
}


def get_builtin_failure_policies() -> dict[str, dict[str, Any]]:
    """Return built-in failure policies with user overrides applied."""
    policies = deepcopy(_BUILTIN_FAILURE_POLICIES)
    user_policies = _load_user_library("failure_policies.yml")
    policies.update(user_policies)
    return policies


def _load_user_library(file_name: str) -> dict[str, Any]:
    """Load user failure policies from ``lib/<file_name>`` if present."""
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
