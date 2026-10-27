"""Builtin failure policies definitions."""

from __future__ import annotations

from typing import Any

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
