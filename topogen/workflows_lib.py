"""Builtin workflows definitions."""

from __future__ import annotations

from typing import Any

_BUILTIN_WORKFLOWS: dict[str, list[dict[str, Any]]] = {
    "design_analysis_brief": [
        {"type": "NetworkStats", "name": "network_statistics"},
        {
            "type": "MaximumSupportedDemand",
            "name": "msd_baseline",
            "demand_set": "baseline_traffic_matrix",
            "alpha_start": 1.0,
            "growth_factor": 2.0,
            "alpha_min": 1e-3,
            "alpha_max": 1e6,
            "alpha_tolerance": 0.05,
            "max_bracket_iters": 16,
            "max_bisect_iters": 32,
        },
        {
            "type": "TrafficMatrixPlacement",
            "name": "tm_placement",
            "demand_set": "baseline_traffic_matrix",
            "failure_policy": "mc_baseline",
            "iterations": 1000,
            "workers": "auto",
            "failure_trace": False,
            "outputs": {"cost_distribution": True},
            "alpha_from_step": "msd_baseline",
        },
        {
            "type": "CostPower",
            "name": "cost_power",
            "include_disabled": True,
            "aggregation_level": 2,
        },
    ]
}
