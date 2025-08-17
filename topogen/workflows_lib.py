"""Built-in workflow definitions.

Provides workflows used by the scenario pipeline and merges overrides from
``cwd/lib/workflows.yml`` when present. The user file must be direct mapping:
name -> list of step definitions. User entries override built-ins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Built-in workflow definitions required by the pipeline
_BUILTIN_WORKFLOWS: dict[str, list[dict[str, Any]]] = {
    "design_analysis_brief": [
        {"step_type": "NetworkStats", "name": "network_statistics"},
        {
            "step_type": "MaximumSupportedDemand",
            "name": "msd_baseline",
            "matrix_name": "baseline_traffic_matrix",
            "acceptance_rule": "hard",
            "alpha_start": 1.0,
            "growth_factor": 2.0,
            "alpha_min": 1e-3,
            "alpha_max": 1e6,
            "resolution": 0.05,
            "max_bracket_iters": 16,
            "max_bisect_iters": 32,
            "seeds_per_alpha": 1,
            "placement_rounds": 2,
        },
        {
            "step_type": "TrafficMatrixPlacement",
            "name": "tm_placement",
            "seed": 42,
            "matrix_name": "baseline_traffic_matrix",
            "failure_policy": "mc_baseline",
            "iterations": 1000,
            "parallelism": 7,
            "placement_rounds": "auto",
            "baseline": True,
            "store_failure_patterns": False,
            "include_flow_details": True,
            "include_used_edges": False,
            "alpha_from_step": "msd_baseline",
            "alpha_from_field": "data.alpha_star",
        },
        # {
        #     "step_type": "MaxFlow",
        #     "name": "node_to_node_capacity_matrix",
        #     "seed": 42,
        #     "source_path": "(metro[0-9]+/dc[0-9]+)",
        #     "sink_path": "(metro[0-9]+/dc[0-9]+)",
        #     "mode": "pairwise",
        #     "failure_policy": "mc_baseline",
        #     "iterations": 100,
        #     "parallelism": 7,
        #     "shortest_path": False,
        #     "flow_placement": "PROPORTIONAL",
        #     "baseline": True,
        #     "store_failure_patterns": False,
        #     "include_flow_details": True,
        #     "include_min_cut": False,
        # },
        {
            "step_type": "CostPower",
            "name": "cost_power",
            "include_disabled": True,
            "aggregation_level": 2,
        },
    ]
}


def _load_user_library(file_name: str) -> dict[str, Any]:
    """Load user workflow library from ``lib/<file_name>`` if present."""
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


def get_builtin_workflows() -> dict[str, list[dict[str, Any]]]:
    """Return workflow library merged with user overrides."""
    import copy

    workflows = copy.deepcopy(_BUILTIN_WORKFLOWS)
    user_workflows = _load_user_library("workflows.yml")
    # Support only direct mapping: name -> definition (list of steps)
    workflows.update(user_workflows)
    return workflows
