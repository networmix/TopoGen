"""Built-in workflows with overrides from ``cwd/lib/workflows.yml``.

The YAML file maps names to step lists; each entry replaces the matching built-in.
NetGraph validates step arguments before the library is returned.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ngraph.workflow.parse import build_workflow_steps

from topogen.library_io import load_user_library

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
            "resolution": 0.05,
            "max_bracket_iters": 16,
            "max_bisect_iters": 32,
        },
        {
            "type": "TrafficMatrixPlacement",
            "name": "tm_placement",
            "seed": 42,
            "demand_set": "baseline_traffic_matrix",
            "failure_policy": "mc_baseline",
            "iterations": 1000,
            "parallelism": "auto",
            "store_failure_patterns": False,
            "include_flow_details": True,
            "include_used_edges": False,
            "alpha_from_step": "msd_baseline",
            "alpha_from_field": "data.alpha_star",
        },
        {
            "type": "CostPower",
            "name": "cost_power",
            "include_disabled": True,
            "aggregation_level": 2,
        },
    ]
}


def get_builtin_workflows(
    matrix_name: str = "baseline_traffic_matrix",
) -> dict[str, list[dict[str, Any]]]:
    """Return workflows, binding built-in steps to the configured demand set.

    User overrides retain their explicit demand-set names, including workflows
    that reference several sampled matrices.
    """
    workflows = deepcopy(_BUILTIN_WORKFLOWS)
    for steps in workflows.values():
        for step in steps:
            if "demand_set" in step:
                step["demand_set"] = matrix_name
    user_workflows = load_user_library("workflows.yml")
    workflows.update(user_workflows)
    for name, steps in workflows.items():
        if not isinstance(steps, list) or any(not isinstance(s, dict) for s in steps):
            raise ValueError(f"Workflow '{name}' must contain a list of step mappings")
        try:
            build_workflow_steps(steps, derive_seed=lambda _: None)
        except ValueError as exc:
            raise ValueError(f"Workflow '{name}': {exc}") from exc
    return workflows
