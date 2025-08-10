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
    "basic_capacity_analysis": [
        {"step_type": "NetworkStats", "name": "network_statistics"},
        {
            "step_type": "CapacityEnvelopeAnalysis",
            "name": "capacity_envelope_baseline",
            "source_path": "(metro[0-9]+/dc[0-9]+)",
            "sink_path": "(metro[0-9]+/dc[0-9]+)",
            "mode": "pairwise",
            "parallelism": 4,
            "shortest_path": False,
            "flow_placement": "PROPORTIONAL",
            "seed": 42,
            "iterations": 100,
            "baseline": True,
            "failure_policy": "single_random_link_failure",
        },
    ],
    "network_stats_only": [{"step_type": "NetworkStats", "name": "network_statistics"}],
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
