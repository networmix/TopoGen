"""Built-in workflow definitions used by the scenario pipeline."""

from __future__ import annotations

from typing import Any

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


def get_builtin_workflows() -> dict[str, list[dict[str, Any]]]:
    """Return deep-copied built-in workflow definitions.

    Returns:
        Dictionary mapping workflow names to their step definitions.
    """
    import copy

    return copy.deepcopy(_BUILTIN_WORKFLOWS)
