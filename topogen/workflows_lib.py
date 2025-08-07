"""Built-in workflow definitions for TopoGen.

This module provides a library of common analysis workflows that can be used
in NetGraph scenarios. Workflows define the sequence of analysis steps to
execute on the network.

Example Usage:
    from topogen.workflows_lib import get_builtin_workflows

    workflows = get_builtin_workflows()
    basic_analysis = workflows["basic_capacity_analysis"]
"""

from typing import Any

# Built-in workflow definitions
_BUILTIN_WORKFLOWS: dict[str, list[dict[str, Any]]] = {
    "basic_capacity_analysis": [
        {"step_type": "NetworkStats", "name": "network_statistics"},
        {
            "step_type": "CapacityEnvelopeAnalysis",
            "name": "capacity_envelope_baseline",
            "source_path": "^(core_.+)",
            "sink_path": "^(core_.+)",
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
    "comprehensive_resilience_analysis": [
        {"step_type": "NetworkStats", "name": "network_statistics"},
        {
            "step_type": "CapacityEnvelopeAnalysis",
            "name": "baseline_capacity",
            "source_path": "^(core_.+)",
            "sink_path": "^(core_.+)",
            "mode": "pairwise",
            "parallelism": 8,
            "shortest_path": False,
            "flow_placement": "PROPORTIONAL",
            "seed": 42,
            "iterations": 1,
            "baseline": True,
            "failure_policy": "no_failures",
        },
        {
            "step_type": "CapacityEnvelopeAnalysis",
            "name": "single_link_failure_analysis",
            "source_path": "^(core_.+)",
            "sink_path": "^(core_.+)",
            "mode": "pairwise",
            "parallelism": 8,
            "shortest_path": False,
            "flow_placement": "PROPORTIONAL",
            "seed": 42,
            "iterations": 500,
            "baseline": False,
            "failure_policy": "single_random_link_failure",
        },
        {
            "step_type": "CapacityEnvelopeAnalysis",
            "name": "dual_link_failure_analysis",
            "source_path": "^(core_.+)",
            "sink_path": "^(core_.+)",
            "mode": "pairwise",
            "parallelism": 8,
            "shortest_path": False,
            "flow_placement": "PROPORTIONAL",
            "seed": 43,
            "iterations": 200,
            "baseline": False,
            "failure_policy": "dual_random_link_failure",
        },
    ],
    "fast_network_analysis": [
        {"step_type": "NetworkStats", "name": "network_statistics"},
        {
            "step_type": "CapacityEnvelopeAnalysis",
            "name": "fast_capacity_check",
            "source_path": "^(core_.+)",
            "sink_path": "^(core_.+)",
            "mode": "combine",  # Faster than pairwise
            "parallelism": 2,
            "shortest_path": True,  # Faster than full flow
            "flow_placement": "PROPORTIONAL",
            "seed": 42,
            "iterations": 50,
            "baseline": True,
            "failure_policy": "single_random_link_failure",
        },
    ],
    "node_resilience_analysis": [
        {"step_type": "NetworkStats", "name": "network_statistics"},
        {
            "step_type": "CapacityEnvelopeAnalysis",
            "name": "baseline_capacity",
            "source_path": "^(core_.+)",
            "sink_path": "^(core_.+)",
            "mode": "pairwise",
            "parallelism": 8,
            "shortest_path": False,
            "flow_placement": "PROPORTIONAL",
            "seed": 42,
            "iterations": 1,
            "baseline": True,
            "failure_policy": "no_failures",
        },
        {
            "step_type": "CapacityEnvelopeAnalysis",
            "name": "single_node_failure_analysis",
            "source_path": "^(core_.+)",
            "sink_path": "^(core_.+)",
            "mode": "pairwise",
            "parallelism": 8,
            "shortest_path": False,
            "flow_placement": "PROPORTIONAL",
            "seed": 42,
            "iterations": 300,
            "baseline": False,
            "failure_policy": "single_random_node_failure",
        },
    ],
    "network_stats_only": [{"step_type": "NetworkStats", "name": "network_statistics"}],
}


def get_builtin_workflows() -> dict[str, list[dict[str, Any]]]:
    """Get all built-in workflow definitions.

    Returns:
        Dictionary mapping workflow names to their step definitions.
    """
    # Deep copy to avoid mutations
    import copy

    return copy.deepcopy(_BUILTIN_WORKFLOWS)


def get_builtin_workflow(name: str) -> list[dict[str, Any]]:
    """Get a specific built-in workflow by name.

    Args:
        name: The name of the workflow to retrieve.

    Returns:
        The workflow step definitions.

    Raises:
        KeyError: If the workflow name is not found.
    """
    if name not in _BUILTIN_WORKFLOWS:
        available = list(_BUILTIN_WORKFLOWS.keys())
        raise KeyError(f"Unknown built-in workflow '{name}'. Available: {available}")

    # Deep copy to avoid mutations
    import copy

    return copy.deepcopy(_BUILTIN_WORKFLOWS[name])


def list_builtin_workflow_names() -> list[str]:
    """Get a list of all built-in workflow names.

    Returns:
        Sorted list of workflow names.
    """
    return sorted(_BUILTIN_WORKFLOWS.keys())


def get_workflows_by_step_type(step_type: str) -> dict[str, list[dict[str, Any]]]:
    """Get workflows that include a specific step type.

    Args:
        step_type: The step type to filter by (e.g., "CapacityEnvelopeAnalysis").

    Returns:
        Dictionary of workflows that include the specified step type.
    """
    import copy

    filtered_workflows = {}

    for name, steps in _BUILTIN_WORKFLOWS.items():
        if any(step.get("step_type") == step_type for step in steps):
            filtered_workflows[name] = copy.deepcopy(steps)

    return filtered_workflows


def get_fast_workflows() -> dict[str, list[dict[str, Any]]]:
    """Get workflows designed for fast execution.

    Returns:
        Dictionary of workflows optimized for speed.
    """
    # Identify fast workflows by low iteration counts and fast settings
    import copy

    fast_workflows = {}

    for name, steps in _BUILTIN_WORKFLOWS.items():
        is_fast = False

        # Check if workflow has characteristics of fast execution
        for step in steps:
            if step.get("step_type") == "CapacityEnvelopeAnalysis":
                iterations = step.get("iterations", 1000)
                shortest_path = step.get("shortest_path", False)
                mode = step.get("mode", "pairwise")

                # Consider fast if: low iterations OR shortest_path OR combine mode
                if iterations <= 100 or shortest_path or mode == "combine":
                    is_fast = True
                    break

        # Also include workflows with only NetworkStats
        if not any(
            step.get("step_type") == "CapacityEnvelopeAnalysis" for step in steps
        ):
            is_fast = True

        if is_fast:
            fast_workflows[name] = copy.deepcopy(steps)

    return fast_workflows


def validate_workflow_steps(steps: list[dict[str, Any]]) -> None:
    """Validate that a workflow definition is structurally correct.

    Args:
        steps: List of workflow step definitions.

    Raises:
        ValueError: If the workflow is invalid.
    """
    if not isinstance(steps, list):
        raise ValueError("Workflow must be a list of steps")

    if not steps:
        raise ValueError("Workflow cannot be empty")

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i} must be a dictionary")

        if "step_type" not in step:
            raise ValueError(f"Step {i} missing required 'step_type' field")

        step_type = step["step_type"]
        if not isinstance(step_type, str):
            raise ValueError(f"Step {i} 'step_type' must be a string")

        # Validate known step types have required fields
        if step_type == "CapacityEnvelopeAnalysis":
            required_fields = ["source_path", "sink_path", "mode"]
            for field in required_fields:
                if field not in step:
                    raise ValueError(
                        f"CapacityEnvelopeAnalysis step {i} missing required field '{field}'"
                    )
