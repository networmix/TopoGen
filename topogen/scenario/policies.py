"""Failure policies and workflow sections for scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from topogen.library import Library, thaw

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig


def _build_failure_policy_set_section(
    config: "TopologyConfig", workflow: list[dict[str, Any]], library: Library
) -> dict[str, Any]:
    """Build the ``failures`` section of the scenario."""
    builtin_policies = library.failure_policies
    policies: dict[str, Any] = {}

    default_policy_name = config.failure_policies.assignments.default
    if default_policy_name in builtin_policies:
        policies[default_policy_name] = thaw(builtin_policies[default_policy_name])
    else:
        available = list(builtin_policies.keys())
        raise ValueError(
            f"Default failure policy '{default_policy_name}' not found. Available built-in policies: {available}"
        )

    workflow_name = config.workflows.assignments.default
    for step in workflow:
        policy_name = step.get("failure_policy")
        if policy_name is None:
            continue
        if policy_name not in builtin_policies:
            raise ValueError(
                f"Workflow '{workflow_name}' references unknown failure policy '{policy_name}'"
            )
        policies[policy_name] = thaw(builtin_policies[policy_name])
    return policies


def _build_workflow_section(
    config: "TopologyConfig", library: Library
) -> list[dict[str, Any]]:
    name = config.workflows.assignments.default
    if name not in library.workflows:
        raise ValueError(
            f"Default workflow '{name}' not found. Available workflows: {list(library.workflows)}"
        )
    definition = library.workflows[name]
    steps = thaw(definition.steps)
    if definition.follows_configured_matrix:
        for step in steps:
            if "demand_set" in step:
                step["demand_set"] = config.traffic.matrix_name
    return steps
