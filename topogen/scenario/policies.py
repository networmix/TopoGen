"""Failure policies and workflow sections for scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from topogen.failure_policies_lib import get_builtin_failure_policies
from topogen.workflows_lib import get_builtin_workflows

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig


def _build_failure_policy_set_section(config: "TopologyConfig") -> dict[str, Any]:
    """Build the ``failure_policy_set`` section of the scenario."""
    builtin_policies = get_builtin_failure_policies()
    policies: dict[str, Any] = {}

    default_policy_name = config.failure_policies.assignments.default
    if default_policy_name in builtin_policies:
        policies[default_policy_name] = builtin_policies[default_policy_name]
    else:
        available = list(builtin_policies.keys())
        raise ValueError(
            f"Default failure policy '{default_policy_name}' not found. Available built-in policies: {available}"
        )

    workflows_cfg = getattr(config, "workflows", None)
    if (
        workflows_cfg is not None
        and getattr(workflows_cfg, "assignments", None) is not None
    ):
        workflows = get_builtin_workflows()
        workflow_name = workflows_cfg.assignments.default
        steps = workflows.get(workflow_name, [])
        for step in steps:
            policy_name = step.get("failure_policy")
            if not policy_name:
                continue
            if policy_name not in builtin_policies:
                available2 = list(builtin_policies.keys())
                raise ValueError(
                    f"Workflow '{workflow_name}' references unknown failure policy '{policy_name}'. Available built-in policies: {available2}"
                )
            policies[policy_name] = builtin_policies[policy_name]
    return policies


def _build_workflow_section(config: "TopologyConfig") -> list[dict[str, Any]]:
    """Build the ``workflow`` section of the scenario."""
    builtin_workflows = get_builtin_workflows()
    default_workflow_name = config.workflows.assignments.default
    if default_workflow_name in builtin_workflows:
        return builtin_workflows[default_workflow_name]
    available_builtin = list(builtin_workflows.keys())
    raise ValueError(
        f"Default workflow '{default_workflow_name}' not found. Available built-in workflows: {available_builtin}"
    )
