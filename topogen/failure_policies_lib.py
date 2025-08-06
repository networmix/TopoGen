"""Built-in failure policy definitions for TopoGen.

This module provides a library of common failure policies that can be used
in NetGraph scenarios. Policies define how failures are simulated during
network analysis.

Example Usage:
    from topogen.failure_policies_lib import get_builtin_failure_policies

    policies = get_builtin_failure_policies()
    single_link = policies["single_random_link_failure"]
"""

from typing import Any, Dict

# Built-in failure policy definitions
_BUILTIN_FAILURE_POLICIES: Dict[str, Dict[str, Any]] = {
    "single_random_link_failure": {
        "attrs": {
            "description": "Fails exactly one random link to test network resilience"
        },
        "rules": [{"entity_scope": "link", "rule_type": "choice", "count": 1}],
    },
    "dual_random_link_failure": {
        "attrs": {
            "description": "Fails exactly two random links to test dual-failure resilience"
        },
        "rules": [{"entity_scope": "link", "rule_type": "choice", "count": 2}],
    },
    "single_random_node_failure": {
        "attrs": {
            "description": "Fails exactly one random node to test node-level resilience"
        },
        "rules": [{"entity_scope": "node", "rule_type": "choice", "count": 1}],
    },
    "random_10_percent_link_failure": {
        "attrs": {"description": "Randomly fails approximately 10% of links"},
        "rules": [{"entity_scope": "link", "rule_type": "random", "probability": 0.1}],
    },
    "random_5_percent_combined_failure": {
        "attrs": {
            "description": "Randomly fails approximately 5% of both nodes and links"
        },
        "rules": [
            {"entity_scope": "node", "rule_type": "random", "probability": 0.05},
            {"entity_scope": "link", "rule_type": "random", "probability": 0.05},
        ],
    },
    "no_failures": {
        "attrs": {
            "description": "No failures - baseline analysis without any network failures"
        },
        "rules": [],
    },
}


def get_builtin_failure_policies() -> Dict[str, Dict[str, Any]]:
    """Get all built-in failure policy definitions.

    Returns:
        Dictionary mapping policy names to their definitions.
    """
    return _BUILTIN_FAILURE_POLICIES.copy()


def get_builtin_failure_policy(name: str) -> Dict[str, Any]:
    """Get a specific built-in failure policy by name.

    Args:
        name: The name of the failure policy to retrieve.

    Returns:
        The failure policy definition.

    Raises:
        KeyError: If the failure policy name is not found.
    """
    if name not in _BUILTIN_FAILURE_POLICIES:
        available = list(_BUILTIN_FAILURE_POLICIES.keys())
        raise KeyError(
            f"Unknown built-in failure policy '{name}'. Available: {available}"
        )

    return _BUILTIN_FAILURE_POLICIES[name].copy()


def list_builtin_failure_policy_names() -> list[str]:
    """Get a list of all built-in failure policy names.

    Returns:
        Sorted list of failure policy names.
    """
    return sorted(_BUILTIN_FAILURE_POLICIES.keys())


def get_failure_policies_by_type(entity_scope: str) -> Dict[str, Dict[str, Any]]:
    """Get failure policies that target a specific entity scope.

    Args:
        entity_scope: The entity scope to filter by ("node", "link", or "risk_group").

    Returns:
        Dictionary of failure policies that include rules for the specified entity scope.
    """
    filtered_policies = {}

    for name, policy in _BUILTIN_FAILURE_POLICIES.items():
        rules = policy.get("rules", [])
        if any(rule.get("entity_scope") == entity_scope for rule in rules):
            filtered_policies[name] = policy.copy()

    return filtered_policies


def get_baseline_policies() -> Dict[str, Dict[str, Any]]:
    """Get failure policies suitable for baseline analysis (no failures).

    Returns:
        Dictionary of failure policies with no failure rules.
    """
    baseline_policies = {}

    for name, policy in _BUILTIN_FAILURE_POLICIES.items():
        rules = policy.get("rules", [])
        if not rules:  # No failure rules = baseline
            baseline_policies[name] = policy.copy()

    return baseline_policies
