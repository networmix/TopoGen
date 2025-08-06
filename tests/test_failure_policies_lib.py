"""Tests for the failure policies library module."""

import pytest

from topogen.failure_policies_lib import (
    get_baseline_policies,
    get_builtin_failure_policies,
    get_builtin_failure_policy,
    get_failure_policies_by_type,
    list_builtin_failure_policy_names,
)


class TestGetBuiltinFailurePolicies:
    """Test get_builtin_failure_policies function."""

    def test_returns_dict(self):
        """Test that the function returns a dictionary."""
        policies = get_builtin_failure_policies()
        assert isinstance(policies, dict)

    def test_contains_expected_policies(self):
        """Test that standard policies are included."""
        policies = get_builtin_failure_policies()

        expected_policies = [
            "single_random_link_failure",
            "dual_random_link_failure",
            "single_random_node_failure",
            "no_failures",
        ]

        for policy_name in expected_policies:
            assert policy_name in policies

    def test_returns_copy(self):
        """Test that modifications don't affect the original."""
        policies1 = get_builtin_failure_policies()
        policies2 = get_builtin_failure_policies()

        # Modify one copy
        policies1["test_policy"] = {"test": True}

        # Original should be unchanged
        assert "test_policy" not in policies2

    def test_policy_structure(self):
        """Test that policies have the expected structure."""
        policies = get_builtin_failure_policies()

        for name, policy in policies.items():
            assert isinstance(policy, dict), f"Policy {name} should be a dict"
            assert "rules" in policy, f"Policy {name} should have 'rules'"
            assert isinstance(policy["rules"], list), (
                f"Policy {name} rules should be a list"
            )

            # Check attrs section if present
            if "attrs" in policy:
                assert isinstance(policy["attrs"], dict)
                if "description" in policy["attrs"]:
                    assert isinstance(policy["attrs"]["description"], str)


class TestGetBuiltinFailurePolicy:
    """Test get_builtin_failure_policy function."""

    def test_valid_policy_name(self):
        """Test retrieving a valid policy."""
        policy = get_builtin_failure_policy("single_random_link_failure")
        assert isinstance(policy, dict)
        assert "rules" in policy
        assert len(policy["rules"]) == 1
        assert policy["rules"][0]["entity_scope"] == "link"

    def test_invalid_policy_name(self):
        """Test that unknown policy names raise KeyError."""
        with pytest.raises(
            KeyError, match="Unknown built-in failure policy 'nonexistent'"
        ):
            get_builtin_failure_policy("nonexistent")

    def test_returns_copy(self):
        """Test that modifications don't affect the original."""
        policy1 = get_builtin_failure_policy("single_random_link_failure")
        policy2 = get_builtin_failure_policy("single_random_link_failure")

        # Modify one copy
        policy1["test"] = True

        # Original should be unchanged
        assert "test" not in policy2


class TestListBuiltinFailurePolicyNames:
    """Test list_builtin_failure_policy_names function."""

    def test_returns_list(self):
        """Test that the function returns a list."""
        names = list_builtin_failure_policy_names()
        assert isinstance(names, list)

    def test_contains_expected_names(self):
        """Test that expected policy names are included."""
        names = list_builtin_failure_policy_names()

        expected_names = [
            "single_random_link_failure",
            "dual_random_link_failure",
            "single_random_node_failure",
            "no_failures",
        ]

        for name in expected_names:
            assert name in names

    def test_sorted_output(self):
        """Test that names are returned in sorted order."""
        names = list_builtin_failure_policy_names()
        assert names == sorted(names)


class TestGetFailurePoliciesByType:
    """Test get_failure_policies_by_type function."""

    def test_link_policies(self):
        """Test filtering for link-targeting policies."""
        link_policies = get_failure_policies_by_type("link")

        # Should include policies that target links
        assert "single_random_link_failure" in link_policies
        assert "dual_random_link_failure" in link_policies

        # Verify the policies actually target links
        for name, policy in link_policies.items():
            rules = policy["rules"]
            has_link_rule = any(rule.get("entity_scope") == "link" for rule in rules)
            assert has_link_rule, f"Policy {name} should target links"

    def test_node_policies(self):
        """Test filtering for node-targeting policies."""
        node_policies = get_failure_policies_by_type("node")

        # Should include policies that target nodes
        assert "single_random_node_failure" in node_policies

        # Verify the policies actually target nodes
        for name, policy in node_policies.items():
            rules = policy["rules"]
            has_node_rule = any(rule.get("entity_scope") == "node" for rule in rules)
            assert has_node_rule, f"Policy {name} should target nodes"

    def test_empty_entity_scope(self):
        """Test filtering for non-existent entity scope."""
        policies = get_failure_policies_by_type("nonexistent_scope")
        assert isinstance(policies, dict)
        # Might be empty or not, depending on built-in policies

    def test_returns_copy(self):
        """Test that modifications don't affect the original."""
        policies1 = get_failure_policies_by_type("link")
        policies2 = get_failure_policies_by_type("link")

        # Modify one copy
        if policies1:
            first_policy_name = list(policies1.keys())[0]
            policies1[first_policy_name]["test"] = True

            # Original should be unchanged
            assert "test" not in policies2[first_policy_name]


class TestGetBaselinePolicies:
    """Test get_baseline_policies function."""

    def test_returns_dict(self):
        """Test that the function returns a dictionary."""
        baseline_policies = get_baseline_policies()
        assert isinstance(baseline_policies, dict)

    def test_no_failures_included(self):
        """Test that no_failures policy is included in baseline."""
        baseline_policies = get_baseline_policies()
        assert "no_failures" in baseline_policies

    def test_baseline_policies_have_no_rules(self):
        """Test that baseline policies have empty rules."""
        baseline_policies = get_baseline_policies()

        for name, policy in baseline_policies.items():
            rules = policy.get("rules", [])
            assert not rules, f"Baseline policy {name} should have no rules"

    def test_returns_copy(self):
        """Test that modifications don't affect the original."""
        policies1 = get_baseline_policies()
        policies2 = get_baseline_policies()

        # Modify one copy
        policies1["test_policy"] = {"test": True}

        # Original should be unchanged
        assert "test_policy" not in policies2


class TestFailurePolicyContent:
    """Test the content and structure of specific built-in policies."""

    def test_single_random_link_failure_structure(self):
        """Test single_random_link_failure policy structure."""
        policy = get_builtin_failure_policy("single_random_link_failure")

        assert "attrs" in policy
        assert "description" in policy["attrs"]
        assert "rules" in policy
        assert len(policy["rules"]) == 1

        rule = policy["rules"][0]
        assert rule["entity_scope"] == "link"
        assert rule["rule_type"] == "choice"
        assert rule["count"] == 1

    def test_dual_random_link_failure_structure(self):
        """Test dual_random_link_failure policy structure."""
        policy = get_builtin_failure_policy("dual_random_link_failure")

        assert "rules" in policy
        assert len(policy["rules"]) == 1

        rule = policy["rules"][0]
        assert rule["entity_scope"] == "link"
        assert rule["rule_type"] == "choice"
        assert rule["count"] == 2

    def test_single_random_node_failure_structure(self):
        """Test single_random_node_failure policy structure."""
        policy = get_builtin_failure_policy("single_random_node_failure")

        assert "rules" in policy
        assert len(policy["rules"]) == 1

        rule = policy["rules"][0]
        assert rule["entity_scope"] == "node"
        assert rule["rule_type"] == "choice"
        assert rule["count"] == 1

    def test_no_failures_structure(self):
        """Test no_failures policy structure."""
        policy = get_builtin_failure_policy("no_failures")

        assert "attrs" in policy
        assert "description" in policy["attrs"]
        assert "rules" in policy
        assert len(policy["rules"]) == 0

    def test_random_percentage_policies(self):
        """Test policies using probability-based failures."""
        policies = get_builtin_failure_policies()

        # Find policies with probability rules
        prob_policies = []
        for name, policy in policies.items():
            rules = policy.get("rules", [])
            for rule in rules:
                if rule.get("rule_type") == "random" and "probability" in rule:
                    prob_policies.append((name, rule))

        # Should have at least one probability-based policy
        assert len(prob_policies) > 0

        # Validate probability values
        for policy_name, rule in prob_policies:
            prob = rule["probability"]
            assert isinstance(prob, (int, float))
            assert 0.0 <= prob <= 1.0, (
                f"Policy {policy_name} has invalid probability: {prob}"
            )
