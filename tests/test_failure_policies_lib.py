"""Tests for the failure policies library module (minimal API)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from topogen.failure_policies_lib import get_builtin_failure_policies


class TestBuiltinFailurePoliciesMinimal:
    """Tests for minimal API: get_builtin_failure_policies and merging behavior."""

    def test_returns_dict_and_contains_single_random_link(self) -> None:
        policies = get_builtin_failure_policies()
        assert isinstance(policies, dict)
        assert "single_random_link_failure" in policies
        policy = policies["single_random_link_failure"]
        assert isinstance(policy, dict)
        assert isinstance(policy.get("rules"), list)

    def test_independent_returns(self) -> None:
        p1 = get_builtin_failure_policies()
        p2 = get_builtin_failure_policies()
        p1["X"] = {"rules": []}
        assert "X" not in p2

    def test_user_library_merge_and_override(self, tmp_path: Path) -> None:
        # Create lib/failure_policies.yml in a temp cwd
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir(parents=True)
        user_yaml: dict[str, Any] = {
            "single_random_link_failure": {
                "attrs": {"description": "override"},
                "rules": [{"entity_scope": "link", "rule_type": "choice", "count": 2}],
            },
            "custom_policy": {
                "attrs": {"description": "custom"},
                "rules": [],
            },
        }
        with (lib_dir / "failure_policies.yml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(user_yaml, f)

        # Change cwd to tmp and verify merge
        old_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            merged = get_builtin_failure_policies()
            # Built-in overridden to count=2
            assert merged["single_random_link_failure"]["rules"][0]["count"] == 2
            # Custom added
            assert "custom_policy" in merged
        finally:
            os.chdir(old_cwd)
