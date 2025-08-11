"""Tests for the simplified blueprints library module."""

from __future__ import annotations

from topogen.blueprints_lib import get_builtin_blueprints


class TestBlueprintsLib:
    """Test cases for built-in blueprints retrieval and structure."""

    def test_get_builtin_blueprints_returns_dict(self):
        """Test that get_builtin_blueprints returns a dictionary."""
        blueprints = get_builtin_blueprints()
        assert isinstance(blueprints, dict)
        assert len(blueprints) > 0

    # Intentionally avoid asserting presence of specific blueprint names or
    # exact contents. Library content is allowed to evolve; we only verify
    # basic API shape and internal consistency below.

    def test_blueprint_consistency(self):
        """Test that all blueprints have consistent structure."""
        blueprints = get_builtin_blueprints()

        for name, blueprint in blueprints.items():
            # Every blueprint must have groups and adjacency
            assert "groups" in blueprint, f"Blueprint {name} missing 'groups'"
            assert "adjacency" in blueprint, f"Blueprint {name} missing 'adjacency'"
            assert isinstance(blueprint["groups"], dict), (
                f"Blueprint {name} groups not dict"
            )
            assert isinstance(blueprint["adjacency"], list), (
                f"Blueprint {name} adjacency not list"
            )

            # Every group must have required fields
            for group_name, group_def in blueprint["groups"].items():
                assert "node_count" in group_def, (
                    f"Group {group_name} missing node_count"
                )
                assert "name_template" in group_def, (
                    f"Group {group_name} missing name_template"
                )
                assert isinstance(group_def["node_count"], int), (
                    f"Group {group_name} node_count not int"
                )
