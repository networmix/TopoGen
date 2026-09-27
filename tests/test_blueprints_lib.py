"""Tests for the blueprints library module."""

from __future__ import annotations

from topogen.blueprints_lib import get_builtin_blueprints


class TestBlueprintsLib:
    def test_get_builtin_blueprints_returns_dict(self):
        blueprints = get_builtin_blueprints()
        assert isinstance(blueprints, dict)
        assert len(blueprints) > 0

    def test_blueprint_consistency(self):
        blueprints = get_builtin_blueprints()

        for name, blueprint in blueprints.items():
            assert "nodes" in blueprint, f"Blueprint {name} missing 'nodes'"
            assert "links" in blueprint, f"Blueprint {name} missing 'links'"
            assert isinstance(blueprint["nodes"], dict), (
                f"Blueprint {name} nodes not dict"
            )
            assert isinstance(blueprint["links"], list), (
                f"Blueprint {name} links not list"
            )

            for group_name, group_def in blueprint["nodes"].items():
                assert "count" in group_def, f"Group {group_name} missing count"
                assert "template" in group_def, f"Group {group_name} missing template"
                assert isinstance(group_def["count"], int), (
                    f"Group {group_name} count not int"
                )
