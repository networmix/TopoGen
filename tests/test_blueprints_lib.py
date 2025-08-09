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

    def test_get_builtin_blueprints_contains_expected_blueprints(self):
        """Test that all expected blueprints are present."""
        blueprints = get_builtin_blueprints()
        expected_blueprints = ["SingleRouter", "FullMesh4", "Clos_64_256", "DCRegion"]

        for expected in expected_blueprints:
            assert expected in blueprints

    def test_single_router_blueprint_structure(self):
        """Test the SingleRouter blueprint has correct structure."""
        blueprint = get_builtin_blueprints()["SingleRouter"]

        assert "groups" in blueprint
        assert "adjacency" in blueprint

        # Should have one core group
        assert "core" in blueprint["groups"]
        core_group = blueprint["groups"]["core"]
        assert core_group["node_count"] == 1
        assert core_group["name_template"] == "core"
        assert "attrs" in core_group

        # Should have no adjacency rules (single router)
        assert blueprint["adjacency"] == []

    def test_full_mesh_4_blueprint_structure(self):
        """Test the FullMesh4 blueprint has correct structure."""
        blueprint = get_builtin_blueprints()["FullMesh4"]

        assert "groups" in blueprint
        assert "adjacency" in blueprint

        # Should have one core group with 4 routers
        assert "core" in blueprint["groups"]
        core_group = blueprint["groups"]["core"]
        assert core_group["node_count"] == 4
        assert "core{node_num}" in core_group["name_template"]

        # Should have mesh adjacency
        assert len(blueprint["adjacency"]) == 1
        adj = blueprint["adjacency"][0]
        assert adj["source"] == "/core"
        assert adj["target"] == "/core"
        assert adj["pattern"] == "mesh"

    def test_clos_64_256_blueprint_structure(self):
        """Test the Clos_64_256 blueprint has correct structure."""
        blueprint = get_builtin_blueprints()["Clos_64_256"]

        assert "groups" in blueprint
        assert "adjacency" in blueprint

        # Should have spine and leaf groups
        assert "spine" in blueprint["groups"]
        assert "leaf" in blueprint["groups"]

        spine_group = blueprint["groups"]["spine"]
        leaf_group = blueprint["groups"]["leaf"]

        assert spine_group["node_count"] == 8
        assert leaf_group["node_count"] == 8

        # Should have leaf-spine adjacency
        assert len(blueprint["adjacency"]) == 1
        adj = blueprint["adjacency"][0]
        assert adj["source"] == "/leaf"
        assert adj["target"] == "/spine"
        assert adj["pattern"] == "mesh"

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

    def test_dc_region_blueprint_structure(self):
        """Test the DCRegion blueprint has correct structure."""
        blueprint = get_builtin_blueprints()["DCRegion"]

        assert "groups" in blueprint
        assert "adjacency" in blueprint

        # Should have one dc group
        assert "dc" in blueprint["groups"]
        dc_group = blueprint["groups"]["dc"]
        assert dc_group["node_count"] == 1
        assert dc_group["name_template"] == "dc"
        assert "attrs" in dc_group
        assert dc_group["attrs"]["role"] == "dc"
        assert dc_group["attrs"]["hw_type"] == "dc_node"

        # Should have no adjacency rules (single node)
        assert blueprint["adjacency"] == []
