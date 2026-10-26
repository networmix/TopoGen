"""Tests for component functionality in scenario builder."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from topogen.config import (
    TopologyConfig,
)
from topogen.library import load_library, thaw
from topogen.scenario.document import _components, check_components
from topogen.scenario.records import Metro, MetroSettings
from topogen.scenario.sites import finalize_blueprints


def finalized(names, config, library):
    settings = tuple(
        MetroSettings(
            Metro(i, name, name, name, 0, 0, 1), 1, 0, name, "DCRegion", {}, ()
        )
        for i, name in enumerate(sorted(names), 1)
    )
    return finalize_blueprints(settings, library, config.components.hw_component)


class TestComponentsScenarioBuilder:
    def create_test_config(self) -> TopologyConfig:
        config = TopologyConfig()
        config.components.hw_component = {
            "spine": "CoreRouter",
            "leaf": "CoreRouter",
            "core": "CoreRouter",
            "dc": "CoreRouter",
        }
        config.components.optics = {
            "core->core": "800G-ZR+",
            "leaf->spine": "800G-DR4",
            "spine->leaf": "800G-DR4",
            "core->leaf": "1600G-2xDR4",
            "leaf->core": "1600G-2xDR4",
        }
        return config

    def test_components_basic(self):
        config = self.create_test_config()
        used_blueprints = {"SingleRouter", "FullMesh4"}

        components = _components(
            config,
            finalized(used_blueprints, config, load_library(Path("lib"))),
            load_library(Path("lib")),
        )

        assert isinstance(components, dict)
        assert len(components) > 0

        assert "CoreRouter" in components
        assert "800G-ZR+" in components

    def test_components_with_role_assignments(self):
        config = self.create_test_config()
        # Pick any available Clos-style blueprint dynamically
        bps = thaw(load_library(Path("lib")).blueprints)
        clos_name = next(
            (
                name
                for name, bp in bps.items()
                if {"spine", "leaf"} <= set(bp.get("nodes", {}).keys())
            ),
            None,
        )
        assert clos_name is not None, "No Clos-style blueprint found"
        used_blueprints = {clos_name}

        components = _components(
            config,
            finalized(used_blueprints, config, load_library(Path("lib"))),
            load_library(Path("lib")),
        )

        assert isinstance(components, dict)
        assert "CoreRouter" in components
        assert "800G-ZR+" in components

    def test_components_uses_merged_library(self):
        """The merged library includes the configured built-in components."""
        config = self.create_test_config()
        # Pick any available Clos-style blueprint dynamically
        bps = thaw(load_library(Path("lib")).blueprints)
        clos_name = next(
            (
                name
                for name, bp in bps.items()
                if {"spine", "leaf"} <= set(bp.get("nodes", {}).keys())
            ),
            None,
        )
        assert clos_name is not None, "No Clos-style blueprint found"
        used_blueprints = {clos_name}
        components = _components(
            config,
            finalized(used_blueprints, config, load_library(Path("lib"))),
            load_library(Path("lib")),
        )
        assert "CoreRouter" in components

    def test_components_check_refuses_unknown_platform(self):
        config = self.create_test_config()

        config.components.hw_component["spine"] = "NonExistentChassis"

        used_blueprints = {"SingleRouter"}

        with pytest.raises(ValueError, match="Unknown components"):
            check_components(
                config.components,
                finalized(used_blueprints, config, load_library(Path("lib"))),
                load_library(Path("lib")),
            )

    def testfinalized_basic(self):
        config = self.create_test_config()
        used_blueprints = {"SingleRouter"}

        blueprints = finalized(used_blueprints, config, load_library(Path("lib")))

        assert isinstance(blueprints, Mapping)
        assert "SingleRouter" in blueprints

        single_router = blueprints["SingleRouter"]
        assert "nodes" in single_router

        core_group = single_router["nodes"]["core"]
        assert "attrs" in core_group
        assert core_group["attrs"].get("hardware", {}).get("component") == "CoreRouter"

    def testfinalized_with_role_assignments(self):
        config = self.create_test_config()
        # Pick any available Clos-style blueprint dynamically
        bps = thaw(load_library(Path("lib")).blueprints)
        clos_name = next(
            (
                name
                for name, bp in bps.items()
                if {"spine", "leaf"} <= set(bp.get("nodes", {}).keys())
            ),
            None,
        )
        assert clos_name is not None, "No Clos-style blueprint found"
        used_blueprints = {clos_name}

        blueprints = finalized(used_blueprints, config, load_library(Path("lib")))

        assert clos_name in blueprints

        clos_blueprint = blueprints[clos_name]
        spine_group = clos_blueprint["nodes"]["spine"]
        leaf_group = clos_blueprint["nodes"]["leaf"]

        assert spine_group["attrs"].get("hardware", {}).get("component") == "CoreRouter"
        assert leaf_group["attrs"].get("hardware", {}).get("component") == "CoreRouter"

    def testfinalized_preserves_existing_attrs(self):
        config = self.create_test_config()
        # Pick any available Clos-style blueprint dynamically
        bps = thaw(load_library(Path("lib")).blueprints)
        clos_name = next(
            (
                name
                for name, bp in bps.items()
                if {"spine", "leaf"} <= set(bp.get("nodes", {}).keys())
            ),
            None,
        )
        assert clos_name is not None, "No Clos-style blueprint found"
        used_blueprints = {clos_name}

        blueprints = finalized(used_blueprints, config, load_library(Path("lib")))

        clos_blueprint = blueprints[clos_name]
        spine_group = clos_blueprint["nodes"]["spine"]

        assert spine_group["attrs"]["role"] == "spine"
        assert spine_group["attrs"]["tier"] == "spine"

        assert spine_group["attrs"].get("hardware", {}).get("component") == "CoreRouter"

    def testfinalized_unknown_blueprint(self):
        config = self.create_test_config()
        used_blueprints = {"UnknownBlueprint"}

        with pytest.raises(ValueError, match="Unknown blueprint: UnknownBlueprint"):
            finalized(used_blueprints, config, load_library(Path("lib")))

    def testfinalized_no_role_assignment(self):
        config = TopologyConfig()
        used_blueprints = {"SingleRouter"}

        blueprints = finalized(used_blueprints, config, load_library(Path("lib")))

        assert "SingleRouter" in blueprints

        core_group = blueprints["SingleRouter"]["nodes"]["core"]
        assert "hardware" not in core_group["attrs"]

    def testfinalized_role_inference(self):
        config = self.create_test_config()

        used_blueprints = {"FullMesh4"}

        blueprints = finalized(used_blueprints, config, load_library(Path("lib")))

        full_mesh = blueprints["FullMesh4"]
        core_group = full_mesh["nodes"]["core"]

        assert core_group["attrs"].get("hardware", {}).get("component") == "CoreRouter"

    def test_components_section_includes_only_referenced(self):
        config = self.create_test_config()

        used_blueprints = {"SingleRouter"}
        components = _components(
            config,
            finalized(used_blueprints, config, load_library(Path("lib"))),
            load_library(Path("lib")),
        )

        assert "UnusedChassis1" not in components
        assert "UnusedChassis2" not in components
        assert "UnusedOptic1" not in components

        assert "CoreRouter" in components
        assert "800G-ZR+" in components

    def test_empty_used_blueprints(self):
        config = self.create_test_config()
        used_blueprints = set()

        components = _components(
            config,
            finalized(used_blueprints, config, load_library(Path("lib"))),
            load_library(Path("lib")),
        )
        blueprints = finalized(used_blueprints, config, load_library(Path("lib")))

        assert isinstance(components, dict)
        assert isinstance(blueprints, Mapping)
        assert len(blueprints) == 0
