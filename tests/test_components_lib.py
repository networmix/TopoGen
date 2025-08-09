"""Tests for the simplified components library module."""

from __future__ import annotations

import pytest

from topogen.components_lib import (
    get_builtin_component,
    get_builtin_components,
    get_components_by_role,
    get_components_by_type,
    list_builtin_component_names,
)


class TestComponentsLib:
    """Test component library functionality."""

    def test_get_builtin_components(self):
        """Test getting all built-in components."""
        components = get_builtin_components()

        assert isinstance(components, dict)
        assert len(components) > 0

        # Check that all components have required fields
        for name, comp in components.items():
            assert isinstance(name, str)
            assert isinstance(comp, dict)
            assert "component_type" in comp
            assert "description" in comp
            assert "cost" in comp
            assert "power_watts" in comp

    def test_get_builtin_component_valid(self):
        """Test getting a specific valid component."""
        component = get_builtin_component("CoreRouter")

        assert isinstance(component, dict)
        assert component["component_type"] == "chassis"
        assert component["description"] == "Core router for metro backbone"
        assert component["cost"] == 650_000.0
        assert component["power_watts"] == 40_000.0

    def test_get_builtin_component_invalid(self):
        """Test getting a non-existent component raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_builtin_component("NonExistentComponent")

        assert "Component 'NonExistentComponent' not found" in str(exc_info.value)

    def test_list_builtin_component_names(self):
        """Test listing all component names."""
        names = list_builtin_component_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert "CoreRouter" in names
        assert "800G-ZR+" in names
        assert all(isinstance(name, str) for name in names)

    def test_get_components_by_type_chassis(self):
        """Test filtering components by chassis type."""
        chassis_components = get_components_by_type("chassis")

        assert isinstance(chassis_components, dict)
        assert len(chassis_components) > 0

        for _name, comp in chassis_components.items():
            assert comp["component_type"] == "chassis"

        # Should include known chassis components from the minimal library
        assert "CoreRouter" in chassis_components

    def test_get_components_by_type_optic(self):
        """Test filtering components by optic type."""
        optic_components = get_components_by_type("optic")

        assert isinstance(optic_components, dict)
        assert len(optic_components) > 0

        for _name, comp in optic_components.items():
            assert comp["component_type"] == "optic"

        # Should include known optic components from the minimal library
        assert "800G-ZR+" in optic_components

    def test_get_components_by_type_nonexistent(self):
        """Test filtering by non-existent type returns empty dict."""
        result = get_components_by_type("nonexistent")
        assert result == {}

    def test_get_components_by_role_spine(self):
        """Test filtering components by spine role."""
        spine_components = get_components_by_role("spine")

        assert isinstance(spine_components, dict)
        # Minimal library may not have explicit spine components
        for _name, comp in spine_components.items():
            assert comp.get("attrs", {}).get("role") == "spine"

    def test_get_components_by_role_core(self):
        """Test filtering components by core role."""
        core_components = get_components_by_role("core")

        assert isinstance(core_components, dict)

        for _name, comp in core_components.items():
            assert comp.get("attrs", {}).get("role") == "core"

    def test_get_components_by_role_nonexistent(self):
        """Test filtering by non-existent role returns empty dict."""
        result = get_components_by_role("nonexistent")
        assert result == {}

    def test_component_structure_consistency(self):
        """Test that all components have consistent structure."""
        components = get_builtin_components()

        required_fields = ["component_type", "description", "cost", "power_watts"]

        for name, comp in components.items():
            # Check required fields exist
            for field in required_fields:
                assert field in comp, f"Component '{name}' missing field '{field}'"

            # Check field types
            assert isinstance(comp["component_type"], str)
            assert isinstance(comp["description"], str)
            assert isinstance(comp["cost"], (int, float))
            assert isinstance(comp["power_watts"], (int, float))

            # Check attrs structure if present
            if "attrs" in comp:
                assert isinstance(comp["attrs"], dict)

    def test_component_cost_values(self):
        """Test that component costs are reasonable."""
        components = get_builtin_components()

        for name, comp in components.items():
            cost = comp["cost"]
            assert cost >= 0, f"Component '{name}' has negative cost: {cost}"
            assert cost < 1_000_000, f"Component '{name}' has unrealistic cost: {cost}"

    def test_component_power_values(self):
        """Test that component power values are reasonable."""
        components = get_builtin_components()

        for name, comp in components.items():
            power = comp["power_watts"]
            assert power >= 0, f"Component '{name}' has negative power: {power}"
            # Allow up to 50kW for chassis
            assert power <= 50_000, f"Component '{name}' has unrealistic power: {power}"

            if "power_watts_max" in comp:
                power_max = comp["power_watts_max"]
                assert power_max >= power, (
                    f"Component '{name}' max power less than typical power"
                )
