"""Tests for the components library module."""

from __future__ import annotations

from topogen.components_lib import get_builtin_components


class TestComponentsLib:
    def test_get_builtin_components(self):
        components = get_builtin_components()

        assert isinstance(components, dict)
        assert len(components) > 0

        for name, comp in components.items():
            assert isinstance(name, str)
            assert isinstance(comp, dict)
            assert "component_type" in comp
            assert "description" in comp
            assert "capex" in comp
            assert "power_watts" in comp

    def test_component_structure_consistency(self):
        components = get_builtin_components()

        required_fields = ["component_type", "description", "capex", "power_watts"]

        for name, comp in components.items():
            for field in required_fields:
                assert field in comp, f"Component '{name}' missing field '{field}'"

            assert isinstance(comp["component_type"], str)
            assert isinstance(comp["description"], str)
            assert isinstance(comp["capex"], (int, float))
            assert isinstance(comp["power_watts"], (int, float))

            if "attrs" in comp:
                assert isinstance(comp["attrs"], dict)

    def test_component_capex_values(self):
        components = get_builtin_components()

        for name, comp in components.items():
            capex = comp["capex"]
            assert capex >= 0, f"Component '{name}' has negative capex: {capex}"
            assert capex < 1_000_000, (
                f"Component '{name}' has unrealistic capex: {capex}"
            )

    def test_component_power_values(self):
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
