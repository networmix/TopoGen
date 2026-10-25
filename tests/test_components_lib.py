"""Component definitions are owned and contain the referenced builtin hardware."""

from topogen.library import load_library


def test_component_definitions():
    library = load_library(None)
    assert {"CoreRouter", "800G-ZR+", "800G-DR4"} <= library.components.keys()
    assert library.components["800G-ZR+"]["capacity"] == 800
    assert library.components["CoreRouter"]["ports"] > 0
