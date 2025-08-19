from __future__ import annotations

import math

from topogen.validation.helpers import (
    _build_ig_coord_map,
    _float_or_nan,
    _node_hw_from_attrs,
)


def test_build_ig_coord_map_filters_and_extracts() -> None:
    ig = {
        "nodes": [
            {"node_type": "metro", "name": "A", "x": 1.0, "y": 2.0},
            {"node_type": "metro+highway", "name": "B", "x": 3, "y": 4},
            {"node_type": "highway", "name": "H", "x": 5, "y": 6},  # excluded
            {"node_type": "metro", "name": "", "x": 7, "y": 8},  # no name
            {"node_type": "metro", "name": "C", "x": "bad", "y": 9},  # bad coords
        ]
    }
    mp = _build_ig_coord_map(ig)
    assert mp == {"A": (1.0, 2.0), "B": (3.0, 4.0)}


def test_float_or_nan() -> None:
    assert _float_or_nan(1.25) == 1.25
    assert math.isnan(_float_or_nan("not-a-number"))


def test_node_hw_from_attrs() -> None:
    comp, count = _node_hw_from_attrs({"hardware": {"component": "P", "count": 3}})
    assert comp == "P" and count == 3.0
    # Missing comp
    comp2, count2 = _node_hw_from_attrs({})
    assert comp2 is None and count2 == 0.0
    # Bad count → default 1.0
    comp3, count3 = _node_hw_from_attrs(
        {"hardware": {"component": "X", "count": "bad"}}
    )
    assert comp3 == "X" and count3 == 1.0
