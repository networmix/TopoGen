from __future__ import annotations

import math

import pytest

from topogen.validation.helpers import (
    _build_ig_coord_map,
    _float_or_nan,
    _node_hw_from_attrs,
)


def test_build_ig_coord_map_extracts_and_rejects_invalid_coordinates():
    assert _build_ig_coord_map(
        {"nodes": [{"node_type": "metro", "name": "A", "x": 1, "y": 2}]}
    ) == {"A": (1.0, 2.0)}
    with pytest.raises(ValueError):
        _build_ig_coord_map(
            {"nodes": [{"node_type": "metro", "name": "A", "x": "bad", "y": 2}]}
        )


def test_float_or_nan() -> None:
    assert _float_or_nan(1.25) == 1.25
    assert math.isnan(_float_or_nan("not-a-number"))


def test_node_hw_from_attrs() -> None:
    comp, count = _node_hw_from_attrs({"hardware": {"component": "P", "count": 3}})
    assert comp == "P" and count == 3.0
    # Missing comp
    comp2, count2 = _node_hw_from_attrs({})
    assert comp2 is None and count2 == 0.0
    with pytest.raises(ValueError):
        _node_hw_from_attrs({"hardware": {"component": "X", "count": "bad"}})
