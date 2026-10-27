"""Hardware extraction accepts read-only NetGraph attrs without copying rules."""

from types import MappingProxyType

import pytest

from netlab_topogen.validation.helpers import _node_hw_from_attrs


def test_hardware_from_readonly_attrs():
    attrs = MappingProxyType(
        {"hardware": MappingProxyType({"component": "P", "count": 3})}
    )
    assert _node_hw_from_attrs(attrs) == ("P", 3.0)
    assert _node_hw_from_attrs({}) == (None, 0.0)
    with pytest.raises(ValueError):
        _node_hw_from_attrs({"hardware": {"component": "P", "count": "bad"}})
