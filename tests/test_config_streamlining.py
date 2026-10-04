"""Strict configuration must reject ambiguous and ignored options."""

from pathlib import Path

import pytest
import yaml

from topogen.config import TopologyConfig


def config_dict():
    return yaml.safe_load(Path("examples/small_baseline.yml").read_text())


@pytest.mark.parametrize("key", [0.5, True])
def test_fractional_or_boolean_priority_is_not_truncated(key):
    data = config_dict()
    data["traffic"]["priority_ratios"] = {key: 1}
    with pytest.raises(ValueError):
        TopologyConfig._from_dict(data)


def test_duplicate_normalized_priority_is_rejected():
    data = config_dict()
    data["traffic"]["priority_ratios"] = {0: 0.5, "0": 1.0}
    with pytest.raises(ValueError, match="Duplicate"):
        TopologyConfig._from_dict(data)


def test_duplicate_metro_override_is_rejected():
    data = config_dict()
    data["build"]["build_overrides"] = [
        {"metros": ["New York"], "pop_per_metro": 2},
        {"metros": ["new-york"], "pop_per_metro": 3},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        TopologyConfig._from_dict(data)


def test_unknown_striping_option_is_rejected():
    data = config_dict()
    data["build"]["build_defaults"]["inter_metro_link"]["striping"] = {
        "width": 1,
        "widht": 2,
    }
    with pytest.raises(ValueError):
        TopologyConfig._from_dict(data)
