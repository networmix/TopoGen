"""Supported component assignment settings and obsolete-key rejection."""

import pytest
import yaml

from topogen.config import ComponentsConfig, TopologyConfig


def test_component_defaults():
    assert ComponentsConfig().hw_component == {}
    assert ComponentsConfig().optics == {}


def test_component_maps(tmp_path):
    config = yaml.safe_load(open("examples/small_baseline.yml"))
    config["components"] = {
        "hw_component": {"core": "CoreRouter"},
        "optics": {"core->dc": "400G-LR4", "dc->core": "400G-LR4"},
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    actual = TopologyConfig.from_yaml(path).components
    assert actual.hw_component == config["components"]["hw_component"]
    assert actual.optics == config["components"]["optics"]


@pytest.mark.parametrize("key", ["assignments", "library"])
def test_obsolete_component_controls_rejected(key):
    config = yaml.safe_load(open("examples/small_baseline.yml"))
    config["components"] = {key: {}}
    with pytest.raises(ValueError, match=key):
        TopologyConfig._from_dict(config)
