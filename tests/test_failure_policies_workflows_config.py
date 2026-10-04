"""Selection settings must represent behavior the pipeline actually applies."""

import pytest
import yaml

from topogen.config import TopologyConfig


@pytest.mark.parametrize(
    "section, default",
    [
        ("failure_policies", "single_random_link_failure"),
        ("workflows", "design_analysis_brief"),
    ],
)
def test_policy_selection(section, default):
    raw = yaml.safe_load(open("examples/small_baseline.yml"))
    raw.pop(section, None)
    assert (
        getattr(TopologyConfig._from_dict(raw), section).assignments.default == default
    )
    raw[section] = {"assignments": {"default": "custom"}}
    assert (
        getattr(TopologyConfig._from_dict(raw), section).assignments.default == "custom"
    )


@pytest.mark.parametrize("section", ["failure_policies", "workflows"])
@pytest.mark.parametrize(
    "value",
    [
        "not a mapping",
        {"assignments": "not a mapping"},
        {"assignments": {"scenario_overrides": {}}},
        {"library": {}},
    ],
)
def test_invalid_or_ignored_selection_controls_rejected(section, value):
    raw = yaml.safe_load(open("examples/small_baseline.yml"))
    raw[section] = value
    with pytest.raises(ValueError, match=section):
        TopologyConfig._from_dict(raw)
