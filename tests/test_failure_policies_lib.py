"""An explicit library directory overrides named failure definitions only."""

import yaml

from topogen.library import load_library


def test_failure_policy_override(tmp_path):
    (tmp_path / "failure_policies.yml").write_text(
        yaml.safe_dump(
            {
                "empty": {
                    "modes": [{"weight": 1, "rules": []}],
                    "attrs": {"owner": "custom"},
                }
            }
        )
    )
    loaded = load_library(tmp_path)
    assert loaded.failure_policies["empty"]["attrs"]["owner"] == "custom"
    assert "single_random_link_failure" in loaded.failure_policies
    assert "owner" not in load_library(None).failure_policies["empty"].get("attrs", {})
