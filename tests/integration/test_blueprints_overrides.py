from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from topogen.blueprints_lib import get_builtin_blueprints


def test_user_blueprints_merge_overrides(tmp_path: Path, monkeypatch) -> None:
    # Prepare user library in CWD/lib
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(parents=True)
    user_bp = {
        "SingleRouter": {
            "groups": {
                "core": {
                    "node_count": 2,
                    "name_template": "core{node_num}",
                    "attrs": {"role": "core"},
                }
            },
            "adjacency": [],
        }
    }
    (lib_dir / "blueprints.yml").write_text(yaml.safe_dump(user_bp))
    monkeypatch.chdir(tmp_path)

    bps = get_builtin_blueprints()
    # Ensure override took effect
    sr = bps["SingleRouter"]
    assert sr["groups"]["core"]["node_count"] == 2
    assert sr["groups"]["core"]["name_template"].startswith("core")


def test_user_blueprints_invalid_yaml_raises(tmp_path: Path, monkeypatch) -> None:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(parents=True)
    # Invalid YAML content
    (lib_dir / "blueprints.yml").write_text("groups: [invalid: :\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Failed to parse YAML"):
        get_builtin_blueprints()


def test_user_blueprints_non_mapping_raises(tmp_path: Path, monkeypatch) -> None:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(parents=True)
    # YAML parses to a list, not a mapping
    (lib_dir / "blueprints.yml").write_text("- not: a-mapping\n- another: item\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must be a mapping"):
        get_builtin_blueprints()
