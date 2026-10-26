from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from netlab_topogen.library import load_library, thaw


def test_user_blueprints_merge_overrides(tmp_path: Path, monkeypatch) -> None:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(parents=True)
    user_bp = {
        "SingleRouter": {
            "nodes": {
                "core": {
                    "count": 2,
                    "template": "core{n}",
                    "attrs": {"role": "core"},
                }
            },
            "links": [],
        }
    }
    (lib_dir / "blueprints.yml").write_text(yaml.safe_dump(user_bp))
    monkeypatch.chdir(tmp_path)

    bps = thaw(load_library(Path("lib")).blueprints)
    sr = bps["SingleRouter"]
    assert sr["nodes"]["core"]["count"] == 2
    assert sr["nodes"]["core"]["template"].startswith("core")


def test_user_blueprints_invalid_yaml_raises(tmp_path: Path, monkeypatch) -> None:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "blueprints.yml").write_text("groups: [invalid: :\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Failed to parse YAML"):
        thaw(load_library(Path("lib")).blueprints)


def test_user_blueprints_non_mapping_raises(tmp_path: Path, monkeypatch) -> None:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(parents=True)
    # YAML parses to a list, not a mapping
    (lib_dir / "blueprints.yml").write_text("- not: a-mapping\n- another: item\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must be a mapping"):
        thaw(load_library(Path("lib")).blueprints)
