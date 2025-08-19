from __future__ import annotations

from types import SimpleNamespace

import pytest

from topogen.scenario.config import _determine_metro_settings


def _build_cfg() -> SimpleNamespace:
    # Build a minimal TopologyConfig-like object for testing
    def link(
        cap,
        cost,
        attrs=None,
        match=None,
        role_pairs=None,
        striping=None,
        mode="mesh",
    ):
        return SimpleNamespace(
            capacity=cap,
            cost=cost,
            attrs=({} if attrs is None else attrs),
            match=({} if match is None else match),
            role_pairs=([] if role_pairs is None else role_pairs),
            striping=(striping or {}),
            mode=mode,
        )

    defaults = SimpleNamespace(
        pop_per_metro=2,
        site_blueprint="SingleRouter",
        dc_regions_per_metro=1,
        dc_region_blueprint="DCRegion",
        intra_metro_link=link(
            3200,
            1,
            attrs={"a": 1},
            match={"m": 1},
            role_pairs=["core|core"],
            striping={},
        ),
        inter_metro_link=link(
            3200,
            500,
            attrs={"b": 2},
            match={"n": 2},
            role_pairs=["core|core"],
            striping={},
            mode="mesh",
        ),
        dc_to_pop_link=link(
            3200, 1, attrs={"c": 3}, match={"k": 3}, role_pairs=["core|dc"], striping={}
        ),
    )
    overrides = {
        "denver": {
            "pop_per_metro": 3,
            "site_blueprint": "FullMesh4",
            "dc_regions_per_metro": 2,
            "dc_region_blueprint": "DCRegion",
            "intra_metro_link": {
                "capacity": 6400,
                "cost": 2,
                "attrs": {"x": 9},
                "match": {"m": 7},
                "mode": "mesh",
            },
            "inter_metro_link": {
                "capacity": 6400,
                "cost": 600,
                "attrs": {"y": 8},
                "role_pairs": ["core|core"],
                "mode": "one_to_one",
            },
            "dc_to_pop_link": {
                "capacity": 1600,
                "cost": 1,
                "attrs": {"z": 7},
            },
        }
    }
    return SimpleNamespace(
        build=SimpleNamespace(build_defaults=defaults, build_overrides=overrides)
    )


def test_override_key_validation_errors_on_unknown_slug():
    cfg = _build_cfg()
    metros = [{"name": "A", "name_orig": "A"}]
    # Add an unknown override key
    cfg.build.build_overrides["unknown"] = {}
    with pytest.raises(ValueError):
        _determine_metro_settings(metros, cfg)


def test_defaults_and_overrides_application_and_validation():
    cfg = _build_cfg()
    metros = [{"name": "Denver", "name_orig": "Denver"}, {"name": "B"}]
    settings = _determine_metro_settings(metros, cfg)
    # Denver overridden
    den = settings["Denver"]
    assert den["pop_per_metro"] == 3
    assert den["site_blueprint"] == "FullMesh4"
    assert den["intra_metro_link"]["capacity"] == 6400
    # Attribute merging: attrs updated, match replaced
    assert den["intra_metro_link"]["attrs"]["x"] == 9
    assert den["intra_metro_link"]["match"] == {"m": 7}
    # Non-overridden metro B uses defaults
    b = settings["B"]
    assert b["pop_per_metro"] == 2
    assert b["dc_regions_per_metro"] == 1
    # Validate raises on invalid values
    cfg.build.build_overrides = {"denver": {"pop_per_metro": 0}}
    with pytest.raises(ValueError):
        _determine_metro_settings(metros, cfg)
    cfg = _build_cfg()
    cfg.build.build_overrides = {"denver": {"dc_regions_per_metro": -1}}
    with pytest.raises(ValueError):
        _determine_metro_settings(metros, cfg)
    cfg = _build_cfg()
    cfg.build.build_overrides = {"denver": {"intra_metro_link": {"capacity": 0}}}
    with pytest.raises(ValueError):
        _determine_metro_settings(metros, cfg)
    cfg = _build_cfg()
    cfg.build.build_overrides = {"denver": {"inter_metro_link": {"cost": -1}}}
    with pytest.raises(ValueError):
        _determine_metro_settings(metros, cfg)
    cfg = _build_cfg()
    cfg.build.build_overrides = {"denver": {"dc_to_pop_link": {"capacity": 0}}}
    with pytest.raises(ValueError):
        _determine_metro_settings(metros, cfg)
