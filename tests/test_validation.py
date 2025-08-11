"""Tests for scenario validation utilities."""

from __future__ import annotations

from pathlib import Path

from topogen.validation import validate_scenario_dict, validate_scenario_yaml


def _minimal_scenario() -> dict:
    # Minimal valid-ish structure with one metro pop and dc group
    return {
        "network": {
            "groups": {
                "metro1/pop[2]": {
                    "use_blueprint": "SingleRouter",
                    "attrs": {
                        "metro_name": "Denver",
                        "metro_name_orig": "Denver",
                        "metro_id": 1,
                        "location_x": 10.0,
                        "location_y": 20.0,
                    },
                },
                "metro1/dc[1]": {
                    "use_blueprint": "DCRegion",
                    "attrs": {
                        "metro_name": "Denver",
                        "metro_name_orig": "Denver",
                        "metro_id": 1,
                        "location_x": 10.0,
                        "location_y": 20.0,
                        "mw_per_dc_region": 50.0,
                        "gbps_per_mw": 250.0,
                    },
                },
            },
            "adjacency": [],
        },
        # Provide empty sections to satisfy reference checks by default
        "failure_policy_set": {},
        "traffic_matrix_set": {},
        "workflow": [],
    }


def test_validate_scenario_dict_attr_mismatch_detected():
    data = _minimal_scenario()
    # Introduce mismatch
    data["network"]["groups"]["metro1/dc[1]"]["attrs"]["location_x"] = 11.0
    issues = validate_scenario_dict(data)
    assert any("location_x mismatch" in s for s in issues)


def test_validate_scenario_dict_missing_required_dc_attrs():
    data = _minimal_scenario()
    # Remove required attribute
    del data["network"]["groups"]["metro1/dc[1]"]["attrs"]["mw_per_dc_region"]
    issues = validate_scenario_dict(data)
    assert any("dc attrs missing required 'mw_per_dc_region'" in s for s in issues)


def test_validate_scenario_yaml_workflow_references_checked(tmp_path: Path):
    # Craft YAML string referencing missing items
    yaml_text = """
network:
  groups:
    metro1/pop[2]:
      use_blueprint: SingleRouter
      attrs:
        metro_name: Denver
        metro_name_orig: Denver
        metro_id: 1
        location_x: 10.0
        location_y: 20.0
    metro1/dc[1]:
      use_blueprint: DCRegion
      attrs:
        metro_name: Denver
        metro_name_orig: Denver
        metro_id: 1
        location_x: 10.0
        location_y: 20.0
        mw_per_dc_region: 50.0
        gbps_per_mw: 250.0
workflow:
  - step_type: TrafficMatrixPlacementAnalysis
    name: tm
    matrix_name: missing_matrix
    failure_policy: missing_policy
"""
    issues = validate_scenario_yaml(
        yaml_text, integrated_graph_path=None, run_ngraph=False
    )
    assert any("references missing traffic matrix" in s for s in issues)
    assert any("references missing failure_policy" in s for s in issues)


def test_validate_scenario_dict_does_not_flag_adjacency_strings():
    data = _minimal_scenario()
    data["network"]["adjacency"].append(
        {"source": "missing", "target": "metro1/pop[2]"}
    )
    data["network"]["adjacency"].append(
        {"source": "metro1/dc[1]", "target": "missing2"}
    )
    # String-level adjacency checks are not performed anymore; rely on ngraph
    issues = validate_scenario_dict(data)
    assert not any("adjacency references missing group" in s for s in issues)


def test_validate_scenario_yaml_isolated_nodes_flagged():
    # No adjacency at all -> groups appear isolated at scenario level
    yaml_text = """
network:
  groups:
    metro1/pop[1]:
      use_blueprint: SingleRouter
      attrs:
        metro_name: Denver
        metro_name_orig: Denver
        metro_id: 1
        location_x: 10.0
        location_y: 20.0
    metro1/dc[1]:
      use_blueprint: DCRegion
      attrs:
        metro_name: Denver
        metro_name_orig: Denver
        metro_id: 1
        location_x: 10.0
        location_y: 20.0
        mw_per_dc_region: 50.0
        gbps_per_mw: 250.0
"""
    # Without adjacency, fallback scenario-level isolation should be reported.
    issues = validate_scenario_yaml(
        yaml_text, integrated_graph_path=None, run_ngraph=False
    )
    assert any("appears isolated" in s for s in issues)
