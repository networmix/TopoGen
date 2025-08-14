"""Tests for scenario validation utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

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


def test_dc_capacity_vs_demand_validation():
    # Build a simple scenario with one DC and a DC->PoP adjacency that has limited capacity
    data = _minimal_scenario()
    # Add one dc_to_pop adjacency with target_capacity 1000
    data["network"]["adjacency"] = [
        {
            "source": "metro1/dc1",
            "target": "metro1/pop1",
            "pattern": "one_to_one",
            "link_params": {
                "capacity": 1000.0,
                "cost": 1,
                "attrs": {
                    "link_type": "dc_to_pop",
                    "source_metro": "Denver",
                    "target_metro": "Denver",
                    "target_capacity": 1000.0,
                },
            },
        }
    ]
    # Traffic matrix with a single class that demands 1200 out of the DC and 800 into the DC
    data["traffic_matrix_set"] = {
        "tm": [
            {
                "source_path": "^metro1/dc1/.*",
                "sink_path": "^metro1/dc1/.*",
                "mode": "pairwise",
                "priority": 0,
                "demand": 1200.0,
            },
            {
                "source_path": "^metro2/dc1/.*",
                "sink_path": "^metro1/dc1/.*",
                "mode": "pairwise",
                "priority": 0,
                "demand": 800.0,
            },
        ]
    }
    issues = validate_scenario_dict(data)
    # Egress 1200 > capacity 1000 => violation
    assert any(
        "egress demand" in s and "exceeds adjacency capacity" in s for s in issues
    )
    # Ingress 800 <= capacity 1000 => no ingress violation for metro1/dc1
    assert not any("ingress demand 800.0" in s for s in issues)


def test_groups_that_expand_to_zero_nodes_are_flagged(monkeypatch):
    # Patch DSL expansion to return zero nodes and zero links
    class _FakeNet:
        def __init__(self) -> None:
            self.nodes = {}
            self.links = {}

    def _fake_expand(_dsl: dict):  # noqa: ANN001
        return _FakeNet()

    monkeypatch.setattr(
        "ngraph.dsl.blueprints.expand.expand_network_dsl", _fake_expand, raising=True
    )

    # Patch Scenario.from_yaml to avoid schema errors and isolation noise
    class _FakeScenario:
        @classmethod
        def from_yaml(cls, _y: str):  # noqa: ANN001
            return cls()

        class network:  # noqa: N801 - name per production API
            @staticmethod
            def to_strict_multidigraph(add_reverse: bool = True):  # noqa: FBT001, FBT002
                class _G:
                    @staticmethod
                    def get_nodes():
                        return {}

                    @staticmethod
                    def get_edges():
                        return {}

                return _G()

    monkeypatch.setattr("ngraph.scenario.Scenario", _FakeScenario, raising=True)
    # A group with zero count via an invalid blueprint should be flagged.
    # Use a blueprint that exists but set an impossible range in the group path.
    data = {
        "blueprints": {},
        "network": {
            "groups": {
                # Invalid range [1-0] yields zero nodes
                "metro1/pop[1-0]": {
                    "use_blueprint": "SingleRouter",
                    "attrs": {
                        "metro_name": "X",
                        "metro_name_orig": "X",
                        "metro_id": 1,
                        "location_x": 0.0,
                        "location_y": 0.0,
                    },
                }
            },
            "adjacency": [],
        },
        "failure_policy_set": {},
        "traffic_matrix_set": {},
        "workflow": [],
    }
    issues = validate_scenario_yaml(
        yaml.safe_dump(data, sort_keys=False),
        integrated_graph_path=None,
        run_ngraph=True,
    )
    assert any("expands to 0 nodes" in s for s in issues)


def test_adjacencies_that_expand_to_zero_links_are_flagged(monkeypatch):
    # Patch DSL expansion to return zero nodes and zero links
    class _FakeNet:
        def __init__(self) -> None:
            self.nodes = {}
            self.links = {}

    def _fake_expand(_dsl: dict):  # noqa: ANN001
        return _FakeNet()

    monkeypatch.setattr(
        "ngraph.dsl.blueprints.expand.expand_network_dsl", _fake_expand, raising=True
    )

    # Patch Scenario.from_yaml to avoid schema errors and isolation noise
    class _FakeScenario:
        @classmethod
        def from_yaml(cls, _y: str):  # noqa: ANN001
            return cls()

        class network:  # noqa: N801 - name per production API
            @staticmethod
            def to_strict_multidigraph(add_reverse: bool = True):  # noqa: FBT001, FBT002
                class _G:
                    @staticmethod
                    def get_nodes():
                        return {}

                    @staticmethod
                    def get_edges():
                        return {}

                return _G()

    monkeypatch.setattr("ngraph.scenario.Scenario", _FakeScenario, raising=True)
    # Construct a scenario with a valid group but an adjacency using nonexistent endpoints
    data = {
        "network": {
            "groups": {
                "metro1/pop[1]": {
                    "use_blueprint": "SingleRouter",
                    "attrs": {
                        "metro_name": "Y",
                        "metro_name_orig": "Y",
                        "metro_id": 1,
                        "location_x": 0.0,
                        "location_y": 0.0,
                    },
                },
            },
            "adjacency": [
                {
                    "source": "metro1/missing",
                    "target": "metro1/also_missing",
                    "pattern": "one_to_one",
                    "link_params": {"capacity": 100.0},
                }
            ],
        },
        "failure_policy_set": {},
        "traffic_matrix_set": {},
        "workflow": [],
    }
    issues = validate_scenario_yaml(
        yaml.safe_dump(data, sort_keys=False),
        integrated_graph_path=None,
        run_ngraph=True,
    )
    assert any("adjacency[0] expands to 0 links" in s for s in issues)


def yaml_dump(d: dict) -> str:
    return yaml.safe_dump(d, sort_keys=False)


def test_node_hardware_presence_audited():
    # Scenario with a node role mapped to hardware but missing assignment on node
    data = {
        "blueprints": {
            "SingleRouter": {
                "groups": {
                    "core": {
                        "node_count": 1,
                        "name_template": "core",
                        "attrs": {"role": "core"},
                    }
                },
                "adjacency": [],
            }
        },
        "components": {
            "CoreRouter": {
                "component_type": "chassis",
                "capacity": 1000.0,
                "ports": 10,
            },
            "hw_component": {"core": "CoreRouter"},
        },
        "network": {
            "groups": {
                "metro1/pop[1]": {
                    "use_blueprint": "SingleRouter",
                    "attrs": {
                        "metro_name": "X",
                        "metro_id": 1,
                        "location_x": 0.0,
                        "location_y": 0.0,
                    },
                }
            },
            "adjacency": [],
        },
    }
    issues = validate_scenario_yaml(
        yaml_dump(data), integrated_graph_path=None, run_ngraph=True
    )
    assert any("node hardware:" in s for s in issues)


def test_link_optics_presence_audited_unordered_and_directional():
    # Build a tiny scenario with two roles A and B via a Clos-like blueprint adjacency
    data = {
        "blueprints": {
            "Clos_2_1": {
                "groups": {
                    "spine": {
                        "node_count": 1,
                        "name_template": "spine{node_num}",
                        "attrs": {"role": "spine"},
                    },
                    "leaf": {
                        "node_count": 2,
                        "name_template": "leaf{node_num}",
                        "attrs": {"role": "leaf"},
                    },
                },
                "adjacency": [
                    {
                        "source": "/leaf",
                        "target": "/spine",
                        "pattern": "mesh",
                        "link_params": {
                            "capacity": 100.0,
                            "cost": 1,
                            "attrs": {"link_type": "leaf_spine"},
                        },
                    }
                ],
            }
        },
        "components": {
            # Provide optics definitions and expose mapping for validation
            "800G-DR4": {"component_type": "optic", "capacity": 800.0, "ports": 4},
            "1600G-2xDR4": {"component_type": "optic", "capacity": 1600.0, "ports": 8},
            # First, unordered only (single 'leaf|spine') should require both ends to be that optic
            "optics": {"leaf|spine": "800G-DR4"},
        },
        "network": {
            "groups": {
                "metro1/pop[1]": {
                    "use_blueprint": "Clos_2_1",
                    "attrs": {
                        "metro_name": "X",
                        "metro_id": 1,
                        "location_x": 0.0,
                        "location_y": 0.0,
                    },
                }
            },
            "adjacency": [],
        },
    }
    # No hardware assigned on links in blueprint -> should flag both ends
    issues = validate_scenario_yaml(
        yaml_dump(data), integrated_graph_path=None, run_ngraph=True
    )
    assert any(
        "optics: missing hardware required by mapping on source end" in s
        for s in issues
    )
    assert any(
        "optics: missing hardware required by mapping on target end" in s
        for s in issues
    )

    # Now add directional override alongside unordered: both present means ordered interpretations.
    data["components"]["optics"] = {
        "leaf|spine": "800G-DR4",
        "spine|leaf": "1600G-2xDR4",
    }
    issues2 = validate_scenario_yaml(
        yaml_dump(data), integrated_graph_path=None, run_ngraph=True
    )
    # Still missing since blueprint didn't assign; we only assert presence of messages (not exact counts)
    assert any("source end" in s for s in issues2)
    assert any("target end" in s for s in issues2)
