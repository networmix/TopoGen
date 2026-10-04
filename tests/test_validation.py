"""Tests for scenario validation utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from topogen.validation import validate_scenario_dict, validate_scenario_yaml


def _minimal_scenario() -> dict:
    # One metro with PoP and DC groups for dictionary-level checks.
    return {
        "network": {
            "nodes": {
                "metro1/pop[2]": {
                    "blueprint": "SingleRouter",
                    "attrs": {
                        "metro_name": "Denver",
                        "metro_name_orig": "Denver",
                        "metro_id": 1,
                        "location_x": 10.0,
                        "location_y": 20.0,
                    },
                },
                "metro1/dc[1]": {
                    "blueprint": "DCRegion",
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
            "links": [],
        },
        # Provide empty sections to satisfy reference checks by default
        "failures": {},
        "demands": {},
        "workflow": [],
    }


def test_validate_scenario_dict_attr_mismatch_detected():
    data = _minimal_scenario()
    data["network"]["nodes"]["metro1/dc[1]"]["attrs"]["location_x"] = 11.0
    issues = validate_scenario_dict(data)
    assert any("location_x mismatch" in s for s in issues)


def test_validate_scenario_dict_missing_required_dc_attrs():
    data = _minimal_scenario()
    del data["network"]["nodes"]["metro1/dc[1]"]["attrs"]["mw_per_dc_region"]
    issues = validate_scenario_dict(data)
    assert any("dc attrs missing required 'mw_per_dc_region'" in s for s in issues)


def test_validate_scenario_yaml_workflow_references_checked(tmp_path: Path):
    yaml_text = """
network:
  nodes:
    metro1/pop[2]:
      blueprint: SingleRouter
      attrs:
        metro_name: Denver
        metro_name_orig: Denver
        metro_id: 1
        location_x: 10.0
        location_y: 20.0
    metro1/dc[1]:
      blueprint: DCRegion
      attrs:
        metro_name: Denver
        metro_name_orig: Denver
        metro_id: 1
        location_x: 10.0
        location_y: 20.0
        mw_per_dc_region: 50.0
        gbps_per_mw: 250.0
workflow:
  - type: TrafficMatrixPlacementAnalysis
    name: tm
    demand_set: missing_matrix
    failure_policy: missing_policy
"""
    issues = validate_scenario_yaml(
        yaml_text, integrated_graph_path=None, run_ngraph=False
    )
    assert any("references missing traffic matrix" in s for s in issues)
    assert any("references missing failure_policy" in s for s in issues)


def test_validate_scenario_dict_does_not_flag_adjacency_strings():
    data = _minimal_scenario()
    data["network"]["links"].append({"source": "missing", "target": "metro1/pop[2]"})
    data["network"]["links"].append({"source": "metro1/dc[1]", "target": "missing2"})
    issues = validate_scenario_dict(data)
    assert not any("adjacency references missing group" in s for s in issues)


def test_validate_scenario_yaml_isolated_nodes_flagged():
    # No adjacency at all -> groups appear isolated at scenario level
    yaml_text = """
network:
  nodes:
    metro1/pop[1]:
      blueprint: SingleRouter
      attrs:
        metro_name: Denver
        metro_name_orig: Denver
        metro_id: 1
        location_x: 10.0
        location_y: 20.0
    metro1/dc[1]:
      blueprint: DCRegion
      attrs:
        metro_name: Denver
        metro_name_orig: Denver
        metro_id: 1
        location_x: 10.0
        location_y: 20.0
        mw_per_dc_region: 50.0
        gbps_per_mw: 250.0
"""
    # Metadata-only validation reports site groups without any adjacency rules.
    issues = validate_scenario_yaml(
        yaml_text, integrated_graph_path=None, run_ngraph=False
    )
    assert any("appears isolated" in s for s in issues)


def test_dc_capacity_vs_demand_validation():
    data = {
        "network": {
            "nodes": {
                name: {"attrs": {"role": "dc"}}
                for name in ("metro1/dc1/r", "metro2/dc1/r")
            },
            "links": [
                {"source": "metro1/dc1/r", "target": "metro2/dc1/r", "capacity": 1000}
            ],
        },
        "demands": {
            "tm": [
                {
                    "source": "^metro1/dc1/.*",
                    "target": "^metro2/dc1/.*",
                    "mode": "pairwise",
                    "volume": 1200,
                },
                {
                    "source": "^metro2/dc1/.*",
                    "target": "^metro1/dc1/.*",
                    "mode": "pairwise",
                    "volume": 800,
                },
            ]
        },
    }
    issues = validate_scenario_yaml(yaml.safe_dump(data))
    assert any(
        "metro1/dc1 egress demand" in issue and "exceeds adjacency capacity" in issue
        for issue in issues
    )
    assert not any("metro1/dc1 ingress demand" in issue for issue in issues)


def test_invalid_group_range_is_rejected():
    data = {"network": {"nodes": {"metro1/pop[1-0]": {}}, "links": []}}
    issues = validate_scenario_yaml(yaml.safe_dump(data), run_ngraph=True)
    assert any("Invalid range '1-0'" in issue for issue in issues)


def test_adjacencies_that_expand_to_zero_links_are_flagged():
    data = {
        "network": {
            "nodes": {"metro1/pop1": {"attrs": {"role": "core"}}},
            "links": [
                {
                    "source": "metro1/missing",
                    "target": "metro1/also_missing",
                    "pattern": "one_to_one",
                    "capacity": 100.0,
                }
            ],
        },
    }
    issues = validate_scenario_yaml(yaml.safe_dump(data), run_ngraph=True)
    assert any("adjacency[0] expands to 0 links" in issue for issue in issues)


def yaml_dump(d: dict) -> str:
    return yaml.safe_dump(d, sort_keys=False)


def test_node_hardware_presence_audited():
    # Scenario with a node role mapped to hardware but missing assignment on node
    data = {
        "blueprints": {
            "SingleRouter": {
                "nodes": {
                    "core": {
                        "count": 1,
                        "template": "core",
                        "attrs": {"role": "core"},
                    }
                },
                "links": [],
            }
        },
        "components": {
            "CoreRouter": {
                "component_type": "chassis",
                "capacity": 1000.0,
                "ports": 10,
            },
        },
        "network": {
            "nodes": {
                "metro1/pop[1]": {
                    "blueprint": "SingleRouter",
                    "attrs": {
                        "metro_name": "X",
                        "metro_id": 1,
                        "location_x": 0.0,
                        "location_y": 0.0,
                    },
                }
            },
            "links": [],
        },
    }
    issues = validate_scenario_yaml(
        yaml_dump(data),
        integrated_graph_path=None,
        run_ngraph=True,
        hw_component_map={"core": "CoreRouter"},
    )
    assert any("node hardware:" in s for s in issues)


def test_invalid_network_nodes_type_is_rejected():
    invalid = {
        "network": {
            # wrong type: nodes should be a mapping, make it a list to fail
            "nodes": [
                {
                    "blueprint": "SingleRouter",
                    "attrs": {
                        "metro_name": "A",
                        "metro_id": 1,
                        "location_x": 0.0,
                        "location_y": 0.0,
                    },
                }
            ],
            # links should be a list of mappings; keep as valid list to isolate the nodes error
            "links": [],
        },
    }

    issues = validate_scenario_yaml(
        yaml_dump(invalid), integrated_graph_path=None, run_ngraph=True
    )
    assert any("nodes" in s and "mapping" in s for s in issues)


def test_link_optics_presence_audited_unordered_and_directional():
    data = {
        "blueprints": {
            "Clos_2_1": {
                "nodes": {
                    "spine": {
                        "count": 1,
                        "template": "spine{n}",
                        "attrs": {"role": "spine"},
                    },
                    "leaf": {
                        "count": 2,
                        "template": "leaf{n}",
                        "attrs": {"role": "leaf"},
                    },
                },
                "links": [
                    {
                        "source": "/leaf",
                        "target": "/spine",
                        "pattern": "mesh",
                        "capacity": 100.0,
                        "cost": 1,
                        "attrs": {"link_type": "leaf_spine"},
                    }
                ],
            }
        },
        "components": {
            # Provide optics definitions and expose mapping for validation
            "800G-DR4": {"component_type": "optic", "capacity": 800.0, "ports": 4},
            "1600G-2xDR4": {"component_type": "optic", "capacity": 1600.0, "ports": 8},
            # First, unordered only (single 'leaf|spine') should require both ends to be that optic
            "optics": {"leaf->spine": "800G-DR4", "spine->leaf": "800G-DR4"},
        },
        "network": {
            "nodes": {
                "metro1/pop[1]": {
                    "blueprint": "Clos_2_1",
                    "attrs": {
                        "metro_name": "X",
                        "metro_id": 1,
                        "location_x": 0.0,
                        "location_y": 0.0,
                    },
                }
            },
            "links": [],
        },
    }
    # No hardware assigned on links in blueprint -> should flag both ends
    optics_map = data["components"].pop("optics")
    issues = validate_scenario_yaml(
        yaml_dump(data),
        integrated_graph_path=None,
        run_ngraph=True,
        optics_map=optics_map,
    )
    assert any(
        "optics: missing hardware required by mapping on source end" in s
        for s in issues
    )
    assert any(
        "optics: missing hardware required by mapping on target end" in s
        for s in issues
    )

    issues2 = validate_scenario_yaml(
        yaml_dump(data),
        optics_map={"leaf->spine": "800G-DR4", "spine->leaf": "1600G-2xDR4"},
    )
    assert any("source end" in issue for issue in issues2)
    assert any("target end" in issue for issue in issues2)


def test_port_budget_detects_platform_port_overuse():
    # Blueprint with explicit node hardware and an adjacency that requires more optics than ports
    data = {
        "blueprints": {
            "Tiny_1_1": {
                "nodes": {
                    "spine": {
                        "count": 1,
                        "template": "spine{n}",
                        "attrs": {
                            "role": "spine",
                            "hardware": {"component": "SpineRouter", "count": 1},
                        },
                    },
                    "leaf": {
                        "count": 1,
                        "template": "leaf{n}",
                        "attrs": {
                            "role": "leaf",
                            "hardware": {"component": "LeafRouter", "count": 1},
                        },
                    },
                },
                "links": [
                    {
                        "source": "/leaf",
                        "target": "/spine",
                        "pattern": "mesh",
                        # 54.4 Tb/s per link requires ceil(54400/800)=68 modules at each end
                        "capacity": 54_400.0,
                        "cost": 1,
                        "attrs": {
                            "link_type": "leaf_spine",
                            "hardware": {
                                end: {"component": "800G-DR4", "count": 68}
                                for end in ("source", "target")
                            },
                        },
                    }
                ],
            }
        },
        "components": {
            # Use optics mapping to infer per-end optics when link hardware isn't specified
            "optics": {"leaf->spine": "800G-DR4", "spine->leaf": "800G-DR4"},
        },
        "network": {
            "nodes": {
                "metro1/pop[1]": {
                    "blueprint": "Tiny_1_1",
                    "attrs": {
                        "metro_name": "Z",
                        "metro_id": 1,
                        "location_x": 0.0,
                        "location_y": 0.0,
                    },
                }
            },
            "links": [],
        },
    }
    from topogen.components_lib import get_builtin_components

    optics_map = data["components"].pop("optics")
    data["components"].update(get_builtin_components())
    issues = validate_scenario_yaml(
        yaml_dump(data),
        integrated_graph_path=None,
        run_ngraph=True,
        optics_map=optics_map,
    )
    # Expect a port budget violation referencing LeafRouter (64 ports) needing 68
    assert any("hardware ports:" in s and "LeafRouter" in s for s in issues)
    assert any("requires 68 ports" in s for s in issues)
    # Spine has 64 ports and the same need; it should also be flagged
    assert any("SpineRouter" in s for s in issues)
    assert any("requires 68 ports" in s for s in issues)
