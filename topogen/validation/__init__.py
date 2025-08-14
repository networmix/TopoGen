"""Scenario validation package.

This package provides validation helpers to check a generated NetGraph scenario for:

- Intra-metro attribute consistency between PoP and DC groups
- Presence of required DC attributes (``mw_per_dc_region``, ``gbps_per_mw``)
- Existence of referenced traffic matrices and failure policies in workflow
- Optional cross-check of metro coordinates against the integrated graph
- Optional schema validation via ``ngraph.scenario.Scenario`` and expansion/audits

Public API:
    - validate_scenario_dict
    - validate_scenario_yaml
"""

from __future__ import annotations

from .scenario_dict import validate_scenario_dict
from .yaml_validation import validate_scenario_yaml

__all__ = ["validate_scenario_dict", "validate_scenario_yaml"]
