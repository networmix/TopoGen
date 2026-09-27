"""Validate generated scenarios against schema, topology, and hardware constraints.

``validate_scenario_dict`` checks attributes, coordinates, and references.
With ``run_ngraph=True``, ``validate_scenario_yaml`` also checks the schema,
constructs a NetGraph Scenario, and audits expansion, hardware, optics, and ports.
"""

from __future__ import annotations

from .scenario_dict import validate_scenario_dict
from .yaml_validation import validate_scenario_yaml

__all__ = ["validate_scenario_dict", "validate_scenario_yaml"]
