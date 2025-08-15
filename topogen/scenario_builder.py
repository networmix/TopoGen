"""NetGraph scenario builder facade.

Public entry points for building scenarios live in ``topogen.scenario``. This
module re-exports the functions to keep the stable import path
``topogen.scenario_builder``.
"""

from __future__ import annotations

from topogen.scenario.assembly import _add_adjacency_comments, build_scenario
from topogen.scenario.config import _determine_metro_settings
from topogen.scenario.libraries import (
    _build_blueprints_section,
    _build_components_section,
)
from topogen.scenario.network import _extract_metros_from_graph
from topogen.scenario.policies import (
    _build_failure_policy_set_section,
    _build_workflow_section,
)
from topogen.scenario.risk import _build_risk_groups_section
from topogen.scenario.traffic import _build_traffic_matrix_section

__all__ = [
    "build_scenario",
    "_add_adjacency_comments",
    "_determine_metro_settings",
    "_build_blueprints_section",
    "_build_components_section",
    "_extract_metros_from_graph",
    "_build_failure_policy_set_section",
    "_build_workflow_section",
    "_build_risk_groups_section",
    "_build_traffic_matrix_section",
]
