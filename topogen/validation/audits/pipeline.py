"""Construct the complete NetGraph scenario once and audit its network."""

from __future__ import annotations

from typing import Any

import yaml
from jsonschema import ValidationError
from ngraph.dsl.blueprints.expand import expand_network_dsl
from ngraph.scenario import Scenario

from topogen.log_config import get_logger

from .dc_capacity import check_dc_capacity
from .expand_checks import check_groups_adjacency_blueprints
from .hw_capacity import check_node_hw_capacity
from .node_hw_presence import check_node_hw_presence
from .node_role import check_node_roles
from .optics_checks import check_link_optics
from .port_budget import audit_port_budget

logger = get_logger(__name__)


def run_ngraph_audits(
    scenario_yaml: str,
    *,
    hw_component_map: dict[str, str] | None = None,
    optics_map: dict[str, str] | None = None,
) -> list[str]:
    """Validate schema/workflows/failures, then audit the constructed network."""
    try:
        scenario = Scenario.from_yaml(scenario_yaml)
    except ValidationError as exc:
        return [f"ngraph schema: {exc.message}"]
    except Exception as exc:
        return [f"ngraph scenario: {exc}"]
    data = yaml.safe_load(scenario_yaml)
    net = scenario.network
    issues: list[str] = []
    engaged = {
        endpoint
        for link in net.links.values()
        for endpoint in (link.source, link.target)
    }
    isolated = [node for node in net.nodes if node not in engaged]
    if isolated:
        issues.append(
            f"{len(isolated)} isolated nodes found in built network "
            f"(e.g., {', '.join(isolated[:10])})"
        )
    components = data.get("components", {})
    role_map = {} if hw_component_map is None else hw_component_map
    optic_assignments = {} if optics_map is None else optics_map
    checks: list[tuple[str, Any]] = [
        ("DC capacity", lambda: check_dc_capacity(net, scenario.demand_set)),
        (
            "adjacency/group expansion",
            lambda: check_groups_adjacency_blueprints(data, expand_network_dsl, logger),
        ),
        ("node roles", lambda: check_node_roles(net)),
        ("node hardware", lambda: check_node_hw_presence(net, role_map, components)),
        ("link optics", lambda: check_link_optics(net, optic_assignments, components)),
        ("hardware capacity", lambda: check_node_hw_capacity(net, components)),
        ("port budget", lambda: audit_port_budget(net, components)),
    ]
    for label, check in checks:
        try:
            issues.extend(check())
        except Exception as exc:
            issues.append(f"{label} audit failed: {exc}")
    return issues
