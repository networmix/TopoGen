"""Run NetGraph construction, topology, and hardware audits."""

from __future__ import annotations

import yaml

from topogen.log_config import get_logger

from .expand_checks import check_groups_adjacency_blueprints
from .hw_capacity import check_node_hw_capacity
from .ngraph_schema import check_schema_and_isolation
from .node_hw_presence import check_node_hw_presence
from .node_role import check_node_roles
from .optics_checks import check_link_optics
from .port_budget import audit_port_budget

logger = get_logger(__name__)


def run_ngraph_audits(
    scenario_yaml: str,
    *,
    hw_component_map: dict[str, object] | None = None,
    optics_map: dict[str, object] | None = None,
) -> list[str]:
    """Validate Scenario construction and rule expansion, then audit the network.

    Check node roles, hardware assignments, optics, attached capacity, and ports.
    Return all collected issue strings.
    """
    issues: list[str] = []

    try:
        issues.extend(check_schema_and_isolation(scenario_yaml))
    except Exception as e:
        issues.append(f"ngraph schema: {e}")

    try:
        from ngraph.dsl.blueprints.expand import (  # type: ignore[import-untyped]
            expand_network_dsl as _ng_expand,
        )

        from topogen.components_lib import (  # type: ignore[import-untyped]
            get_builtin_components as _get_components_lib,
        )

        d = yaml.safe_load(scenario_yaml) or {}
        dsl = {
            "blueprints": (d.get("blueprints") or {}),
            "network": (d.get("network") or {}),
        }

        try:
            issues.extend(check_groups_adjacency_blueprints(dsl, _ng_expand, logger))
        except Exception as e:
            issues.append(f"adjacency/group expansion audit failed: {e}")

        net = _ng_expand(dsl)
        comp_lib = _get_components_lib()

        try:
            issues.extend(check_node_roles(net))
        except Exception as e:
            issues.append(f"node roles audit failed: {e}")

        try:
            # Prefer explicit mapping provided by caller (from config)
            if isinstance(hw_component_map, dict) and hw_component_map:
                comps_section = hw_component_map
            else:
                comps_section = (d.get("components") or {}).get("hw_component", {})
            issues.extend(check_node_hw_presence(net, comps_section, comp_lib))
        except Exception as e:
            issues.append(f"node hardware audit failed: {e}")

        try:
            # When override provided, inject into a shallow copy for optics checks
            if isinstance(optics_map, dict) and optics_map:
                d2 = dict(d)
                comps = dict(d2.get("components") or {})
                comps["optics"] = optics_map
                d2["components"] = comps
                issues.extend(check_link_optics(net, d2, comp_lib))
            else:
                issues.extend(check_link_optics(net, d, comp_lib))
        except Exception as e:
            issues.append(f"link optics audit failed: {e}")

        try:
            issues.extend(check_node_hw_capacity(net, comp_lib))
        except Exception as e:
            issues.append(f"hardware capacity audit failed: {e}")

        try:
            issues.extend(audit_port_budget(net, d, comp_lib))
        except Exception as e:
            issues.append(f"port budget audit failed: {e}")

    except Exception as e:
        issues.append(f"ngraph explorer: {e}")

    return issues
