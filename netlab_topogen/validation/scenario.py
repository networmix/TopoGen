"""Prepare every demand set, then run ordered NetLab-TopoGen audits."""

from __future__ import annotations

from dataclasses import dataclass

from ngraph import PreparedScenario, Scenario, prepare

from netlab_topogen.config import ComponentsConfig

from .audits.dc_capacity import check_dc_capacity
from .audits.hw_capacity import check_node_hw_capacity
from .audits.node_hw_presence import check_node_hw_presence
from .audits.node_role import check_node_roles
from .audits.optics_checks import check_link_optics
from .audits.port_budget import audit_port_budget


@dataclass(frozen=True, slots=True)
class Validation:
    issues: tuple[str, ...]
    prepared: PreparedScenario | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))


def validate_scenario(text: str, components: ComponentsConfig) -> Validation:
    """Validate the exact written text; a NetGraph refusal is the only issue."""
    try:
        scenario = Scenario.from_yaml(text)
        prepared = prepare(scenario, demand_sets=tuple(scenario.demand_sets))
    except Exception as exc:
        return Validation((f"ngraph scenario: {exc}",), None)
    net = prepared.scenario.network
    specs = {
        name: {"capacity": component.capacity, "ports": component.ports}
        for name, component in prepared.scenario.components_library.components.items()
    }
    issues = []
    for name, demands in prepared.demand_sets.items():
        for accounting in demands.accounting:
            if accounting.discarded_units:
                issues.append(
                    f"demand set '{name}' {accounting.base_id}: {accounting.discarded_units} discarded units"
                )
    issues.extend(check_dc_capacity(prepared))
    engaged = {
        endpoint
        for link in net.links.values()
        for endpoint in (link.source, link.target)
    }
    isolated = [node for node in net.nodes if node not in engaged]
    if not net.nodes:
        issues.append("No network.nodes found in scenario")
    if isolated:
        issues.append(
            f"{len(isolated)} isolated nodes found in built network (e.g., {', '.join(isolated[:10])})"
        )
    for label, check in (
        ("node roles", lambda: check_node_roles(net)),
        (
            "node hardware",
            lambda: check_node_hw_presence(net, components.hw_component, specs),
        ),
        ("link optics", lambda: check_link_optics(net, components.optics, specs)),
        ("hardware capacity", lambda: check_node_hw_capacity(net, specs)),
        ("port budget", lambda: audit_port_budget(net, specs)),
    ):
        try:
            issues.extend(check())
        except Exception as exc:
            issues.append(f"{label} audit failed: {exc}")
    return Validation(tuple(issues), prepared)
