"""Build a complete NetGraph scenario and detect isolated nodes."""

from __future__ import annotations


def check_schema_and_isolation(scenario_yaml: str) -> list[str]:
    """Return NetGraph Scenario construction errors and isolated-node issues."""
    issues: list[str] = []
    try:
        from ngraph.scenario import Scenario
    except Exception as exc:  # pragma: no cover - environment import error
        return [f"ngraph explorer: {exc}"]
    try:
        net = Scenario.from_yaml(scenario_yaml).network
    except Exception as exc:
        return [f"ngraph scenario: {exc}"]

    node_names = [str(n.name) for n in net.nodes.values()]
    engaged: set[str] = set()
    for link in net.links.values():
        engaged.add(str(link.source))
        engaged.add(str(link.target))
    isolated_nodes = [n for n in node_names if n not in engaged]

    if isolated_nodes:
        preview = ", ".join(isolated_nodes[:10])
        issues.append(
            f"{len(isolated_nodes)} isolated nodes found in built network (e.g., {preview})"
        )

    return issues
