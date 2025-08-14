"""Schema-based checks via ngraph.Scenario."""

from __future__ import annotations


def check_schema_and_isolation(scenario_yaml: str) -> list[str]:
    """Build Scenario and flag isolated nodes in the built network graph."""
    issues: list[str] = []
    from ngraph.scenario import Scenario  # type: ignore[import-untyped]

    scenario = Scenario.from_yaml(scenario_yaml)

    # Detect isolated nodes using built network graph
    graph = scenario.network.to_strict_multidigraph(add_reverse=True)
    node_names = list((graph.get_nodes() or {}).keys())
    edges = list((graph.get_edges() or {}).values())
    engaged: set[str] = set()
    for src, dst, _key, _attr in edges:
        engaged.add(str(src))
        engaged.add(str(dst))
    isolated_nodes = [n for n in node_names if n not in engaged]

    if isolated_nodes:
        preview = ", ".join(list(map(str, isolated_nodes))[:10])
        issues.append(
            f"{len(isolated_nodes)} isolated nodes found in built network (e.g., {preview})"
        )

    return issues
