"""Schema-based checks using DSL expansion for isolation detection.

This module avoids parsing the full scenario (including workflows) with
``ngraph.scenario.Scenario`` to prevent strict type parsing issues (e.g.,
string values like "auto" in workflow configs). Instead, it expands only the
``blueprints`` and ``network`` sections via the DSL and computes isolation on
the expanded network graph.
"""

from __future__ import annotations

from typing import Any

import yaml


def check_schema_and_isolation(scenario_yaml: str) -> list[str]:
    """Expand DSL and flag isolated nodes in the built network graph.

    Args:
        scenario_yaml: Full scenario YAML string.

    Returns:
        List of human-readable issues regarding isolated nodes.
    """
    issues: list[str] = []
    try:
        data: dict[str, Any] = yaml.safe_load(scenario_yaml) or {}
    except Exception as exc:  # noqa: BLE001
        return [f"YAML parse error: {exc}"]

    try:
        from ngraph.dsl.blueprints.expand import (  # type: ignore[import-untyped]
            expand_network_dsl as _ng_expand,
        )
    except Exception as exc:  # pragma: no cover - environment import error
        return [f"ngraph explorer: {exc}"]

    dsl = {
        "blueprints": (data.get("blueprints") or {}),
        "network": (data.get("network") or {}),
    }

    try:
        net = _ng_expand(dsl)
    except Exception as exc:
        # Surface expansion error as-is for clarity
        return [f"ngraph explorer: {exc}"]

    # Detect isolated nodes using expanded network
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
