"""Node role presence checks."""

from __future__ import annotations

from typing import Any


def check_node_roles(net: Any) -> list[str]:
    """Ensure each node has a non-empty role attribute."""
    issues: list[str] = []
    missing_role_count = 0
    missing_role_samples: list[str] = []
    for node in net.nodes.values():
        nname = str(getattr(node, "name", ""))
        role = str(getattr(node, "attrs", {}).get("role", "")).strip()
        if not role:
            missing_role_count += 1
            if (
                nname
                and len(missing_role_samples) < 5
                and nname not in missing_role_samples
            ):
                missing_role_samples.append(nname)
    if missing_role_count:
        if missing_role_samples:
            examples = ", ".join("'" + n + "'" for n in missing_role_samples)
            issues.append(
                f"node roles: {missing_role_count} node(s) missing role (e.g., {examples})"
            )
        else:
            issues.append("node roles: nodes missing role detected")
    return issues
