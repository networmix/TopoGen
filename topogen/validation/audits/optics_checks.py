"""Audit explicit link hardware and optional expected role-pair coverage."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ngraph import Network

from topogen.roles import role_optics


def check_link_optics(
    net: Network, assignments: dict[str, str], comp_lib: dict[str, Any]
) -> list[str]:
    try:
        expected = role_optics(assignments)
    except ValueError as exc:
        return [f"optics mapping: {exc}"]
    failures: Counter[str] = Counter()
    for link in net.links.values():
        hardware = link.attrs.get("hardware")
        if hardware is None and not expected:
            continue
        for end, node_name, other_name in (
            ("source", link.source, link.target),
            ("target", link.target, link.source),
        ):
            role = net.nodes[node_name].attrs.get("role", "")
            other = net.nodes[other_name].attrs.get("role", "")
            label = f"for roles ({role},{other})"
            if hardware is None:
                issue = (
                    f"optics: missing hardware required by mapping on {end} end"
                    if (role, other) in expected
                    else f"optics mapping: missing {end}-end mapping"
                )
                failures[f"{issue} {label}"] += 1
                continue
            assignment = hardware.get(end)
            if not assignment:
                failures[
                    f"optics (blueprint): missing hardware on {end} end {label}"
                ] += 1
                continue
            name = assignment["component"]
            if name not in comp_lib:
                failures[
                    f"optics (blueprint): unknown component on {end} end {label}: {name}"
                ] += 1
                continue
            count = float(assignment.get("count", 1))
            capacity = float(comp_lib[name]["capacity"]) * count
            if (
                not math.isfinite(count)
                or count <= 0
                or not math.isfinite(capacity)
                or capacity < link.capacity
            ):
                failures[
                    f"optics (blueprint): hardware capacity shortfall {label}"
                ] += 1
    return [f"{issue} - {count} link ends" for issue, count in sorted(failures.items())]
