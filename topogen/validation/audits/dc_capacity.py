"""Check each demand set against actual external capacity of each DC site."""

from __future__ import annotations

import math
import re
from collections import defaultdict

from ngraph import Network
from ngraph.analysis.demand import expand_demands
from ngraph.model.demand.matrix import DemandSet

_DC = re.compile(r"^(metro\d+/dc\d+)(?:/|$)")


def check_dc_capacity(network: Network, demand_sets: DemandSet) -> list[str]:
    """Resolve selectors with NetGraph and audit separate demand sets separately.

    Count external device links, not DSL selector rules or internal DC links.
    Pairwise demand endpoints have exact per-site volumes. A combine endpoint
    contributes a guaranteed per-DC load only when all members belong to that DC;
    placement across multiple DCs requires a routing analysis.
    """
    sites = {
        name: match.group(1) if (match := _DC.match(name)) else None
        for name in network.nodes
    }
    capacity: dict[str, float] = defaultdict(float)
    for link in network.links.values():
        source, target = sites[link.source], sites[link.target]
        if (
            source == target
            or link.disabled
            or network.nodes[link.source].disabled
            or network.nodes[link.target].disabled
        ):
            continue
        for site in (source, target):
            if site is not None:
                capacity[site] += link.capacity
    issues = []
    for name, demands in demand_sets.sets.items():
        egress: dict[str, float] = defaultdict(float)
        ingress: dict[str, float] = defaultdict(float)
        for index, demand in enumerate(demands):
            label = f"demand set '{name}' entry {index}"
            if not math.isfinite(demand.volume) or demand.volume <= 0:
                issues.append(f"{label}: volume must be positive and finite")
                continue
            try:
                expanded = expand_demands(network, [demand])
            except ValueError as exc:
                issues.append(f"{label}: {exc}")
                continue
            endpoints: dict[str, set[str | None]] = {}
            for edge in expanded.augmentations:
                if edge.source in sites:
                    endpoints.setdefault(edge.target, set()).add(sites[edge.source])
                if edge.target in sites:
                    endpoints.setdefault(edge.source, set()).add(sites[edge.target])
            for flow in expanded.demands:
                source = (
                    {sites[flow.src_name]}
                    if flow.src_name in sites
                    else endpoints[flow.src_name]
                )
                target = (
                    {sites[flow.dst_name]}
                    if flow.dst_name in sites
                    else endpoints[flow.dst_name]
                )
                for local, remote, totals in (
                    (source, target, egress),
                    (target, source, ingress),
                ):
                    if len(local) == 1:
                        site = next(iter(local))
                        if site is not None and site not in remote:
                            totals[site] += flow.volume
        for direction, totals in (("egress", egress), ("ingress", ingress)):
            for site, volume in sorted(totals.items()):
                available = capacity[site]
                if volume > available and not math.isclose(
                    volume, available, rel_tol=1e-9, abs_tol=1e-9
                ):
                    issues.append(
                        f"dc capacity: demand set '{name}', {site} {direction} demand {volume:,.6g} "
                        f"exceeds adjacency capacity {available:,.6g}"
                    )
    return issues
