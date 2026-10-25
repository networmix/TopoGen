"""Audit resolved demand endpoints against each DC site's external capacity."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterator
from fractions import Fraction

from ngraph import (
    Endpoints,
    ExpansionLeaf,
    ExpansionSplit,
    PreparedScenario,
    resolve_demands,
)

from topogen.naming import site_of
from topogen.numeric import decimal


def _declared_leaves(
    root: ExpansionSplit | ExpansionLeaf, volume: Fraction
) -> Iterator[tuple[Endpoints, Fraction]]:
    """Apportion physical declarations through the public, unquantized tree."""
    if isinstance(root, ExpansionSplit):
        for child in root.children:
            yield from _declared_leaves(child, volume / len(root.children))
    elif root.endpoints is not None:
        yield root.endpoints, volume


def check_dc_capacity(prepared: PreparedScenario) -> list[str]:
    network = prepared.scenario.network
    sites = {
        name: site_of(name) if re.fullmatch(r"metro\d+/dc\d+", site_of(name)) else None
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
    for name, specs in prepared.scenario.demand_sets.items():
        for index, spec in enumerate(specs.demands):
            if spec.volume <= 0 or not math.isfinite(spec.volume):
                issues.append(
                    f"demand set '{name}' entry {index}: volume must be positive and finite"
                )
        resolved = resolve_demands(prepared, specs)
        egress: dict[str, Fraction] = defaultdict(Fraction)
        ingress: dict[str, Fraction] = defaultdict(Fraction)
        for spec, parent in zip(
            specs.demands, resolved.expansion_template, strict=True
        ):
            if spec.volume <= 0 or not math.isfinite(spec.volume):
                continue
            for endpoints, volume in _declared_leaves(
                parent.root, decimal(spec.volume)
            ):
                source = {sites[node] for node in endpoints.sources}
                target = {sites[node] for node in endpoints.destinations}
                for local, remote, totals in (
                    (source, target, egress),
                    (target, source, ingress),
                ):
                    if len(local) == 1:
                        site = next(iter(local))
                        if site is not None and site not in remote:
                            totals[site] += volume
        for direction, totals in (("egress", egress), ("ingress", ingress)):
            for site, volume in sorted(totals.items()):
                available = capacity[site]
                if volume > available and not math.isclose(
                    volume, available, rel_tol=1e-9, abs_tol=1e-9
                ):
                    issues.append(
                        f"dc capacity: demand set '{name}', {site} {direction} demand {float(volume):,.6g} "
                        f"exceeds adjacency capacity {available:,.6g}"
                    )
    return issues
