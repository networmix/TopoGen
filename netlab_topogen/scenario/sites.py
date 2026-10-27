"""Read corridor records, resolve settings and plan ordered site adjacencies."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import networkx as nx

from netlab_topogen.config import (
    LINK_TYPES,
    BuildConfig,
    LinkParams,
    TrafficConfig,
    parse_link_params,
)
from netlab_topogen.library import Library, freeze, thaw
from netlab_topogen.naming import metro_slug, site_edge_id, site_id
from netlab_topogen.roles import role_pair

from .records import (
    Corridor,
    CorridorGraph,
    LinkSettings,
    Metro,
    MetroSettings,
    Site,
    SiteLink,
    SitePlan,
    StripeGroups,
)


def read_corridors(graph: nx.MultiGraph) -> CorridorGraph:
    """Snapshot graph order and metadata without retaining graph containers."""
    if not graph.is_multigraph() or graph.is_directed():
        raise TypeError("Expected an undirected corridor MultiGraph")
    required = ("name", "name_orig", "metro_id", "x", "y", "radius_km")
    metros = {}
    names: set[str] = set()
    ids: set[str] = set()
    for index, (node, data) in enumerate(graph.nodes(data=True), 1):
        if data.get("node_type") != "metro":
            raise ValueError(f"Expected metro node at {node}")
        if missing := set(required) - data.keys():
            raise ValueError(
                f"Metro node {node} missing required attributes: {sorted(missing)}"
            )
        for key, seen in (("name", names), ("metro_id", ids)):
            value = data[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Metro {node} requires nonempty string '{key}'")
            if value in seen:
                raise ValueError(f"Duplicate metro {key}: {value}")
            seen.add(value)
        for key in ("x", "y", "radius_km"):
            value = data[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Metro {node} requires finite numeric '{key}'")
        if data["radius_km"] < 0:
            raise ValueError(f"Metro {node} has negative radius_km")
        metros[node] = Metro(index, **{key: data[key] for key in required})
    return CorridorGraph(
        tuple(metros.values()),
        tuple(
            Corridor(
                metros[u],
                metros[v],
                key,
                data["length_km"],
                data.get("euclidean_km"),
                data.get("detour_ratio"),
                tuple(data.get("risk_groups", ())),
            )
            for u, v, key, data in graph.edges(keys=True, data=True)
        ),
    )


def resolve_metro_settings(
    corridors: CorridorGraph, build: BuildConfig, traffic: TrafficConfig
) -> tuple[MetroSettings, ...]:
    """Resolve owned link settings and every DC site's power exactly once."""
    available = {
        metro_slug(name)
        for metro in corridors.metros
        for name in (metro.name, metro.name_orig)
    }
    if corridors.metros and (unknown := set(build.build_overrides) - available):
        raise ValueError(
            f"Build override references unknown metro {sorted(unknown)}. Available metro slugs: {sorted(available)}"
        )
    powers = traffic.gravity.mw_per_dc_region_overrides
    known = {metro.name for metro in corridors.metros}
    result = []
    for metro in corridors.metros:
        name = metro.name
        override = build.build_overrides.get(
            metro_slug(name), build.build_overrides.get(metro_slug(metro.name_orig), {})
        )
        resolved = asdict(build.build_defaults)
        resolved.update(
            {key: value for key, value in override.items() if key not in LINK_TYPES}
        )
        links = {}
        for kind in LINK_TYPES:
            link = parse_link_params(
                override.get(kind, {}), LinkParams(**resolved[kind]), kind
            )
            if (
                link.capacity <= 0
                or link.cost < 0
                or (kind == "intra_metro_link" and link.cost == 0)
            ):
                raise ValueError(f"Metro '{name}' has invalid {kind} capacity or cost")
            links[kind] = LinkSettings(**asdict(link))
        pops, dcs = resolved["pop_per_metro"], resolved["dc_regions_per_metro"]
        if pops < 1 or dcs < 0:
            raise ValueError(f"Metro '{name}' has invalid PoP or DC count")
        prefix = name.lower().replace(" ", "-")
        power = []
        for ordinal in range(1, dcs + 1):
            key = f"{prefix}/dc{ordinal}"
            known.add(key)
            mass = powers.get(key, powers.get(name, traffic.mw_per_dc_region))
            if not math.isfinite(mass) or mass < 0:
                raise ValueError(
                    f"DC power for {name}/dc{ordinal} must be finite and non-negative"
                )
            power.append(mass)
        result.append(
            MetroSettings(
                metro,
                pops,
                dcs,
                resolved["site_blueprint"],
                resolved["dc_region_blueprint"],
                links,
                tuple(power),
            )
        )
    if unknown := set(powers) - known:
        raise ValueError(
            f"DC power override references unknown metro or DC site: {sorted(unknown)}"
        )
    return tuple(result)


def finalize_blueprints(
    settings: tuple[MetroSettings, ...],
    library: Library,
    hw_component: Mapping[str, str],
) -> Mapping[str, Any]:
    """Include used dependencies and assign configured platforms before expansion."""
    from copy import deepcopy

    if not isinstance(hw_component, Mapping):
        raise ValueError("components.hw_component must be a mapping")
    definitions = thaw(library.blueprints)
    used = {
        name
        for setting in settings
        for count, name in (
            (setting.pops, setting.site_blueprint),
            (setting.dcs, setting.dc_blueprint),
        )
        if count > 0
    }
    result: dict[str, Any] = {}
    visiting: set[str] = set()

    def include(name: str) -> None:
        if name in visiting:
            raise ValueError(f"Cyclic blueprint reference: {name}")
        if name in result:
            return
        if name not in definitions:
            raise ValueError(f"Unknown blueprint: {name}")
        visiting.add(name)
        blueprint = deepcopy(definitions[name])
        for group_name, group in blueprint["nodes"].items():
            if "blueprint" in group:
                include(group["blueprint"])
            attrs = group.setdefault("attrs", {})
            role = attrs.get("role")
            if (not isinstance(role, str) or not role) and "blueprint" not in group:
                raise ValueError(
                    f"Blueprint '{name}' group '{group_name}' is missing required 'role' attribute"
                )
            if hardware := hw_component.get(role):
                attrs["hardware"] = {"component": hardware, "count": 1}
        visiting.remove(name)
        result[name] = blueprint

    for name in sorted(used):
        include(name)
    return freeze(result)


def role_match(link: LinkSettings) -> Mapping[str, Any]:
    """Role pairs select the union of their roles at both ends."""
    roles = sorted({role for pair in link.role_pairs for role in role_pair(pair)})
    return (
        {
            "conditions": [
                {"attr": "role", "op": "==", "value": role} for role in roles
            ],
            "logic": "or",
        }
        if roles
        else link.match
    )


def _ring_cost(count: int, source: int, target: int, radius: float) -> int:
    delta = abs(source - target) % count
    return math.ceil(min(delta, count - delta) * math.tau * radius / count)


def plan_sites(
    corridors: CorridorGraph, settings: tuple[MetroSettings, ...], stripes: StripeGroups
) -> SitePlan:
    """Plan sites, then intra-metro, DC-to-PoP and corridor links in that order."""
    sites = []
    for setting in settings:
        metro = setting.metro
        if (setting.pops > 1 or setting.dcs > 0) and metro.radius_km <= 0:
            raise ValueError(
                f"Metro '{metro.name}' has radius_km={metro.radius_km}; expected > 0 for ring-based adjacency"
            )
        for kind, count, blueprint, fraction, phase in (
            ("pop", setting.pops, setting.site_blueprint, 0.7, 0.0),
            ("dc", setting.dcs, setting.dc_blueprint, 0.4, math.pi / 4),
        ):
            for ordinal in range(1, count + 1):
                theta = phase + math.tau * (ordinal - 1) / count
                radius = 1000.0 * metro.radius_km
                sites.append(
                    Site(
                        site_id(metro.index, kind, ordinal),
                        metro,
                        kind,
                        ordinal,
                        blueprint,
                        setting.dc_power_mw[ordinal - 1] if kind == "dc" else None,
                        (
                            metro.x + fraction * radius * math.cos(theta),
                            metro.y + fraction * radius * math.sin(theta),
                        ),
                    )
                )
    order = {site.id: index for index, site in enumerate(sites)}
    by_metro = {setting.metro.index: setting for setting in settings}
    links: list[SiteLink] = []
    rules: dict[str, dict[str, str]] = {}

    def attach(
        site: str, attribute: str, groups: Mapping[str, tuple[str, ...]]
    ) -> None:
        for label, members in groups.items():
            for member in members:
                path = f"^{re.escape(site + '/' + member)}$"
                attrs = rules.setdefault(path, {})
                if attribute in attrs and attrs[attribute] != label:
                    raise ValueError(
                        f"Conflicting stripe assignment for {site}/{member}"
                    )
                attrs[attribute] = label

    def add(
        source: str,
        target: str,
        key: str,
        link: LinkSettings,
        kind: str,
        distance: int,
        adjacency: str,
        match: Mapping[str, Any],
        corridor: Corridor | None = None,
    ) -> None:
        if order[source] > order[target]:
            source, target = target, source
        links.append(
            SiteLink(
                site_edge_id(source, target, key),
                kind,
                source,
                target,
                max(distance, link.cost),
                distance,
                link.capacity,
                match,
                link.attrs,
                adjacency,
                link.role_pairs,
                corridor.risk_groups if corridor else (),
                corridor.euclidean_km if corridor else None,
                corridor.detour_ratio if corridor else None,
            )
        )

    def stripe_match(attribute: str, label: str) -> dict[str, Any]:
        return {"conditions": [{"attr": attribute, "op": "==", "value": label}]}

    for setting in settings:
        metro = setting.metro
        link = setting.links["intra_metro_link"]
        adjacency = f"intra_metro:{metro.name}"
        for a, b in itertools.combinations(range(1, setting.pops + 1), 2):
            add(
                site_id(metro.index, "pop", a),
                site_id(metro.index, "pop", b),
                f"{adjacency}:{a}-{b}",
                link,
                "intra_metro",
                _ring_cost(setting.pops, a, b, metro.radius_km),
                adjacency,
                role_match(link),
            )
    for setting in settings:
        metro = setting.metro
        if not setting.dcs:
            continue
        link = setting.links["dc_to_pop_link"]
        match = role_match(link)
        adjacency = f"dc_to_pop:{metro.name}"
        if link.striping:
            pop_groups = stripes.groups[
                metro.index, "dc_to_pop_link", setting.site_blueprint
            ]
            dc_groups = stripes.groups[
                metro.index, "dc_to_pop_link", setting.dc_blueprint
            ]
            asymmetric = (
                link.striping.get("mode", "width") == "width"
                and link.striping["width"] == 1
            )
            if not asymmetric and pop_groups.keys() != dc_groups.keys():
                raise ValueError(
                    f"DC-to-PoP striping mismatch in {metro.name}: pop={list(pop_groups)}, dc={list(dc_groups)}"
                )
            attribute = f"stripe_dc_{metro.index}"
            for kind, count, groups in (
                ("pop", setting.pops, pop_groups),
                ("dc", setting.dcs, dc_groups),
            ):
                for ordinal in range(1, count + 1):
                    attach(site_id(metro.index, kind, ordinal), attribute, groups)
            match = stripe_match(attribute, next(iter(pop_groups)))
        for dc in range(1, setting.dcs + 1):
            fraction = (dc - 1) * setting.pops / setting.dcs
            candidates = {
                math.floor(fraction) % setting.pops + 1,
                math.ceil(fraction) % setting.pops + 1,
            }
            for pop in range(1, setting.pops + 1):
                distance = max(
                    1,
                    min(
                        _ring_cost(setting.pops, pop, candidate, metro.radius_km)
                        for candidate in candidates
                    ),
                )
                add(
                    site_id(metro.index, "dc", dc),
                    site_id(metro.index, "pop", pop),
                    f"{adjacency}:{dc}-{pop}",
                    link,
                    "dc_to_pop",
                    distance,
                    adjacency,
                    match,
                )
    for corridor in corridors.corridors:
        a, b = corridor.source, corridor.target
        left, right = by_metro[a.index], by_metro[b.index]
        link = left.links["inter_metro_link"]
        match = role_match(link)
        if link.mode == "one_to_one":
            pairs = [(p, p) for p in range(1, min(left.pops, right.pops) + 1)]
        elif link.mode == "mesh":
            pairs = list(
                itertools.product(range(1, left.pops + 1), range(1, right.pops + 1))
            )
        else:
            raise ValueError(f"Unknown inter-metro mode: {link.mode}")
        distance = math.ceil(corridor.length_km)
        adjacency = f"inter_metro:{a.index}-{b.index}:path{corridor.path}"
        attribute = f"stripe_im_{a.index}_{b.index}"
        labels = []
        if link.striping:
            lg = stripes.groups[a.index, "inter_metro_link", left.site_blueprint]
            rg = stripes.groups[a.index, "inter_metro_link", right.site_blueprint]
            if lg.keys() != rg.keys():
                raise ValueError(
                    f"Inter-metro striping mismatch: {a.name}={list(lg)}, {b.name}={list(rg)}"
                )
            labels = list(lg)
            for setting, groups in ((left, lg), (right, rg)):
                for ordinal in range(1, setting.pops + 1):
                    attach(
                        site_id(setting.metro.index, "pop", ordinal), attribute, groups
                    )
        for ordinal, (p, q) in enumerate(pairs):
            label = labels[ordinal % len(labels)] if labels else None
            add(
                site_id(a.index, "pop", p),
                site_id(b.index, "pop", q),
                f"{adjacency}:{p}-{q}",
                link,
                "inter_metro_corridor",
                distance,
                f"{adjacency}:{label}" if label is not None else adjacency,
                stripe_match(attribute, label) if label is not None else match,
                corridor,
            )
    return SitePlan(
        tuple(sites),
        tuple(links),
        tuple({"path": path, "attrs": attrs} for path, attrs in rules.items()),
    )
