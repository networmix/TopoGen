"""Expand device-link identities once and assign exact budgets and optics."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import networkx as nx
import yaml
from ngraph import Scenario

from topogen.config import ComponentsConfig
from topogen.library import Library, freeze
from topogen.naming import site_edge_id
from topogen.numeric import optic_count, written_count
from topogen.roles import role_optics

from .records import DeviceLink, LinkHardware, Sizing


def expand_sites(
    plan: nx.MultiGraph, document: dict[str, Any]
) -> Mapping[str, tuple[DeviceLink, ...]]:
    """Refuse empty site/adjacency/blueprint rules in the one E2 expansion."""
    doc = copy.deepcopy(document)
    rules = {}
    for name, blueprint in doc["blueprints"].items():
        for index, rule in enumerate(blueprint.get("links", [])):
            label = f"blueprint '{name}' link rule {index}"
            rule.setdefault("attrs", {})["_topogen_rule"] = label
            rules[label] = False
    scenario = Scenario.from_yaml(yaml.safe_dump(doc, sort_keys=False))
    network = scenario.network
    for site in plan:
        if not any(node.startswith(site + "/") for node in network.nodes):
            raise ValueError(f"Site '{site}' expands to no devices")
    by_edge: dict[str, list[DeviceLink]] = defaultdict(list)
    for link in network.links.values():
        if label := link.attrs.get("_topogen_rule"):
            rules[label] = True
        if edge := link.attrs.get("site_edge"):
            by_edge[edge].append(
                DeviceLink(
                    link.source,
                    link.target,
                    network.nodes[link.source].attrs.get("role", ""),
                    network.nodes[link.target].attrs.get("role", ""),
                )
            )
    for label, produced in rules.items():
        if not produced:
            raise ValueError(f"{label} expands to no links")
    for u, v, key in plan.edges(keys=True):
        edge = site_edge_id(u, v, key)
        if not by_edge[edge]:
            raise ValueError(f"Adjacency '{edge}' expands to no device links")
    return freeze({edge: tuple(links) for edge, links in by_edge.items()})


def assign_hardware(
    plan: nx.MultiGraph,
    sizing: Sizing,
    expansion: Mapping[str, tuple[DeviceLink, ...]],
    library: Library,
    components: ComponentsConfig,
) -> Mapping[str, LinkHardware]:
    """Divide each budget exactly; cover its written device share at both ends."""
    optics = role_optics(components.optics)
    result = {}
    for u, v, key in plan.edges(keys=True):
        edge = site_edge_id(u, v, key)
        links = expansion[edge]
        capacity = sizing.capacity[edge] / len(links)
        written = float(capacity)
        if not math.isfinite(written) or written <= 0:
            raise ValueError(f"Adjacency '{edge}' has invalid capacity {written}")
        ends = []
        for end in ("source", "target"):
            if not optics:
                ends.append((None, 0))
                continue
            pairs = {
                (link.source_role, link.target_role)
                if end == "source"
                else (link.target_role, link.source_role)
                for link in links
            }
            if missing := pairs - optics.keys():
                raise ValueError(
                    f"Adjacency '{edge}' has no {end}-end optic mapping for {sorted(missing)}"
                )
            names = {optics[pair] for pair in pairs}
            if len(names) != 1:
                raise ValueError(
                    f"Adjacency '{edge}' requires different optics on its {end} ends; select one role pair per adjacency"
                )
            name = next(iter(names))
            # Components are normalized to floats by NetGraph when E2 reads them.
            held = float(library.components[name]["capacity"])
            label = f"Adjacency '{edge}' {links[0].source}->{links[0].target} {end} optic '{name}'"
            if held <= 0:
                raise ValueError(
                    f"{label}: optic capacity must be positive, got {held!r}"
                )
            count = optic_count(written, held)
            written_count(count, label)
            ends.append((name, count))
        result[edge] = LinkHardware(
            links, capacity, ends[0][0], ends[0][1], ends[1][0], ends[1][1]
        )
    return freeze(result)
