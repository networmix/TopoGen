"""Immutable results of scenario construction stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import networkx as nx

from topogen.library import freeze, thaw


def readonly_graph(graph: nx.MultiGraph) -> nx.MultiGraph:
    """Own graph topology and freeze every nested attribute container."""
    owned: Any = nx.MultiGraph()
    owned.graph.update(thaw(graph.graph))
    for name, attrs in graph.nodes(data=True):
        owned.add_node(name, **thaw(attrs))
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        owned.add_edge(source, target, key=key, **thaw(attrs))
    owned.graph = freeze(owned.graph)
    for name, attrs in owned._node.items():
        owned._node[name] = freeze(attrs)
    for source, target, key in list(owned.edges(keys=True)):
        attrs = freeze(owned._adj[source][target][key])
        owned._adj[source][target][key] = attrs
        owned._adj[target][source][key] = attrs
    return nx.freeze(owned)


@dataclass(frozen=True, slots=True)
class SiteDemand:
    source: str
    target: str
    priority: int
    volume: Fraction
    preset: str | None = None
    euclidean_km: int | None = None


@dataclass(frozen=True, slots=True)
class TrafficMatrix:
    name: str
    demands: tuple[SiteDemand, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "demands", tuple(self.demands))


@dataclass(frozen=True, slots=True)
class Sizing:
    load: Mapping[str, Fraction]
    capacity: Mapping[str, Fraction]

    def __post_init__(self) -> None:
        object.__setattr__(self, "load", freeze(self.load))
        object.__setattr__(self, "capacity", freeze(self.capacity))


@dataclass(frozen=True, slots=True)
class DeviceLink:
    source: str
    target: str
    source_role: str
    target_role: str


@dataclass(frozen=True, slots=True)
class LinkHardware:
    device_links: tuple[DeviceLink, ...]
    capacity: Fraction
    source_optic: str | None = None
    source_count: int = 0
    target_optic: str | None = None
    target_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_links", tuple(self.device_links))


@dataclass(frozen=True, slots=True)
class Build:
    yaml: str
    traffic: tuple[TrafficMatrix, ...]
    sizing: Sizing
    hardware: Mapping[str, LinkHardware]
    site_plan: nx.MultiGraph

    def __post_init__(self) -> None:
        object.__setattr__(self, "traffic", tuple(self.traffic))
        object.__setattr__(self, "hardware", freeze(self.hardware))
        object.__setattr__(self, "site_plan", readonly_graph(self.site_plan))
