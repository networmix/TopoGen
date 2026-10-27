"""Immutable results of scenario construction stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from netlab_topogen.library import freeze


@dataclass(frozen=True, slots=True)
class Metro:
    index: int
    name: str
    name_orig: str
    metro_id: str
    x: float
    y: float
    radius_km: float


@dataclass(frozen=True, slots=True)
class Corridor:
    source: Metro
    target: Metro
    path: Any
    length_km: float
    euclidean_km: float | None
    detour_ratio: float | None
    risk_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", freeze(self.path))
        object.__setattr__(self, "risk_groups", tuple(self.risk_groups))


@dataclass(frozen=True, slots=True)
class CorridorGraph:
    metros: tuple[Metro, ...]
    corridors: tuple[Corridor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metros", tuple(self.metros))
        object.__setattr__(self, "corridors", tuple(self.corridors))


@dataclass(frozen=True, slots=True)
class LinkSettings:
    capacity: int
    cost: int
    attrs: Mapping[str, Any]
    match: Mapping[str, Any]
    role_pairs: tuple[str, ...]
    striping: Mapping[str, Any]
    mode: str

    def __post_init__(self) -> None:
        for name in ("attrs", "match", "striping"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        object.__setattr__(self, "role_pairs", tuple(self.role_pairs))


@dataclass(frozen=True, slots=True)
class MetroSettings:
    metro: Metro
    pops: int
    dcs: int
    site_blueprint: str
    dc_blueprint: str
    links: Mapping[str, LinkSettings]
    dc_power_mw: tuple[int | float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "links", freeze(self.links))
        object.__setattr__(self, "dc_power_mw", tuple(self.dc_power_mw))


@dataclass(frozen=True, slots=True)
class StripeGroups:
    groups: Mapping[tuple[int, str, str], Mapping[str, tuple[str, ...]]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", freeze(self.groups))


@dataclass(frozen=True, slots=True)
class Site:
    id: str
    metro: Metro
    kind: str
    ordinal: int
    blueprint: str
    power_mw: int | float | None
    position: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", tuple(self.position))


@dataclass(frozen=True, slots=True)
class SiteLink:
    id: str
    kind: str
    source: str
    target: str
    cost: int
    distance_km: int
    base_capacity: int
    match: Mapping[str, Any]
    attrs: Mapping[str, Any]
    adjacency_id: str
    role_pairs: tuple[str, ...]
    risk_groups: tuple[str, ...]
    euclidean_km: float | None
    detour_ratio: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "match", freeze(self.match))
        object.__setattr__(self, "attrs", freeze(self.attrs))
        object.__setattr__(self, "role_pairs", tuple(self.role_pairs))
        object.__setattr__(self, "risk_groups", tuple(self.risk_groups))


@dataclass(frozen=True, slots=True)
class SitePlan:
    sites: tuple[Site, ...]
    links: tuple[SiteLink, ...]
    node_rules: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "node_rules", freeze(self.node_rules))


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
class SiteExpansion:
    links: Mapping[str, tuple[DeviceLink, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "links", freeze(self.links))


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
    site_plan: SitePlan

    def __post_init__(self) -> None:
        object.__setattr__(self, "traffic", tuple(self.traffic))
        object.__setattr__(self, "hardware", freeze(self.hardware))
