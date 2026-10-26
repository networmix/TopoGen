"""Expand finalized devices once, then group them for each resolved policy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import yaml
from ngraph import Scenario
from ngraph.model.selectors import normalize_selector, select_nodes

from netlab_topogen.library import thaw

from .records import CorridorGraph, LinkSettings, MetroSettings, StripeGroups
from .sites import role_match


def _natural_key(path: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)
    )


def _group_devices(
    selected: Mapping[str, Any], striping: Mapping[str, Any], blueprint: str
) -> dict[str, tuple[str, ...]]:
    if not selected:
        raise ValueError(f"No eligible stripe devices in blueprint '{blueprint}'")
    names = sorted(selected, key=_natural_key)
    mode = striping.get("mode", "width")
    if mode == "width":
        width = striping["width"]
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or len(names) % width
        ):
            raise ValueError(
                f"Eligible device count {len(names)} requires a positive divisible striping.width"
            )
        return {
            f"g{offset // width + 1}": tuple(names[offset : offset + width])
            for offset in range(0, len(names), width)
        }
    if mode != "by_attr":
        raise ValueError(f"Unknown striping.mode: {mode}")
    attribute = striping["attribute"]
    labels: dict[str, list[str]] = {}
    for name in names:
        if attribute not in selected[name]:
            raise ValueError(
                f"Stripe device '{blueprint}/{name}' lacks attribute '{attribute}'"
            )
        labels.setdefault(str(selected[name][attribute]), []).append(name)
    return {label: tuple(labels[label]) for label in sorted(labels)}


def stripe_groups(
    corridors: CorridorGraph,
    settings: tuple[MetroSettings, ...],
    blueprints: Mapping[str, Any],
) -> StripeGroups:
    """Cache device inventory by blueprint; retain grouping per metro and link kind."""
    contexts: dict[tuple[int, str, str], LinkSettings] = {}
    by_metro = {setting.metro.index: setting for setting in settings}
    for setting in settings:
        link = setting.links["dc_to_pop_link"]
        if setting.dcs and link.striping:
            for name in (setting.site_blueprint, setting.dc_blueprint):
                contexts[setting.metro.index, "dc_to_pop_link", name] = link
    for corridor in corridors.corridors:
        policy = by_metro[corridor.source.index]
        link = policy.links["inter_metro_link"]
        if link.striping:
            for setting in (policy, by_metro[corridor.target.index]):
                contexts[
                    policy.metro.index, "inter_metro_link", setting.site_blueprint
                ] = link
    if not contexts:
        return StripeGroups({})
    prefixes = {
        name: f"stripe_probe{index}"
        for index, name in enumerate(sorted({key[2] for key in contexts}))
    }
    network = Scenario.from_yaml(
        yaml.safe_dump(
            {
                "blueprints": thaw(blueprints),
                "network": {
                    "nodes": {
                        prefix: {"blueprint": name} for name, prefix in prefixes.items()
                    }
                },
            }
        )
    ).network
    result = {}
    for key, link in contexts.items():
        blueprint = key[2]
        prefix = prefixes[blueprint]
        groups = select_nodes(
            network,
            normalize_selector(
                {"path": "^" + prefix + "/", "match": thaw(role_match(link))}, "link"
            ),
            default_active_only=False,
        )
        selected = {
            node.name.removeprefix(prefix + "/"): node.attrs
            for nodes in groups.values()
            for node in nodes
        }
        result[key] = _group_devices(selected, link.striping, blueprint)
    return StripeGroups(result)
