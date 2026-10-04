"""Resolve typed build defaults and per-metro overrides."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from topogen.config import LINK_TYPES, LinkParams, TopologyConfig, parse_link_params
from topogen.naming import metro_slug


def _determine_metro_settings(
    metros: list[dict[str, Any]], config: TopologyConfig
) -> dict[str, dict[str, Any]]:
    """Resolve independent settings for each metro, rejecting invalid budgets."""
    defaults = config.build.build_defaults
    overrides = config.build.build_overrides
    available = {
        metro_slug(name)
        for metro in metros
        for name in (metro["name"], metro.get("name_orig", metro["name"]))
    }
    if metros and (unknown := set(overrides) - available):
        raise ValueError(
            f"Build override references unknown metro {sorted(unknown)}. "
            f"Available metro slugs: {sorted(available)}"
        )
    settings = {}
    for metro in metros:
        name = metro["name"]
        override = overrides.get(
            metro_slug(name),
            overrides.get(metro_slug(metro.get("name_orig", name)), {}),
        )
        resolved = asdict(defaults)
        resolved.update(
            {key: value for key, value in override.items() if key not in LINK_TYPES}
        )
        for key in LINK_TYPES:
            link = parse_link_params(
                override.get(key, {}), LinkParams(**resolved[key]), key
            )
            if (
                link.capacity <= 0
                or link.cost < 0
                or (key == "intra_metro_link" and link.cost == 0)
            ):
                raise ValueError(f"Metro '{name}' has invalid {key} capacity or cost")
            resolved[key] = asdict(link)
        if resolved["pop_per_metro"] < 1 or resolved["dc_regions_per_metro"] < 0:
            raise ValueError(f"Metro '{name}' has invalid PoP or DC count")
        settings[name] = resolved
    return settings
