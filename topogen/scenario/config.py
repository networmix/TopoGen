"""Configuration resolution for scenario building.

Determines per-metro settings from the global configuration and overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from topogen.naming import metro_slug

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig


def _determine_metro_settings(
    metros: list[dict[str, Any]], config: "TopologyConfig"
) -> dict[str, dict[str, Any]]:
    """Determine per-metro configuration settings from config and overrides.

    Args:
        metros: List of metro dictionaries.
        config: Topology configuration with build settings.

    Returns:
        Dictionary mapping metro names to their resolved settings.

    Raises:
        ValueError: If override references unknown metro name or if resolved
            settings contain invalid values.
    """
    build_config = config.build
    defaults = build_config.build_defaults
    overrides = build_config.build_overrides

    # Strictly validate override keys against available metro slugs
    if metros:
        available_slugs = set()
        for m in metros:
            available_slugs.add(metro_slug(m.get("name", "")))
            available_slugs.add(metro_slug(m.get("name_orig", m.get("name", ""))))
        for override_key in overrides.keys():
            if override_key not in available_slugs:
                available_list = ", ".join(sorted(available_slugs))
                raise ValueError(
                    "Build override references unknown metro "
                    f"'{override_key}'. Available metro slugs: {available_list}"
                )

    settings: dict[str, dict[str, Any]] = {}
    for metro in metros:
        metro_name = metro["name"]

        # Seed with defaults
        metro_settings = {
            "pop_per_metro": defaults.pop_per_metro,
            "site_blueprint": defaults.site_blueprint,
            "dc_regions_per_metro": defaults.dc_regions_per_metro,
            "dc_region_blueprint": defaults.dc_region_blueprint,
            "intra_metro_link": {
                "capacity": defaults.intra_metro_link.capacity,
                "cost": defaults.intra_metro_link.cost,
                "attrs": defaults.intra_metro_link.attrs.copy(),
            },
            "inter_metro_link": {
                "capacity": defaults.inter_metro_link.capacity,
                "cost": defaults.inter_metro_link.cost,
                "attrs": defaults.inter_metro_link.attrs.copy(),
            },
            "dc_to_pop_link": {
                "capacity": defaults.dc_to_pop_link.capacity,
                "cost": defaults.dc_to_pop_link.cost,
                "attrs": defaults.dc_to_pop_link.attrs.copy(),
            },
        }

        # Apply overrides (exact slug match only)
        metro_name_orig = metro.get("name_orig", metro_name)
        override = None
        slug_sanitized = metro_slug(metro_name)
        slug_original = metro_slug(metro_name_orig)
        if slug_sanitized in overrides:
            override = overrides[slug_sanitized]
        elif slug_original in overrides:
            override = overrides[slug_original]

        if override:
            if "pop_per_metro" in override:
                metro_settings["pop_per_metro"] = override["pop_per_metro"]
            if "site_blueprint" in override:
                metro_settings["site_blueprint"] = override["site_blueprint"]
            if "dc_regions_per_metro" in override:
                metro_settings["dc_regions_per_metro"] = override[
                    "dc_regions_per_metro"
                ]
            if "dc_region_blueprint" in override:
                metro_settings["dc_region_blueprint"] = override["dc_region_blueprint"]
            if "intra_metro_link" in override:
                override_intra = override["intra_metro_link"]
                if "attrs" in override_intra:
                    metro_settings["intra_metro_link"]["attrs"].update(
                        override_intra["attrs"]
                    )
                for key, value in override_intra.items():
                    if key != "attrs":
                        metro_settings["intra_metro_link"][key] = value
            if "inter_metro_link" in override:
                override_inter = override["inter_metro_link"]
                if "attrs" in override_inter:
                    metro_settings["inter_metro_link"]["attrs"].update(
                        override_inter["attrs"]
                    )
                for key, value in override_inter.items():
                    if key != "attrs":
                        metro_settings["inter_metro_link"][key] = value
            if "dc_to_pop_link" in override:
                override_dc_pop = override["dc_to_pop_link"]
                if "attrs" in override_dc_pop:
                    metro_settings["dc_to_pop_link"]["attrs"].update(
                        override_dc_pop["attrs"]
                    )
                for key, value in override_dc_pop.items():
                    if key != "attrs":
                        metro_settings["dc_to_pop_link"][key] = value

        # Validate values
        if metro_settings["pop_per_metro"] < 1:
            raise ValueError(
                f"Metro '{metro_name}' has invalid pop_per_metro: {metro_settings['pop_per_metro']}"
            )
        if metro_settings["dc_regions_per_metro"] < 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid dc_regions_per_metro: {metro_settings['dc_regions_per_metro']}"
            )

        intra_link = metro_settings["intra_metro_link"]
        if intra_link["capacity"] <= 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid intra_metro_link capacity: {intra_link['capacity']}"
            )
        if intra_link["cost"] <= 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid intra_metro_link cost: {intra_link['cost']}"
            )

        inter_link = metro_settings["inter_metro_link"]
        if inter_link["capacity"] <= 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid inter_metro_link capacity: {inter_link['capacity']}"
            )
        if inter_link["cost"] < 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid inter_metro_link cost: {inter_link['cost']}"
            )

        dc_pop_link = metro_settings["dc_to_pop_link"]
        if dc_pop_link["capacity"] <= 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid dc_to_pop_link capacity: {dc_pop_link['capacity']}"
            )
        if dc_pop_link["cost"] < 0:
            raise ValueError(
                f"Metro '{metro_name}' has invalid dc_to_pop_link cost: {dc_pop_link['cost']}"
            )

        settings[metro_name] = metro_settings

    return settings
