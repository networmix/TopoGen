"""Stripe actual blueprint devices and emit one exact node rule per device."""

from __future__ import annotations

import re
from typing import Any

import yaml
from ngraph import Scenario
from ngraph.model.selectors import normalize_selector, select_nodes


def _natural_key(path: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)
    )


class StripePlanner:
    """Cache each blueprint inventory and merge stripe attributes within one build."""

    def __init__(self, blueprints: dict[str, Any], names: set[str]) -> None:
        self._prefixes = {
            name: f"stripe_probe{index}" for index, name in enumerate(sorted(names))
        }
        self._rules: dict[str, dict[str, str]] = {}
        self._network = (
            Scenario.from_yaml(
                yaml.safe_dump(
                    {
                        "blueprints": blueprints,
                        "network": {
                            "nodes": {
                                prefix: {"blueprint": name}
                                for name, prefix in self._prefixes.items()
                            }
                        },
                    }
                )
            ).network
            if names
            else None
        )

    def groups(
        self, blueprint: str, striping: dict[str, Any], match: dict[str, Any]
    ) -> dict[str, list[str]]:
        prefix = self._prefixes[blueprint]
        assert self._network is not None
        groups = select_nodes(
            self._network,
            normalize_selector({"path": "^" + prefix + "/", "match": match}, "link"),
            default_active_only=False,
        )
        selected = {
            node.name.removeprefix(prefix + "/"): node.attrs
            for nodes in groups.values()
            for node in nodes
        }
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
                f"g{offset // width + 1}": names[offset : offset + width]
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
        return {label: labels[label] for label in sorted(labels)}

    def attach(self, site: str, attribute: str, groups: dict[str, list[str]]) -> None:
        for label, members in groups.items():
            for member in members:
                path = f"^{re.escape(site + '/' + member)}$"
                attrs = self._rules.setdefault(path, {})
                if attribute in attrs and attrs[attribute] != label:
                    raise ValueError(
                        f"Conflicting stripe assignment for {site}/{member}"
                    )
                attrs[attribute] = label

    def node_rules(self) -> list[dict[str, Any]]:
        return [{"path": path, "attrs": attrs} for path, attrs in self._rules.items()]
