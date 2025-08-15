"""Helpers for deterministic striping of inter-site adjacencies.

Compute stripe groupings (by fixed width or by blueprint attribute) and emit
``node_overrides`` that assign per-adjacency stripe identifiers to nodes. The
graph pipeline then matches on a single stripe attribute, so role-based
eligibility is encoded upfront in the grouping selection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _natural_key(name: str) -> Tuple[str, int]:
    """Return a key for natural ordering by alpha prefix and numeric suffix.

    Example: "leaf12" -> ("leaf", 12)
    Fallback numeric part is 0 if none present.
    """

    prefix_chars: List[str] = []
    digits: List[str] = []
    for ch in name:
        if ch.isdigit():
            digits.append(ch)
        else:
            prefix_chars.append(ch)
    prefix = "".join(prefix_chars)
    try:
        num = int("".join(digits)) if digits else 0
    except Exception:
        num = 0
    return prefix, num


def eligible_device_names_from_blueprint(
    blueprint: Dict[str, Any], roles: set[str] | None
) -> List[str]:
    """List eligible device names from a blueprint filtered by roles.

    Args:
        blueprint: Blueprint definition with ``groups``.
        roles: Allowed roles or None for all roles.

    Returns:
        Sorted device names (e.g., ["leaf1", "leaf2"]).
    """

    names: List[str] = []
    groups = blueprint.get("groups", {}) if isinstance(blueprint, dict) else {}
    for _gname, gdef in groups.items():
        attrs = gdef.get("attrs", {}) if isinstance(gdef, dict) else {}
        role = str(attrs.get("role", ""))
        if roles and role not in roles:
            continue
        try:
            n = int(gdef.get("node_count", 0))
        except Exception:
            n = 0
        template = str(gdef.get("name_template", ""))
        for i in range(1, n + 1):
            names.append(template.replace("{node_num}", str(i)))
    names.sort(key=_natural_key)
    return names


def group_by_width(names: List[str], width: int) -> List[List[str]]:
    """Partition names into contiguous groups of size ``width``.

    Raises:
        ValueError: If ``width`` <= 0 or ``len(names) % width != 0``.
    """

    if width <= 0:
        raise ValueError("striping.width must be positive")
    total = len(names)
    if total % width != 0:
        raise ValueError(
            f"Eligible device count {total} is not divisible by width {width}"
        )
    groups: List[List[str]] = []
    for i in range(0, total, width):
        groups.append(names[i : i + width])
    return groups


def group_by_attr(
    blueprint: Dict[str, Any], *, attr: str, roles: set[str] | None
) -> Dict[str, List[str]]:
    """Group device names by a blueprint group attribute value.

    Uses per-blueprint group attribute (constant for the subgroup) as label.
    Returns a mapping label -> list of device names sorted by natural order.
    """

    labels: Dict[str, List[str]] = {}
    groups = blueprint.get("groups", {}) if isinstance(blueprint, dict) else {}
    for _gname, gdef in groups.items():
        attrs = gdef.get("attrs", {}) if isinstance(gdef, dict) else {}
        role = str(attrs.get("role", ""))
        if roles and role not in roles:
            continue
        label = attrs.get(attr)
        if label is None:
            continue
        label_str = str(label)
        try:
            n = int(gdef.get("node_count", 0))
        except Exception:
            n = 0
        template = str(gdef.get("name_template", ""))
        for i in range(1, n + 1):
            namestr = template.replace("{node_num}", str(i))
            labels.setdefault(label_str, []).append(namestr)
    # Sort each bucket
    for lab in list(labels.keys()):
        labels[lab].sort(key=_natural_key)
    return labels


def build_node_overrides_for_site(
    site_path: str, stripe_attr: str, label_to_names: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Create DSL ``node_overrides`` to assign a stripe attribute to nodes.

    Matches nodes by anchored regex on full path with exactly one subgroup level
    (blueprint group name), e.g., ``^metro3/pop1/[^/]+/leaf3$``.
    """

    overrides: List[Dict[str, Any]] = []
    for label, names in label_to_names.items():
        for nodename in names:
            pattern = f"^{site_path}/[^/]+/{nodename}$"
            overrides.append({"path": pattern, "attrs": {stripe_attr: label}})
    return overrides


def make_stripe_attr_name(scope: str) -> str:
    """Return a stable stripe attribute name for a scope string.

    Example scopes:
      - ``im_5_7`` for inter-metro between indices 5 and 7
      - ``dc_5`` for dc-to-pop inside metro 5
    """

    return f"stripe_{scope}"
