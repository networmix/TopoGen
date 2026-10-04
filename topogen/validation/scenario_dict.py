"""Intra-scenario dictionary validation (no ngraph dependencies)."""

from __future__ import annotations

import re
from typing import Any

from .helpers import _float_or_nan


def validate_scenario_dict(
    data: dict[str, Any], ig_coords: dict[str, tuple[float, float]] | None = None
) -> list[str]:
    """Validate a parsed scenario dictionary and return issues.

    Args:
        data: Parsed scenario dictionary.
        ig_coords: Optional map of metro name to coordinates from the
            integrated graph for cross-checking metro locations.

    Returns:
        A list of human-readable issue strings. Empty when no issues found.
    """
    issues: list[str] = []

    network = (data or {}).get("network", {})
    if not isinstance(network, dict):
        return ["network must be a mapping"]
    groups: dict[str, Any] = network.get("nodes", {})
    if not isinstance(groups, dict) or not groups:
        issues.append("No network.nodes found in scenario")
        groups = {}

    metro_pattern = re.compile(r"^metro(\d+)")
    pop_groups: dict[str, dict[str, Any]] = {}
    dc_groups: dict[str, dict[str, Any]] = {}

    for name, entry in groups.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            issues.append("network.nodes must map string paths to node definitions")
            continue
        m = metro_pattern.match(str(name))
        if not m:
            continue
        idx = m.group(1)
        if "/pop[" in name:
            pop_groups[idx] = entry
        elif "/dc[" in name:
            dc_groups[idx] = entry

    for idx in sorted(set(pop_groups.keys()) | set(dc_groups.keys()), key=int):
        pop = pop_groups.get(idx)
        dc = dc_groups.get(idx)
        # PoP group is required per metro; DC group is optional.
        if not pop:
            issues.append(f"metro{idx}: missing pop group")
            continue
        pa = pop.get("attrs", {})
        da = dc.get("attrs", {}) if dc is not None else {}
        if not isinstance(pa, dict) or not isinstance(da, dict):
            issues.append(f"metro{idx}: site attrs must be mappings")
            continue

        if ig_coords is not None:
            for kind, attrs in (("pop", pa), ("dc", da)):
                if kind == "dc" and dc is None:
                    continue
                name = str(attrs.get("metro_name", ""))
                if name not in ig_coords:
                    issues.append(
                        f"metro{idx}: {kind} metro '{name}' missing from integrated graph"
                    )
                elif (
                    _float_or_nan(attrs.get("location_x")),
                    _float_or_nan(attrs.get("location_y")),
                ) != ig_coords[name]:
                    issues.append(
                        f"metro{idx}: {kind} location differs from integrated graph for {name}"
                    )
        if dc is None:
            continue
        for key in ("metro_name", "metro_name_orig", "metro_id"):
            if pa.get(key) != da.get(key):
                issues.append(
                    f"metro{idx}: attribute mismatch for {key}: pop={pa.get(key)} dc={da.get(key)}"
                )
        for key in ("location_x", "location_y"):
            if _float_or_nan(pa.get(key)) != _float_or_nan(da.get(key)):
                issues.append(
                    f"metro{idx}: {key} mismatch: pop={pa.get(key)} dc={da.get(key)}"
                )
        for key in ("mw_per_dc_region", "gbps_per_mw"):
            if key not in da:
                issues.append(f"metro{idx}: dc attrs missing required '{key}'")

    failure_set = (data or {}).get("failures") or {}
    traffic_set = (data or {}).get("demands") or {}
    for label, section in (("failures", failure_set), ("demands", traffic_set)):
        if not isinstance(section, dict):
            issues.append(f"{label} must be a mapping")
    if not isinstance(failure_set, dict):
        failure_set = {}
    if not isinstance(traffic_set, dict):
        traffic_set = {}
    workflows = (data or {}).get("workflow") or []
    if isinstance(workflows, list):
        for step in workflows:
            if not isinstance(step, dict):
                continue
            step_name = str(step.get("name") or step.get("type") or "step").strip()
            policy_ref = step.get("failure_policy")
            if policy_ref and (
                not isinstance(policy_ref, str) or policy_ref not in failure_set
            ):
                issues.append(
                    f"workflow step '{step_name}' references missing failure_policy '{policy_ref}'"
                )
            matrix_ref = step.get("demand_set")
            if matrix_ref and (
                not isinstance(matrix_ref, str) or matrix_ref not in traffic_set
            ):
                issues.append(
                    f"workflow step '{step_name}' references missing traffic matrix '{matrix_ref}'"
                )

    adjacency = network.get("links", []) or []
    if not adjacency:
        key_re = re.compile(r"^/?(metro\d+)/(pop|dc)\b")
        for gkey in groups.keys():
            if key_re.match(str(gkey)):
                issues.append(f"{gkey} appears isolated (no adjacency references)")

    return issues
