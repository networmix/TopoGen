"""Intra-scenario dictionary validation (no ngraph dependencies)."""

from __future__ import annotations

import re
from typing import Any

from topogen.log_config import get_logger

from .helpers import _float_or_nan

logger = get_logger(__name__)


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

    # Validate groups structure
    network = (data or {}).get("network", {})
    groups: dict[str, Any] = (
        network.get("groups", {}) if isinstance(network, dict) else {}
    )
    if not isinstance(groups, dict) or not groups:
        issues.append("No network.groups found in scenario")
        groups = {}

    # Build per-metro references to PoP and DC groups
    metro_pattern = re.compile(r"^metro(\d+)")
    pop_groups: dict[str, dict[str, Any]] = {}
    dc_groups: dict[str, dict[str, Any]] = {}

    for name, entry in groups.items():
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
        if dc is None:
            continue

        pa = (pop.get("attrs", {}) if isinstance(pop, dict) else {}) or {}
        da = (dc.get("attrs", {}) if isinstance(dc, dict) else {}) or {}

        # Identity attributes must match
        for key in ("metro_name", "metro_name_orig", "metro_id"):
            if pa.get(key) != da.get(key):
                issues.append(
                    f"metro{idx}: attribute mismatch for {key}: pop={pa.get(key)} dc={da.get(key)}"
                )

        # Coordinates must match between PoP and DC
        for key in ("location_x", "location_y"):
            pv = _float_or_nan(pa.get(key))
            dv = _float_or_nan(da.get(key))
            if not (pv == dv):
                issues.append(
                    f"metro{idx}: {key} mismatch: pop={pa.get(key)} dc={da.get(key)}"
                )

        # Optional cross-check against integrated graph coordinates
        if ig_coords:
            name = str(pa.get("metro_name", "")).strip()
            if name in ig_coords:
                ix, iy = ig_coords[name]
                px = _float_or_nan(pa.get("location_x", 0.0))
                py = _float_or_nan(pa.get("location_y", 0.0))
                if not (px == ix and py == iy):
                    issues.append(
                        f"metro{idx}: pop location differs from integrated graph for {name}"
                    )
                if dc is not None:
                    dx = _float_or_nan(da.get("location_x", 0.0))
                    dy = _float_or_nan(da.get("location_y", 0.0))
                    if not (dx == ix and dy == iy):
                        issues.append(
                            f"metro{idx}: dc location differs from integrated graph for {name}"
                        )

        # Required DC attributes (only when DC exists)
        if dc is not None:
            for key in ("mw_per_dc_region", "gbps_per_mw"):
                if key not in da:
                    issues.append(f"metro{idx}: dc attrs missing required '{key}'")

    # Validate workflow references to failure policies and traffic matrices
    failure_set = (data or {}).get("failure_policy_set") or {}
    traffic_set = (data or {}).get("traffic_matrix_set") or {}
    workflows = (data or {}).get("workflow") or []
    if isinstance(workflows, list):
        for step in workflows:
            if not isinstance(step, dict):
                continue
            step_name = str(step.get("name") or step.get("step_type") or "step").strip()
            policy_ref = step.get("failure_policy")
            if policy_ref and policy_ref not in failure_set:
                issues.append(
                    f"workflow step '{step_name}' references missing failure_policy '{policy_ref}'"
                )
            matrix_ref = step.get("matrix_name")
            if matrix_ref and matrix_ref not in traffic_set:
                issues.append(
                    f"workflow step '{step_name}' references missing traffic matrix '{matrix_ref}'"
                )

    # Simple isolation hint if adjacency is absent
    adjacency = network.get("adjacency", []) or []
    if not adjacency:
        key_re = re.compile(r"^/?(metro\d+)/(pop|dc)\b")
        for gkey in groups.keys():
            if key_re.match(str(gkey)):
                issues.append(f"{gkey} appears isolated (no adjacency references)")

    # --- DC demand vs adjacency capacity check ---
    try:

        def _endpoint_path(ep: Any) -> str:
            if isinstance(ep, str):
                return ep
            if isinstance(ep, dict):
                p = ep.get("path")
                return str(p) if p is not None else ""
            return ""

        dc_re = re.compile(r"\^?/?(metro\d+)/dc(\d+)")

        def _parse_dc_key(path: str) -> str | None:
            m = dc_re.search(str(path))
            if not m:
                return None
            metro = m.group(1)
            dc_idx = m.group(2)
            return f"{metro}/dc{int(dc_idx)}"

        dc_capacity: dict[str, float] = {}
        if isinstance(adjacency, list):
            for rule in adjacency:
                if not isinstance(rule, dict):
                    continue
                src = _endpoint_path(rule.get("source"))
                dst = _endpoint_path(rule.get("target"))
                lp = (
                    rule.get("link_params", {})
                    if isinstance(rule.get("link_params"), dict)
                    else {}
                )
                attrs = lp.get("attrs", {}) if isinstance(lp.get("attrs"), dict) else {}
                try:
                    cap_val = float(
                        attrs.get("target_capacity", lp.get("capacity", 0.0))
                    )
                except Exception:
                    cap_val = float(lp.get("capacity", 0.0))
                for endpoint in (src, dst):
                    key = _parse_dc_key(endpoint)
                    if key:
                        dc_capacity[key] = dc_capacity.get(key, 0.0) + cap_val

        dc_egress: dict[str, float] = {}
        dc_ingress: dict[str, float] = {}
        tm_set = (data or {}).get("traffic_matrix_set") or {}
        if isinstance(tm_set, dict):
            for _mname, entries in tm_set.items():
                if not isinstance(entries, list):
                    continue
                for d in entries:
                    if not isinstance(d, dict):
                        continue
                    try:
                        demand_val = float(d.get("demand", 0.0))
                    except Exception:
                        demand_val = 0.0
                    if demand_val <= 0.0:
                        continue
                    s_path = str(d.get("source_path", ""))
                    t_path = str(d.get("sink_path", ""))
                    s_dc = _parse_dc_key(s_path)
                    t_dc = _parse_dc_key(t_path)
                    if s_dc:
                        dc_egress[s_dc] = dc_egress.get(s_dc, 0.0) + demand_val
                    if t_dc:
                        dc_ingress[t_dc] = dc_ingress.get(t_dc, 0.0) + demand_val

        all_dcs = set(dc_capacity) | set(dc_egress) | set(dc_ingress)
        for dc_key in sorted(all_dcs):
            cap = float(dc_capacity.get(dc_key, 0.0))
            eg = float(dc_egress.get(dc_key, 0.0))
            ing = float(dc_ingress.get(dc_key, 0.0))
            try:
                logger.info(
                    "dc capacity check: %s capacity=%s egress_demand=%s ingress_demand=%s",
                    dc_key,
                    f"{cap:,.1f}",
                    f"{eg:,.1f}",
                    f"{ing:,.1f}",
                )
            except Exception:
                pass
            eps = 1e-9
            if eg > cap + eps:
                issues.append(
                    f"dc capacity: {dc_key} egress demand {eg:,.1f} exceeds adjacency capacity {cap:,.1f}"
                )
            if ing > cap + eps:
                issues.append(
                    f"dc capacity: {dc_key} ingress demand {ing:,.1f} exceeds adjacency capacity {cap:,.1f}"
                )
    except Exception as e:
        issues.append(f"dc capacity audit failed: {e}")

    return issues
