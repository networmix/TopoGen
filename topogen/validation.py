"""Scenario validation utilities.

Provides validation helpers to check a generated NetGraph scenario for:

- Intra-metro attribute consistency between PoP and DC groups
- Presence of required DC attributes (``mw_per_dc_region``, ``gbps_per_mw``)
- Existence of referenced traffic matrices and failure policies in workflow
- Optional cross-check of metro coordinates against the integrated graph
- Optional schema validation via ``ngraph.scenario.Scenario``

These functions are pure and return issue strings; callers decide how to
surface or handle validation errors.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from topogen.log_config import get_logger

logger = get_logger(__name__)


def _build_ig_coord_map(ig_json: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Return metro name to (x, y) coordinate map from integrated graph JSON.

    Args:
        ig_json: Parsed JSON object for the integrated graph.

    Returns:
        Mapping from metro display name to coordinate tuple in the target CRS.
    """

    mapping: dict[str, tuple[float, float]] = {}
    for node in ig_json.get("nodes", []) or []:
        try:
            node_type = node.get("node_type")
        except AttributeError:
            continue
        if node_type not in ("metro", "metro+highway"):
            continue
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        try:
            x = float(node.get("x", 0.0))
            y = float(node.get("y", 0.0))
        except Exception:
            continue
        mapping[name] = (x, y)
    return mapping


def _float_or_nan(value: Any) -> float:
    """Convert value to float or return NaN if conversion fails.

    Args:
        value: Arbitrary input value.

    Returns:
        Floating-point value or NaN when conversion fails.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def validate_scenario_dict(
    data: dict[str, Any], ig_coords: dict[str, tuple[float, float]] | None = None
) -> list[str]:
    """Validate scenario dictionary and return a list of issue strings.

    Args:
        data: Parsed scenario YAML as a dictionary.
        ig_coords: Optional mapping from metro name to (x, y) for cross-check.

    Returns:
        List of human-readable issue strings. Empty list when no issues.
    """

    issues: list[str] = []

    # Validate groups structure
    network = (data or {}).get("network", {})
    groups: dict[str, Any] = (
        network.get("groups", {}) if isinstance(network, dict) else {}
    )
    if not isinstance(groups, dict) or not groups:
        return ["No network.groups found in scenario"]

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
            # If there's no PoP, skip further checks for this metro.
            continue
        # If DC group is absent, treat as valid (some metros have 0 DC regions).
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
                # Only check DC coords when DC exists
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

    # Only perform a simple, non-DSL-specific fallback: if there is no adjacency
    # section or it is empty, every group is isolated by definition at the
    # scenario level. Rich endpoint parsing is intentionally avoided here.
    adjacency = network.get("adjacency", []) or []
    if not adjacency:
        key_re = re.compile(r"^/?(metro\d+)/(pop|dc)\b")
        for gkey in groups.keys():
            if key_re.match(str(gkey)):
                issues.append(f"{gkey} appears isolated (no adjacency references)")

    # --- DC demand vs adjacency capacity check ---
    try:
        # Helper: extract concrete path string from endpoint (string or mapping)
        def _endpoint_path(ep: Any) -> str:
            """Return path string from endpoint spec.

            Args:
                ep: Endpoint spec: string path or mapping with 'path'.

            Returns:
                String path; empty when not found.
            """

            if isinstance(ep, str):
                return ep
            if isinstance(ep, dict):
                p = ep.get("path")
                return str(p) if p is not None else ""
            return ""

        # Helper: parse DC identifier from a path or regex like '^metro3/dc2/.*'
        dc_re = re.compile(r"\^?/?(metro\d+)/dc(\d+)")

        def _parse_dc_key(path: str) -> str | None:
            """Return canonical DC key 'metroN/dcM' when present in the path.

            Args:
                path: Endpoint path or regex.

            Returns:
                Canonical dc key or None when not a DC path.
            """

            m = dc_re.search(str(path))
            if not m:
                return None
            metro = m.group(1)
            dc_idx = m.group(2)
            return f"{metro}/dc{int(dc_idx)}"

        # Aggregate per-DC total adjacency capacity (sum of DC->PoP adjacencies)
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
                # Prefer target_capacity (total intended per-adjacency capacity); fallback to per-link capacity
                try:
                    cap_val = float(
                        attrs.get("target_capacity", lp.get("capacity", 0.0))
                    )
                except Exception:
                    cap_val = float(lp.get("capacity", 0.0))
                # Attribute link_type can guide us, but robustly rely on path parsing
                for endpoint in (src, dst):
                    key = _parse_dc_key(endpoint)
                    if key:
                        dc_capacity[key] = dc_capacity.get(key, 0.0) + cap_val

        # Aggregate per-DC egress/ingress demand across all matrices
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
                    # Only count demands that explicitly target a single DC endpoint
                    if s_dc:
                        dc_egress[s_dc] = dc_egress.get(s_dc, 0.0) + demand_val
                    if t_dc:
                        dc_ingress[t_dc] = dc_ingress.get(t_dc, 0.0) + demand_val

        # Compare per-DC totals; log and record violations
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
                    (
                        f"dc capacity: {dc_key} egress demand {eg:,.1f} exceeds adjacency capacity {cap:,.1f}"
                    )
                )
            if ing > cap + eps:
                issues.append(
                    (
                        f"dc capacity: {dc_key} ingress demand {ing:,.1f} exceeds adjacency capacity {cap:,.1f}"
                    )
                )
    except Exception as e:
        # Never fail overall validation due to DC capacity audit; report as issue
        issues.append(f"dc capacity audit failed: {e}")

    return issues


def validate_scenario_yaml(
    scenario_yaml: str,
    integrated_graph_path: Path | None = None,
    *,
    run_ngraph: bool = True,
) -> list[str]:
    """Validate scenario YAML and return a list of issue strings.

    Args:
        scenario_yaml: Complete scenario YAML string.
        integrated_graph_path: Optional path to integrated graph JSON for
            cross-check of metro coordinates.
        run_ngraph: If True, attempt to instantiate
            ``ngraph.scenario.Scenario`` for schema validation.

    Returns:
        List of human-readable issue strings. Empty list when no issues.
    """

    issues: list[str] = []

    try:
        data = yaml.safe_load(scenario_yaml) or {}
    except Exception as e:  # YAML parse error
        return [f"YAML parse error: {e}"]

    ig_coords: dict[str, tuple[float, float]] | None = None
    if integrated_graph_path is not None and integrated_graph_path.exists():
        try:
            text = integrated_graph_path.read_text(encoding="utf-8")
            ig_json = json.loads(text)
            ig_coords = _build_ig_coord_map(ig_json)
        except Exception as e:
            issues.append(f"Failed to read integrated graph: {e}")

    # Intra-scenario validation
    issues.extend(validate_scenario_dict(data, ig_coords))

    # Optional ngraph validation and topology checks
    if run_ngraph:
        # Basic schema/graph build check + isolation detection
        try:
            from ngraph.scenario import Scenario  # type: ignore[import-untyped]

            scenario = Scenario.from_yaml(scenario_yaml)

            # Detect isolated nodes using built network graph
            graph = scenario.network.to_strict_multidigraph(add_reverse=True)
            # Use StrictMultiDiGraph helpers to avoid ambiguous typing on degree views
            node_names = list((graph.get_nodes() or {}).keys())
            edges = list((graph.get_edges() or {}).values())
            engaged: set[str] = set()
            for src, dst, _key, _attr in edges:
                engaged.add(str(src))
                engaged.add(str(dst))
            isolated_nodes = [n for n in node_names if n not in engaged]

            if isolated_nodes:
                preview = ", ".join(list(map(str, isolated_nodes))[:10])
                issues.append(
                    f"{len(isolated_nodes)} isolated nodes found in built network (e.g., {preview})"
                )
        except Exception as e:
            issues.append(f"ngraph schema: {e}")

        # Hardware capacity feasibility check using DSL expansion (Explorer analogue)
        try:
            from ngraph.dsl.blueprints.expand import (  # type: ignore[import-untyped]
                expand_network_dsl as _ng_expand,
            )

            from topogen.components_lib import (
                get_builtin_components as _get_components_lib,
            )

            # Build a minimal DSL consisting of blueprints + network only
            d = yaml.safe_load(scenario_yaml) or {}
            dsl = {
                "blueprints": (d.get("blueprints") or {}),
                "network": (d.get("network") or {}),
            }
            net = _ng_expand(dsl)

            comp_lib = _get_components_lib()

            # Pre-compute total attached capacity per node (sum of incident link capacities)
            attached: dict[str, float] = {}
            for link in net.links.values():
                cap = float(getattr(link, "capacity", 0.0) or 0.0)
                src = str(getattr(link, "source", ""))
                dst = str(getattr(link, "target", ""))
                if src:
                    attached[src] = attached.get(src, 0.0) + cap
                if dst:
                    attached[dst] = attached.get(dst, 0.0) + cap

            # Helper to extract node hardware (component name and count)
            def _node_hw(node_attrs: dict[str, object]) -> tuple[str | None, float]:
                hw = node_attrs.get("hardware")
                if isinstance(hw, dict):
                    comp_name = str(hw.get("component", "")).strip()
                    if comp_name:
                        try:
                            count = float(hw.get("count", 1.0))
                        except Exception:
                            count = 1.0
                        return comp_name, count
                # Fallback to role->platform mapping if present in scenario components
                role = str(node_attrs.get("role", "")).strip()
                comps_section = (d.get("components") or {}).get("hw_component", {})
                if isinstance(comps_section, dict) and role in comps_section:
                    comp_name = str(comps_section.get(role) or "").strip()
                    if comp_name:
                        return comp_name, 1.0
                return None, 0.0

            # Check each node against its hardware capacity
            for node in net.nodes.values():
                node_name = str(getattr(node, "name", ""))
                if not node_name:
                    continue
                comp_name, count = _node_hw(getattr(node, "attrs", {}) or {})
                if not comp_name or count <= 0.0:
                    # No hardware assignment for this node; skip feasibility
                    continue
                comp = comp_lib.get(comp_name)
                if comp is None:
                    issues.append(
                        f"hardware capacity: node '{node_name}' references unknown component '{comp_name}'"
                    )
                    continue
                hw_cap = float(comp.get("capacity", 0.0)) * float(count)
                total_attached = float(attached.get(node_name, 0.0))
                if total_attached > hw_cap + 1e-9:
                    issues.append(
                        (
                            "hardware capacity: node '\n"
                            f"{node_name}\n' total attached capacity {total_attached:,.0f} "
                            f"exceeds hardware capacity {hw_cap:,.0f} from component "
                            f"'{comp_name}' (hw_count={count:g})."
                        ).replace("\n", "")
                    )
        except Exception as e:
            # Include context but do not stop other validations
            issues.append(f"ngraph explorer: {e}")

    return issues
