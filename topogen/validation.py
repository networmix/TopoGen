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

    # Only perform a simple, non-DSL-specific check: if there is no adjacency
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
                # Prefer target_capacity (total intended per-adjacency capacity); otherwise use per-link capacity
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
            from copy import deepcopy as _deepcopy

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

            # --- Strict checks: groups must yield nodes; adjacencies must yield links ---
            try:
                # Tag groups with unique ids and expand once
                groups_orig = (
                    _deepcopy((dsl.get("network") or {}).get("groups", {}))
                    if isinstance((dsl.get("network") or {}).get("groups", {}), dict)
                    else {}
                )
                gid_to_path: dict[int, str] = {}
                if isinstance(groups_orig, dict) and groups_orig:
                    for _i, (gpath, gdef) in enumerate(groups_orig.items()):
                        if not isinstance(gdef, dict):
                            continue
                        attrs = gdef.get("attrs")
                        if not isinstance(attrs, dict):
                            attrs = {}
                            gdef["attrs"] = attrs
                        attrs["_tg_group_id"] = _i
                        gid_to_path[_i] = str(gpath)
                    dsl_groups = {
                        "blueprints": dsl.get("blueprints") or {},
                        "network": {
                            "groups": groups_orig,
                            "adjacency": (dsl.get("network") or {}).get(
                                "adjacency", []
                            ),
                        },
                    }
                    net_groups = _ng_expand(dsl_groups)
                    counts: dict[int, int] = {k: 0 for k in gid_to_path}
                    for node in net_groups.nodes.values():
                        try:
                            gid = int(
                                getattr(node, "attrs", {}).get("_tg_group_id", -1)
                            )
                        except Exception:
                            gid = -1
                        if gid in counts:
                            counts[gid] += 1
                    for gid, cnt in counts.items():
                        if cnt <= 0:
                            gpath = gid_to_path[gid]
                            try:
                                logger.error(
                                    "validation: group expands to zero nodes: %s",
                                    gpath,
                                )
                            except Exception:
                                pass
                            issues.append(f"group '{gpath}' expands to 0 nodes")
                # Tag all scenario-level adjacency rules and expand once
                adj_list = (dsl.get("network") or {}).get("adjacency", [])
                if isinstance(adj_list, list) and adj_list:
                    tagged_adj: list[dict[str, Any]] = []
                    tag_attr = "_tg_adj_tag"
                    for idx, rule in enumerate(adj_list):
                        if not isinstance(rule, dict):
                            continue
                        r = _deepcopy(rule)
                        lp = r.get("link_params")
                        if not isinstance(lp, dict):
                            lp = {}
                            r["link_params"] = lp
                        attrs = lp.get("attrs")
                        if not isinstance(attrs, dict):
                            attrs = {}
                            lp["attrs"] = attrs
                        attrs[tag_attr] = f"adj_{idx}"
                        tagged_adj.append(r)
                    dsl_adj = {
                        "blueprints": dsl.get("blueprints") or {},
                        "network": {
                            "groups": (
                                _deepcopy((dsl.get("network") or {}).get("groups", {}))
                                if isinstance(
                                    (dsl.get("network") or {}).get("groups", {}), dict
                                )
                                else {}
                            ),
                            "adjacency": tagged_adj,
                        },
                    }
                    net_adj = _ng_expand(dsl_adj)
                    adj_counts_by_tag: dict[str, int] = {}
                    for link in net_adj.links.values():
                        tag = getattr(link, "attrs", {}).get(tag_attr)
                        if isinstance(tag, str):
                            adj_counts_by_tag[tag] = adj_counts_by_tag.get(tag, 0) + 1
                    for idx, _rule in enumerate(adj_list):
                        tag = f"adj_{idx}"
                        if adj_counts_by_tag.get(tag, 0) <= 0:
                            # Include a brief rule summary
                            try:
                                src = (
                                    _rule.get("source")
                                    if isinstance(_rule, dict)
                                    else None
                                )
                                dst = (
                                    _rule.get("target")
                                    if isinstance(_rule, dict)
                                    else None
                                )
                                patt = (
                                    _rule.get("pattern")
                                    if isinstance(_rule, dict)
                                    else None
                                )
                            except Exception:
                                src = dst = patt = None
                            try:
                                logger.error(
                                    (
                                        "validation: adjacency[%d] expands to zero "
                                        "links: source=%r target=%r pattern=%r"
                                    ),
                                    idx,
                                    src,
                                    dst,
                                    patt,
                                )
                            except Exception:
                                pass
                            issues.append(
                                f"adjacency[{idx}] expands to 0 links (source={src}, target={dst}, pattern={patt})"
                            )
                # For each blueprint present, tag each adjacency rule and expand once
                blueprints = dsl.get("blueprints") or {}
                if isinstance(blueprints, dict) and blueprints:
                    for bp_name, bp_def in blueprints.items():
                        if not isinstance(bp_def, dict):
                            continue
                        bp_adj = bp_def.get("adjacency", [])
                        if not isinstance(bp_adj, list) or not bp_adj:
                            continue
                        bp_copy = _deepcopy(bp_def)
                        tag_attr = "_tg_bp_adj_tag"
                        expected_tags: list[str] = []
                        for i, rule in enumerate(bp_copy.get("adjacency", [])):
                            if not isinstance(rule, dict):
                                continue
                            lp = rule.get("link_params")
                            if not isinstance(lp, dict):
                                lp = {}
                                rule["link_params"] = lp
                            attrs = lp.get("attrs")
                            if not isinstance(attrs, dict):
                                attrs = {}
                                lp["attrs"] = attrs
                            tag_val = f"{bp_name}#{i}"
                            expected_tags.append(tag_val)
                            attrs[tag_attr] = tag_val
                        dsl_bp = {
                            "blueprints": {str(bp_name): bp_copy},
                            "network": {
                                "groups": {"__check__": {"use_blueprint": str(bp_name)}}
                            },
                        }
                        net_bp = _ng_expand(dsl_bp)
                        seen: set[str] = set()
                        for link in net_bp.links.values():
                            tag_val = getattr(link, "attrs", {}).get(tag_attr)
                            if isinstance(tag_val, str):
                                seen.add(tag_val)
                        for i, tag_val in enumerate(expected_tags):
                            if tag_val not in seen:
                                try:
                                    logger.error(
                                        (
                                            "validation: blueprint '%s' "
                                            "adjacency[%d] expands to zero links"
                                        ),
                                        bp_name,
                                        i,
                                    )
                                except Exception:
                                    pass
                                issues.append(
                                    f"blueprint '{bp_name}' adjacency[{i}] expands to 0 links"
                                )
            except Exception as e:
                issues.append(f"adjacency/group expansion audit failed: {e}")

            # Continue with hardware capacity feasibility and assignment checks
            net = _ng_expand(dsl)

            comp_lib = _get_components_lib()

            # --- Node role presence audit (roles are required by design) ---
            try:
                missing_role_count = 0
                missing_role_samples: list[str] = []
                for node in net.nodes.values():
                    nname = str(getattr(node, "name", ""))
                    role = str(getattr(node, "attrs", {}).get("role", "")).strip()
                    if not role:
                        missing_role_count += 1
                        if (
                            nname
                            and len(missing_role_samples) < 5
                            and nname not in missing_role_samples
                        ):
                            missing_role_samples.append(nname)
                if missing_role_count:
                    if missing_role_samples:
                        examples = ", ".join(
                            "'" + n + "'" for n in missing_role_samples
                        )
                        issues.append(
                            f"node roles: {missing_role_count} node(s) missing role (e.g., {examples})"
                        )
                    else:
                        issues.append("node roles: nodes missing role detected")
            except Exception as e:
                issues.append(f"node roles audit failed: {e}")

            # --- Node hardware presence audit (assignment completeness) ---
            try:
                comps_section = (d.get("components") or {}).get("hw_component", {})
                if isinstance(comps_section, dict) and comps_section:
                    for node in net.nodes.values():
                        nname = str(getattr(node, "name", ""))
                        attrs = getattr(node, "attrs", {}) or {}
                        role = str(attrs.get("role", "")).strip()
                        if not role or role not in comps_section:
                            continue
                        hw = attrs.get("hardware")
                        if (
                            not isinstance(hw, dict)
                            or not str(hw.get("component", "")).strip()
                        ):
                            issues.append(
                                f"node hardware: role '{role}' mapped in components.hw_component but node '{nname}' has no hardware assignment"
                            )
                        else:
                            # If present, also check capacity sufficiency against attached capacity
                            try:
                                comp_name = str(hw.get("component", "")).strip()
                                count = float(hw.get("count", 1.0))
                                comp = comp_lib.get(comp_name)
                                if comp is None:
                                    issues.append(
                                        f"node hardware: node '{nname}' references unknown component '{comp_name}'"
                                    )
                                else:
                                    hw_cap = float(comp.get("capacity", 0.0)) * float(
                                        count
                                    )
                                    # total_attached computed later; defer compare by recording needed
                            except (
                                Exception
                            ) as e2:  # pragma: no cover - be conservative
                                issues.append(
                                    f"node hardware: capacity calculation error for node '{nname}': {e2}"
                                )
            except Exception as e:  # pragma: no cover - validation should not crash
                issues.append(f"node hardware audit failed: {e}")

            # --- Link optics presence audit (per-end, unordered/directional semantics) ---
            try:
                # Build directional lookup: (src_role, dst_role) -> optic for source end
                optics_cfg = (d.get("components") or {}).get("optics", {})
                optics_map: dict[tuple[str, str], str] = {}
                optics_values: dict[tuple[str, str], set[str]] = {}
                mapping_conflicts: dict[tuple[str, str], set[str]] = {}
                if isinstance(optics_cfg, dict) and optics_cfg:
                    # First, collect '|' (unordered) declarations by pair signature
                    unordered_by_pair: dict[
                        frozenset[str], list[tuple[str, str, str]]
                    ] = {}
                    for key, val in optics_cfg.items():
                        if not isinstance(val, str):
                            continue
                        k = str(key)
                        if "-" in k and "|" not in k:
                            a, b = [p.strip() for p in k.split("-", 1)]
                            if a and b:
                                key = (a, b)
                                ov = str(val)
                                if key not in optics_map:
                                    optics_map[key] = ov
                                if key not in optics_values:
                                    optics_values[key] = set()
                                optics_values[key].add(ov)
                        elif "|" in k:
                            a, b = [p.strip() for p in k.split("|", 1)]
                            if a and b:
                                sig = frozenset((a, b))
                                unordered_by_pair.setdefault(sig, []).append(
                                    (a, b, str(val))
                                )
                    # Resolve unordered declarations
                    for _sig, entries in unordered_by_pair.items():
                        if len(entries) == 1:
                            a, b, ov = entries[0]
                            # Single unordered spec applies to both ends
                            for key in ((a, b), (b, a)):
                                if key not in optics_map:
                                    optics_map[key] = ov
                                if key not in optics_values:
                                    optics_values[key] = set()
                                optics_values[key].add(ov)
                        else:
                            # Both directions present with '|': treat as ordered by left role
                            for a, b, ov in entries:
                                key = (a, b)
                                if key not in optics_map:
                                    optics_map[key] = ov
                                if key not in optics_values:
                                    optics_values[key] = set()
                                optics_values[key].add(ov)

                    # Detect conflicts: any pair assigned multiple different optics
                    for key, values in optics_values.items():
                        if len(values) > 1:
                            mapping_conflicts[key] = values

                if optics_map:
                    # Helper to get role of a node by name
                    def _role_of(node_name: str) -> str:
                        try:
                            return str(
                                getattr(net.nodes[node_name], "attrs", {}).get(
                                    "role", ""
                                )
                            )
                        except Exception:
                            return ""

                    from collections import defaultdict as _dd

                    miss_src: dict[tuple[str, str], int] = _dd(int)
                    miss_tgt: dict[tuple[str, str], int] = _dd(int)
                    miss_src_samples: dict[tuple[str, str], list[str]] = _dd(list)
                    miss_tgt_samples: dict[tuple[str, str], list[str]] = _dd(list)
                    # Missing mapping when hardware absent
                    miss_map_src: dict[tuple[str, str], int] = _dd(int)
                    miss_map_tgt: dict[tuple[str, str], int] = _dd(int)
                    miss_map_src_samples: dict[tuple[str, str], list[str]] = _dd(list)
                    miss_map_tgt_samples: dict[tuple[str, str], list[str]] = _dd(list)
                    # Aggregation for blueprint-provided hardware checks
                    bhw_missing_src: dict[tuple[str, str], int] = _dd(int)
                    bhw_missing_tgt: dict[tuple[str, str], int] = _dd(int)
                    bhw_cap_short: dict[tuple[str, str], int] = _dd(int)
                    bhw_unknown_src: dict[tuple[str, str], int] = _dd(int)
                    bhw_unknown_tgt: dict[tuple[str, str], int] = _dd(int)
                    bhw_calc_err: dict[tuple[str, str], int] = _dd(int)
                    bhw_samples: dict[tuple[str, str], list[str]] = _dd(list)

                    for link in net.links.values():
                        src = str(getattr(link, "source", ""))
                        dst = str(getattr(link, "target", ""))
                        if not src or not dst:
                            continue
                        rs = _role_of(src)
                        rd = _role_of(dst)
                        if not rs or not rd:
                            continue
                        attrs_link = getattr(link, "attrs", {}) or {}
                        aid = str(attrs_link.get("adjacency_id", "")).strip()
                        if not aid:
                            aid = str(attrs_link.get("link_type", "")).strip()
                        hw = attrs_link.get("hardware")
                        # If hardware is present on the link, validate presence and capacity
                        # and skip mapping enforcement for this link.
                        if isinstance(hw, dict):
                            src_hw = (
                                hw.get("source")
                                if isinstance(hw.get("source"), dict)
                                else None
                            )
                            tgt_hw = (
                                hw.get("target")
                                if isinstance(hw.get("target"), dict)
                                else None
                            )
                            if not src_hw:
                                key = (rs, rd)
                                bhw_missing_src[key] += 1
                                if (
                                    aid
                                    and len(bhw_samples[key]) < 3
                                    and aid not in bhw_samples[key]
                                ):
                                    bhw_samples[key].append(aid)
                                continue
                            if not tgt_hw:
                                key = (rd, rs)
                                bhw_missing_tgt[key] += 1
                                if (
                                    aid
                                    and len(bhw_samples[key]) < 3
                                    and aid not in bhw_samples[key]
                                ):
                                    bhw_samples[key].append(aid)
                                continue
                            # Capacity feasibility: component capacity * count must cover per-link capacity
                            try:
                                comp_src = str(src_hw.get("component", "")).strip()
                                comp_tgt = str(tgt_hw.get("component", "")).strip()
                                c_src = comp_lib.get(comp_src) if comp_src else None
                                c_tgt = comp_lib.get(comp_tgt) if comp_tgt else None
                                if comp_src and not c_src:
                                    key = (rs, rd)
                                    bhw_unknown_src[key] += 1
                                    if (
                                        aid
                                        and len(bhw_samples[key]) < 3
                                        and aid not in bhw_samples[key]
                                    ):
                                        bhw_samples[key].append(aid)
                                if comp_tgt and not c_tgt:
                                    key = (rd, rs)
                                    bhw_unknown_tgt[key] += 1
                                    if (
                                        aid
                                        and len(bhw_samples[key]) < 3
                                        and aid not in bhw_samples[key]
                                    ):
                                        bhw_samples[key].append(aid)
                                cnt_src = float(src_hw.get("count", 1.0))
                                cnt_tgt = float(tgt_hw.get("count", 1.0))
                                av_src = (
                                    (float(c_src.get("capacity", 0.0)) * cnt_src)
                                    if c_src
                                    else 0.0
                                )
                                av_tgt = (
                                    (float(c_tgt.get("capacity", 0.0)) * cnt_tgt)
                                    if c_tgt
                                    else 0.0
                                )
                                need = float(getattr(link, "capacity", 0.0) or 0.0)
                                if av_src + 1e-9 < need or av_tgt + 1e-9 < need:
                                    key = (rs, rd)
                                    bhw_cap_short[key] += 1
                                    if (
                                        aid
                                        and len(bhw_samples[key]) < 3
                                        and aid not in bhw_samples[key]
                                    ):
                                        bhw_samples[key].append(aid)
                            except Exception:
                                key = (rs, rd)
                                bhw_calc_err[key] += 1
                                if (
                                    aid
                                    and len(bhw_samples[key]) < 3
                                    and aid not in bhw_samples[key]
                                ):
                                    bhw_samples[key].append(aid)
                            continue

                        # Enforce mapping only when hardware is not already specified on the link
                        req_src = optics_map.get((rs, rd))
                        req_tgt = optics_map.get((rd, rs))
                        if isinstance(req_src, str):
                            key = (rs, rd)
                            miss_src[key] += 1
                            if (
                                aid
                                and len(miss_src_samples[key]) < 3
                                and aid not in miss_src_samples[key]
                            ):
                                miss_src_samples[key].append(aid)
                        else:
                            # Mapping missing for source end
                            key = (rs, rd)
                            miss_map_src[key] += 1
                            if (
                                aid
                                and len(miss_map_src_samples[key]) < 3
                                and aid not in miss_map_src_samples[key]
                            ):
                                miss_map_src_samples[key].append(aid)
                        if isinstance(req_tgt, str):
                            key = (rd, rs)
                            miss_tgt[key] += 1
                            if (
                                aid
                                and len(miss_tgt_samples[key]) < 3
                                and aid not in miss_tgt_samples[key]
                            ):
                                miss_tgt_samples[key].append(aid)
                        else:
                            # Mapping missing for target end
                            key = (rd, rs)
                            miss_map_tgt[key] += 1
                            if (
                                aid
                                and len(miss_map_tgt_samples[key]) < 3
                                and aid not in miss_map_tgt_samples[key]
                            ):
                                miss_map_tgt_samples[key].append(aid)

                    # Emit aggregated issues
                    # Mapping conflicts
                    for (a, b), vals in sorted(mapping_conflicts.items()):
                        vstr = ", ".join(sorted(vals))
                        issues.append(
                            f"optics mapping: conflicting values for roles ({a},{b}) -> {{{vstr}}}"
                        )
                    for (a, b), cnt in sorted(bhw_missing_src.items()):
                        samples = bhw_samples.get((a, b), [])
                        prefix = f"optics (blueprint): missing hardware on source end for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join("'" + s + "'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
                    for (a, b), cnt in sorted(bhw_missing_tgt.items()):
                        samples = bhw_samples.get((a, b), [])
                        prefix = f"optics (blueprint): missing hardware on target end for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join("'" + s + "'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
                    for (a, b), cnt in sorted(bhw_unknown_src.items()):
                        samples = bhw_samples.get((a, b), [])
                        prefix = f"optics (blueprint): unknown component on source end for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join("'" + s + "'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
                    for (a, b), cnt in sorted(bhw_unknown_tgt.items()):
                        samples = bhw_samples.get((a, b), [])
                        prefix = f"optics (blueprint): unknown component on target end for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join("'" + s + "'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
                    for (a, b), cnt in sorted(bhw_cap_short.items()):
                        samples = bhw_samples.get((a, b), [])
                        prefix = f"optics (blueprint): hardware capacity shortfall for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join("'" + s + "'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
                    for (a, b), cnt in sorted(bhw_calc_err.items()):
                        samples = bhw_samples.get((a, b), [])
                        prefix = f"optics (blueprint): capacity calculation error for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join("'" + s + "'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
                    # Missing-mapping issues when hardware absent
                    for (a, b), cnt in sorted(miss_map_src.items()):
                        samples = miss_map_src_samples.get((a, b), [])
                        prefix = f"optics mapping: missing source-end mapping for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join("'" + s + "'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
                    for (a, b), cnt in sorted(miss_map_tgt.items()):
                        samples = miss_map_tgt_samples.get((a, b), [])
                        prefix = f"optics mapping: missing target-end mapping for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join("'" + s + "'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
                    for (a, b), cnt in sorted(miss_src.items()):
                        samples = miss_src_samples.get((a, b), [])
                        prefix = f"optics: missing hardware required by mapping on source end for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join(f"'{s}'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
                    for (a, b), cnt in sorted(miss_tgt.items()):
                        samples = miss_tgt_samples.get((a, b), [])
                        prefix = f"optics: missing hardware required by mapping on target end for roles ({a},{b}) - {cnt} links"
                        if samples:
                            sample_str = ", ".join(f"'{s}'" for s in samples)
                            issues.append(f"{prefix} (e.g., adj={sample_str})")
                        else:
                            issues.append(prefix)
            except Exception as e:  # pragma: no cover - validation should not crash
                issues.append(f"link optics audit failed: {e}")

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
                # No fallback to mapping here: presence is checked separately; capacity only uses explicit node HW
                return None, 0.0

            # Check each node against its hardware capacity (explicit node HW only)
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

    # Log all issues at ERROR level for consistency
    try:
        for _msg in issues:
            logger.error(_msg)
    except Exception:  # pragma: no cover - logging must not break validation
        pass

    return issues
