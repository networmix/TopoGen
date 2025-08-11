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
        if not pop or not dc:
            issues.append(f"metro{idx}: missing {'pop' if not pop else 'dc'} group")
            continue

        pa = pop.get("attrs", {}) or {}
        da = dc.get("attrs", {}) or {}

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
                dx = _float_or_nan(da.get("location_x", 0.0))
                dy = _float_or_nan(da.get("location_y", 0.0))
                if not (px == ix and py == iy):
                    issues.append(
                        f"metro{idx}: pop location differs from integrated graph for {name}"
                    )
                if not (dx == ix and dy == iy):
                    issues.append(
                        f"metro{idx}: dc location differs from integrated graph for {name}"
                    )

        # Required DC attributes
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

    return issues
