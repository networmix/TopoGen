"""Top-level YAML validation entry point."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from topogen.log_config import get_logger

from .audits import run_ngraph_audits as _run_ngraph_audits
from .helpers import _build_ig_coord_map
from .scenario_dict import validate_scenario_dict as _validate_scenario_dict

logger = get_logger(__name__)


def validate_scenario_yaml(
    scenario_yaml: str,
    integrated_graph_path: Path | None = None,
    *,
    run_ngraph: bool = True,
    hw_component_map: dict[str, str] | None = None,
    optics_map: dict[str, str] | None = None,
) -> list[str]:
    """Validate scenario YAML and return a list of issue strings.

    Args:
        scenario_yaml: Complete scenario YAML string.
        integrated_graph_path: Optional path to integrated graph JSON for
            cross-check of metro coordinates.
        run_ngraph: Check the NetGraph schema, construct a Scenario, and run
            topology, per-demand-set DC capacity and hardware audits. False
            checks metadata/references only; it does not resolve demand selectors.
        hw_component_map: Optional role-to-platform assignments for hardware
            coverage checks. Components must be defined in the scenario.
        optics_map: Optional directional role assignments for optic coverage
            checks. Explicit blueprint hardware is audited directly.

    Returns:
        List of human-readable issue strings. Empty list when no issues.
    """
    issues: list[str] = []

    try:
        data = yaml.safe_load(scenario_yaml) or {}
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    if not isinstance(data, dict):
        return ["Scenario must be a YAML mapping"]

    ig_coords: dict[str, tuple[float, float]] | None = None
    if integrated_graph_path is not None:
        try:
            text = integrated_graph_path.read_text(encoding="utf-8")
            ig_json = json.loads(text)
            ig_coords = _build_ig_coord_map(ig_json)
        except Exception as e:
            issues.append(f"Failed to read integrated graph: {e}")

    issues.extend(_validate_scenario_dict(data, ig_coords))

    if run_ngraph:
        issues.extend(
            _run_ngraph_audits(
                scenario_yaml,
                hw_component_map=hw_component_map,
                optics_map=optics_map,
            )
        )

    for message in issues:
        logger.error(message)
    return issues
