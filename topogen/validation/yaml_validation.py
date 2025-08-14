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


def validate_scenario_yaml(  # noqa: C901, PLR0912, PLR0915
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
            ``ngraph.scenario.Scenario`` and run additional audits.

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

    # Intra-scenario validation (pure dict checks)
    issues.extend(_validate_scenario_dict(data, ig_coords))

    # Optional ngraph validation and topology checks
    if run_ngraph:
        issues.extend(_run_ngraph_audits(scenario_yaml))

    # Log all issues at ERROR level for consistency
    try:
        for _msg in issues:
            logger.error(_msg)
    except Exception:  # pragma: no cover
        pass

    return issues
