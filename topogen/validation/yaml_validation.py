"""Top-level YAML validation entry point."""

from __future__ import annotations

import json
from importlib import resources as res
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
    hw_component_map: dict[str, object] | None = None,
    optics_map: dict[str, object] | None = None,
) -> list[str]:
    """Validate scenario YAML and return a list of issue strings.

    Args:
        scenario_yaml: Complete scenario YAML string.
        integrated_graph_path: Optional path to integrated graph JSON for
            cross-check of metro coordinates.
        run_ngraph: If True, attempt to instantiate
            ``ngraph.scenario.Scenario`` and run additional audits.
        hw_component_map: Optional override for role->platform mapping sourced
            from configuration. When provided, audits use this instead of any
            mapping present in the scenario.
        optics_map: Optional override for role-pair->optic mapping sourced from
            configuration. When provided, audits use this instead of any mapping
            present in the scenario.

    Returns:
        List of human-readable issue strings. Empty list when no issues.
    """
    issues: list[str] = []

    try:
        data = yaml.safe_load(scenario_yaml) or {}
    except Exception as e:  # YAML parse error
        return [f"YAML parse error: {e}"]

    # Strict JSON Schema validation using embedded ngraph schema
    if run_ngraph:
        try:
            import jsonschema  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover - environment import error
            issues.append(f"ngraph schema: {e}")
        else:
            try:
                with (
                    res.files("ngraph.schemas")
                    .joinpath("scenario.json")
                    .open("r", encoding="utf-8") as f
                ):
                    schema = json.load(f)
            except Exception as e:
                issues.append(f"ngraph schema: {e}")
            else:
                try:
                    jsonschema.validate(data, schema)  # type: ignore[arg-type]
                except Exception as e:
                    issues.append(f"ngraph schema: {e}")

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
        issues.extend(
            _run_ngraph_audits(
                scenario_yaml,
                hw_component_map=hw_component_map,
                optics_map=optics_map,
            )
        )

    # Log all issues at ERROR level for consistency
    try:
        for _msg in issues:
            logger.error(_msg)
    except Exception:  # pragma: no cover
        pass

    return issues
