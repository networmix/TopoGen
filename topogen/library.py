"""Explicit, owned snapshots of scenario libraries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from ngraph.workflow.parse import build_workflow_steps

from topogen.blueprints_lib import _BUILTIN_BLUEPRINTS
from topogen.components_lib import _BUILTIN_COMPONENTS
from topogen.failure_policies_lib import _BUILTIN_FAILURE_POLICIES
from topogen.workflows_lib import _BUILTIN_WORKFLOWS


def freeze(value: Any) -> Any:
    """Own recursively immutable containers without changing scalar values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(child) for child in value)
    return value


def thaw(value: Any) -> Any:
    """Make serialization containers from an owned snapshot."""
    if isinstance(value, Mapping):
        return {key: thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return {thaw(child) for child in value}
    return value


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    steps: tuple[Mapping[str, Any], ...]
    follows_configured_matrix: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", freeze(self.steps))


@dataclass(frozen=True, slots=True)
class Library:
    blueprints: Mapping[str, Any]
    components: Mapping[str, Any]
    failure_policies: Mapping[str, Any]
    workflows: Mapping[str, WorkflowDefinition]

    def __post_init__(self) -> None:
        for field in ("blueprints", "components", "failure_policies", "workflows"):
            object.__setattr__(self, field, freeze(getattr(self, field)))


def load_library(directory: Path | None) -> Library:
    """Load builtins and optional explicit overrides; validate every workflow."""
    tables = {}
    replaced = set()
    for name, builtin in (
        ("blueprints", _BUILTIN_BLUEPRINTS),
        ("components", _BUILTIN_COMPONENTS),
        ("failure_policies", _BUILTIN_FAILURE_POLICIES),
        ("workflows", _BUILTIN_WORKFLOWS),
    ):
        table = thaw(builtin)
        path = directory / f"{name}.yml" if directory is not None else None
        if path is not None and path.exists():
            try:
                overrides = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ValueError(f"Failed to parse YAML: {path}") from exc
            if not isinstance(overrides, dict):
                raise ValueError(f"User library YAML must be a mapping: {path}")
            table.update(overrides)
            if name == "workflows":
                replaced.update(overrides)
        tables[name] = table
    workflows = {}
    for name, steps in tables["workflows"].items():
        if not isinstance(steps, list) or any(not isinstance(s, dict) for s in steps):
            raise ValueError(f"Workflow '{name}' must contain a list of step mappings")
        try:
            build_workflow_steps(steps, derive_seed=lambda _: None)
        except ValueError as exc:
            raise ValueError(f"Workflow '{name}': {exc}") from exc
        workflows[name] = WorkflowDefinition(tuple(steps), name not in replaced)
    return Library(
        tables["blueprints"],
        tables["components"],
        tables["failure_policies"],
        workflows,
    )
