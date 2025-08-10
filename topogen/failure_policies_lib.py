"""Built-in failure policy definitions.

Provides the minimal API used by the scenario pipeline and merges overrides
from ``cwd/lib/failure_policies.yml`` when present. The user file must be
direct mapping: name -> definition. User entries override built-ins.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# Built-in failure policy definitions
_BUILTIN_FAILURE_POLICIES: dict[str, dict[str, Any]] = {
    "empty": {},
    "single_random_link_failure": {
        "attrs": {
            "description": "Fails exactly one random link to test network resilience"
        },
        "rules": [{"entity_scope": "link", "rule_type": "choice", "count": 1}],
    },
}


def get_builtin_failure_policies() -> dict[str, dict[str, Any]]:
    """Return failure policies library merged with user overrides.

    Returns:
        Dictionary mapping policy names to their definitions.
    """
    policies = deepcopy(_BUILTIN_FAILURE_POLICIES)
    user_policies = _load_user_library("failure_policies.yml")
    # Support only direct mapping: name -> definition
    policies.update(user_policies)
    return policies


def _load_user_library(file_name: str) -> dict[str, Any]:
    """Load user failure policies from ``lib/<file_name>`` if present."""
    lib_path = Path.cwd() / "lib" / file_name
    if not lib_path.exists():
        return {}

    try:
        with lib_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse YAML: {lib_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"User library YAML must be a mapping: {lib_path}")

    return data
