"""Read optional user libraries with one strict mapping contract."""

from pathlib import Path
from typing import Any

import yaml


def load_user_library(file_name: str) -> dict[str, Any]:
    path = Path.cwd() / "lib" / file_name
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"User library YAML must be a mapping: {path}")
    return data
