"""Utilities for handling YAML parsing quirks and common operations."""

from typing import Any, Dict, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def normalize_yaml_dict_keys(data: Dict[Any, V]) -> Dict[str, V]:
    """Normalize dictionary keys from YAML parsing to ensure consistent string keys.

    YAML 1.1 boolean keys (e.g., true, false, yes, no, on, off) get converted to
    Python True/False boolean values. This function converts them to predictable
    string representations ("True"/"False") and ensures all keys are strings.

    Args:
        data: Dictionary that may contain boolean or other non-string keys from YAML parsing

    Returns:
        Dictionary with all keys converted to strings, boolean keys converted to "True"/"False"

    Examples:
        >>> normalize_yaml_dict_keys({True: "value1", False: "value2", "normal": "value3"})
        {"True": "value1", "False": "value2", "normal": "value3"}

        >>> # In YAML: true:, yes:, on: all become Python True
        >>> # In YAML: false:, no:, off: all become Python False
    """
    normalized = {}
    for key, value in data.items():
        # Handle YAML parsing quirks: YAML 1.1 boolean keys (e.g., true, false,
        # yes, no, on, off) get converted to Python True/False. Convert them to
        # predictable string representations.
        if isinstance(key, bool):
            key = str(key)  # Convert True/False to "True"/"False"
        key = str(key)  # Ensure all keys are strings
        normalized[key] = value
    return normalized
