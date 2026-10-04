"""Explicit role selectors and directional optic endpoint assignments."""

from __future__ import annotations


def role_pair(value: str) -> tuple[str, str]:
    parts = value.split("|")
    if len(parts) != 2 or any(not part.strip() for part in parts):
        raise ValueError(f"Expected role pair 'role|role', got {value!r}")
    return parts[0].strip(), parts[1].strip()


def role_optics(assignments: dict[str, str]) -> dict[tuple[str, str], str]:
    """Map the local and remote endpoint roles to the local optic type."""
    resolved: dict[tuple[str, str], str] = {}
    for pair, component in assignments.items():
        parts = pair.split("->")
        if len(parts) != 2 or any(not part.strip() for part in parts):
            raise ValueError(
                f"Expected optic endpoint mapping 'role->role', got {pair!r}"
            )
        key = parts[0].strip(), parts[1].strip()
        if key in resolved:
            raise ValueError(f"Duplicate optic endpoint mapping: {key}")
        resolved[key] = component
    return resolved
