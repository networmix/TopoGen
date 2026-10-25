"""Decimal arithmetic at configuration and written-hardware boundaries."""

from __future__ import annotations

import math
from fractions import Fraction


def decimal(value: int | float | Fraction) -> Fraction:
    """Read a numeric scalar as the decimal its repr writes."""
    return value if isinstance(value, Fraction) else Fraction(repr(value))


def optic_ratio(written_capacity: float, held_optic_capacity: float) -> Fraction:
    """Exact number of optics needed to cover a written capacity."""
    capacity = decimal(held_optic_capacity)
    if capacity <= 0:
        raise ValueError(f"Optic capacity must be positive: {held_optic_capacity!r}")
    return decimal(written_capacity) / capacity


def optic_count(written_capacity: float, held_optic_capacity: float) -> int:
    return math.ceil(optic_ratio(written_capacity, held_optic_capacity))


def covers(count: float, held_optic_capacity: float, written_capacity: float) -> bool:
    return decimal(count) >= optic_ratio(written_capacity, held_optic_capacity)


def written_count(count: int, label: str) -> float:
    """Refuse any count whose float output would change the chosen integer."""
    output = 0.0
    try:
        output = float(count)
        writable = math.isfinite(output) and decimal(output) == count
    except OverflowError:
        writable = False
    if not writable:
        raise ValueError(f"{label}: optic count {count} cannot be written exactly")
    return output
