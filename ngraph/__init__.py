"""NetGraph: Network modeling and analysis library.

Provides interfaces for network topology modeling, traffic analysis, and
capacity planning. Exposes selected modules and artifact types at the package
root for convenience.
"""

from __future__ import annotations

from . import cli, config, logging
from .results.artifacts import CapacityEnvelope, PlacementResultSet, TrafficMatrixSet

__all__ = [
    "cli",
    "config",
    "logging",
    "CapacityEnvelope",
    "PlacementResultSet",
    "TrafficMatrixSet",
]
