"""Traffic section builder for scenario assembly.

Provides a thin adapter that constructs the ``traffic_matrix_set`` section
using the traffic generation algorithms in ``topogen.traffic_matrix``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from topogen.traffic_matrix import generate_traffic_matrix

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig


def _build_traffic_matrix_section(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: "TopologyConfig",
) -> dict[str, list[dict[str, Any]]]:
    """Build the ``traffic_matrix_set`` section if enabled.

    Args:
        metros: Extracted metro descriptors.
        metro_settings: Per-metro settings including DC region counts.
        config: Full topology configuration.

    Returns:
        Mapping for the scenario ``traffic_matrix_set`` section. Returns empty
        mapping when traffic generation is disabled or no DC regions exist.
    """

    return generate_traffic_matrix(metros, metro_settings, config)
