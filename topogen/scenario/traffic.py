"""Map generated traffic matrices to the NetGraph ``demands`` section."""

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
    """Return NetGraph demand sets, or {} when traffic is disabled or no DCs exist."""

    # Map traffic generator fields to NetGraph demand fields.
    field_names = {
        "source_path": "source",
        "sink_path": "target",
        "demand": "volume",
        "flow_policy_config": "flow_policy",
    }
    return {
        name: [
            {field_names.get(key, key): value for key, value in demand.items()}
            for demand in demands
        ]
        for name, demands in generate_traffic_matrix(
            metros, metro_settings, config
        ).items()
    }
