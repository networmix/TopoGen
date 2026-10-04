"""Map generated traffic matrices to the NetGraph demands contract."""

from typing import Any


def to_demand_sets(
    matrices: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Translate the same matrices used for sizing without regenerating traffic."""
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
        for name, demands in matrices.items()
    }
