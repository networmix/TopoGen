"""Traffic matrix generation algorithms.

Provides functions to generate DC-to-DC traffic matrices for scenarios.

The primary entrypoint is ``generate_traffic_matrix`` which implements two
models:

- "uniform": emit a class-level pairwise demand across all DC nodes using
  regex paths.
- "gravity": compute per-pair allocations proportional to a gravity-like
  kernel over Euclidean metro distances in kilometers with optional jitter,
  top-K pruning, and rounding with conservation via largest remainders.

Time complexity is dominated by pair enumeration between DC nodes which is
O(N_dc^2). Memory usage is O(N_dc^2) for intermediate weights in the gravity
model.
"""

from __future__ import annotations

import math
from typing import Any

from topogen.config import TopologyConfig


def _safe_metro_to_path(metro_name: str) -> str:
    """Return a stable, sanitized slug from a metro display name.

    Args:
        metro_name: Human-readable metro name (e.g., "Salt Lake City").

    Returns:
        Lowercase slug with spaces replaced by hyphens (e.g., "salt-lake-city").
    """

    return metro_name.lower().replace(" ", "-")


def generate_traffic_matrix(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Generate traffic_matrix_set mapping from configuration and metros.

    Builds DC inventory from ``metro_settings``, computes the offered load, and
    emits traffic matrices according to the configured model. When the gravity
    model is selected, per-pair allocations are proportional to
    (m_i^alpha * m_j^alpha) / (dist_ij^beta) using Euclidean distances in km.

    Args:
        metros: Extracted metro descriptors with keys ``name``, ``x``, ``y``.
        metro_settings: Per-metro settings including ``dc_regions_per_metro``.
        config: Full topology configuration.

    Returns:
        Mapping from matrix name to list of demand dicts. Returns empty dict
        when traffic generation is disabled or no DCs exist.

    Raises:
        ValueError: If the gravity model yields zero total weight or if top-K
            pruning removes all pairs.
    """

    traffic_cfg = getattr(config, "traffic", None)
    if not traffic_cfg or not getattr(traffic_cfg, "enabled", False):
        return {}

    # Build DC inventory
    dc_nodes: list[tuple[str, int]] = []  # (metro_name, dc_index)
    for metro_name, settings in metro_settings.items():
        dc_count = int(settings.get("dc_regions_per_metro", 0))
        for dc_idx in range(1, dc_count + 1):
            dc_nodes.append((metro_name, dc_idx))

    if not dc_nodes:
        return {}

    # Offered traffic in Gbps
    gravity_enabled = getattr(traffic_cfg, "model", "uniform") == "gravity"

    def _power_for_dc(metro_name: str, dc_path: str) -> float:
        # Override by full path or by metro name; else default
        overrides = getattr(traffic_cfg.gravity, "mw_per_dc_region_overrides", {})
        if dc_path in overrides:
            return float(overrides[dc_path])
        if metro_name in overrides:
            return float(overrides[metro_name])
        return float(traffic_cfg.mw_per_dc_region)

    # Compute per-DC masses and total power
    dc_mass: dict[tuple[str, int], float] = {}
    total_power_mw = 0.0
    for metro_name, dc_idx in dc_nodes:
        dc_path = f"{_safe_metro_to_path(metro_name)}/dc{dc_idx}"
        mw = _power_for_dc(metro_name, dc_path)
        dc_mass[(metro_name, dc_idx)] = mw
        total_power_mw += mw

    offered_gbps = float(traffic_cfg.gbps_per_mw) * float(total_power_mw)

    if not gravity_enabled:
        # Uniform model emission using regex selection across all DCs
        source_regex = "(metro[0-9]+/dc[0-9]+)"
        sink_regex = "(metro[0-9]+/dc[0-9]+)"
        demands: list[dict[str, Any]] = []
        for priority, ratio in sorted(traffic_cfg.priority_ratios.items()):
            class_demand = offered_gbps * float(ratio)
            demands.append(
                {
                    "source_path": source_regex,
                    "sink_path": sink_regex,
                    "mode": "pairwise",
                    "priority": int(priority),
                    "demand": float(class_demand),
                }
            )
        return {traffic_cfg.matrix_name: demands}

    # Gravity model configuration
    gcfg = traffic_cfg.gravity

    # Metro coordinates (EPSG:5070 meters); convert to km for distance
    coords: dict[str, tuple[float, float]] = {
        m["name"]: (float(m.get("x", 0.0)), float(m.get("y", 0.0))) for m in metros
    }

    def _distance_km(m1: str, m2: str) -> float:
        if m1 == m2:
            return max(float(gcfg.min_distance_km), 0.0)
        (x1, y1) = coords.get(m1, (0.0, 0.0))
        (x2, y2) = coords.get(m2, (0.0, 0.0))
        dx = x1 - x2
        dy = y1 - y2
        return max(math.hypot(dx, dy) / 1000.0, float(gcfg.min_distance_km))

    # Build undirected weights for pairs of DC nodes
    weights: dict[tuple[tuple[str, int], tuple[str, int]], float] = {}
    total_w = 0.0
    for i, (m1, d1) in enumerate(dc_nodes):
        for j in range(i + 1, len(dc_nodes)):
            m2, d2 = dc_nodes[j]
            if gcfg.exclude_same_metro and m1 == m2:
                continue
            m_i = dc_mass[(m1, d1)]
            m_j = dc_mass[(m2, d2)]
            dist = _distance_km(m1, m2)
            dist_eff = max(dist, float(gcfg.min_distance_km))
            w = (
                (m_i ** float(gcfg.alpha))
                * (m_j ** float(gcfg.alpha))
                / (dist_eff ** float(gcfg.beta))
            )
            if w <= 0.0:
                continue
            key = ((m1, d1), (m2, d2))
            weights[key] = w
            total_w += w

    if total_w <= 0.0:
        raise ValueError(
            "Gravity traffic model produced zero total weight across DC pairs"
        )

    # Optional top-K pruning per DC
    if gcfg.max_partners_per_dc is not None:
        k = int(gcfg.max_partners_per_dc)
        partners: dict[tuple[str, int], list[tuple[tuple[str, int], float]]] = {}
        for (a, b), w in weights.items():
            partners.setdefault(a, []).append((b, w))
            partners.setdefault(b, []).append((a, w))
        keep: set[tuple[tuple[str, int], tuple[str, int]]] = set()
        for node, lst in partners.items():
            lst_sorted = sorted(lst, key=lambda t: t[1], reverse=True)[:k]
            for other, _w in lst_sorted:
                pair = tuple(sorted([node, other]))  # undirected key
                keep.add((pair[0], pair[1]))
        weights = {pair: w for pair, w in weights.items() if pair in keep}
        total_w = sum(weights.values())
        if total_w <= 0.0:
            raise ValueError(
                "After top-K pruning, no DC pairs remain for gravity model"
            )

    # Map metro name to 1-based index consistent with network group naming
    metro_idx_map = {m["name"]: idx for idx, m in enumerate(metros, 1)}

    # Emit explicit per-pair demands; apply jitter and rounding per class with conservation
    demands: list[dict[str, Any]] = []
    for priority, ratio in sorted(traffic_cfg.priority_ratios.items()):
        D_c = offered_gbps * float(ratio)
        # Raw allocations
        allocs: list[tuple[tuple[tuple[str, int], tuple[str, int]], float]] = []
        for pair, w in weights.items():
            alloc = D_c * (w / total_w)
            allocs.append((pair, alloc))

        # Optional jitter (lognormal with sigma=jitter_stddev, mu set for mean=1)
        if gcfg.jitter_stddev > 0.0:
            import random as _r

            sigma = float(gcfg.jitter_stddev)
            mu = -0.5 * sigma * sigma
            jittered: list[tuple[tuple[tuple[str, int], tuple[str, int]], float]] = []
            total_after = 0.0
            for pair, v in allocs:
                factor = _r.lognormvariate(mu, sigma)
                val = v * factor
                jittered.append((pair, val))
                total_after += val
            if total_after > 0:
                allocs = [(pair, v * (D_c / total_after)) for pair, v in jittered]

        # Apply rounding if requested and repair to conserve totals via largest remainders
        rounding = float(gcfg.rounding_gbps)
        if rounding > 0.0:
            floored: list[
                tuple[tuple[tuple[str, int], tuple[str, int]], float, float]
            ] = []
            total_floor = 0.0
            for pair, v in allocs:
                q = math.floor(v / rounding) * rounding
                rem = v - q
                floored.append((pair, q, rem))
                total_floor += q
            remainder = D_c - total_floor
            steps = int(round(remainder / rounding)) if rounding > 0 else 0
            # Distribute leftover to pairs with largest remainders
            floored.sort(key=lambda t: t[2], reverse=True)
            final_map: dict[tuple[tuple[str, int], tuple[str, int]], float] = {
                pair: q for pair, q, _ in floored
            }
            idx = 0
            while steps > 0 and idx < len(floored):
                pair, q, _rem = floored[idx]
                final_map[pair] = q + rounding
                steps -= 1
                idx += 1
            allocs = list(final_map.items())

        # Emit entries for each pair in both directions using explicit paths
        for pair, v in allocs:
            (m1, d1), (m2, d2) = pair
            i1 = metro_idx_map[m1]
            i2 = metro_idx_map[m2]
            src = f"^metro{i1}/dc{d1}/.*"
            dst = f"^metro{i2}/dc{d2}/.*"
            # Symmetric split: half each direction; render with up to 2 decimals
            demand_each = round(float(v) / 2.0, 2)
            if demand_each <= 0.0:
                continue
            demands.append(
                {
                    "source_path": src,
                    "sink_path": dst,
                    "mode": "pairwise",
                    "priority": int(priority),
                    "demand": demand_each,
                }
            )
            demands.append(
                {
                    "source_path": dst,
                    "sink_path": src,
                    "mode": "pairwise",
                    "priority": int(priority),
                    "demand": demand_each,
                }
            )

    return {traffic_cfg.matrix_name: demands}
