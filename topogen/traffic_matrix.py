"""Generate deterministic uniform, gravity and hose demand sets.

Gravity and hose share pair selection, rounding and symmetric emission. Their
pair inventories are quadratic in DC count; uniform emits one selector per class.
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from topogen.config import TopologyConfig, TrafficConfig, _validate_flow_policy_names
from topogen.log_config import get_logger

logger = get_logger(__name__)
Pair = tuple[int, int]


def _fit_hose_matrix(
    matrix: list[list[float]], targets: list[float]
) -> list[list[float]]:
    """Fit supported pairs to both margins, rejecting infeasible or unfinished fits."""
    values = np.asarray(matrix, dtype=float)
    margins = np.asarray(targets, dtype=float)
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(margins))
        or np.any(values < 0)
        or np.any(margins < 0)
    ):
        raise ValueError("hose requires finite non-negative weights and totals")
    total = math.fsum(targets)
    largest = int(np.argmax(margins))
    rest = math.fsum(value for index, value in enumerate(targets) if index != largest)
    if margins[largest] > rest and not math.isclose(
        margins[largest], rest, rel_tol=1e-12, abs_tol=0
    ):
        raise ValueError(
            "hose margins are infeasible: one DC exceeds all peers combined"
        )
    # At equality, only a star through the largest DC can satisfy both margins.
    if math.isclose(margins[largest], rest, rel_tol=1e-12, abs_tol=0):
        star = np.zeros_like(values)
        for index, target in enumerate(targets):
            if index == largest or target == 0:
                continue
            if values[largest, index] <= 0 or values[index, largest] <= 0:
                raise ValueError("hose support is infeasible for the requested margins")
            star[largest, index] = star[index, largest] = target
        return star.tolist()
    for _ in range(2000):
        for axis in (1, 0):
            sums = values.sum(axis=axis)
            if np.any((sums == 0) & (margins > 0)):
                raise ValueError(
                    "hose support is infeasible: a DC has no eligible peers"
                )
            factors = np.divide(
                margins, sums, out=np.zeros_like(margins), where=sums > 0
            )
            values *= factors[:, None] if axis == 1 else factors[None, :]
        if all(
            np.allclose(values.sum(axis=axis), margins, rtol=1e-9, atol=total * 1e-12)
            for axis in (0, 1)
        ):
            return values.tolist()
    raise ValueError(
        "hose fitting did not converge after 2000 iterations; adjust DC totals or retained partners"
    )


@dataclass(frozen=True, slots=True)
class _DC:
    metro: str
    path: str
    mass: float


def _inventory(
    metros: list[dict[str, Any]],
    settings: dict[str, dict[str, Any]],
    config: TrafficConfig,
) -> list[_DC]:
    inventory = []
    overrides = config.gravity.mw_per_dc_region_overrides
    for metro_index, metro in enumerate(metros, 1):
        name = metro["name"]
        prefix = name.lower().replace(" ", "-")
        for ordinal in range(1, settings[name]["dc_regions_per_metro"] + 1):
            mass = float(
                overrides.get(
                    f"{prefix}/dc{ordinal}",
                    overrides.get(name, config.mw_per_dc_region),
                )
            )
            if not math.isfinite(mass) or mass < 0:
                raise ValueError(
                    f"DC power for {name}/dc{ordinal} must be finite and non-negative"
                )
            inventory.append(
                _DC(
                    name,
                    f"^metro{metro_index}/dc{ordinal}/.*",
                    mass,
                )
            )
    return inventory


def _distances(
    dcs: list[_DC], metros: list[dict[str, Any]], minimum: float
) -> np.ndarray:
    positions = {metro["name"]: (metro["x"], metro["y"]) for metro in metros}
    xy = np.array([positions[dc.metro] for dc in dcs], dtype=float)
    return np.maximum(
        np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        / 1000,
        minimum,
    )


def _top_pairs(weights: dict[Pair, float], k: int) -> dict[Pair, float]:
    """Keep the union of each DC's strongest K partners, with stable ties."""
    if k <= 0:
        raise ValueError("Partner count must be positive")
    partners: dict[int, list[tuple[Pair, float]]] = {}
    for pair, value in weights.items():
        for endpoint in pair:
            partners.setdefault(endpoint, []).append((pair, value))
    keep = {
        pair
        for entries in partners.values()
        for pair, _ in heapq.nlargest(k, entries, key=lambda item: item[1])
    }
    return {pair: value for pair, value in weights.items() if pair in keep}


def _quantize(
    allocations: dict[Pair, float], total: float, step: float, policy: str
) -> dict[Pair, float]:
    """Quantize pair totals and distribute positive residual by largest remainder."""
    if step == 0:
        return allocations
    quantizer = {"nearest": round, "ceil": math.ceil, "floor": math.floor}[policy]
    rounded = [
        (pair, quantizer(value / step) * step, value)
        for pair, value in allocations.items()
    ]
    rounded.sort(key=lambda item: item[2] - item[1], reverse=True)
    steps = max(0, int(round((total - sum(value for _, value, _ in rounded)) / step)))
    return {
        pair: value + (step if index < steps else 0)
        for index, (pair, value, _) in enumerate(rounded)
    }


def _emit_pairs(
    dcs: list[_DC],
    pairs: dict[Pair, float],
    distances: np.ndarray,
    config: TrafficConfig,
    offered: float,
    *,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    demands = []
    gravity = config.gravity
    step = gravity.rounding_gbps
    for priority, ratio in sorted(config.priority_ratios.items()):
        total = offered * ratio
        if total <= 0:
            continue
        allocations = {pair: value * ratio for pair, value in pairs.items()}
        if rng is not None and gravity.jitter_stddev > 0:
            sigma = gravity.jitter_stddev
            allocations = {
                pair: value * rng.lognormvariate(-0.5 * sigma * sigma, sigma)
                for pair, value in allocations.items()
            }
            jittered_total = sum(allocations.values())
            if not math.isfinite(jittered_total) or jittered_total <= 0:
                raise ValueError("Gravity jitter produced an invalid total")
            allocations = {
                pair: value * (total / jittered_total)
                for pair, value in allocations.items()
            }
        allocations = _quantize(allocations, total, step, gravity.rounding_policy)
        for (source, target), value in allocations.items():
            each = value / 2
            if step > 0:
                each = round(each / (step / 2)) * (step / 2)
            if each <= 0:
                continue
            distance = math.ceil(float(distances[source, target]))
            for a, b in ((source, target), (target, source)):
                entry = {
                    "source_path": dcs[a].path,
                    "sink_path": dcs[b].path,
                    "mode": "pairwise",
                    "priority": priority,
                    "demand": each,
                    "attrs": {"euclidean_km": distance},
                }
                if priority in config.flow_policy_config:
                    entry["flow_policy_config"] = config.flow_policy_config[priority]
                demands.append(entry)
    return demands


def _gravity_pairs(
    dcs: list[_DC], distances: np.ndarray, config: TrafficConfig, offered: float
) -> dict[Pair, float]:
    gravity = config.gravity
    weights = {}
    for i, a in enumerate(dcs):
        for j in range(i + 1, len(dcs)):
            b = dcs[j]
            if gravity.exclude_same_metro and a.metro == b.metro:
                continue
            value = (
                (a.mass**gravity.alpha)
                * (b.mass**gravity.alpha)
                / (float(distances[i, j]) ** gravity.beta)
            )
            if value > 0:
                weights[i, j] = value
    if gravity.max_partners_per_dc is not None:
        weights = _top_pairs(weights, gravity.max_partners_per_dc)
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(
            "Gravity traffic model produced zero or non-finite total weight across DC pairs"
        )
    return {pair: offered * (value / total) for pair, value in weights.items()}


def _random_hose(
    dcs: list[_DC],
    distances: np.ndarray,
    config: TrafficConfig,
    rng: random.Random,
    keep: set[Pair] | None = None,
) -> list[list[float]]:
    hose = config.hose
    n = len(dcs)
    matrix = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(dcs):
        for j, b in enumerate(dcs):
            if i == j or (keep is not None and (min(i, j), max(i, j)) not in keep):
                continue
            value = 1e-9 + rng.random()
            if hose.tilt_exponent > 0:
                # This setting excludes same-metro pairs from the tilt kernel,
                # while keeping positive support for marginal fitting.
                tilt = (
                    1e-12
                    if hose.exclude_same_metro and a.metro == b.metro
                    else (1.0 / float(distances[i, j]) ** hose.beta)
                    ** hose.tilt_exponent
                )
                value *= tilt
            matrix[i][j] = value
    return _fit_hose_matrix(matrix, [dc.mass * config.gbps_per_mw for dc in dcs])


def _hose_pairs(matrix: list[list[float]]) -> dict[Pair, float]:
    return {
        (i, j): matrix[i][j] + matrix[j][i]
        for i in range(len(matrix))
        for j in range(i + 1, len(matrix))
        if matrix[i][j] + matrix[j][i] > 0
    }


def generate_traffic_matrix(
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    config: TopologyConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Return named demand lists, without file I/O or global PRNG mutation.

    Uniform shares offered volume over ordered DC pairs through NetGraph's
    selectors. Gravity weights pairs by power/distance; hose fits each DC's
    margins. Both emit symmetric pairs with shared quantization semantics.
    """
    traffic = config.traffic
    if not traffic.enabled:
        return {}
    _validate_flow_policy_names(traffic.flow_policy_config)
    if traffic.model not in {"uniform", "gravity", "hose"}:
        raise ValueError(f"Unknown traffic model: {traffic.model}")
    dcs = _inventory(metros, metro_settings, traffic)
    if not dcs:
        return {}
    offered = sum(dc.mass for dc in dcs) * traffic.gbps_per_mw
    if not math.isfinite(offered) or offered < 0:
        raise ValueError("Offered traffic must be finite and non-negative")
    name = traffic.matrix_name
    if traffic.model == "uniform":
        demands = []
        for priority, ratio in sorted(traffic.priority_ratios.items()):
            if (volume := offered * ratio) <= 0:
                continue
            entry = {
                "source_path": "(metro[0-9]+/dc[0-9]+)",
                "sink_path": "(metro[0-9]+/dc[0-9]+)",
                "mode": "pairwise",
                "group_mode": "group_pairwise",
                "priority": priority,
                "demand": volume,
            }
            if priority in traffic.flow_policy_config:
                entry["flow_policy_config"] = traffic.flow_policy_config[priority]
            demands.append(entry)
        result = {name: demands}
    elif traffic.model == "gravity":
        distances = _distances(dcs, metros, traffic.gravity.min_distance_km)
        pairs = _gravity_pairs(dcs, distances, traffic, offered)
        result = {
            name: _emit_pairs(
                dcs,
                pairs,
                distances,
                traffic,
                offered,
                rng=random.Random(config.output.scenario_seed),
            )
        }
    else:
        if offered == 0:
            return {}
        distances = _distances(dcs, metros, traffic.hose.min_distance_km)
        result = {}
        for sample in range(1, traffic.samples + 1):
            rng = random.Random(config.output.scenario_seed * 1000003 + sample)
            pairs = _hose_pairs(_random_hose(dcs, distances, traffic, rng))
            if traffic.hose.carve_top_k is not None:
                keep = set(_top_pairs(pairs, traffic.hose.carve_top_k))
                pairs = _hose_pairs(_random_hose(dcs, distances, traffic, rng, keep))
            matrix_name = name if traffic.samples == 1 else f"{name}_{sample}"
            result[matrix_name] = _emit_pairs(dcs, pairs, distances, traffic, offered)
    logger.info(
        "Generated %s traffic: %d DCs, %d matrices, %d demands",
        traffic.model,
        len(dcs),
        len(result),
        sum(map(len, result.values())),
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Traffic matrices:\n%s", json.dumps(result, indent=2))
    return result
