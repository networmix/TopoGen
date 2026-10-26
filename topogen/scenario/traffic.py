"""Generate deterministic uniform, gravity and hose demand sets.

Gravity and hose share pair selection, rounding and symmetric emission. Their
pair inventories are quadratic in DC count; uniform emits ordered distinct site pairs.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass

import numpy as np

from topogen.config import TrafficConfig
from topogen.log_config import get_logger
from topogen.numeric import decimal

from .records import SiteDemand, SitePlan, TrafficMatrix

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
    mass: int | float


def _inventory(plan: SitePlan) -> list[_DC]:
    result = []
    for site in plan.sites:
        if site.kind == "dc":
            assert site.power_mw is not None
            result.append(_DC(site.metro.name, site.id, site.power_mw))
    return result


def _distances(plan: SitePlan, minimum: float) -> np.ndarray:
    xy = np.array(
        [(site.metro.x, site.metro.y) for site in plan.sites if site.kind == "dc"],
        dtype=float,
    )
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
) -> tuple[SiteDemand, ...]:
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
                demands.append(
                    SiteDemand(
                        dcs[a].path,
                        dcs[b].path,
                        priority,
                        decimal(each),
                        config.flow_policy_config.get(priority),
                        distance,
                    )
                )
    return tuple(demands)


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


def generate_traffic(
    plan: SitePlan,
    traffic: TrafficConfig,
    seed: int,
) -> tuple[TrafficMatrix, ...]:
    """Generate site demands from metro centers, without I/O or global PRNG state."""
    if not traffic.enabled:
        return ()
    if traffic.model not in {"uniform", "gravity", "hose"}:
        raise ValueError(f"Unknown traffic model: {traffic.model}")
    dcs = _inventory(plan)
    if not dcs:
        return ()
    if traffic.model == "uniform":
        offered_exact = sum(decimal(dc.mass) for dc in dcs) * decimal(
            traffic.gbps_per_mw
        )
        try:
            finite = math.isfinite(offered_exact)
        except OverflowError:
            finite = False
        if not finite or offered_exact < 0:
            raise ValueError("Offered traffic must be finite and non-negative")
        n = len(dcs)
        if offered_exact > 0 and n < 2:
            raise ValueError(
                f"uniform traffic requires at least two DC sites: sites={n}, volume={float(offered_exact)!r}"
            )
        demands = tuple(
            SiteDemand(
                a.path,
                b.path,
                priority,
                offered_exact * decimal(ratio) / (n * (n - 1)),
                traffic.flow_policy_config.get(priority),
            )
            for priority, ratio in sorted(traffic.priority_ratios.items())
            if offered_exact > 0 and ratio > 0
            for a in dcs
            for b in dcs
            if a.path != b.path
        )
        return (TrafficMatrix(traffic.matrix_name, demands),)
    dcs = [_DC(dc.metro, dc.path, float(dc.mass)) for dc in dcs]
    offered = sum(dc.mass for dc in dcs) * traffic.gbps_per_mw
    if not math.isfinite(offered) or offered < 0:
        raise ValueError("Offered traffic must be finite and non-negative")
    if traffic.model == "gravity":
        distances = _distances(plan, traffic.gravity.min_distance_km)
        pairs = _gravity_pairs(dcs, distances, traffic, offered)
        return (
            TrafficMatrix(
                traffic.matrix_name,
                _emit_pairs(
                    dcs, pairs, distances, traffic, offered, rng=random.Random(seed)
                ),
            ),
        )
    if offered == 0:
        return ()
    distances = _distances(plan, traffic.hose.min_distance_km)
    matrices = []
    for sample in range(1, traffic.samples + 1):
        rng = random.Random(seed * 1000003 + sample)
        pairs = _hose_pairs(_random_hose(dcs, distances, traffic, rng))
        if traffic.hose.carve_top_k is not None:
            keep = set(_top_pairs(pairs, traffic.hose.carve_top_k))
            pairs = _hose_pairs(_random_hose(dcs, distances, traffic, rng, keep))
        name = (
            traffic.matrix_name
            if traffic.samples == 1
            else f"{traffic.matrix_name}_{sample}"
        )
        matrices.append(
            TrafficMatrix(name, _emit_pairs(dcs, pairs, distances, traffic, offered))
        )
    return tuple(matrices)
