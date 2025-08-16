"""Traffic matrix generation algorithms.

Provides functions to generate DC-to-DC traffic matrices for scenarios.

The primary entrypoint is ``generate_traffic_matrix`` which implements three
models:

- "uniform": emit a class-level pairwise demand across all DC nodes using
  regex paths.
- "gravity": compute per-pair allocations proportional to a gravity-like
  kernel over Euclidean metro distances in kilometers with optional jitter,
  top-K pruning, and rounding with conservation via largest remainders.
- "hose": sample one or more randomized matrices that satisfy per-DC ingress
  and egress totals via iterative proportional fitting (IPF), then emit
  symmetric directed demands split equally per pair.

Time complexity is dominated by pair enumeration between DC nodes which is
O(N_dc^2). Memory usage is O(N_dc^2) for intermediate weights in the gravity
model.
"""

from __future__ import annotations

import json
import math
import random as _random
from pathlib import Path
from typing import Any

from topogen.config import TopologyConfig
from topogen.log_config import get_logger

logger = get_logger(__name__)


def _fmt(value: float, *, decimals: int = 2) -> str:
    """Return a string with thousands separators for logging.

    Args:
        value: Number to format.
        decimals: Number of fractional digits.

    Returns:
        Formatted number as a string with thousands separators.
    """

    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(float(value))


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
        logger.debug("Traffic generation disabled or no traffic configuration present")
        return {}

    # Build DC inventory
    dc_nodes: list[tuple[str, int]] = []  # (metro_name, dc_index)
    for metro_name, settings in metro_settings.items():
        dc_count = int(settings.get("dc_regions_per_metro", 0))
        for dc_idx in range(1, dc_count + 1):
            dc_nodes.append((metro_name, dc_idx))

    if not dc_nodes:
        logger.debug("No DC regions configured; skipping traffic matrix generation")
        return {}

    # Offered traffic in Gbps
    model = str(getattr(traffic_cfg, "model", "uniform")).strip()

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

    # Summary of inputs
    try:
        logger.debug(
            "Traffic model=%s matrix_name=%s gbps_per_mw=%s mw_per_dc_region=%s",
            getattr(traffic_cfg, "model", "uniform"),
            getattr(traffic_cfg, "matrix_name", "default"),
            _fmt(float(traffic_cfg.gbps_per_mw), decimals=3),
            _fmt(float(traffic_cfg.mw_per_dc_region), decimals=3),
        )
        logger.debug(
            "DC inventory (MW) total=%s: %s",
            _fmt(total_power_mw, decimals=3),
            ", ".join(
                f"{m}/dc{d}={_fmt(dc_mass[(m, d)], decimals=3)}"
                for (m, d) in sorted(dc_mass.keys())
            ),
        )
        logger.debug("Offered traffic (Gbps)=%s", _fmt(offered_gbps))
    except Exception:  # pragma: no cover - logging only
        # Guard against accidental formatting failures in debug path
        pass

    if model == "uniform":
        # Uniform model emission using regex selection across all DCs
        source_regex = "(metro[0-9]+/dc[0-9]+)"
        sink_regex = "(metro[0-9]+/dc[0-9]+)"
        demands: list[dict[str, Any]] = []
        for priority, ratio in sorted(traffic_cfg.priority_ratios.items()):
            class_demand = offered_gbps * float(ratio)
            if class_demand <= 0.0:
                # Skip zero or negative class demand entries
                continue
            entry = {
                "source_path": source_regex,
                "sink_path": sink_regex,
                "mode": "pairwise",
                "priority": int(priority),
                "demand": float(class_demand),
            }
            # Optional per-priority flow policy config passthrough
            try:
                fpc = getattr(traffic_cfg, "flow_policy_config", {})
                if isinstance(fpc, dict) and int(priority) in fpc:
                    entry["flow_policy_config"] = str(fpc[int(priority)])
            except Exception:  # pragma: no cover - logging only
                pass
            demands.append(entry)
        # Debug summary for uniform model
        try:
            for priority, ratio in sorted(traffic_cfg.priority_ratios.items()):
                logger.debug(
                    "uniform: class priority=%s ratio=%s class_demand_gbps=%s",
                    int(priority),
                    _fmt(float(ratio), decimals=6),
                    _fmt(offered_gbps * float(ratio)),
                )
            pretty = {
                str(traffic_cfg.matrix_name): [
                    {
                        "source_path": d["source_path"],
                        "sink_path": d["sink_path"],
                        "mode": d["mode"],
                        "priority": int(d["priority"]),
                        "demand_gbps": _fmt(float(d["demand"]))
                        if isinstance(d.get("demand"), (int, float))
                        else str(d.get("demand")),
                    }
                    for d in demands
                ]
            }
            logger.debug(
                "traffic_matrix pretty:\n%s",
                json.dumps(pretty, indent=2, ensure_ascii=True),
            )
        except Exception:  # pragma: no cover - logging only
            pass
        return {traffic_cfg.matrix_name: demands}

    if model == "hose":
        # Hose model: sample one or more randomized matrices satisfying per-DC totals
        if total_power_mw <= 0.0:
            logger.debug("hose: total_power_mw is zero; skipping traffic generation")
            return {}

        # Directed per-DC totals (egress == ingress for symmetric hose)
        T: dict[tuple[str, int], float] = {
            k: offered_gbps * (float(mw) / float(total_power_mw))
            for k, mw in dc_mass.items()
        }

        # Map metro name to 1-based index for regex paths
        metro_idx_map_hose = {m["name"]: idx for idx, m in enumerate(metros, 1)}

        # Hose tilt configuration (gravity-tilted initialization)
        hcfg = getattr(traffic_cfg, "hose", object())
        try:
            tilt_exp = float(getattr(hcfg, "tilt_exponent", 0.0))
            hose_beta = float(getattr(hcfg, "beta", 1.0))
            hose_min_km = float(getattr(hcfg, "min_distance_km", 1.0))
            hose_excl_same = bool(getattr(hcfg, "exclude_same_metro", False))
            carve_top_k = getattr(hcfg, "carve_top_k", None)
        except Exception:
            tilt_exp = 0.0
            hose_beta = 1.0
            hose_min_km = 1.0
            hose_excl_same = False
            carve_top_k = None

        # Coordinates for Euclidean distance (meters)
        coords_hose: dict[str, tuple[float, float]] = {
            m["name"]: (float(m.get("x", 0.0)), float(m.get("y", 0.0))) for m in metros
        }

        def _hose_distance_km(name_a: str, name_b: str) -> float:
            if name_a == name_b:
                return max(hose_min_km, 0.0)
            (x1, y1) = coords_hose.get(name_a, (0.0, 0.0))
            (x2, y2) = coords_hose.get(name_b, (0.0, 0.0))
            return max(math.hypot(x1 - x2, y1 - y2) / 1000.0, hose_min_km)

        # Seeded RNG for reproducibility; vary with sample index
        try:
            base_seed = int(
                getattr(getattr(config, "output", object()), "scenario_seed", 42)
            )
        except Exception:
            base_seed = 42

        def _ipf_directed_matrix(rng: _random.Random) -> list[list[float]]:
            """Return directed matrix D with zero diagonal and row/col sums == T_i.

            Uses iterative proportional fitting on a positive random initialization.
            """

            n = len(dc_nodes)
            t_vec = [T[dc_nodes[i]] for i in range(n)]
            D = [[0.0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                (mi, di) = dc_nodes[i]
                for j in range(n):
                    if i == j:
                        continue
                    (mj, dj) = dc_nodes[j]
                    # Random base > 0
                    base = 1e-9 + rng.random()
                    # Optional gravity tilt on unordered pair
                    if tilt_exp > 0.0:
                        if hose_excl_same and mi == mj:
                            weight = 0.0
                        else:
                            dist_km = _hose_distance_km(mi, mj)
                            weight = 1.0 / (dist_km**hose_beta) if dist_km > 0 else 0.0
                        if weight <= 0.0:
                            # Keep strictly positive but very small to retain feasibility
                            tilt = 1e-12
                        else:
                            tilt = weight**tilt_exp
                        D[i][j] = base * tilt
                    else:
                        D[i][j] = base

            max_iter = 2000
            tol = 1e-6
            for _ in range(max_iter):
                # Scale rows
                for i in range(n):
                    row_sum = sum(D[i][j] for j in range(n) if j != i)
                    if row_sum > 0.0:
                        factor = t_vec[i] / row_sum
                        if factor != 1.0:
                            for j in range(n):
                                if i != j:
                                    D[i][j] *= factor
                # Scale columns and track max deviation
                max_dev = 0.0
                for j in range(n):
                    col_sum = sum(D[i][j] for i in range(n) if i != j)
                    if col_sum > 0.0:
                        factor = t_vec[j] / col_sum
                        if factor != 1.0:
                            for i in range(n):
                                if i != j:
                                    D[i][j] *= factor
                    col_sum2 = sum(D[i][j] for i in range(n) if i != j)
                    max_dev = max(max_dev, abs(col_sum2 - t_vec[j]))
                for i in range(n):
                    row_sum2 = sum(D[i][j] for j in range(n) if j != i)
                    max_dev = max(max_dev, abs(row_sum2 - t_vec[i]))
                if max_dev <= tol:
                    break
            return D

        num_samples = int(getattr(traffic_cfg, "samples", 1))
        base_name = str(getattr(traffic_cfg, "matrix_name", "default"))
        # Reuse gravity rounding settings for undirected totals, then split equally
        rounding = float(
            getattr(getattr(traffic_cfg, "gravity", object()), "rounding_gbps", 0.0)
        )
        rounding_policy = str(
            getattr(
                getattr(traffic_cfg, "gravity", object()), "rounding_policy", "nearest"
            )
        )

        result_hose: dict[str, list[dict[str, Any]]] = {}
        for s_idx in range(1, num_samples + 1):
            rng = _random.Random(base_seed * 1000003 + s_idx)
            D = _ipf_directed_matrix(rng)
            # Optional gravity carve step: keep top-K partners per DC by undirected total
            if carve_top_k is not None:
                k = int(carve_top_k)
                n = len(dc_nodes)
                # Build undirected totals
                und: dict[tuple[int, int], float] = {}
                for i in range(n):
                    for j in range(i + 1, n):
                        u = float(D[i][j] + D[j][i])
                        if u > 0.0:
                            und[(i, j)] = u
                # Partners per node
                partner_map: dict[int, list[tuple[int, float]]] = {}
                for (i, j), u in und.items():
                    partner_map.setdefault(i, []).append((j, u))
                    partner_map.setdefault(j, []).append((i, u))
                # Build keep set
                keep_pairs: set[tuple[int, int]] = set()
                for i, lst in partner_map.items():
                    lst_sorted = sorted(lst, key=lambda t: t[1], reverse=True)[:k]
                    for j, _ in lst_sorted:
                        a, b = (i, j) if i < j else (j, i)
                        keep_pairs.add((a, b))
                # Zero out non-kept pairs in the initialization and rerun IPF to restore T_i
                # Start from a fresh initialization using the same tilt to avoid drift
                D = [[0.0 for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    (mi, di) = dc_nodes[i]
                    for j in range(n):
                        if i == j:
                            continue
                        (mj, dj) = dc_nodes[j]
                        a, b = (i, j) if i < j else (j, i)
                        if (a, b) not in keep_pairs:
                            continue
                        base = 1e-9 + rng.random()
                        if tilt_exp > 0.0:
                            if hose_excl_same and mi == mj:
                                weight = 0.0
                            else:
                                dist_km = _hose_distance_km(mi, mj)
                                weight = (
                                    1.0 / (dist_km**hose_beta) if dist_km > 0 else 0.0
                                )
                            tilt = (weight**tilt_exp) if weight > 0.0 else 1e-12
                            D[i][j] = base * tilt
                        else:
                            D[i][j] = base
                # Run IPF again to satisfy row/col sums
                # Reuse inner scaling loop
                max_iter = 2000
                tol = 1e-6
                for _ in range(max_iter):
                    for i in range(n):
                        row_sum = sum(D[i][j] for j in range(n) if j != i)
                        if row_sum > 0.0:
                            factor = T[dc_nodes[i]] / row_sum
                            if factor != 1.0:
                                for j in range(n):
                                    if i != j:
                                        D[i][j] *= factor
                    max_dev = 0.0
                    for j in range(n):
                        col_sum = sum(D[i][j] for i in range(n) if i != j)
                        if col_sum > 0.0:
                            factor = T[dc_nodes[j]] / col_sum
                            if factor != 1.0:
                                for i in range(n):
                                    if i != j:
                                        D[i][j] *= factor
                        col_sum2 = sum(D[i][j] for i in range(n) if i != j)
                        max_dev = max(max_dev, abs(col_sum2 - T[dc_nodes[j]]))
                    for i in range(n):
                        row_sum2 = sum(D[i][j] for j in range(n) if j != i)
                        max_dev = max(max_dev, abs(row_sum2 - T[dc_nodes[i]]))
                    if max_dev <= tol:
                        break
            # Build undirected totals for i<j
            n = len(dc_nodes)
            undirected: list[tuple[tuple[int, int], float]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    total_ij = float(D[i][j] + D[j][i])
                    if total_ij > 0.0:
                        undirected.append(((i, j), total_ij))

            demands: list[dict[str, Any]] = []
            # Emit per class
            for priority, ratio in sorted(traffic_cfg.priority_ratios.items()):
                D_c = offered_gbps * float(ratio)
                allocs_hose = [((i, j), u * float(ratio)) for ((i, j), u) in undirected]

                # Optional rounding with conservation via largest remainders (undirected)
                if rounding > 0.0:
                    floored_hose: list[tuple[tuple[int, int], float, float]] = []
                    total_floor = 0.0
                    for (i, j), v in allocs_hose:
                        if rounding_policy == "ceil":
                            q = math.ceil(v / rounding) * rounding
                        elif rounding_policy == "floor":
                            q = math.floor(v / rounding) * rounding
                        else:
                            q = round(v / rounding) * rounding
                        rem = v - q
                        floored_hose.append(((i, j), q, rem))
                        total_floor += q
                    remainder = D_c - total_floor
                    steps = int(round(remainder / rounding)) if rounding > 0 else 0
                    floored_hose.sort(key=lambda t: t[2], reverse=True)
                    final_map_hose: dict[tuple[int, int], float] = {
                        key: q for key, q, _ in floored_hose
                    }
                    idx = 0
                    while steps > 0 and idx < len(floored_hose):
                        key, q, _ = floored_hose[idx]
                        final_map_hose[key] = q + rounding
                        steps -= 1
                        idx += 1
                    allocs_hose = list(final_map_hose.items())

                dir_step = max(rounding / 2.0, 0.0)
                for (i, j), v in allocs_hose:
                    (m1, d1) = dc_nodes[i]
                    (m2, d2) = dc_nodes[j]
                    i1 = metro_idx_map_hose[m1]
                    i2 = metro_idx_map_hose[m2]
                    src = f"^metro{i1}/dc{d1}/.*"
                    dst = f"^metro{i2}/dc{d2}/.*"
                    # Euclidean km for attrs parity with gravity model
                    dist_km = _hose_distance_km(m1, m2)
                    demand_each_raw = float(v) / 2.0
                    if dir_step > 0.0:
                        demand_each = round(demand_each_raw / dir_step) * dir_step
                    else:
                        demand_each = round(demand_each_raw, 2)
                    if demand_each <= 0.0:
                        continue
                    entry_fwd = {
                        "source_path": src,
                        "sink_path": dst,
                        "mode": "pairwise",
                        "priority": int(priority),
                        "demand": float(demand_each),
                        "attrs": {"euclidean_km": int(math.ceil(float(dist_km)))},
                    }
                    entry_rev = {
                        "source_path": dst,
                        "sink_path": src,
                        "mode": "pairwise",
                        "priority": int(priority),
                        "demand": float(demand_each),
                        "attrs": {"euclidean_km": int(math.ceil(float(dist_km)))},
                    }
                    fpc = getattr(traffic_cfg, "flow_policy_config", {})
                    if isinstance(fpc, dict) and int(priority) in fpc:
                        entry_fwd["flow_policy_config"] = str(fpc[int(priority)])
                        entry_rev["flow_policy_config"] = str(fpc[int(priority)])
                    demands.append(entry_fwd)
                    demands.append(entry_rev)

            matrix_name = base_name if num_samples == 1 else f"{base_name}_{s_idx}"
            result_hose[matrix_name] = demands

        # Pretty-print the first sample for visibility
        try:
            first = next(iter(result_hose.keys())) if result_hose else base_name
            pretty = {
                str(first): [
                    {
                        "source_path": d["source_path"],
                        "sink_path": d["sink_path"],
                        "mode": d["mode"],
                        "priority": int(d["priority"]),
                        "demand_gbps": _fmt(float(d["demand"]))
                        if isinstance(d.get("demand"), (int, float))
                        else str(d.get("demand")),
                    }
                    for d in result_hose.get(first, [])
                ]
            }
            logger.debug(
                "traffic_matrix pretty (hose sample):\n%s",
                json.dumps(pretty, indent=2, ensure_ascii=True),
            )
        except Exception:  # pragma: no cover - logging only
            pass

        return result_hose

    # Early return for hose handled above; remaining branch is gravity
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
    logger.debug(
        "Distance model: Euclidean straight-line km (EPSG:5070 coords), min_distance_km=%s",
        _fmt(float(gcfg.min_distance_km), decimals=3),
    )
    logger.debug("traffic: diagnostics begin")
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

    # Log gravity parameters and a brief weight summary
    try:
        logger.debug(
            (
                "gravity: alpha=%s beta=%s min_distance_km=%s "
                "exclude_same_metro=%s jitter_stddev=%s rounding_gbps=%s "
                "directional_step_gbps=%s max_partners_per_dc=%s"
            ),
            _fmt(float(gcfg.alpha), decimals=6),
            _fmt(float(gcfg.beta), decimals=6),
            _fmt(float(gcfg.min_distance_km), decimals=3),
            bool(gcfg.exclude_same_metro),
            _fmt(float(gcfg.jitter_stddev), decimals=6),
            _fmt(float(gcfg.rounding_gbps), decimals=3),
            _fmt(float(gcfg.rounding_gbps) / 2.0, decimals=3),
            (
                None
                if gcfg.max_partners_per_dc is None
                else int(gcfg.max_partners_per_dc)
            ),
        )
        # Show a few top-weight pairs for orientation
        top_pairs = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for (m1, d1), (m2, d2) in [p[0] for p in top_pairs]:
            w = weights[((m1, d1), (m2, d2))]
            frac = w / total_w if total_w > 0 else 0.0
            logger.debug(
                "gravity: pair %s/dc%s <-> %s/dc%s weight=%s frac=%s",
                m1,
                d1,
                m2,
                d2,
                _fmt(w, decimals=6),
                _fmt(frac, decimals=6),
            )
    except Exception:  # pragma: no cover - logging only
        pass

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

        logger.debug(
            "gravity: applied top-K pruning with k=%d; remaining_pairs=%d",
            k,
            len(weights),
        )

    # Map metro name to 1-based index consistent with network group naming
    metro_idx_map = {m["name"]: idx for idx, m in enumerate(metros, 1)}
    try:
        logger.debug(
            "regex index map: %s",
            ", ".join(
                f"metro{idx} -> {name}"
                for name, idx in sorted(metro_idx_map.items(), key=lambda x: x[1])
            ),
        )
    except Exception:  # pragma: no cover - logging only
        pass

    # Emit explicit per-pair demands; apply jitter and rounding per class with conservation
    demands: list[dict[str, Any]] = []
    debug_entries: list[dict[str, Any]] = []
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

        logger.debug(
            "gravity: class priority=%d ratio=%s class_total_gbps=%s",
            int(priority),
            _fmt(float(ratio), decimals=6),
            _fmt(D_c),
        )

        # Apply rounding if requested and repair to conserve totals via largest remainders
        rounding = float(gcfg.rounding_gbps)
        if rounding > 0.0:
            floored: list[
                tuple[tuple[tuple[str, int], tuple[str, int]], float, float]
            ] = []
            total_floor = 0.0
            for pair, v in allocs:
                # Quantize according to policy
                if getattr(gcfg, "rounding_policy", "nearest") == "ceil":
                    q = math.ceil(v / rounding) * rounding
                elif getattr(gcfg, "rounding_policy", "nearest") == "floor":
                    q = math.floor(v / rounding) * rounding
                else:
                    # nearest
                    q = round(v / rounding) * rounding
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
            try:
                sum_final = sum(final_map.values())
                delta = D_c - sum_final
                logger.debug(
                    (
                        "gravity: class priority=%d rounding delta_gbps=%s "
                        "(exact-total - rounded-total) policy=%s"
                    ),
                    int(priority),
                    _fmt(delta),
                    getattr(gcfg, "rounding_policy", "nearest"),
                )
            except Exception:  # pragma: no cover - logging only
                pass
            logger.debug(
                "gravity: class priority=%d applied rounding_gbps=%s",
                int(priority),
                _fmt(rounding, decimals=3),
            )

        # Emit entries for each pair in both directions using explicit paths
        class_directed_sum = 0.0
        dir_step = max(float(gcfg.rounding_gbps) / 2.0, 0.0)
        for pair, v in allocs:
            (m1, d1), (m2, d2) = pair
            i1 = metro_idx_map[m1]
            i2 = metro_idx_map[m2]
            src = f"^metro{i1}/dc{d1}/.*"
            dst = f"^metro{i2}/dc{d2}/.*"
            # Symmetric split: half each direction; align to configured rounding step per direction
            demand_each_raw = float(v) / 2.0
            if dir_step > 0.0:
                demand_each = round(demand_each_raw / dir_step) * dir_step
            else:
                demand_each = round(demand_each_raw, 2)
            if demand_each <= 0.0:
                continue
            # Precompute Euclidean distance in km for attrs and logging
            dist_km = _distance_km(m1, m2)
            # Pair-specific debug lines (both directions) with key parameters
            try:
                w = weights[pair]
                frac = w / total_w if total_w > 0 else 0.0
                m_i = dc_mass[(m1, d1)]
                m_j = dc_mass[(m2, d2)]
                # Forward
                logger.debug(
                    (
                        "entry: prio=%d src=%s dst=%s demand_gbps=%s "
                        "m_i_MW=%s m_j_MW=%s euclidean_km=%s weight=%s frac=%s "
                        "alpha=%s beta=%s class_total_gbps=%s rounding_gbps=%s "
                        "jitter_stddev=%s"
                    ),
                    int(priority),
                    src,
                    dst,
                    _fmt(demand_each),
                    _fmt(m_i, decimals=3),
                    _fmt(m_j, decimals=3),
                    _fmt(dist_km, decimals=3),
                    _fmt(w, decimals=6),
                    _fmt(frac, decimals=6),
                    _fmt(float(gcfg.alpha), decimals=6),
                    _fmt(float(gcfg.beta), decimals=6),
                    _fmt(D_c),
                    _fmt(float(gcfg.rounding_gbps), decimals=3),
                    _fmt(float(gcfg.jitter_stddev), decimals=6),
                )
                debug_entries.append(
                    {
                        "priority": int(priority),
                        "source_path": src,
                        "sink_path": dst,
                        "demand_gbps": float(demand_each),
                        "m_i_MW": float(m_i),
                        "m_j_MW": float(m_j),
                        "euclidean_km": float(dist_km),
                        "weight": float(w),
                        "weight_fraction": float(frac),
                        "alpha": float(gcfg.alpha),
                        "beta": float(gcfg.beta),
                        "class_total_gbps": float(D_c),
                        "rounding_gbps": float(gcfg.rounding_gbps),
                        "jitter_stddev": float(gcfg.jitter_stddev),
                    }
                )
                # Reverse
                logger.debug(
                    (
                        "entry: prio=%d src=%s dst=%s demand_gbps=%s "
                        "m_i_MW=%s m_j_MW=%s euclidean_km=%s weight=%s frac=%s "
                        "alpha=%s beta=%s class_total_gbps=%s rounding_gbps=%s "
                        "jitter_stddev=%s"
                    ),
                    int(priority),
                    dst,
                    src,
                    _fmt(demand_each),
                    _fmt(m_j, decimals=3),
                    _fmt(m_i, decimals=3),
                    _fmt(dist_km, decimals=3),
                    _fmt(w, decimals=6),
                    _fmt(frac, decimals=6),
                    _fmt(float(gcfg.alpha), decimals=6),
                    _fmt(float(gcfg.beta), decimals=6),
                    _fmt(D_c),
                    _fmt(float(gcfg.rounding_gbps), decimals=3),
                    _fmt(float(gcfg.jitter_stddev), decimals=6),
                )
                debug_entries.append(
                    {
                        "priority": int(priority),
                        "source_path": dst,
                        "sink_path": src,
                        "demand_gbps": float(demand_each),
                        "m_i_MW": float(m_j),
                        "m_j_MW": float(m_i),
                        "euclidean_km": float(dist_km),
                        "weight": float(w),
                        "weight_fraction": float(frac),
                        "alpha": float(gcfg.alpha),
                        "beta": float(gcfg.beta),
                        "class_total_gbps": float(D_c),
                        "rounding_gbps": float(gcfg.rounding_gbps),
                        "jitter_stddev": float(gcfg.jitter_stddev),
                    }
                )
            except Exception:  # pragma: no cover - logging only
                pass
            demands.append(
                {
                    "source_path": src,
                    "sink_path": dst,
                    "mode": "pairwise",
                    "priority": int(priority),
                    "demand": demand_each,
                    "attrs": {"euclidean_km": int(math.ceil(float(dist_km)))},
                }
            )
            # Optional per-priority flow policy config passthrough
            if isinstance(getattr(traffic_cfg, "flow_policy_config", {}), dict):
                fpc = getattr(traffic_cfg, "flow_policy_config", {})
                if int(priority) in fpc:
                    demands[-1]["flow_policy_config"] = str(fpc[int(priority)])
            demands.append(
                {
                    "source_path": dst,
                    "sink_path": src,
                    "mode": "pairwise",
                    "priority": int(priority),
                    "demand": demand_each,
                    "attrs": {"euclidean_km": int(math.ceil(float(dist_km)))},
                }
            )
            if isinstance(getattr(traffic_cfg, "flow_policy_config", {}), dict):
                fpc = getattr(traffic_cfg, "flow_policy_config", {})
                if int(priority) in fpc:
                    demands[-1]["flow_policy_config"] = str(fpc[int(priority)])
            # Track post-split rounded directed sum for delta diagnostics
            class_directed_sum += 2.0 * demand_each

        # Post-split rounding delta relative to class exact total (using directional rounding quanta)
        try:
            delta_dir = D_c - class_directed_sum
            logger.debug(
                "gravity: class priority=%d post-split rounding delta_gbps=%s (exact-total - sum(directed))",
                int(priority),
                _fmt(delta_dir),
            )
        except Exception:  # pragma: no cover - logging only
            pass

    result = {traffic_cfg.matrix_name: demands}
    try:
        # Summary counts and distribution statistics
        logger.debug(
            "traffic: counts dc_regions=%d undirected_pairs=%d directed_entries=%d",
            len(dc_nodes),
            len(weights),
            len(demands),
        )
        values = [float(d.get("demand", 0.0)) for d in demands]
        if values:
            svals = sorted(values)
            n = len(svals)
            total = sum(svals)
            mean = total / n
            median = (
                svals[n // 2]
                if n % 2 == 1
                else 0.5 * (svals[n // 2 - 1] + svals[n // 2])
            )
            logger.debug(
                "traffic: stats min=%s median=%s mean=%s max=%s",
                _fmt(svals[0]),
                _fmt(median),
                _fmt(mean),
                _fmt(svals[-1]),
            )
            by_class: dict[int, list[float]] = {}
            for d in demands:
                by_class.setdefault(int(d["priority"]), []).append(float(d["demand"]))
            for prio in sorted(by_class):
                lst = sorted(by_class[prio])
                n_c = len(lst)
                tot_c = sum(lst)
                mean_c = tot_c / n_c
                med_c = (
                    lst[n_c // 2]
                    if n_c % 2 == 1
                    else 0.5 * (lst[n_c // 2 - 1] + lst[n_c // 2])
                )
                logger.debug(
                    "class %d: entries=%d total=%s min=%s median=%s mean=%s max=%s",
                    prio,
                    n_c,
                    _fmt(tot_c),
                    _fmt(lst[0]),
                    _fmt(med_c),
                    _fmt(mean_c),
                    _fmt(lst[-1]),
                )
        # Optional export of detailed entries and weights
        debug_dir = getattr(config, "_debug_dir", None)
        if debug_dir is not None:
            export: dict[str, Any] = {
                "model": getattr(traffic_cfg, "model", "uniform"),
                "matrix_name": getattr(traffic_cfg, "matrix_name", "default"),
                "gbps_per_mw": float(traffic_cfg.gbps_per_mw),
                "mw_per_dc_region": float(traffic_cfg.mw_per_dc_region),
                "offered_gbps": float(offered_gbps),
                "gravity": {
                    "alpha": float(gcfg.alpha),
                    "beta": float(gcfg.beta),
                    "min_distance_km": float(gcfg.min_distance_km),
                    "exclude_same_metro": bool(gcfg.exclude_same_metro),
                    "jitter_stddev": float(gcfg.jitter_stddev),
                    "rounding_gbps": float(gcfg.rounding_gbps),
                    "max_partners_per_dc": (
                        None
                        if gcfg.max_partners_per_dc is None
                        else int(gcfg.max_partners_per_dc)
                    ),
                },
                "dc_masses_MW": {
                    f"{m}/dc{d}": float(dc_mass[(m, d)]) for (m, d) in sorted(dc_mass)
                },
                "metro_index_map": {
                    f"metro{idx}": name for name, idx in metro_idx_map.items()
                },
                "weights": {
                    f"{a[0]}/dc{a[1]} <-> {b[0]}/dc{b[1]}": float(w)
                    for (a, b), w in weights.items()
                },
                "entries": debug_entries,
            }
            out_dir = Path(str(debug_dir))
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = getattr(config, "_source_stem", None)
            if stem is None:
                src = getattr(config, "_source_path", None)
                if isinstance(src, (str, Path)):
                    stem = Path(src).stem
                else:
                    stem = "scenario"
            out_path = out_dir / f"{stem}_traffic_debug.json"
            try:
                out_path.write_text(json.dumps(export, indent=2))
                logger.debug("traffic: wrote debug JSON to %s", str(out_path))
            except Exception:
                logger.debug("traffic: failed to write debug JSON to %s", str(out_path))
    except Exception:  # pragma: no cover - logging only
        pass
    # Pretty-print the final matrix for visibility in verbose mode (single emission)
    try:
        pretty = {
            str(traffic_cfg.matrix_name): [
                {
                    "source_path": d["source_path"],
                    "sink_path": d["sink_path"],
                    "mode": d["mode"],
                    "priority": int(d["priority"]),
                    "demand_gbps": _fmt(float(d["demand"]))
                    if isinstance(d.get("demand"), (int, float))
                    else str(d.get("demand")),
                }
                for d in demands
            ]
        }
        logger.debug(
            "traffic_matrix pretty:\n%s",
            json.dumps(pretty, indent=2, ensure_ascii=True),
        )
    except Exception:  # pragma: no cover - logging only
        pass

    logger.debug("traffic: diagnostics end")

    return result
