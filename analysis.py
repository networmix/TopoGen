#!/usr/bin/env python3
"""
analysis.py — Orchestrator for metric computation and reporting.

What this script does (in plain words)
- It reads all scenario results, one seed at a time.
- It validates that each seed has a clear “no‑failure” baseline.
- It computes a small set of interpretable, failure‑aware metrics per seed.
- It summarizes those metrics across seeds (per scenario) and across scenarios (project‑level).
- It also prints a baseline‑normalized view so you can say “this topology is 1.2× better than baseline.”

Architecture
- Discovery: find all *.results.json under a root, group them by scenario and seed.
- Validation: require steps 'msd_baseline' and 'tm_placement'; optionally 'node_to_node_capacity_matrix'
  (enabled via --enable-maxflow). Enforce that each required step has metadata.baseline==true and that
  the first iteration in flow_results is the explicit baseline (failure_id=='baseline').
- Per-seed metrics (computed independently for each seed):
  1) alpha_star (Maximum Supported Demand): from msd_baseline.
  2) BAC (tm_placement): sum delivered bandwidth per iteration; normalize to the step's baseline delivered
     (no-failure); compute BW@p (p50, p90, p99, p99.9, p99.99) and AUC (mean of min(delivered/baseline, 1)).
  3) Latency stretch (tm_placement only): derive per-pair best-path costs from tm_placement baseline
     cost_distribution; per iteration, compute volume-weighted tails (p50, p95, p99); summarize as:
       - lat_base_p50 (baseline p50)
       - lat_fail_p99 (median p99 across failure iterations)
       - lat_TD99 = p99_fail / p99_base
       - lat_SLO_1_2_drop = drop in P(stretch ≤ 1.2) baseline→failures
       - lat_best_path_drop = drop in best-path share baseline→failures
       - lat_WES_delta = change in weighted excess stretch E[(stretch - 1)+]
  4) Cost/Power: totals from cost_power; normalize to offered@α=1 (sum of base_demands) and to reliable
     bandwidth at p99.9 from BAC; report USD/Gb_offered, W/Gb_offered, USD/Gb_p99.9, W/Gb_p99.9.
  5) Optional MaxFlow (when enabled): compute structural survivability (SPS) and write per-pair percentile
     matrices for tm_placement (always) and MaxFlow (if enabled) to CSV for debugging.

Architecture (step by step)
- Discovery
  - Find *.results.json under the chosen root.
  - Group files by scenario name and seed.
- Validation
  - Require steps: 'msd_baseline' and 'tm_placement' (always), 'node_to_node_capacity_matrix' (if --enable-maxflow).
  - Enforce: metadata.baseline == true and flow_results[0].failure_id == 'baseline' for required steps.
- Per‑seed metrics (computed independently for every seed)
  - alpha_star (Maximum Supported Demand)
    - From msd_baseline; measures how much we can scale base demands.
  - BAC (Bandwidth–Availability Curve, tm_placement)
    - For each failure iteration: sum delivered bandwidth; normalize by the step’s baseline delivered.
    - Report BW@p (p50, p90, p99, p99.9, p99.99) and AUC (mean of min(delivered/baseline,1)).
  - Latency stretch (tm_placement only)
    - Build per‑pair best‑path baseline from tm_placement baseline cost_distribution.
    - For each iteration: compute volume‑weighted tails (p50, p95, p99) of stretch = path_cost / best_path_cost.
    - Summaries we keep:
      * lat_base_p50 (baseline p50)
      * lat_fail_p99 (median p99 across failures)
      * lat_TD99 = p99_fail / p99_base
      * lat_SLO_1_2_drop = drop in P(stretch ≤ 1.2) baseline→failures
      * lat_best_path_drop = drop in best‑path share baseline→failures
      * lat_WES_delta = change in E[(stretch − 1)+]
  - Cost / Power
    - Read totals from cost_power.
    - Normalize by offered@α=1 (sum base_demands) and by reliable_p999 (BAC absolute p99.9 delivered):
      * USD/Gb_offered, W/Gb_offered
      * USD/Gb_p99.9,  W/Gb_p99.9
  - Optional MaxFlow (only if --enable-maxflow)
    - For diagnostics, compute structural survivability (SPS) and write per‑pair percentile matrices
      for tm_placement (always) and MaxFlow (if enabled).

- Cross‑seed per scenario
  - Write: alpha_summary.json; bac_summary.json (tails, BW@p, AUC);
    latency_summary.csv (baseline/failure fields and drops); costpower_summary.csv; provenance.json.

- Project‑level outputs
  - Print a consolidated table: seeds, alpha*, BW@99%, BW@99.9%, BAC AUC, latency fields,
    cost/power per Gbit (offered and p99.9), and CapEx.
  - Print and save a baseline‑normalized table: per‑seed scenario/baseline ratios (medians) and drops.

Final metrics (concise definitions)
- alpha_star: factor computed by msd_baseline indicating the maximum supported demand.
- BW@p: delivered bandwidth threshold met with probability p across iterations, normalized to tm_placement baseline
  delivered (no-failure). Implemented as the (1-p) lower quantile of delivered, divided by baseline delivered.
- BAC AUC: mean of min(delivered/baseline, 1.0) across failure iterations.
- lat_base_p50: baseline volume-weighted median stretch (≈1.0 in healthy state).
- lat_fail_p99: p99 stretch under failures, median across iterations.
- lat_TD99: ratio p99_fail / p99_base (tail amplification factor).
- lat_SLO_1_2_drop: decrease in P(stretch ≤ 1.2) from baseline to failures.
- lat_best_path_drop: decrease in volume share on best paths (stretch ≈ 1.0) from baseline to failures.
- lat_WES_delta: change in weighted excess stretch E[(stretch−1)+] baseline→failures.
- USD_per_Gbit_offered / Watt_per_Gbit_offered: totals divided by offered@α=1 (sum of base_demands).
- USD_per_Gbit_p999 / Watt_per_Gbit_p999: totals divided by reliable_p999 (BAC p99.9 absolute delivered).

Baselines
- All normalizations and latency baselines use the tm_placement baseline (same step and seed).
- MaxFlow is optional and used only for structural diagnostics and per‑pair matrix dumps.

### Simple view of how we compare scenarios
- Per-seed first
  - For each scenario and each seed, we compute the final metrics (availability BW@p, BAC AUC, latency tails and drops, cost/power per Gbit).

- Normalize to the baseline per seed
  - For every seed, we compare a scenario directly to the baseline scenario from the same seed:
    - Ratios (bigger is better): `BW@99%`, `BW@99.9%`, `BAC AUC`.
    - Ratios (smaller is better): `lat_fail_p99`, `USD/Gb offered`, `W/Gb offered`, `USD/Gb p99.9`, `W/Gb p99.9`.
    - Deltas (closer to 0 is better): `SLO≤1.2 drop`, `best-path drop`, `WES Δ`.
    - Passthrough (already a ratio vs its own baseline): `TD99`.

- Aggregate across seeds
  - We take the median of those per-seed comparisons to get one number per scenario.

- Two tables
  - Absolute table: raw medians (useful context).
  - Baseline-normalized table: scenario vs baseline (easy to read: “1.25× better” or “0.8× the cost”).

- One simple significance check
  - We run a single test on the normalized values per scenario:
    - Ratios vs 1.0, deltas vs 0.0.
  - Then show:
    - All comparisons.
    - A filtered list of big effects (e.g., ≥10% ratio change, ≥0.05 absolute delta) to highlight what matters.

How to read it
- Ratio > 1.0 is better for BW/AUC; < 1.0 is better for latency and cost/power.
- Drops/deltas closer to 0 are better.
- This cancels seed noise, is comparable across metrics, and is easy to interpret.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import re
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics.summary as summary_mod

# ---- Local modules ----
from metrics.aggregate import (
    summarize_across_seeds,
    write_csv_atomic,
    write_json_atomic,
)
from metrics.bac import BacResult, compute_bac, plot_bac
from metrics.costpower import (
    CostPowerResult,
    compute_cost_power,
    plot_cost_power,
)
from metrics.latency import LatencyResult, compute_latency_stretch, plot_latency
from metrics.matrixdump import compute_pair_matrices
from metrics.msd import AlphaResult, compute_alpha_star
from metrics.sps import SpsResult, compute_sps

# ---- Global plotting defaults ----
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (8.0, 5.0)
plt.rcParams["savefig.bbox"] = "tight"
# Consistent font sizes for publication-ready figures
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

# --------- discovery & grouping helpers ---------

SEED_STEM_RE = re.compile(r"^(?P<stem>.+)__seed(?P<seed>\d+)_scenario$")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_results_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.results.json"))


def parse_seeded_stem(stem: str) -> Tuple[str, Optional[int]]:
    """
    Input example: 'small_test_clos__seed11_scenario'
    Returns: ('small_test_clos', 11)
    """
    m = SEED_STEM_RE.match(stem)
    if not m:
        return stem, None
    return m.group("stem"), int(m.group("seed"))


def group_by_scenario(files: List[Path]) -> Dict[str, Dict[int, Path]]:
    """
    Returns: { scenario_stem: { seed: path } }
    """
    grouped: Dict[str, Dict[int, Path]] = {}
    for p in files:
        s = p.stem  # '..._scenario.results'
        if s.endswith(".results"):
            s = s[:-8]
        scenario_stem, seed = parse_seeded_stem(s)
        if seed is None:
            # Try extracting seed from file content as a fallback
            try:
                data = load_json(p)
                seed = int(data.get("scenario", {}).get("seed"))
            except Exception:
                continue
        grouped.setdefault(scenario_stem, {})[seed] = p
    return grouped


# --------- main orchestration ---------


@dataclass
class ScenarioOutputs:
    alpha: Dict[int, AlphaResult] = field(default_factory=dict)
    bac_place: Dict[int, BacResult] = field(default_factory=dict)
    bac_maxflow: Dict[int, BacResult] = field(default_factory=dict)
    latency: Dict[int, LatencyResult] = field(default_factory=dict)
    costpower: Dict[int, CostPowerResult] = field(default_factory=dict)
    sps: Dict[int, SpsResult] = field(default_factory=dict)


def analyze_one_seed(
    results: dict, out_dir: Path, do_plots: bool
) -> Tuple[
    AlphaResult,
    BacResult,
    Optional[BacResult],
    LatencyResult,
    CostPowerResult,
]:
    # Robust input validations: raise on clearly invalid inputs to avoid misleading metrics
    def _require_steps(res: dict, require_maxflow: bool) -> None:
        steps = res.get("steps", {})
        required = ["msd_baseline", "tm_placement"] + (
            ["node_to_node_capacity_matrix"] if require_maxflow else []
        )
        for step in required:
            if step not in steps:
                raise ValueError(f"Missing required step in results: {step}")

    def _validate_alpha_and_base_demands(res: dict) -> float:
        msd = res.get("steps", {}).get("msd_baseline", {}).get("data", {}) or {}
        alpha_star = msd.get("alpha_star", None)
        if alpha_star is None:
            raise ValueError("Missing alpha_star in msd_baseline.data")
        try:
            alpha_star = float(alpha_star)
        except Exception as e:
            raise ValueError(f"alpha_star is not a number: {alpha_star}") from e
        if not np.isfinite(alpha_star) or alpha_star <= 0:
            raise ValueError(
                f"alpha_star must be a positive finite number; got {alpha_star}"
            )

        base_demands = msd.get("base_demands", []) or []
        if not isinstance(base_demands, list) or not base_demands:
            raise ValueError("msd_baseline.data.base_demands missing or empty")

        offenders_zero: List[str] = []
        offenders_neg: List[str] = []
        offenders_nan: List[str] = []
        duplicates: List[str] = []
        seen = set()
        base_total = 0.0
        for rec in base_demands:
            src = str(rec.get("source_path", "")).strip()
            dst = str(rec.get("sink_path", "")).strip()
            if not src or not dst:
                raise ValueError("Empty source_path/sink_path in base_demands entry")
            key = (src, dst, rec.get("mode", None), rec.get("priority", None))
            if key in seen:
                duplicates.append(f"{src}→{dst}")
            seen.add(key)
            try:
                dem = float(rec.get("demand", float("nan")))
            except Exception:
                dem = float("nan")
            if not np.isfinite(dem):
                offenders_nan.append(f"{src}→{dst}")
                continue
            if dem < 0.0:
                offenders_neg.append(f"{src}→{dst}")
            elif dem == 0.0:
                offenders_zero.append(f"{src}→{dst}")
            else:
                base_total += dem

        if offenders_nan:
            sample = ", ".join(offenders_nan[:5])
            more = (
                "" if len(offenders_nan) <= 5 else f" (+{len(offenders_nan) - 5} more)"
            )
            raise ValueError(
                f"Invalid traffic matrix: non-numeric demand for {len(offenders_nan)} entr"
                f"{'y' if len(offenders_nan) == 1 else 'ies'}: {sample}{more}"
            )
        if offenders_neg:
            sample = ", ".join(offenders_neg[:5])
            more = (
                "" if len(offenders_neg) <= 5 else f" (+{len(offenders_neg) - 5} more)"
            )
            raise ValueError(
                f"Invalid traffic matrix: negative demand for {len(offenders_neg)} entr"
                f"{'y' if len(offenders_neg) == 1 else 'ies'}: {sample}{more}"
            )
        if offenders_zero:
            sample = ", ".join(offenders_zero[:5])
            more = (
                ""
                if len(offenders_zero) <= 5
                else f" (+{len(offenders_zero) - 5} more)"
            )
            raise ValueError(
                "Invalid traffic matrix: "
                f"{len(offenders_zero)} zero-demand entr{'y' if len(offenders_zero) == 1 else 'ies'} detected: "
                f"{sample}{more}. TM generation must not emit entries with zero demand."
            )
        if duplicates:
            sample = ", ".join(duplicates[:5])
            more = "" if len(duplicates) <= 5 else f" (+{len(duplicates) - 5} more)"
            raise ValueError(
                f"Invalid traffic matrix: duplicate entries detected for {len(duplicates)} pair"
                f"{' ' if len(duplicates) == 1 else 's'}: {sample}{more}"
            )
        return base_total * alpha_star

    def _validate_tm_placement_baseline(
        res: dict, expected_total_at_alpha: Optional[float]
    ) -> None:
        # Enforce metadata.baseline == True
        tm_step = res.get("steps", {}).get("tm_placement", {}) or {}
        tm_meta = tm_step.get("metadata", {}) or {}
        if bool(tm_meta.get("baseline")) is not True:
            raise ValueError(
                "tm_placement.metadata.baseline must be true and baseline must be included"
            )

        tm = tm_step.get("data", {}) or {}
        fr = tm.get("flow_results", []) or []
        if not isinstance(fr, list) or not fr:
            raise ValueError("tm_placement.data.flow_results missing or empty")
        # Enforce baseline as first element and with failure_id == 'baseline'
        first = fr[0]
        if str(first.get("failure_id", "")) != "baseline":
            raise ValueError(
                "tm_placement baseline must be first (flow_results[0].failure_id == 'baseline')"
            )
        baseline = first

        flows = baseline.get("flows", []) or []
        if not isinstance(flows, list) or not flows:
            raise ValueError("tm_placement baseline has no flows")

        # Check per-flow integrity and that placed+dropped ~= demand
        def _is_close(a: float, b: float) -> bool:
            if not (np.isfinite(a) and np.isfinite(b)):
                return False
            diff = abs(a - b)
            return diff <= max(1e-6, 1e-3 * max(abs(a), abs(b), 1.0))

        for rec in flows:
            s = rec.get("source", "")
            d = rec.get("destination", "")
            if not s or not d or s == d:
                raise ValueError(
                    "tm_placement baseline contains invalid flow endpoints"
                )
            try:
                dem = float(rec.get("demand", float("nan")))
                pla = float(rec.get("placed", float("nan")))
                drp = float(rec.get("dropped", float("nan")))
            except Exception as e:
                raise ValueError(
                    "tm_placement baseline has non-numeric demand/placed/dropped"
                ) from e
            if not (np.isfinite(dem) and np.isfinite(pla) and np.isfinite(drp)):
                raise ValueError("tm_placement baseline has NaN/Inf values")
            if dem <= 0 or pla < 0 or drp < 0:
                raise ValueError(
                    "tm_placement baseline has non-positive demand or negative placed/dropped"
                )
            if not _is_close(pla + drp, dem):
                raise ValueError(
                    "tm_placement baseline violates placed + dropped ≈ demand"
                )

            cdist = rec.get("cost_distribution", {}) or {}
            if cdist:
                vol_sum = 0.0
                for k, v in cdist.items():
                    try:
                        # cost keys and volume values must be numeric; volume non-negative
                        float(k)
                        vv = float(v)
                    except Exception as e:
                        raise ValueError(
                            "tm_placement baseline has non-numeric cost_distribution"
                        ) from e
                    if vv < 0:
                        raise ValueError(
                            "tm_placement baseline has negative volume in cost_distribution"
                        )
                    vol_sum += vv
                if not _is_close(vol_sum, pla):
                    raise ValueError(
                        "tm_placement baseline cost_distribution volume sum does not equal placed"
                    )

        # Cross-check totals if available
        total_dem = baseline.get("summary", {}).get("total_demand", None)
        if expected_total_at_alpha is not None and total_dem is not None:
            try:
                td = float(total_dem)
            except Exception:
                td = float("nan")
            if np.isfinite(td) and not _is_close(td, float(expected_total_at_alpha)):
                raise ValueError(
                    "tm_placement baseline total_demand does not match base_demands × alpha_star"
                )

    def _validate_maxflow_baseline(res: dict) -> None:
        # Enforce metadata.baseline == True
        mf_step = res.get("steps", {}).get("node_to_node_capacity_matrix", {}) or {}
        mf_meta = mf_step.get("metadata", {}) or {}
        if bool(mf_meta.get("baseline")) is not True:
            raise ValueError(
                "node_to_node_capacity_matrix.metadata.baseline must be true and baseline must be included"
            )

        mf = mf_step.get("data", {}) or {}
        fr = mf.get("flow_results", []) or []
        if not isinstance(fr, list) or not fr:
            raise ValueError(
                "node_to_node_capacity_matrix.data.flow_results missing or empty"
            )
        # Enforce baseline as first element and with failure_id == 'baseline'
        first = fr[0]
        if str(first.get("failure_id", "")) != "baseline":
            raise ValueError(
                "node_to_node_capacity_matrix baseline must be first (flow_results[0].failure_id == 'baseline')"
            )

        # Basic integrity across iterations: when endpoints look valid, ensure placed is sane
        for it in fr:
            for rec in it.get("flows", []) or []:
                s = rec.get("source", "")
                d = rec.get("destination", "")
                # Some implementations may include bookkeeping entries; ignore those
                if not s or not d or s == d:
                    continue
                try:
                    placed = float(rec.get("placed", 0.0))
                except Exception as e:
                    raise ValueError(
                        "maxflow results contain non-numeric placed value"
                    ) from e
                if not np.isfinite(placed) or placed < 0.0:
                    raise ValueError(
                        "maxflow results contain negative or non-finite placed value"
                    )

    # Run validations
    require_maxflow = bool(
        os.environ.get("NGRAPH_ENABLE_MAXFLOW", "0").strip()
        not in ("", "0", "false", "False")
    )
    _require_steps(results, require_maxflow)
    expected_total_at_alpha = _validate_alpha_and_base_demands(results)
    _validate_tm_placement_baseline(results, expected_total_at_alpha)
    if require_maxflow:
        _validate_maxflow_baseline(results)

    # Metrics
    alpha = compute_alpha_star(results)

    bac_place = compute_bac(results, step_name="tm_placement", mode="auto")
    bac_max = (
        compute_bac(results, step_name="node_to_node_capacity_matrix", mode="auto")
        if require_maxflow
        else None
    )

    latency = compute_latency_stretch(results)

    # Optional structural diagnostic (SPS); currently not surfaced in tables
    if require_maxflow:
        _ = compute_sps(results)

    # For cost/power normalization use the SAME alpha (α*):
    # offered_at_alpha_star = alpha* × sum(base_demands); reliable = BAC p99.9 absolute delivered at α*
    offered_alpha_star = None
    if not np.isnan(alpha.base_total_demand) and np.isfinite(alpha.alpha_star):
        offered_alpha_star = float(alpha.base_total_demand * alpha.alpha_star)
    # Reliable denominator should use Bandwidth-at-Probability (inverse availability)
    reliable_p999 = bac_place.bw_at_probability_abs.get(99.9, np.nan)
    costpower = compute_cost_power(
        results, offered_at_alpha1=offered_alpha_star, reliable_at_p999=reliable_p999
    )

    # Plots
    if do_plots:
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        plot_bac(
            bac_place,
            overlay=bac_max if bac_max is not None else None,
            save_to=out_dir / "bac.png",
        )
        plot_latency(latency, save_to=out_dir / "latency.png")
        # SPS is scalar per-iteration; no plot for now
        plot_cost_power(costpower, save_to=out_dir / "costpower.png")

    return alpha, bac_place, bac_max, latency, costpower


def main():
    ap = argparse.ArgumentParser(description="NetGraph modular analysis")
    ap.add_argument(
        "root",
        type=str,
        help="Root directory containing *.results.json (e.g., scenarios_3)",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated scenario stems to include (e.g., small_baseline_hose,backbone_clos_hose)",
    )
    ap.add_argument(
        "--no-plots", action="store_true", help="Skip PNG charts generation"
    )
    ap.add_argument(
        "--enable-maxflow",
        action="store_true",
        help="Enable MaxFlow-based metrics (SPS, BAC overlay)",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print the summary table and generate figures in the provided root path: "
            "BAC.png (pooled BAC), Latency_p99.png (pooled latency), and abs_*/norm_* charts."
        ),
    )
    args = ap.parse_args()

    root = Path(args.root)
    # If CLI flag is provided, set env var used by analyzers
    if args.enable_maxflow:
        os.environ["NGRAPH_ENABLE_MAXFLOW"] = "1"
    # Derive output root as <root>_metrics side-by-side with input root
    out_root = root.parent / f"{root.name}_metrics"
    only_set = (
        set([s.strip() for s in args.only.split(",") if s.strip()])
        if args.only
        else None
    )
    do_plots = not args.no_plots

    # Discover result files
    files = find_results_files(root)
    if not files:
        print(f"[WARN] No *.results.json found under {root}. Nothing to do.")
        return

    grouped = group_by_scenario(files)
    if only_set:
        grouped = {k: v for k, v in grouped.items() if k in only_set}

    # Run per scenario, aggregate across seeds, write outputs
    # Determine once for the run whether maxflow is enabled
    require_maxflow = bool(
        os.environ.get("NGRAPH_ENABLE_MAXFLOW", "0").strip()
        not in ("", "0", "false", "False")
    )
    for scenario_stem, seed_map in grouped.items():
        print(f"\n=== Scenario: {scenario_stem} (seeds={sorted(seed_map)}) ===")
        scenario_out = ScenarioOutputs()

        for seed, path in sorted(seed_map.items()):
            results = load_json(path)
            seed_dir = out_root / scenario_stem / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            alpha, bac_p, bac_m, latency, cp = analyze_one_seed(
                results, seed_dir, do_plots
            )
            scenario_out.alpha[seed] = alpha
            scenario_out.bac_place[seed] = bac_p
            if bac_m is not None:
                scenario_out.bac_maxflow[seed] = bac_m
            scenario_out.latency[seed] = latency
            scenario_out.costpower[seed] = cp

            # Per-seed outputs
            write_csv_atomic(
                seed_dir / "bac_series.csv",
                bac_p.series.to_frame(name="delivered"),
            )
            write_json_atomic(seed_dir / "bac.json", bac_p.to_jsonable())
            write_json_atomic(seed_dir / "alpha.json", alpha.to_jsonable())
            write_json_atomic(seed_dir / "latency.json", latency.to_jsonable())
            # paircap removed from outputs; SPS is computed but not plotted
            write_json_atomic(seed_dir / "costpower.json", cp.to_jsonable())
            # Always dump per-pair matrices for debugging (tm_placement; plus maxflow if enabled)
            tm_abs, tm_norm, mf_abs, mf_norm = compute_pair_matrices(
                results, include_maxflow=require_maxflow
            )
            try:
                if not tm_abs.empty:
                    write_csv_atomic(seed_dir / "pairs_tm_abs.csv", tm_abs)
                if not tm_norm.empty:
                    write_csv_atomic(seed_dir / "pairs_tm_norm.csv", tm_norm)
                if mf_abs is not None and not mf_abs.empty:
                    write_csv_atomic(seed_dir / "pairs_mf_abs.csv", mf_abs)
                if mf_norm is not None and not mf_norm.empty:
                    write_csv_atomic(seed_dir / "pairs_mf_norm.csv", mf_norm)
            except Exception:
                pass

        # Cross-seed summaries
        scen_dir = out_root / scenario_stem
        scen_dir.mkdir(parents=True, exist_ok=True)

        # Summaries and paired stats (if we have multiple stems with the same seeds)
        # 1) Alpha*
        alpha_summary = summarize_across_seeds(
            {k: v.alpha_star for k, v in scenario_out.alpha.items()}, label="alpha_star"
        )
        write_json_atomic(scen_dir / "alpha_summary.json", alpha_summary)

        # 2) BAC (Placement) key points
        series_by_seed = {
            seed: br.series for seed, br in scenario_out.bac_place.items()
        }
        bac_summary = summarize_across_seeds(series_by_seed, label="bac_delivered")
        bac_tail = {
            "p50": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.50, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "p90": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.90, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "p99": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.99, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "p999": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.999, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "p9999": float(
                np.nanmedian(
                    [
                        v.quantiles_pct.get(0.9999, np.nan)
                        for v in scenario_out.bac_place.values()
                    ]
                )
            ),
            "auc_norm": float(
                np.nanmedian(
                    [v.auc_normalized for v in scenario_out.bac_place.values()]
                )
            ),
        }

        # Also include Bandwidth-at-Probability (ratio to baseline/offered)
        # e.g., bw_p999_pct is the proportion of baseline bandwidth that is met/exceeded
        # with probability 99.9% across failure iterations (cross-seed median here).
        def _bw_med(pct: float, _scenario_out: ScenarioOutputs = scenario_out) -> float:
            vals = []
            for v in _scenario_out.bac_place.values():
                try:
                    vals.append(float(v.bw_at_probability_pct.get(pct, np.nan)))
                except Exception:
                    vals.append(float("nan"))
            return float(np.nanmedian(vals)) if vals else float("nan")

        bac_tail.update(
            {
                "bw_p90_pct": _bw_med(90.0),
                "bw_p95_pct": _bw_med(95.0),
                "bw_p99_pct": _bw_med(99.0),
                "bw_p999_pct": _bw_med(99.9),
                "bw_p9999_pct": _bw_med(99.99),
            }
        )
        write_json_atomic(
            scen_dir / "bac_summary.json",
            {"per_seed": bac_summary, "tail": bac_tail},
        )

        # 2b) BAC pooled empirical across seeds (recommended aggregation)
        try:
            # Gather normalized samples per seed (percent of offered, capped at 100)
            pooled_samples: list[float] = []
            per_seed_samples: dict[int, list[float]] = {}
            for seed, br in scenario_out.bac_place.items():
                offered = float(br.offered)
                s = np.asarray(br.series.astype(float).values, dtype=float)
                if np.isfinite(offered) and offered > 0.0 and s.size > 0:
                    norm = np.minimum(s / offered, 1.0) * 100.0
                    vals = [float(x) for x in norm if np.isfinite(x)]
                    if vals:
                        pooled_samples.extend(vals)
                        per_seed_samples[seed] = vals

            pooled_tail = {}
            grid = np.linspace(0.0, 100.0, 401)
            pooled_grid_x = []
            pooled_grid_a = []
            iqr_q25 = []
            iqr_q75 = []
            if pooled_samples:
                xs = np.sort(np.asarray(pooled_samples, dtype=float))
                cdf = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
                avail = 1.0 - cdf
                pooled_grid_x = xs.tolist()
                pooled_grid_a = avail.tolist()
                # Tails using inverse availability (lower-tail quantile)
                for p in (90.0, 95.0, 99.0, 99.9, 99.99):
                    q = max(0.0, 1.0 - (p / 100.0))
                    try:
                        thr = float(np.quantile(xs, q, method="lower"))
                    except Exception:
                        thr = float("nan")
                    pooled_tail[f"bw_p{str(p).rstrip('0').rstrip('.')}_pct"] = (
                        thr / 100.0
                    )
                # Pooled AUC on normalized axis (mean of min(norm,1))
                pooled_tail["auc_norm"] = float(np.mean(xs / 100.0))

                # Cross-seed IQR on a common grid
                if len(per_seed_samples) >= 3:
                    mat = []
                    for vals in per_seed_samples.values():
                        # Availability on grid via step-post interpolation
                        sv = np.sort(np.asarray(vals, dtype=float))
                        cdf_s = np.arange(1, sv.size + 1, dtype=float) / float(sv.size)
                        a_s = 1.0 - cdf_s
                        a_on_grid = np.interp(grid, sv, a_s, left=a_s[0], right=a_s[-1])
                        mat.append(a_on_grid)
                    mat = np.asarray(mat, dtype=float)
                    iqr_q25 = np.nanpercentile(mat, 25, axis=0).tolist()
                    iqr_q75 = np.nanpercentile(mat, 75, axis=0).tolist()

            # Append pooled artifacts to existing bac_summary.json
            pooled_payload = {
                "pooled_tail": pooled_tail,
                "pooled_grid": {"x_pct": pooled_grid_x, "availability": pooled_grid_a},
            }
            if iqr_q25 and iqr_q75:
                pooled_payload["pooled_iqr"] = {
                    "x_pct": grid.tolist(),
                    "a_q25": iqr_q25,
                    "a_q75": iqr_q75,
                }
            # Merge into the previously written file
            try:
                import json as _json

                path = scen_dir / "bac_summary.json"
                cur = (
                    _json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else {}
                )
                cur.update(pooled_payload)
                write_json_atomic(path, cur)
            except Exception:
                pass
        except Exception:
            pass

        # 3) Latency summary per seed (baseline vs failures + derived deltas)
        rows = []
        for seed, s in scenario_out.latency.items():
            b = s.baseline or {}
            f = s.failures or {}
            d = s.derived or {}
            rows.append(
                {
                    "seed": seed,
                    "base_p50": float(b.get("p50", np.nan)),
                    "fail_p99": float(f.get("p99", np.nan)),
                    "TD99": float(d.get("TD99", np.nan)),
                    "SLO_1_2_drop": float(d.get("SLO_1_2_drop", np.nan)),
                    "best_path_share_drop": float(
                        d.get("best_path_share_drop", np.nan)
                    ),
                    "WES_delta": float(d.get("WES_delta", np.nan)),
                }
            )
        if rows:
            lat_sum = pd.DataFrame(rows).set_index("seed").sort_index()
            write_csv_atomic(scen_dir / "latency_summary.csv", lat_sum)

        # 4) Network stats summary per scenario (node/link counts)
        ns_rows = []
        for seed, path in sorted(seed_map.items()):
            res2 = load_json(path)
            ns = res2.get("steps", {}).get("network_statistics", {}).get("data", {})
            if not ns:
                raise ValueError(f"Missing network_statistics.data in {path}")
            if ns.get("node_count") is None or ns.get("link_count") is None:
                raise ValueError(
                    f"Incomplete network_statistics (node/link counts) in {path}"
                )
            ns_rows.append(
                {
                    "seed": seed,
                    "node_count": int(ns.get("node_count")),
                    "link_count": int(ns.get("link_count")),
                }
            )
        ns_df = pd.DataFrame(ns_rows).set_index("seed").sort_index()
        write_csv_atomic(scen_dir / "network_stats_summary.csv", ns_df)

        # 5) Provenance metadata for reproducibility
        provenance = {
            "generated_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "analysis_script": str(Path(__file__).resolve()),
        }
        try:
            # Capture git commit if available
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
            provenance["git_commit"] = commit
        except Exception:
            pass
        write_json_atomic(scen_dir / "provenance.json", provenance)

        # 4) Cost/Power summaries
        cp_df = pd.DataFrame(
            {
                seed: scenario_out.costpower[seed].flat_series()
                for seed in sorted(scenario_out.costpower)
            }
        ).T
        write_csv_atomic(scen_dir / "costpower_summary.csv", cp_df)

    # (Legacy paired-comparison block removed; baseline-normalized view supersedes it.)

    # Consolidated summary + baseline-normalized view and insights (printed and also saved to text)
    df = summary_mod.build_project_summary_table(out_root)
    if df.empty:
        print("(no scenarios summarized)")
        return
    summary_txt = out_root / "summary.txt"
    buf = io.StringIO()
    with redirect_stdout(buf):
        # Absolute consolidated table
        summary_mod.print_pretty_table(df, title="Consolidated project metrics")
        print("\n\n")
        # Baseline-normalized table + CSV + normalized insights (all + filtered)
        try:
            base_df = summary_mod.build_baseline_normalized_table(out_root)
            if not base_df.empty:
                summary_mod.print_pretty_table(
                    base_df, title="Baseline-normalized metrics (scenario / baseline)"
                )
                print("\n\n")
                # Save normalized CSV alongside outputs
                summary_mod.write_csv_atomic(
                    (out_root / "project_baseline_normalized.csv"),
                    base_df.reset_index(),
                )
                # Two-step normalized insights
                try:
                    summary_mod._print_normalized_insights(out_root)
                    print("\n\n")
                except Exception as e:
                    print(f"[WARN] Failed to compute normalized insights: {e}")
        except Exception as e:
            print(f"[WARN] Failed to build baseline-normalized table: {e}")
    text = buf.getvalue()
    try:
        summary_txt.write_text(text, encoding="utf-8")
        print(f"Wrote text summary: {summary_txt}")
    except Exception:
        pass
    print(text, end="")
    saved_csv = summary_mod.save_project_csv_incremental(df, cwd=out_root)
    print(f"Wrote project CSV: {saved_csv}")

    # Optional: generate publishable figures when requested
    if args.summary:
        try:
            # Prefer cross-seed pooled BAC overlay for the summary figure
            from metrics.plot_bac_delta_vs_baseline import (
                plot_bac_delta_vs_baseline as _plot_bac_delta_vs_baseline,
            )
            from metrics.plot_cross_seed_bac import (
                plot_cross_seed_bac as _plot_cross_seed_bac,
            )
            from metrics.plot_cross_seed_latency import (
                plot_cross_seed_latency as _plot_cross_seed_latency,
            )

            # Figures go to the analysis output root (e.g., 'scenarios_metrics')
            fig_dir = out_root
            fig_dir.mkdir(parents=True, exist_ok=True)

            # BAC figure (PNG)
            out_bac_png = fig_dir / "BAC.png"
            bac_path = _plot_cross_seed_bac(out_root, save_to=out_bac_png)
            if bac_path is not None:
                print(f"Saved BAC summary figure: {bac_path}")
            else:
                print("[WARN] No BAC data found to plot.")

            # Latency availability figure (p99; PNG)
            out_lat_png = fig_dir / "Latency_p99.png"
            lat_path = _plot_cross_seed_latency(
                out_root, metric="p99", save_to=out_lat_png
            )
            if lat_path is not None:
                print(f"Saved latency summary figure: {lat_path}")
            else:
                print("[WARN] No latency data found to plot.")

            # BAC Δ-availability vs baseline (80–100%) with top-left legend
            out_delta_png = fig_dir / "BAC_delta_vs_baseline.png"
            delta_path = _plot_bac_delta_vs_baseline(
                out_root,
                baseline="baseline_SingleRouter",
                grid_min=80.0,
                grid_max=100.0,
                legend_loc="upper left",
                save_to=out_delta_png,
            )
            if delta_path is not None:
                print(f"Saved BAC Δ-availability figure: {delta_path}")
            else:
                print("[WARN] No data to plot for BAC Δ-availability.")

            # Project-level bar charts for key scalar metrics
            # Reuse df (absolute) and baseline-normalized table
            def _plot_bar_abs(column: str, title: str, ylabel: str, fname: str) -> None:
                if column not in df.columns:
                    return
                try:
                    import seaborn as sns  # local to avoid mandatory dep warnings

                    plt.figure(figsize=(8.0, 5.0))
                    data = df[[column]].copy()
                    data = data.sort_values(by=column, ascending=False)
                    data = data.reset_index().rename(columns={"index": "scenario"})
                    ax = sns.barplot(data=data, x="scenario", y=column)
                    ax.set_title(title)
                    ax.set_ylabel(ylabel)
                    ax.set_xlabel("scenario")
                    ax.grid(True, linestyle=":", linewidth=0.5, axis="y")
                    plt.xticks(rotation=20, ha="right")
                    outp = fig_dir / fname
                    plt.tight_layout()
                    plt.savefig(outp)
                    plt.close()
                    print(f"Saved project metric figure: {outp}")
                except Exception as _e:
                    pass

            def _plot_bar_norm(
                column: str, title: str, ylabel: str, fname: str
            ) -> None:
                try:
                    base_df_local = summary_mod.build_baseline_normalized_table(
                        out_root
                    )
                except Exception:
                    return
                if base_df_local.empty or column not in base_df_local.columns:
                    return
                try:
                    import seaborn as sns  # local to avoid mandatory dep warnings

                    plt.figure(figsize=(8.0, 5.0))
                    data = base_df_local[[column]].copy()
                    data = data.sort_values(by=column, ascending=False)
                    data = data.reset_index().rename(columns={"index": "scenario"})
                    ax = sns.barplot(data=data, x="scenario", y=column)
                    ax.set_title(title)
                    ax.set_ylabel(ylabel)
                    ax.set_xlabel("scenario")
                    ax.grid(True, linestyle=":", linewidth=0.5, axis="y")
                    plt.xticks(rotation=20, ha="right")
                    outp = fig_dir / fname
                    plt.tight_layout()
                    plt.savefig(outp)
                    plt.close()
                    print(f"Saved project metric figure: {outp}")
                except Exception as _e:
                    pass

            # Absolute plots (prefixed with abs_)
            _plot_bar_abs("node_count", "Node count", "nodes", "abs_nodes.png")
            _plot_bar_abs("link_count", "Link count", "links", "abs_links.png")
            _plot_bar_abs(
                "bac_auc",
                title="BAC AUC (median across seeds)",
                ylabel="AUC (0..1)",
                fname="abs_AUC.png",
            )
            _plot_bar_abs(
                "bw_p99",
                title="Bandwidth at 99% (ratio to offered)",
                ylabel="ratio",
                fname="abs_BW_p99.png",
            )
            _plot_bar_abs(
                "USD_per_Gbit_offered",
                title="Cost per Gbit (offered)",
                ylabel="USD/Gb",
                fname="abs_USD_per_Gbit_offered.png",
            )
            _plot_bar_abs(
                "USD_per_Gbit_p999",
                title="Cost per Gbit at p99.9",
                ylabel="USD/Gb",
                fname="abs_USD_per_Gbit_p999.png",
            )
            _plot_bar_abs(
                "lat_fail_p99",
                title="Latency p99 under failures (median across seeds)",
                ylabel="stretch (×)",
                fname="abs_Latency_fail_p99.png",
            )
            _plot_bar_abs(
                "capex_total",
                title="Total CapEx",
                ylabel="USD",
                fname="abs_CapEx.png",
            )

            # Normalized plots vs baseline (prefixed with norm_)
            _plot_bar_norm(
                "node_count_r",
                "Nodes (relative to baseline)",
                "ratio",
                "norm_nodes.png",
            )
            _plot_bar_norm(
                "link_count_r",
                "Links (relative to baseline)",
                "ratio",
                "norm_links.png",
            )
            _plot_bar_norm(
                "auc_norm_r",
                "BAC AUC (relative)",
                "ratio",
                "norm_AUC.png",
            )
            _plot_bar_norm(
                "bw_p99_pct_r",
                "BW@99% (relative)",
                "ratio",
                "norm_BW_p99.png",
            )
            _plot_bar_norm(
                "USD_per_Gbit_offered_r",
                "Cost per Gbit (offered, relative)",
                "ratio",
                "norm_USD_per_Gbit_offered.png",
            )
            _plot_bar_norm(
                "USD_per_Gbit_p999_r",
                "Cost per Gbit p99.9 (relative)",
                "ratio",
                "norm_USD_per_Gbit_p999.png",
            )
            _plot_bar_norm(
                "lat_fail_p99_r",
                "Latency p99 under failures (relative)",
                "ratio",
                "norm_Latency_fail_p99.png",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to generate summary figures: {e}")


if __name__ == "__main__":
    main()
