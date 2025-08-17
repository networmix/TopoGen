#!/usr/bin/env python3
"""
runner.py

Streamlined driver:

1) Build each master once via ./build.sh, writing base scenario and artefacts under
   scenarios/<master_stem>/<master_stem>/ ...

2) Create per-seed scenario copies by replacing 'seed: <n>' lines in the base scenario,
   saved under scenarios/<master_stem>/<master_stem>__seed<SEED>/.

3) Run ngraph (inspect → run) per-seed. Default is sequential (single-process). Reporting is optional via --report.

Usage:
  python3 runner.py \
      --masters-dir topogen_configs \
      --seeds 11,19,23,29,31,37,41,43 \
      [--scenarios-dir scenarios] [--run-jobs 1] [--report] [--force]

Notes:
- Expects build.sh and run.sh in the current working directory.
- No external Python deps; standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple

# -------------------------------
# File system helpers
# -------------------------------


def die(msg: str, code: int = 1) -> NoReturn:
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(code)


def sh(
    cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None
) -> None:
    """Run a subprocess, streaming output. Raises on nonzero exit."""
    print(f"▶ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit {ret}: {' '.join(cmd)}")


def find_master_yaml_files(masters_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in masters_dir.iterdir()
            if p.is_file() and p.suffix in (".yml", ".yaml")
        ]
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Scenario seeding (post-build)
# -------------------------------

_SCENARIO_FILE_CANDIDATES = (
    "{stem}_scenario.yml",
    "{stem}_scenario.yaml",
)

_SCENARIO_SEED_LINE = re.compile(r"^(\s*seed\s*:\s*)\d+\s*$")


def _load_base_scenario(master_stem: str, master_scenarios_root: Path) -> Path:
    """Locate the single base scenario YAML produced by TopoGen build."""
    base_dir = master_scenarios_root / master_stem
    if not base_dir.exists():
        die(f"Base scenario directory not found: {base_dir}")
    for pat in _SCENARIO_FILE_CANDIDATES:
        p = base_dir / pat.format(stem=master_stem)
        if p.exists():
            return p
    die(f"No base scenario YAML found under: {base_dir}")


def _rewrite_all_seed_lines(yaml_text: str, new_seed: int) -> str:
    """Replace every line of the form 'seed: <int>' with the provided seed."""
    lines = yaml_text.splitlines()
    for i, line in enumerate(lines):
        m = _SCENARIO_SEED_LINE.match(line)
        if m:
            prefix = m.group(1)
            lines[i] = f"{prefix}{new_seed}"
    return "\n".join(lines) + ("\n" if yaml_text.endswith("\n") else "")


def create_seeded_scenarios(
    master_stem: str,
    master_scenarios_root: Path,
    seeds: List[int],
) -> List[Path]:
    """Create per-seed scenario YAMLs by cloning the base scenario and rewriting seed lines.

    Returns a list of created scenario YAML paths.
    """
    base_yaml = _load_base_scenario(master_stem, master_scenarios_root)
    base_text = base_yaml.read_text(encoding="utf-8")
    created: List[Path] = []
    for s in seeds:
        dest_dir = master_scenarios_root / f"{master_stem}__seed{s}"
        ensure_dir(dest_dir)
        dest_yaml = dest_dir / f"{master_stem}__seed{s}_scenario.yml"
        mutated = _rewrite_all_seed_lines(base_text, s)
        dest_yaml.write_text(mutated, encoding="utf-8")
        created.append(dest_yaml)
    return created


# -------------------------------
# Parallel build helpers
# -------------------------------


def _build_one_master(
    build_sh: Path,
    master_generated_dir: Path,
    master_scenarios_root: Path,
    include_pat: str,
    force: bool,
    build_only: bool,
    log_path: Path,
    timeout_secs: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Execute build.sh for a single master, redirecting stdout/stderr to a log file.
    Returns (log_path, return_code).
    """
    ensure_dir(master_scenarios_root)
    ensure_dir(log_path.parent)
    cmd: List[str] = ["bash", str(build_sh)]
    if force:
        cmd.append("--force")
    elif build_only:
        cmd.append("--build-only")
    cmd += [
        "--include",
        include_pat,
        str(master_generated_dir),
        str(master_scenarios_root),
    ]
    print(f"▶ [build] {' '.join(cmd)}  | log → {log_path}")
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        try:
            ret = proc.wait(timeout=timeout_secs) if timeout_secs else proc.wait()
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            # Append timeout note to the log
            try:
                with log_path.open("a", encoding="utf-8") as af:
                    af.write("\n⏱️ build.sh timed out and was terminated by runner.py\n")
            except Exception:
                pass
            ret = 124  # conventional timeout exit code
    return (str(log_path), ret)


# (aggregation logic removed; runner focuses on orchestration only)


# -------------------------------
# Main workflow
# -------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--masters-dir",
        default=Path("topogen_configs"),
        type=Path,
        help="Folder with TopoGen master YAMLs (default: topogen_configs)",
    )
    ap.add_argument(
        "--seeds", default="11,19,23,29,31,37,41,43", help="Comma-separated seeds"
    )
    # generated-dir removed: generate outputs are stored under scenarios now
    ap.add_argument("--scenarios-dir", default=Path("scenarios"), type=Path)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force both build and run (passthrough to build.sh and run.sh)",
    )
    ap.add_argument(
        "--force-build",
        action="store_true",
        help="Force TopoGen build (passthrough --force to build.sh only)",
    )
    ap.add_argument(
        "--force-run",
        action="store_true",
        help="Force ngraph run (passthrough --force to run.sh only)",
    )
    ap.add_argument(
        "--build-jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Max concurrent builds (per master). Default ~= half of CPU cores.",
    )
    ap.add_argument(
        "--build-logs-dir",
        type=Path,
        default=Path("scenarios/_build_logs"),
        help="Directory to write per-master build logs",
    )
    ap.add_argument(
        "--build-timeout",
        type=int,
        default=0,
        help="Per-master build timeout in seconds (0 disables timeout)",
    )
    ap.add_argument(
        "--run-jobs",
        type=int,
        default=1,
        help="Max concurrent ngraph inspect+run jobs per master (default: 1 - sequential)",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Also run ngraph report after run (default: off)",
    )
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        die("No seeds specified")

    masters = find_master_yaml_files(args.masters_dir)
    if not masters:
        die(f"No YAMLs under {args.masters_dir}")

    # Validate helper scripts
    build_sh = Path("./build.sh")
    run_sh = Path("./run.sh")
    if not build_sh.exists():
        die("build.sh not found in current directory")
    if not run_sh.exists():
        die("run.sh not found in current directory")

    ensure_dir(args.scenarios_dir)

    # High-level execution plan
    print("\n===== Experiment Plan =====")
    print(f"Masters dir:    {args.masters_dir}")
    print(f"Masters found:   {len(masters)}")
    print(f"Seeds:           {seeds}  (count: {len(seeds)})")
    print(f"Scenarios dir:   {args.scenarios_dir}")
    print(f"Build jobs:      {max(1, args.build_jobs)}")
    print(f"Run jobs:        {max(1, args.run_jobs)} (per master)")
    if getattr(args, "build_timeout", 0):
        print(f"Build timeout:   {args.build_timeout}s per master")
    else:
        print("Build timeout:   disabled")
    print("Behavior:")
    print(" - Build once per master (parallel across masters)")
    print(" - Create per-seed scenario copies by replacing 'seed: <n>' lines")
    print(" - Run per-seed scenarios in parallel per master")
    if args.report:
        print(" - Report: generate HTML + Notebook for each scenario")
    # Effective force controls
    if args.force or args.force_build or args.force_run:
        print(
            f" - Force controls: build={bool(args.force or args.force_build)} run={bool(args.force or args.force_run)}"
        )
    print("==========================\n")

    # 1) Prepare contexts per master
    master_contexts: List[Dict[str, Any]] = []
    for master_yaml in masters:
        master_stem = master_yaml.stem
        print(f"\n====== Master: {master_yaml.name} ======")
        # Build will be executed once per master using the original master config
        master_generated_dir = args.masters_dir
        master_scenarios_root = args.scenarios_dir / master_stem
        ensure_dir(master_scenarios_root)
        master_contexts.append(
            {
                "stem": master_stem,
                "yaml": master_yaml,
                "generated_dir": master_generated_dir,
                "scenarios_root": master_scenarios_root,
            }
        )

    # 2) Parallel build once per master
    ensure_dir(args.build_logs_dir)
    futures = []
    print(
        f"🚧 Building {len(master_contexts)} masters in parallel (jobs={max(1, args.build_jobs)})"
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.build_jobs)
    ) as ex:
        for ctx in master_contexts:
            stem = ctx["stem"]
            log_path = args.build_logs_dir / f"{stem}.build.log"
            fut = ex.submit(
                _build_one_master,
                build_sh,
                ctx["generated_dir"],
                ctx["scenarios_root"],
                ctx["yaml"].name,
                bool(args.force),
                bool(args.force_build and not args.force),
                log_path,
                args.build_timeout
                if args.build_timeout and args.build_timeout > 0
                else None,
            )
            futures.append((stem, log_path, fut))

        # Wait and verify
        build_errors: List[str] = []
        for stem, _log_path, fut in futures:
            log_file, ret = fut.result()
            if ret != 0:
                build_errors.append(f"{stem} (log: {log_file})")
            else:
                print(f"✅ Build completed: {stem} (log: {log_file})")

    if build_errors:
        die(
            "One or more builds failed: "
            + "; ".join(build_errors)
            + ". Inspect logs for details."
        )

    # 3) Create per-seed scenario YAMLs by string replacement (no extra builds)
    for ctx in master_contexts:
        master_stem = ctx["stem"]
        master_scenarios_root = ctx["scenarios_root"]
        created = create_seeded_scenarios(master_stem, master_scenarios_root, seeds)
        ctx["scenarios"] = created
        print(f"📝 Seeded scenarios created for {master_stem}: {len(created)}")

    # Prepare a directory to collect per-master run summaries emitted by run.sh
    run_summaries_dir = args.scenarios_dir / "_run_summaries"
    ensure_dir(run_summaries_dir)

    # 4) Run per master: parallel ngraph inspect+run per scenario
    ngraph_start_wall = time.time()
    ngraph_start = time.perf_counter()
    ngraph_invoke = _detect_ngraph_invoke()
    for ctx in master_contexts:
        master_stem = ctx["stem"]
        scenarios: List[Path] = ctx.get("scenarios", [])
        if not scenarios:
            print(f"⚠️  No scenarios to run for {master_stem}")
            continue

        print(
            f"🧪 Running scenarios (inspect+run) for master: {master_stem} (jobs={max(1, args.run_jobs)})"
        )
        # Initialize per-master summary buffer
        summary_tsv = run_summaries_dir / f"{master_stem}.tsv"
        summary_tsv.write_text(
            "ScenarioDir\tScenario\tInspect\tRun\tReport\n", encoding="utf-8"
        )

        # Launch inspect+run in parallel
        run_results: List[
            Tuple[Path, str, str]
        ] = []  # (scenario_yaml, inspect_status, run_status)
        futures2 = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.run_jobs)
        ) as ex:
            for scn in scenarios:
                fut = ex.submit(
                    _inspect_run_one,
                    ngraph_invoke,
                    scn,
                    bool(args.force or args.force_run),
                )
                futures2.append(fut)
            for fut in futures2:
                scn_path, ins_status, run_status = fut.result()
                run_results.append((scn_path, ins_status, run_status))

        # After run, optionally generate reports sequentially
        for scn_path, ins_status, run_status in run_results:
            if args.report:
                rep_status = _report_one(
                    ngraph_invoke, scn_path, bool(args.force or args.force_run)
                )
            else:
                rep_status = "⏭️ skipped"
            scn_dir = scn_path.parent
            scn_name = scn_path.name
            # Append row to master summary TSV
            with summary_tsv.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"{scn_dir}\t{scn_name}\t{ins_status}\t{run_status}\t{rep_status}\n"
                )
        print(f"📦 Run finished: {master_stem}")

    # 5) Overall consolidated summary across all masters
    _print_overall_summary(run_summaries_dir)

    # 6) Record overall ngraph run timing
    ngraph_end_wall = time.time()
    ngraph_elapsed = time.perf_counter() - ngraph_start
    start_iso = datetime.fromtimestamp(ngraph_start_wall, tz=timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(ngraph_end_wall, tz=timezone.utc).isoformat()
    timing_tsv = run_summaries_dir / "_overall_ngraph_time.tsv"
    timing_tsv.write_text(
        "Scope\tStartUTC\tEndUTC\tElapsedSec\n"
        f"ngraph\t{start_iso}\t{end_iso}\t{ngraph_elapsed:.3f}\n",
        encoding="utf-8",
    )
    print(f"⏱️ Overall ngraph run time: {ngraph_elapsed:.3f}s")


def _print_overall_summary(run_summaries_dir: Path) -> None:
    # Collect all rows
    rows: List[Tuple[str, str, str, str, str]] = []
    for tsv in sorted(run_summaries_dir.glob("*.tsv")):
        try:
            content = tsv.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(content):
            if i == 0:
                # header
                continue
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            rows.append((parts[0], parts[1], parts[2], parts[3], parts[4]))

    if not rows:
        print("\n======================")
        print("📋 Overall Summary")
        print("(no scenarios discovered in run summaries)")
        print("======================\n")
        return

    # Compute column widths
    w_name = max(8, max(len(r[1]) for r in rows))
    w_ins = max(7, max(len(r[2]) for r in rows))
    w_run = max(3, max(len(r[3]) for r in rows))
    w_rep = max(6, max(len(r[4]) for r in rows))

    # Totals
    total = len(rows)
    ins_ok = ins_fail = ins_cached = ins_skipped = 0
    run_ok = run_fail = run_cached = run_skipped = 0
    rep_ok = rep_fail = rep_cached = rep_skipped = 0

    def _bump(status: str) -> Tuple[int, int, int, int]:
        ok = 1 if status == "✅" else 0
        fail = 1 if status == "❌" else 0
        cached = 1 if status.startswith("⏭️ cached") else 0
        skipped = 1 if status.startswith("⏭️ skipped") else 0
        return ok, fail, cached, skipped

    print("\n======================")
    print("📋 Overall Summary")
    # Header
    print(
        f"{'Scenario'.ljust(w_name)}  {'Inspect'.ljust(w_ins)}  {'Run'.ljust(w_run)}  {'Report'.ljust(w_rep)}"
    )
    dash_len = w_name + 2 + w_ins + 2 + w_run + 2 + w_rep
    print("-" * dash_len)

    for _, scenario_name, ins, run, rep in rows:
        print(
            f"{scenario_name.ljust(w_name)}  {ins.ljust(w_ins)}  {run.ljust(w_run)}  {rep.ljust(w_rep)}"
        )
        ok, fail, cached, skipped = _bump(ins)
        ins_ok += ok
        ins_fail += fail
        ins_cached += cached
        ins_skipped += skipped
        ok, fail, cached, skipped = _bump(run)
        run_ok += ok
        run_fail += fail
        run_cached += cached
        run_skipped += skipped
        ok, fail, cached, skipped = _bump(rep)
        rep_ok += ok
        rep_fail += fail
        rep_cached += cached
        rep_skipped += skipped

    print("----------------------")
    print(f"🧮 Totals: {total} scenarios")
    print(
        f"   • Inspect: ✅ {ins_ok} | ⏭️ {ins_skipped} (skipped) | ⏭️ {ins_cached} (cached) | ❌ {ins_fail}"
    )
    print(
        f"   • Run:     ✅ {run_ok} | ⏭️ {run_skipped} (skipped) | ⏭️ {run_cached} (cached) | ❌ {run_fail}"
    )
    print(
        f"   • Report:  ✅ {rep_ok} | ⏭️ {rep_skipped} (skipped) | ⏭️ {rep_cached} (cached) | ❌ {rep_fail}"
    )
    print("======================\n")


# -------------------------------
# ngraph helpers (inspect, run, report)
# -------------------------------


def _detect_ngraph_invoke() -> List[str]:
    """Detect ngraph CLI or fall back to python -m ngraph."""
    if shutil.which("ngraph"):
        return ["ngraph"]
    if shutil.which("python3"):
        return ["python3", "-m", "ngraph"]
    if shutil.which("python"):
        return ["python", "-m", "ngraph"]
    die(
        "Neither 'ngraph' nor Python found on PATH. Activate your venv or install ngraph."
    )


def _run_to_log(cmd: List[str], cwd: Path, log_path: Path) -> int:
    """Run a command, writing stdout/stderr to a log file; return the exit code."""
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as fh:
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT
            )
            return proc.wait()
        except Exception as e:
            try:
                fh.write(f"\n❌ runner.py: failed to invoke: {' '.join(cmd)}\n{e}\n")
            except Exception:
                pass
            return 1


def _scenario_io_paths(scn_yaml: Path) -> Tuple[Path, str, Path, Path, Path]:
    """Return absolute paths: (dir, stem, results_json, html_path, ipynb_path) for a scenario YAML."""
    scn_yaml_abs = scn_yaml.resolve()
    scn_dir = scn_yaml_abs.parent
    scn_name = scn_yaml_abs.name
    scn_stem = scn_name[: scn_name.rfind(".")]
    results_json = scn_dir / f"{scn_stem}.results.json"
    html_path = scn_dir / f"{scn_stem}.results.html"
    ipynb_path = scn_dir / f"{scn_stem}.results.ipynb"
    return scn_dir, scn_stem, results_json, html_path, ipynb_path


def _has_cached(scn_dir: Path, results_json: Path, html_path: Path) -> bool:
    return results_json.exists() and html_path.exists()


def _inspect_run_one(
    ngraph_invoke: List[str], scn_yaml: Path, force: bool
) -> Tuple[Path, str, str]:
    scn_dir, scn_stem, results_json, html_path, _ = _scenario_io_paths(scn_yaml)
    scn_yaml_abs = scn_yaml.resolve()
    log_ins = scn_dir / f"{scn_stem}.inspect.log"
    log_run = scn_dir / f"{scn_stem}.run.log"

    if not force and _has_cached(scn_dir, results_json, html_path):
        return scn_yaml, "⏭️ cached", "⏭️ cached"

    # Inspect
    ec_ins = _run_to_log(
        ngraph_invoke + ["inspect", "-o", str(scn_dir), str(scn_yaml_abs)],
        scn_dir,
        log_ins,
    )
    if ec_ins != 0:
        return scn_yaml, "❌", "⏭️ skipped"

    # Run
    ec_run = _run_to_log(
        ngraph_invoke
        + ["run", "-o", str(scn_dir), "-r", str(results_json), str(scn_yaml_abs)],
        scn_dir,
        log_run,
    )
    if ec_run != 0:
        return scn_yaml, "✅", "❌"
    return scn_yaml, "✅", "✅"


def _report_one(ngraph_invoke: List[str], scn_yaml: Path, force: bool) -> str:
    scn_dir, scn_stem, results_json, html_path, ipynb_path = _scenario_io_paths(
        scn_yaml
    )
    log_rep = scn_dir / f"{scn_stem}.report.log"
    if not force and _has_cached(scn_dir, results_json, html_path):
        return "⏭️ cached"
    if not results_json.exists():
        return "⏭️ skipped"
    ec_rep = _run_to_log(
        ngraph_invoke
        + [
            "report",
            "-o",
            str(scn_dir),
            str(results_json),
            "--html",
            str(html_path),
            "--notebook",
            str(ipynb_path),
        ],
        scn_dir,
        log_rep,
    )
    return "✅" if ec_rep == 0 else "❌"


if __name__ == "__main__":
    main()
