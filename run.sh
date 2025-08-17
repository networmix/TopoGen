#!/usr/bin/env bash

# Execute ngraph on generated scenarios: inspect → run → report --html
#
# Usage:
#   ./run.sh [--include PATTERN ...] [--exclude PATTERN ...] [--force] <root_dir>
#
# Examples:
#   ./run.sh scenarios
#   ./run.sh --include "*_scenario.yml" scenarios
#   ./run.sh --exclude "*clos*" scenarios
#   ./run.sh --include "small_*" --exclude "*clos*" scenarios
#   ./run.sh --force scenarios
#
# Behavior:
# - Recursively finds *_scenario.yml / *_scenario.yaml files under <root_dir>.
# - For each scenario, runs in its directory:
#     ngraph inspect -o <dir> <file>
#     ngraph run -o <dir> <file>
#     ngraph report -o <dir> <results.json> --html --notebook <path>
# - Logs for each step are saved next to the scenario file:
#     <stem>.inspect.log, <stem>.run.log, <stem>.report.log
# - If --force is not set and cached artifacts are found (results JSON + HTML),
#   the scenario is marked cached and steps are skipped.

set -u -o pipefail

die() {
  echo "❌ $*" >&2
  exit 1
}

# Resolve an absolute path without relying on realpath.
abs_path() {
  local p="$1"
  local dir base
  dir=$(cd "$(dirname -- "$p")" && pwd) || return 1
  base=$(basename -- "$p")
  printf '%s/%s' "$dir" "$base"
}

# Option parsing
INCLUDE_PATTERNS=()
EXCLUDE_PATTERNS=()
FORCE=false

print_usage() {
  cat >&2 <<EOF
Usage: $0 [--include PATTERN ...] [--exclude PATTERN ...] [--force] <root_dir>

Examples:
  $0 scenarios
  $0 --include "*_scenario.yml" scenarios
  $0 --exclude "*clos*" scenarios
  $0 --include "small_*" --exclude "*clos*" scenarios
  $0 --force scenarios

Notes:
  - PATTERNs are shell globs matched against the scenario basename.
  - Multiple --include patterns act as OR; --exclude patterns remove matches.
  - --force ignores cached artifacts (results JSON + HTML) and re-runs all steps.
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --include)
      shift || true
      [[ ${1-} ]] || die "Missing PATTERN after --include"
      INCLUDE_PATTERNS+=("$1"); shift || true ;;
    --exclude)
      shift || true
      [[ ${1-} ]] || die "Missing PATTERN after --exclude"
      EXCLUDE_PATTERNS+=("$1"); shift || true ;;
    --force)
      FORCE=true; shift || true ;;
    -h|--help)
      print_usage; exit 0 ;;
    --)
      shift; break ;;
    --*)
      die "Unknown option: $1" ;;
    *)
      ARGS+=("$1"); shift || true ;;
  esac
done

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

if [[ ${#ARGS[@]} -ne 1 ]]; then
  print_usage
  exit 2
fi

ROOT_DIR_RAW="${ARGS[0]}"
[[ -d "$ROOT_DIR_RAW" ]] || die "Root directory not found: $ROOT_DIR_RAW"
ROOT_DIR=$(abs_path "$ROOT_DIR_RAW")

# Detect ngraph CLI
NGRAPH_INVOKE=(ngraph)
if ! command -v "${NGRAPH_INVOKE[0]}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    NGRAPH_INVOKE=(python3 -m ngraph)
  elif command -v python >/dev/null 2>&1; then
    NGRAPH_INVOKE=(python -m ngraph)
  else
    die "Neither 'ngraph' nor Python found on PATH. Activate your venv or install ngraph."
  fi
fi

echo "🧭 Scenario batch run"
echo "📂 Root:    ${ROOT_DIR%/.}"
if [[ ${#INCLUDE_PATTERNS[@]} -gt 0 ]]; then
  echo "🔎 Include: ${INCLUDE_PATTERNS[*]}"
else
  echo "🔎 Include: (none)"
fi
if [[ ${#EXCLUDE_PATTERNS[@]} -gt 0 ]]; then
  echo "🔎 Exclude: ${EXCLUDE_PATTERNS[*]}"
else
  echo "🔎 Exclude: (none)"
fi
echo "⚙️  Force:   $FORCE"
echo

passes_filters() {
  # 0 if basename passes filters
  local name="$1"
  if [[ ${#INCLUDE_PATTERNS[@]} -gt 0 ]]; then
    local any=1
    for pat in "${INCLUDE_PATTERNS[@]}"; do
      if [[ $name == $pat ]]; then any=0; break; fi
    done
    if (( any != 0 )); then return 1; fi
  fi
  if [[ ${#EXCLUDE_PATTERNS[@]} -gt 0 ]]; then
    for pat in "${EXCLUDE_PATTERNS[@]}"; do
      if [[ $name == $pat ]]; then return 1; fi
    done
  fi
  return 0
}

has_cached() {
  # Cached if expected results and HTML report exist
  local dir="$1"; local results_prefix="$2"
  if [[ -f "$dir/$results_prefix.json" && -f "$dir/$results_prefix.html" ]]; then
    return 0
  fi
  return 1
}

# Summary accumulators
total=0
ins_ok=0; ins_fail=0; ins_cached=0
run_ok=0; run_fail=0; run_skipped=0; run_cached=0
rep_ok=0; rep_fail=0; rep_skipped=0; rep_cached=0

row_names=()
row_inspect=()
row_run=()
row_report=()
w_name=8; w_ins=7; w_run=3; w_rep=6

# Enumerate scenarios (recursive)
while IFS= read -r -d '' scn; do
  scn_abs=$(abs_path "$scn")
  scn_name=$(basename -- "$scn")
  if ! passes_filters "$scn_name"; then
    continue
  fi
  total=$((total + 1))
  scn_dir=$(cd "$(dirname -- "$scn")" && pwd)
  scn_base_noext=${scn_name%.*}
  scn_stem=${scn_base_noext}
  results_prefix="$scn_stem.results"
  results_json="$scn_dir/$results_prefix.json"
  html_path="$scn_dir/$results_prefix.html"
  ipynb_path="$scn_dir/$results_prefix.ipynb"

  echo "➡️  Scenario: $scn_name"
  echo "   📁 Dir:   $scn_dir"

  log_ins="$scn_dir/$scn_stem.inspect.log"
  log_run="$scn_dir/$scn_stem.run.log"
  log_rep="$scn_dir/$scn_stem.report.log"

  if [[ "$FORCE" == "false" ]] && has_cached "$scn_dir" "$results_prefix"; then
    echo "⏭️  Cached: results + report found, skipping"
    ins_status="⏭️ cached"; run_status="⏭️ cached"; rep_status="⏭️ cached"
    ins_cached=$((ins_cached + 1))
    run_cached=$((run_cached + 1))
    rep_cached=$((rep_cached + 1))
  else
    # Inspect
    (cd "$scn_dir" && "${NGRAPH_INVOKE[@]}" inspect -o "$scn_dir" "$scn_abs") 2>&1 | tee "$log_ins"
    ins_ec=${PIPESTATUS[0]}
    if [[ $ins_ec -eq 0 ]]; then
      ins_status="✅"; ins_ok=$((ins_ok + 1))
      # Run
      (cd "$scn_dir" && "${NGRAPH_INVOKE[@]}" run -o "$scn_dir" -r "$results_json" "$scn_abs") 2>&1 | tee "$log_run"
      run_ec=${PIPESTATUS[0]}
      if [[ $run_ec -eq 0 ]]; then
        run_status="✅"; run_ok=$((run_ok + 1))
        # Report
        # Generate both HTML and Notebook under the scenario directory using the results-derived prefix
        (cd "$scn_dir" && "${NGRAPH_INVOKE[@]}" report -o "$scn_dir" "$results_json" --html "$html_path" --notebook "$ipynb_path") 2>&1 | tee "$log_rep"
        rep_ec=${PIPESTATUS[0]}
        if [[ $rep_ec -eq 0 ]]; then
          rep_status="✅"; rep_ok=$((rep_ok + 1))
        else
          rep_status="❌"; rep_fail=$((rep_fail + 1))
        fi
      else
        run_status="❌"; run_fail=$((run_fail + 1))
        rep_status="⏭️ skipped"; rep_skipped=$((rep_skipped + 1))
      fi
    else
      ins_status="❌"; ins_fail=$((ins_fail + 1))
      run_status="⏭️ skipped"; run_skipped=$((run_skipped + 1))
      rep_status="⏭️ skipped"; rep_skipped=$((rep_skipped + 1))
    fi
  fi

  row_names+=("$scn_name")
  row_inspect+=("$ins_status")
  row_run+=("$run_status")
  row_report+=("$rep_status")
  # Optional: append a machine-readable summary row for consolidation by runner.py
  if [[ -n "${RUNNER_SUMMARY_TSV:-}" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\n" "$scn_dir" "$scn_name" "$ins_status" "$run_status" "$rep_status" >> "$RUNNER_SUMMARY_TSV"
  fi
  # Update widths
  nlen=$(printf %s "$scn_name" | wc -m | tr -d ' ')
  ilen=$(printf %s "$ins_status" | wc -m | tr -d ' ')
  rulen=$(printf %s "$run_status" | wc -m | tr -d ' ')
  replen=$(printf %s "$rep_status" | wc -m | tr -d ' ')
  (( nlen > w_name )) && w_name=$nlen
  (( ilen > w_ins )) && w_ins=$ilen
  (( rulen > w_run )) && w_run=$rulen
  (( replen > w_rep )) && w_rep=$replen

  echo
done < <(find "$ROOT_DIR" -type f \( -name '*_scenario.yml' -o -name '*_scenario.yaml' \) -print0)

if [[ $total -eq 0 ]]; then
  echo "⚠️  No scenario files matched under $ROOT_DIR" >&2
  echo "   Include: ${INCLUDE_PATTERNS[*]:-(none)}" >&2
  echo "   Exclude: ${EXCLUDE_PATTERNS[*]:-(none)}" >&2
  exit 2
fi

echo "======================"
echo "📋 Summary"
printf "%-*s  %-*s  %-*s  %-*s\n" "$w_name" "Scenario" "$w_ins" "Inspect" "$w_run" "Run" "$w_rep" "Report"
dash_len=$((w_name + 2 + w_ins + 2 + w_run + 2 + w_rep))
printf '%*s\n' "$dash_len" '' | tr ' ' '-'
for i in "${!row_names[@]}"; do
  printf "%-*s  %-*s  %-*s  %-*s\n" "$w_name" "${row_names[$i]}" "$w_ins" "${row_inspect[$i]}" "$w_run" "${row_run[$i]}" "$w_rep" "${row_report[$i]}"
done
echo "----------------------"
echo "🧮 Totals: $total scenarios"
echo "   • Inspect: ✅ $ins_ok | ⏭️ $ins_cached (cached) | ❌ $ins_fail"
echo "   • Run:     ✅ $run_ok | ⏭️ $run_skipped (skipped) | ⏭️ $run_cached (cached) | ❌ $run_fail"
echo "   • Report:  ✅ $rep_ok | ⏭️ $rep_skipped (skipped) | ⏭️ $rep_cached (cached) | ❌ $rep_fail"
echo "======================"
echo "Done. ✨"

echo "Done. ✨"
