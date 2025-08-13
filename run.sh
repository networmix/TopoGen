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

#!/usr/bin/env bash

# Batch runner for TopoGen: run generate+build for each config in a folder.
#
# Usage:
#   ./build.sh [--include PATTERN ...] [--exclude PATTERN ...] [--force] <configs_dir> <output_dir>
#
# Example:
#   ./build.sh examples scenarios
#   ./build.sh --include "small_*" --exclude "*clos*" examples scenarios
#   ./build.sh --force examples scenarios
#
# Behavior:
# - Finds .yml/.yaml files directly under <configs_dir> (no recursion).
# - For each config, creates <output_dir>/<config_stem>/ and runs both stages
#   in that directory so artefacts are kept together:
#     <config_stem>_integrated_graph.json
#     <config_stem>_scenario.yml
#     generate.log, build.log
# - Prints a concise emoji summary at the end.

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

# Parse options: --include/--exclude support multiple occurrences
INCLUDE_PATTERNS=()
EXCLUDE_PATTERNS=()
FORCE=false

print_usage() {
  cat >&2 <<EOF
Usage: $0 [--include PATTERN ...] [--exclude PATTERN ...] [--force] <configs_dir> <output_dir>

Examples:
  $0 examples scenarios
  $0 --include "small_*" examples scenarios
  $0 --exclude "*clos*" examples scenarios
  $0 --include "small_*" --exclude "*clos*" examples scenarios
  $0 --force examples scenarios

Notes:
  - PATTERNs are shell globs matched against the config file basename (e.g., small_test.yml).
  - Multiple --include patterns act as OR; --exclude patterns remove matches.
  - --force ignores cached integrated graphs and runs generation unconditionally.
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

# Append any remaining args
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

if [[ ${#ARGS[@]} -ne 2 ]]; then
  print_usage
  exit 2
fi

CONFIGS_DIR_RAW="${ARGS[0]}"
OUTPUT_DIR_RAW="${ARGS[1]}"

[[ -d "$CONFIGS_DIR_RAW" ]] || die "Configs directory not found: $CONFIGS_DIR_RAW"
mkdir -p "$OUTPUT_DIR_RAW" || die "Cannot create output directory: $OUTPUT_DIR_RAW"

# Canonical absolute paths
CONFIGS_DIR=$(abs_path "$CONFIGS_DIR_RAW")
OUTPUT_DIR=$(abs_path "$OUTPUT_DIR_RAW")

# Detect TopoGen invoker once: prefer installed CLI, fallback to python -m.
TOPGEN_INVOKE=(topogen)
if ! command -v "${TOPGEN_INVOKE[0]}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    TOPGEN_INVOKE=(python3 -m topogen)
  elif command -v python >/dev/null 2>&1; then
    TOPGEN_INVOKE=(python -m topogen)
  else
    die "Neither 'topogen' nor Python found on PATH. Activate your venv or install TopoGen."
  fi
fi

# Resolve project root as the directory of this script for relative data paths
SCRIPT_DIR=$(cd "$(dirname -- "$0")" && pwd)

echo "🚀 TopoGen batch run"
echo "📁 Configs: $CONFIGS_DIR"
OUTPUT_DIR_PRINT=${OUTPUT_DIR%/.}
CONFIGS_DIR_PRINT=${CONFIGS_DIR%/.}
echo "📁 Configs: $CONFIGS_DIR_PRINT"
echo "📂 Output:  $OUTPUT_DIR_PRINT"
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

# Collect summary statistics
total=0
gen_ok=0
gen_fail=0
gen_config=0
gen_cached=0
build_ok=0
build_validation=0
build_runtime=0
build_config=0
build_skipped=0
build_other=0

summary_lines=()
row_names=()
row_gen=()
row_build=()
w_name=6   # min width for header 'Config'
w_gen=8    # min width for header 'Generate'
w_build=5  # min width for header 'Build'

passes_filters() {
  # return 0 if basename passes include/exclude filters, else 1
  local name="$1"
  # Includes: if provided, require any to match
  if [[ ${#INCLUDE_PATTERNS[@]} -gt 0 ]]; then
    local any=1
    for pat in "${INCLUDE_PATTERNS[@]}"; do
      if [[ $name == $pat ]]; then any=0; break; fi
    done
    if (( any != 0 )); then return 1; fi
  fi
  # Excludes: drop if any matches
  if [[ ${#EXCLUDE_PATTERNS[@]} -gt 0 ]]; then
    for pat in "${EXCLUDE_PATTERNS[@]}"; do
      if [[ $name == $pat ]]; then return 1; fi
    done
  fi
  return 0
}

# Enumerate configs (non-recursive)
while IFS= read -r -d '' cfg; do
  cfg_abs=$(abs_path "$cfg")
  cfg_name=$(basename -- "$cfg")
  # Apply include/exclude filters on basename
  if ! passes_filters "$cfg_name"; then
    continue
  fi
  total=$((total + 1))
  stem=${cfg_name%.*}
  workdir="$OUTPUT_DIR/$stem"
  mkdir -p "$workdir" || die "Cannot create work directory: $workdir"

  echo "➡️  Processing $cfg_name"
  echo "   📦 Workdir: $(abs_path "$workdir")"

  graph_work="$workdir/${stem}_integrated_graph.json"
  graph_repo="$SCRIPT_DIR/${stem}_integrated_graph.json"
  vis_repo="$SCRIPT_DIR/${stem}_integrated_graph.jpg"

  gen_ec=0
  if [[ "$FORCE" == "true" ]]; then
    run_generate=true
  else
    if [[ -f "$graph_work" || -f "$graph_repo" ]]; then
      run_generate=false
    else
      run_generate=true
    fi
  fi

  if [[ "$run_generate" == "true" ]]; then
    # Run generate in project root so relative data paths in configs work
    # Artefacts are produced in project root; we will move them to workdir
    (cd "$SCRIPT_DIR" && "${TOPGEN_INVOKE[@]}" generate "$cfg_abs") 2>&1 | tee "$workdir/generate.log"
    gen_ec=${PIPESTATUS[0]}
  else
    # Use cached artefacts
    if [[ -f "$graph_repo" && ! -f "$graph_work" ]]; then
      mv -f "$graph_repo" "$workdir/" || die "Failed to move integrated graph to $workdir"
    fi
    # Move optional visualization if present
    if [[ -f "$vis_repo" && ! -f "$workdir/${stem}_integrated_graph.jpg" ]]; then
      mv -f "$vis_repo" "$workdir/" || true
    fi
    echo "⏭️  Skipping generate: found existing ${stem}_integrated_graph.json" | tee "$workdir/generate.log" >/dev/null
    gen_ec=100  # special code for 'cached'
  fi

  if [[ $gen_ec -eq 0 ]]; then
    gen_icon="✅"
    gen_ok=$((gen_ok + 1))
  elif [[ $gen_ec -eq 2 ]]; then
    gen_icon="❌"
    gen_config=$((gen_config + 1))
  elif [[ $gen_ec -eq 100 ]]; then
    gen_icon="⏭️"
    gen_cached=$((gen_cached + 1))
  else
    gen_icon="❌"
    gen_fail=$((gen_fail + 1))
  fi

  # Run build only if generate succeeded
  build_icon="⏭️"
  build_note="skipped"
  build_ec=-1
  scenario_out="$workdir/${stem}_scenario.yml"
  if [[ $gen_ec -eq 0 || $gen_ec -eq 100 ]]; then
    # Move integrated graph (and optional visualization) from project root to workdir
    graph_src="$SCRIPT_DIR/${stem}_integrated_graph.json"
    if [[ -f "$graph_src" && ! -f "$workdir/${stem}_integrated_graph.json" ]]; then
      mv -f "$graph_src" "$workdir/" || die "Failed to move integrated graph to $workdir"
    fi
    vis_src="$SCRIPT_DIR/${stem}_integrated_graph.jpg"
    if [[ -f "$vis_src" && ! -f "$workdir/${stem}_integrated_graph.jpg" ]]; then
      mv -f "$vis_src" "$workdir/" || true
    fi

    # Run build in workdir so it finds the integrated graph by prefix
    pushd "$workdir" >/dev/null || die "Cannot enter $workdir"
    ("${TOPGEN_INVOKE[@]}" build "$cfg_abs" -o "$scenario_out") 2>&1 | tee "$workdir/build.log"
    build_ec=${PIPESTATUS[0]}
    case "$build_ec" in
      0)
        build_icon="✅"; build_note="ok"; build_ok=$((build_ok + 1));;
      3)
        build_icon="⚠️"; build_note="validation failed"; build_validation=$((build_validation + 1));;
      2)
        build_icon="❌"; build_note="config error"; build_config=$((build_config + 1));;
      1)
        build_icon="❌"; build_note="runtime error"; build_runtime=$((build_runtime + 1));;
      *)
        build_icon="❌"; build_note="exit $build_ec"; build_other=$((build_other + 1));;
    esac
    popd >/dev/null || true
  else
    build_skipped=$((build_skipped + 1))
  fi

  # Summary line for this config
  gen_col="$gen_icon"
  if [[ $gen_ec -eq 2 ]]; then gen_col="❌ config"; fi
  if [[ $gen_ec -eq 100 ]]; then gen_col="⏭️ cached"; fi
  if [[ $gen_ec -ne 0 && $gen_ec -ne 2 && $gen_ec -ne 100 ]]; then gen_col="❌ runtime"; fi

  build_col="$build_icon"
  case "$build_ec" in
    0) build_col="✅" ;;
    3) build_col="⚠️ validation" ;;
    2) build_col="❌ config" ;;
    1) build_col="❌ runtime" ;;
   -1) build_col="⏭️ skipped" ;;
     *) build_col="❓ other" ;;
  esac

  row_names+=("$cfg_name")
  row_gen+=("$gen_col")
  row_build+=("$build_col")
  # Update column widths (character counts)
  name_len=$(printf %s "$cfg_name" | wc -m | tr -d ' ')
  gen_len=$(printf %s "$gen_col" | wc -m | tr -d ' ')
  build_len=$(printf %s "$build_col" | wc -m | tr -d ' ')
  (( name_len > w_name )) && w_name=$name_len
  (( gen_len > w_gen )) && w_gen=$gen_len
  (( build_len > w_build )) && w_build=$build_len

  echo
done < <(find "$CONFIGS_DIR" -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) -print0)

if [[ $total -eq 0 ]]; then
  echo "⚠️  No YAML config files matched in $CONFIGS_DIR" >&2
  echo "   Include: ${INCLUDE_PATTERNS[*]:-(none)}" >&2
  echo "   Exclude: ${EXCLUDE_PATTERNS[*]:-(none)}" >&2
  exit 2
fi

echo "======================"
echo "📋 Summary"
printf "%-*s  %-*s  %-*s\n" "$w_name" "Config" "$w_gen" "Generate" "$w_build" "Build"
dash_len=$((w_name + 2 + w_gen + 2 + w_build))
printf '%*s\n' "$dash_len" '' | tr ' ' '-'
for i in "${!row_names[@]}"; do
  printf "%-*s  %-*s  %-*s\n" "$w_name" "${row_names[$i]}" "$w_gen" "${row_gen[$i]}" "$w_build" "${row_build[$i]}"
done
echo "----------------------"
echo "🧮 Totals: $total configs"
echo "   • Generate: ✅ $gen_ok  | ⏭️ $gen_cached (cached)  | ❌ $gen_fail (runtime)  | ❌ $gen_config (config)"
echo "   • Build:    ✅ $build_ok  | ⚠️ $build_validation (validation)  | ❌ $build_runtime (runtime)  | ❌ $build_config (config)  | ⏭️ $build_skipped (skipped)  | ❓ $build_other (other)"
echo "======================"

echo "Done. ✨"
