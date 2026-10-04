#!/usr/bin/env bash

# Batch runner for TopoGen: run generate+build for each config in a folder.
#
# Usage:
#   ./build.sh [--include PATTERN ...] [--exclude PATTERN ...] [--force] [--build-only] <configs_dir> <output_dir>
#
# Example:
#   ./build.sh examples scenarios
#   ./build.sh --include "small_*" --exclude "*clos*" examples scenarios
#   ./build.sh --force examples scenarios
#   ./build.sh --build-only examples scenarios
#
# Behavior:
# - Finds .yml/.yaml files directly under <configs_dir> (no recursion).
# - Writes each config's artifacts under <output_dir>/<config_stem>/:
#     <config_stem>_integrated_graph.json
#     <config_stem>_scenario.yml
#     generate.log, build.log

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
BUILD_ONLY=false

print_usage() {
  cat >&2 <<EOF
Usage: $0 [--include PATTERN ...] [--exclude PATTERN ...] [--force] [--build-only] <configs_dir> <output_dir>

Examples:
  $0 examples scenarios
  $0 --include "small_*" examples scenarios
  $0 --exclude "*clos*" examples scenarios
  $0 --include "small_*" --exclude "*clos*" examples scenarios
  $0 --force examples scenarios
  $0 --build-only examples scenarios

Notes:
  - PATTERNs are shell globs matched against the config file basename (e.g., small_test.yml).
  - Multiple --include patterns act as OR; --exclude patterns remove matches.
  - --force ignores cached integrated graphs and runs generation unconditionally.
  - --build-only skips the generate stage and runs only the build stage.
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
    --build-only)
      BUILD_ONLY=true; shift || true ;;
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

# Use the installed CLI from the active environment.
command -v topogen >/dev/null 2>&1 || die "Activate the TopoGen venv or install TopoGen."

echo "🚀 TopoGen batch run"
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
echo "🏗️  Build-only: $BUILD_ONLY"
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

  gen_ec=0
  if [[ "$BUILD_ONLY" == "true" ]]; then
    # --build-only requires a saved graph.
    echo "⏭️  Skipping generate due to --build-only" | tee "$workdir/generate.log" >/dev/null
    gen_ec=100  # treat as cached/ready so build runs
  else
    if [[ "$FORCE" == "true" ]]; then
      run_generate=true
    else
      if [[ -f "$graph_work" ]]; then
        run_generate=false
      else
        run_generate=true
      fi
    fi

    if [[ "$run_generate" == "true" ]]; then
      # Run generate writing artefacts directly to workdir via -o
      topogen generate "$cfg_abs" -o "$workdir" 2>&1 | tee "$workdir/generate.log"
      gen_ec=${PIPESTATUS[0]}
    else
      # Use cached artefacts
      echo "⏭️  Skipping generate: found existing ${stem}_integrated_graph.json" | tee "$workdir/generate.log" >/dev/null
      gen_ec=100  # special code for 'cached'
    fi
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

  # Build after successful generation or when a saved graph is available.
  build_ec=-1
  scenario_out="$workdir/${stem}_scenario.yml"
  if [[ $gen_ec -eq 0 || $gen_ec -eq 100 ]]; then
    # Run build writing scenario into workdir via -o
    topogen build "$cfg_abs" -o "$scenario_out" 2>&1 | tee "$workdir/build.log"
    build_ec=${PIPESTATUS[0]}
    case "$build_ec" in
      0)
        build_ok=$((build_ok + 1));;
      3)
        build_validation=$((build_validation + 1));;
      2)
        build_config=$((build_config + 1));;
      1)
        build_runtime=$((build_runtime + 1));;
      *)
        build_other=$((build_other + 1));;
    esac
  else
    build_skipped=$((build_skipped + 1))
  fi

  # Summary line for this config
  gen_col="$gen_icon"
  if [[ $gen_ec -eq 2 ]]; then gen_col="❌ config"; fi
  if [[ $gen_ec -eq 100 ]]; then gen_col="⏭️ cached"; fi
  if [[ $gen_ec -ne 0 && $gen_ec -ne 2 && $gen_ec -ne 100 ]]; then gen_col="❌ runtime"; fi

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

if (( build_ok != total )); then
  echo "Batch failed: $((total - build_ok)) config(s) did not build successfully." >&2
  exit 1
fi
echo "Done. ✨"
