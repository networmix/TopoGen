#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode=${1:-help}
case "$mode" in
    setup|check|teardown) ;;
    *) echo 'Usage: bash .superset/workspace.sh {setup|check|teardown}' >&2; exit 2 ;;
esac

# Setup starts no background services and reserves no ports.
if [[ "$mode" == teardown ]]; then
    echo 'TopoGen has no workspace services to stop.'
    exit 0
fi

# Never inherit another workspace's Python import paths.
unset PYTHONPATH PYTHONHOME

if [[ "$mode" == setup ]]; then
    # Superset supplies this path. The fallback also supports manual invocation.
    root=${SUPERSET_ROOT_PATH:-$(git rev-parse --path-format=absolute --git-common-dir)/..}
    root=$(cd "$root" && pwd -P)
    if [[ "$root" != "$(pwd -P)" ]]; then
        for source in "$root"/.env "$root"/.env.*; do
            [[ -f "$source" ]] || continue
            name=${source##*/}
            case "$name" in
                *.example|*.sample|*.template) continue ;;
            esac
            # Only copy local settings; preserve workspace edits on repeated setup.
            if git -C "$root" ls-files --error-unmatch -- "$name" >/dev/null 2>&1; then
                continue
            fi
            if [[ ! -e "$name" && ! -L "$name" ]]; then
                (umask 077; cp "$source" "$name")
                echo "Copied $name from the root checkout."
            fi
        done
    fi

    if [[ ! -x venv/bin/python ]]; then
        # Select an interpreter covered by CI.
        setup_python=${SUPERSET_PYTHON:-}
        if [[ -z "$setup_python" ]]; then
            for candidate in python3.13 python3.12 python3.11 python3; do
                if command -v "$candidate" >/dev/null 2>&1 &&
                    "$candidate" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))' 2>/dev/null; then
                    setup_python=$(command -v "$candidate")
                    break
                fi
            done
        fi
        if [[ -z "$setup_python" ]] && command -v uv >/dev/null 2>&1; then
            uv python install 3.13
            setup_python=$(uv python find 3.13)
        fi
        if [[ -z "$setup_python" ]]; then
            echo 'Install Python 3.11–3.13 (or uv), or set SUPERSET_PYTHON to its executable.' >&2
            exit 1
        fi
        "$setup_python" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))' || {
            echo 'SUPERSET_PYTHON must select Python 3.11–3.13.' >&2
            exit 1
        }
        "$setup_python" -m venv venv
        venv/bin/python -m pip install --upgrade pip setuptools wheel
    fi
fi

[[ -x venv/bin/python ]] || { echo 'Run bash .superset/workspace.sh setup first.' >&2; exit 1; }
export VIRTUAL_ENV="$PWD/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if [[ "$mode" == setup ]]; then
    python -m pip install -e '.[dev]'
    python -m pip check
    python -c 'import topogen, ngraph, netgraph_core; print("Workspace environment ready")'
    echo 'Activate with: source venv/bin/activate'
else
    make check-ci
fi
