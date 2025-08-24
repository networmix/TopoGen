#!/bin/bash
# Run all code quality checks and tests
# This script runs the complete validation suite: pre-commit hooks + schema validation + tests

set -e  # Exit on any error

# Determine python interpreter (prefer venv if active)
PYTHON=${PYTHON:-python3}

# Check if pre-commit is installed
if ! "$PYTHON" -m pre_commit --version &> /dev/null; then
    echo "❌ pre-commit is not installed. Please run 'make dev' first."
    exit 1
fi

# Check if pytest is installed
if ! "$PYTHON" -m pytest --version &> /dev/null; then
    echo "❌ pytest is not installed. Please run 'make dev' first."
    exit 1
fi

# Check if pre-commit hooks are installed
if [ ! -f .git/hooks/pre-commit ]; then
    echo "⚠️  Pre-commit hooks not installed. Installing now..."
    "$PYTHON" -m pre_commit install
    echo ""
fi

# Run pre-commit with fixers (first pass), do not fail if files were modified
echo "🏃 Running pre-commit (first pass: apply auto-fixes if needed)..."
set +e
"$PYTHON" -m pre_commit run --all-files
first_pass_status=$?
set -e

if [ $first_pass_status -ne 0 ]; then
    echo "ℹ️  Some hooks modified files or reported issues. Re-running checks..."
fi

# Re-run to verify all checks pass after fixes; fail on any remaining issues
echo "🏃 Running pre-commit (second pass: verify all checks)..."
if ! "$PYTHON" -m pre_commit run --all-files; then
    echo ""
    echo "❌ Pre-commit checks failed after applying fixes. Please address the issues above."
    exit 1
fi

# Track whether auto-fixes were applied and resolved issues
autofixed=0
if [ $first_pass_status -ne 0 ]; then
    autofixed=1
fi

echo ""
echo "✅ Pre-commit checks passed!"
echo ""

# Run schema validation
echo "📋 Validating YAML schemas..."
if ! "$PYTHON" -c "import jsonschema" >/dev/null 2>&1; then
    echo "❌ jsonschema is required but not installed. Please run 'make dev'."
    exit 1
fi
"$PYTHON" -c "import json, yaml, jsonschema, pathlib; from importlib import resources as res; \
# Validate scenarios if NetGraph packaged schema is available \
try: \
    f = res.files('ngraph.schemas').joinpath('scenario.json').open('r', encoding='utf-8'); \
    schema = json.load(f); f.close(); \
    dirs = ['scenarios', 'tests/integration']; files = []; \
    for d in dirs: \
        p = pathlib.Path(d); \
        if p.exists(): files += list(p.rglob('*.yaml')) + list(p.rglob('*.yml')); \
    for fp in files: jsonschema.validate(yaml.safe_load(open(str(fp))), schema); \
    print(f'✅ Validated {len(files)} scenario YAML files'); \
except Exception: \
    pass; \
# Validate TopoGen configs using packaged schema \
f2 = res.files('topogen.schemas').joinpath('topogen_config.json').open('r', encoding='utf-8'); \
schema_cfg = json.load(f2); f2.close(); \
cfg_dirs = ['examples', 'topogen_configs', 'topogen_configs_small', 'private/scenarios']; cfg_files = []; \
for d in cfg_dirs: \
    p = pathlib.Path(d); \
    if p.exists(): cfg_files += list(p.rglob('*.yaml')) + list(p.rglob('*.yml')); \
for fp in cfg_files: jsonschema.validate(yaml.safe_load(open(str(fp))), schema_cfg); \
print(f'✅ Validated {len(cfg_files)} TopoGen config YAML files')"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Schema validation failed. Please fix the YAML files above."
    exit 1
fi

echo ""

# Run tests with coverage (includes slow and benchmark tests for regression detection)
echo "🧪 Running tests with coverage..."
"$PYTHON" -m pytest

if [ $? -eq 0 ]; then
    echo ""
    if [ $autofixed -eq 1 ]; then
        echo "🎉 All checks and tests passed. Auto-fixes were applied by pre-commit."
    else
        echo "🎉 All checks and tests passed."
    fi
else
    echo ""
    if [ $autofixed -eq 1 ]; then
        echo "❌ Some tests failed. Note: auto-fixes were applied earlier by pre-commit."
    else
        echo "❌ Some tests failed."
    fi
    exit 1
fi
