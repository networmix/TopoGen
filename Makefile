# TopoGen Development Makefile (aligned with NetGraph / NetGraph-Core / NetLab)

.PHONY: help venv clean-venv dev install check check-ci lint format test qt build clean check-dist publish-test publish validate info hooks check-python

.DEFAULT_GOAL := help

# --------------------------------------------------------------------------
# Python interpreter detection
# --------------------------------------------------------------------------
VENV_BIN := $(PWD)/venv/bin

# Supports 3.11-3.13 to match requires-python >=3.11
PY_BEST := $(shell for v in 3.13 3.12 3.11; do command -v python$$v >/dev/null 2>&1 && { echo python$$v; exit 0; }; done; command -v python3 2>/dev/null || command -v python 2>/dev/null)
PY_PATH := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
PYTHON ?= $(if $(wildcard $(VENV_BIN)/python),$(VENV_BIN)/python,$(if $(PY_PATH),$(PY_PATH),$(if $(PY_BEST),$(PY_BEST),python3)))

PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
PRECOMMIT := $(PYTHON) -m pre_commit
PYRIGHT := $(PYTHON) -m pyright

help:
	@echo "🔧 TopoGen Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make venv          - Create a local virtualenv (./venv)"
	@echo "  make dev           - Full development environment (package + dev deps + hooks)"
	@echo "  make install       - Install package for usage (no dev dependencies)"
	@echo "  make clean-venv    - Remove virtual environment"
	@echo ""
	@echo "Code Quality & Testing:"
	@echo "  make check         - Run pre-commit + validation + tests, then lint"
	@echo "  make check-ci      - Run non-mutating lint + validation + tests (CI entrypoint)"
	@echo "  make lint          - Run only linting (non-mutating: ruff + pyright)"
	@echo "  make format        - Auto-format code with ruff"
	@echo "  make validate      - Validate YAML configs against schemas"
	@echo "  make test          - Run tests with coverage (includes slow and benchmark)"
	@echo "  make qt            - Run quick tests only (excludes slow and benchmark)"
	@echo ""
	@echo "Build & Package:"
	@echo "  make build         - Build distribution packages"
	@echo "  make clean         - Clean build artifacts and cache files"
	@echo ""
	@echo "Publishing:"
	@echo "  make check-dist    - Check distribution packages with twine"
	@echo "  make publish-test  - Publish to Test PyPI"
	@echo "  make publish       - Publish to PyPI"
	@echo ""
	@echo "Utilities:"
	@echo "  make hooks         - Run pre-commit on all files"
	@echo "  make check-python  - Check if venv Python matches best available"
	@echo "  make info          - Show project information"

# Setup and Installation

dev:
	@echo "🚀 Setting up development environment..."
	@if [ ! -x "$(VENV_BIN)/python" ]; then \
		if [ -z "$(PY_BEST)" ]; then \
			echo "❌ Error: No Python interpreter found (python3 or python)"; \
			exit 1; \
		fi; \
		echo "🐍 Creating virtual environment with $(PY_BEST) ..."; \
		$(PY_BEST) -m venv venv || { echo "❌ Failed to create venv"; exit 1; }; \
		if [ ! -x "$(VENV_BIN)/python" ]; then \
			echo "❌ Error: venv creation failed - $(VENV_BIN)/python not found"; \
			exit 1; \
		fi; \
		$(VENV_BIN)/python -m pip install -U pip setuptools wheel; \
	fi
	@echo "📦 Installing dev dependencies..."
	@$(VENV_BIN)/python -m pip install -e .'[dev]'
	@echo "🔗 Installing pre-commit hooks..."
	@$(VENV_BIN)/python -m pre_commit install --install-hooks
	@echo "✅ Dev environment ready. Activate with: source venv/bin/activate"
	@$(MAKE) check-python

venv:
	@echo "🐍 Creating virtual environment in ./venv ..."
	@if [ -z "$(PY_BEST)" ]; then \
		echo "❌ Error: No Python interpreter found (python3 or python)"; \
		exit 1; \
	fi
	@$(PY_BEST) -m venv venv || { echo "❌ Failed to create venv"; exit 1; }
	@if [ ! -x "$(VENV_BIN)/python" ]; then \
		echo "❌ Error: venv creation failed - $(VENV_BIN)/python not found"; \
		exit 1; \
	fi
	@$(VENV_BIN)/python -m pip install -U pip setuptools wheel
	@echo "✅ venv ready. Activate with: source venv/bin/activate"

clean-venv:
	@rm -rf venv/

install:
	@echo "📦 Installing package for usage (no dev dependencies)..."
	@$(PIP) install -e .

# Code Quality and Testing
check:
	@echo "🔍 Running complete code quality checks and tests..."
	@$(PRECOMMIT) run --all-files || true
	@$(PRECOMMIT) run --all-files
	@$(MAKE) validate
	@$(MAKE) test
	@$(MAKE) lint

check-ci:
	@echo "🔍 Running CI checks (non-mutating lint + schema validation + tests)..."
	@$(MAKE) lint
	@$(MAKE) validate
	@$(MAKE) test

lint:
	@echo "🧹 Running linting checks (non-mutating)..."
	@$(RUFF) format --check .
	@$(RUFF) check .
	@$(PYRIGHT)

format:
	@echo "✨ Auto-formatting code..."
	@$(RUFF) format .

test:
	@echo "🧪 Running tests with coverage (includes slow and benchmark)..."
	@$(PYTEST)

qt:
	@echo "⚡ Running quick tests only (excludes slow and benchmark)..."
	@$(PYTEST) --no-cov -m "not slow and not benchmark"

validate:
	@echo "📋 Validating TopoGen config YAMLs..."
	@if $(PYTHON) -c "import jsonschema" >/dev/null 2>&1; then \
		$(PYTHON) -c "import json, yaml, jsonschema, pathlib; from importlib import resources as res; f=res.files('topogen.schemas').joinpath('topogen_config.json').open('r', encoding='utf-8'); schema=json.load(f); f.close(); cfg_dirs=['examples']; cfg_files=[]; [cfg_files.extend(list(pathlib.Path(d).rglob('*.yaml'))+list(pathlib.Path(d).rglob('*.yml'))) for d in cfg_dirs if pathlib.Path(d).exists()]; [jsonschema.validate(yaml.safe_load(open(fp)), schema) for fp in cfg_files]; print(f'✅ Validated {len(cfg_files)} TopoGen config YAML files')"; \
	else \
		echo "⚠️  jsonschema not installed. Skipping schema validation"; \
	fi

# Build and Package
build:
	@echo "🏗️  Building distribution packages..."
	@if $(PYTHON) -c "import build" >/dev/null 2>&1; then \
		$(PYTHON) -m build; \
	else \
		echo "❌ build module not installed. Install dev dependencies with: make dev"; \
		exit 1; \
	fi

clean:
	@echo "🧹 Cleaning build artifacts and cache files..."
	@rm -rf build/ dist/ *.egg-info/ || true
	@rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov htmlcov-python .coverage coverage.xml coverage-*.xml .benchmarks .pytest-benchmark || true
	@find . -path "./venv" -prune -o -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -path "./venv" -prune -o -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -path "./venv" -prune -o -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -path "./venv" -prune -o -type f -name "*~" -delete 2>/dev/null || true
	@find . -path "./venv" -prune -o -type f -name "*.orig" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Publishing
check-dist:
	@echo "🔍 Checking distribution packages..."
	@if $(PYTHON) -c "import twine" >/dev/null 2>&1; then \
		$(PYTHON) -m twine check dist/*; \
	else \
		echo "❌ twine not installed. Install dev dependencies with: make dev"; \
		exit 1; \
	fi

publish-test:
	@echo "📦 Publishing to Test PyPI..."
	@if $(PYTHON) -c "import twine" >/dev/null 2>&1; then \
		$(PYTHON) -m twine upload --repository testpypi dist/*; \
	else \
		echo "❌ twine not installed. Install dev dependencies with: make dev"; \
		exit 1; \
	fi

publish:
	@echo "🚀 Publishing to PyPI..."
	@if $(PYTHON) -c "import twine" >/dev/null 2>&1; then \
		$(PYTHON) -m twine upload dist/*; \
	else \
		echo "❌ twine not installed. Install dev dependencies with: make dev"; \
		exit 1; \
	fi

# Project Information
info:
	@echo "📋 TopoGen Project Information"
	@echo "================================"
	@echo ""
	@echo "🐍 Python Environment:"
	@echo "  Python (active): $$($(PYTHON) --version)"
	@echo "  Python (best):   $$($(PY_BEST) --version 2>/dev/null || echo 'missing')"
	@$(MAKE) check-python
	@echo "  Package version: $$($(PYTHON) -c 'import importlib.metadata; print(importlib.metadata.version("topogen"))' 2>/dev/null || echo 'Not installed')"
	@echo "  Virtual environment: $$(echo $$VIRTUAL_ENV | sed 's|.*/||' || echo 'None active')"
	@echo ""
	@echo "🔧 Development Tools:"
	@echo "  Pre-commit: $$($(PRECOMMIT) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Pytest: $$($(PYTEST) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Ruff: $$($(RUFF) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Pyright: $$($(PYTHON) -m pyright --version 2>/dev/null | head -1 || echo 'Not installed')"
	@echo "  Build: $$($(PYTHON) -m build --version 2>/dev/null | sed 's/build //' | sed 's/ (.*//' || echo 'Not installed')"
	@echo "  Twine: $$($(PYTHON) -m twine --version 2>/dev/null | grep -o 'twine version [0-9.]*' | cut -d' ' -f3 || echo 'Not installed')"
	@echo ""
	@echo "📂 Git Repository:"
	@echo "  Current branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "  Status: $$(git status --porcelain | wc -l | tr -d ' ') modified files"
	@if [ "$$(git status --porcelain | wc -l | tr -d ' ')" != "0" ]; then \
		echo "  Modified files:"; \
		git status --porcelain | head -5 | sed 's/^/    /'; \
		if [ "$$(git status --porcelain | wc -l | tr -d ' ')" -gt "5" ]; then \
			echo "    ... and $$(( $$(git status --porcelain | wc -l | tr -d ' ') - 5 )) more"; \
		fi; \
	fi

hooks:
	@echo "🔗 Running pre-commit on all files..."
	@$(PRECOMMIT) run --all-files || (echo "Some pre-commit hooks failed. Fix and re-run." && exit 1)

check-python:
	@if [ -x "$(VENV_BIN)/python" ]; then \
		VENV_VER=$$($(VENV_BIN)/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown"); \
		BEST_VER=$$($(PY_BEST) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown"); \
		if [ -n "$$VENV_VER" ] && [ -n "$$BEST_VER" ] && [ "$$VENV_VER" != "$$BEST_VER" ]; then \
			echo "⚠️  WARNING: venv Python ($$VENV_VER) != best available Python ($$BEST_VER)"; \
			echo "   Run 'make clean-venv && make dev' to recreate venv if desired"; \
		fi; \
	fi
