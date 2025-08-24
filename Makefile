# TopoGen Development Makefile
# This Makefile provides convenient shortcuts for common development tasks

.PHONY: help dev install check check-ci lint format test qt clean build check-dist publish-test publish validate info

# Default target - show help
.DEFAULT_GOAL := help

# Toolchain
VENV_BIN := $(CURDIR)/topogen-venv/bin
PYTHON = $(if $(wildcard $(VENV_BIN)/python),$(VENV_BIN)/python,python3)
PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest
RUFF = $(PYTHON) -m ruff
PRECOMMIT = $(PYTHON) -m pre_commit
PYRIGHT = $(PYTHON) -m pyright

help:
	@echo "🔧 TopoGen Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       - Install package for usage (no dev dependencies)"
	@echo "  make dev           - Full development environment (package + dev deps + hooks)"
	@echo ""
	@echo "Code Quality & Testing:"
	@echo "  make check         - Run lint + pre-commit + tests (includes slow and benchmark)"
	@echo "  make check-ci      - Run non-mutating checks and tests (CI entrypoint)"
	@echo "  make lint          - Run only linting (non-mutating: ruff + pyright)"
	@echo "  make format        - Auto-format code with ruff"
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
	@echo "  make info          - Show project information"

# Setup and Installation
dev:
	@echo "🚀 Setting up development environment..."
	@bash dev/setup-dev.sh

install:
	@echo "📦 Installing package for usage (no dev dependencies)..."
	@$(PIP) install -e .

# Code Quality and Testing
check:
	@echo "🔍 Running complete code quality checks and tests..."
	@$(MAKE) lint
	@PYTHON="$(PYTHON)" bash dev/run-checks.sh

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
		$(PYTHON) -c "import json, yaml, jsonschema, pathlib; from importlib import resources as res; f=res.files('topogen.schemas').joinpath('topogen_config.json').open('r', encoding='utf-8'); schema=json.load(f); f.close(); cfg_dirs=['examples','topogen_configs']; cfg_files=[]; [cfg_files.extend(list(pathlib.Path(d).rglob('*.yaml'))+list(pathlib.Path(d).rglob('*.yml'))) for d in cfg_dirs if pathlib.Path(d).exists()]; [jsonschema.validate(yaml.safe_load(open(fp)), schema) for fp in cfg_files]; print(f'✅ Validated {len(cfg_files)} TopoGen config YAML files')"; \
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
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*~" -delete
	@find . -type f -name "*.orig" -delete
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
	@echo "  Python version: $$($(PYTHON) --version)"
	@echo "  Package version: $$($(PYTHON) -c 'import importlib.metadata; print(importlib.metadata.version("topogen"))' 2>/dev/null || echo 'Not installed')"
	@echo "  Virtual environment: $$(echo $$VIRTUAL_ENV | sed 's|.*/||' || echo 'None active')"
	@echo ""
	@echo "🔧 Development Tools:"
	@echo "  Pre-commit: $$($(PRECOMMIT) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Pytest: $$($(PYTEST) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Ruff: $$($(RUFF) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Pyright: $$($(PYRIGHT) --version 2>/dev/null | head -1 || echo 'Not installed')"
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
