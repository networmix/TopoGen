# TopoGen Development Guide

## Essential Commands

```bash
make dev         # Complete dev environment setup
make check         # Run all quality checks + tests
make test          # Run tests with coverage
make docs          # Generate API documentation
make docs-serve    # Serve docs locally
```

**For all available commands**: `make help`

## Publishing

**Manual**: `make clean && make build && make publish-test && make publish`

**Automated**: Create GitHub release → auto-publishes to PyPI

**Version**: Update `version = "x.y.z"` in `pyproject.toml` before publishing

## Key Development Files

```text
pyproject.toml              # Package config, dependencies, tool settings
Makefile                    # Development commands
.pre-commit-config.yaml     # Code quality hooks
dev/setup-dev.sh           # Development environment setup script
dev/run-checks.sh           # Manual code quality checks
```

## Git Workflows

```text
.github/workflows/
├── python-test.yml         # CI: tests, linting, type checking
├── docs.yml                # Auto-deploy documentation
└── publish.yml             # Auto-publish to PyPI on releases
```
