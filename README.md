# TopoGen

[![Python-test](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml)

Topology generator for US backbone networks with [NetGraph](https://github.com/networmix/NetGraph) scenario output.

## Overview

Generates backbone network topologies from US Census Urban Areas (UAC) and TIGER/Line highway data. Outputs NetGraph scenario YAML with blueprints, components, risk groups, failure policies, and traffic matrices.

## Features

### Graph Generation

- Metro selection from UAC data with size filtering and hub overrides
- Equal-area projection, coordinate snapping, geometry pruning
- Corridor discovery via k-nearest metro adjacency and k-shortest paths
- Corridor risk groups with distance validation

### Scenario Output

- NetGraph scenario YAML with blueprints, components, failure policies, workflows
- Traffic matrix generation with seed metadata
- Schema validation and `ngraph` workflow execution checks

## Installation

### From PyPI

```bash
pip install topogen
```

### From Source

```bash
git clone https://github.com/networmix/TopoGen
cd TopoGen
make dev    # Install in editable mode with dev dependencies
make check  # Run full test suite
```

## Usage

### CLI

```bash
# Show help
topogen --help

# Inspect configuration and data availability
topogen info config.yml

# Generate integrated metro + highway graph
topogen generate config.yml -o out_dir

# Build NetGraph scenario from integrated graph (requires previous generate)
topogen build config.yml -o out_dir

# Print scenario YAML to stdout without validation
topogen build config.yml --print
```

### Python API

```python
from topogen import TopologyConfig, build_integrated_graph, save_to_json

# Load configuration
config = TopologyConfig.from_yaml("config.yml")

# Build integrated graph
graph = build_integrated_graph(config)

# Save to JSON
save_to_json(
    graph,
    "output/integrated_graph.json",
    config.projection.target_crs,
    config.output.formatting,
)
```

## Configuration

YAML config validated by JSON schema (`topogen/schemas/topogen_config.json`). Example configs in `examples/`. Run `make validate` to check.

### Override Libraries

Place files in `./lib/` to override built-in libraries:

| File | Purpose |
|------|---------|
| `blueprints.yml` | Topology templates (e.g., Clos fabrics) |
| `components.yml` | Hardware definitions (routers, optics) |
| `failure_policies.yml` | Failure mode definitions |
| `workflows.yml` | Analysis workflow steps |

User entries override built-ins by name.

## Repository Structure

```
topogen/                # Python package
  cli.py                # Command-line interface
  config.py             # Configuration loading/validation
  integrated_graph.py   # Graph construction
  scenario_builder.py   # NetGraph scenario generation
  scenario/             # Scenario assembly modules
  schemas/              # JSON schemas
  validation/           # Validation audits
  *_lib.py              # Built-in libraries
examples/               # Example configurations
lib/                    # Optional user library overrides
tests/                  # Pytest suite
```

## Development

```bash
make dev        # Setup environment
make check      # Run pre-commit + validation + tests + lint
make check-ci   # Run lint + validation + tests (CI mode)
make test       # Run tests with coverage
make lint       # Run linting only (ruff + pyright)
make validate   # Validate config YAMLs against schema
```

## Requirements

- **Python**: 3.11+
- **ngraph**: >= 0.12.0 (runtime dependency)
- **Geo stack**: geopandas, rasterio, pyproj, shapely (via dependencies)
- **Data inputs** (paths configured in YAML):
  - UAC polygons (Census Urban Areas)
  - TIGER/Line primary roads
  - CONUS boundary (optional, for visualization)

## License

[AGPL-3.0-or-later](LICENSE)
