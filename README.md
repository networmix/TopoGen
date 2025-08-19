# TopoGen

[![Python-test](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml)

TopoGen builds realistic backbone graphs for the continental US and prepares analysis-ready scenarios for `NetGraph`.

## What it does

- **Generator**: Produces an integrated metro + highway graph from public datasets.
- **Builder**: Converts the integrated graph into a `NetGraph` scenario YAML for downstream analysis.

## How it creates realistic graphs

- **Data-grounded**: Uses Census Urban Areas (UAC20) for metro delineation and TIGER/Line primary roads for the backbone substrate.
- **Metro clustering**: Selects major urban areas, applies size filters, and supports explicit overrides for strategic hubs.
- **Highway graph shaping**: Projects to an equal-area CRS, snaps close nodes, prunes invalid geometry, and keeps the largest connected component for a coherent substrate.
- **Corridor discovery**: Connects nearby metros via k-nearest adjacency and computes paths along the highway network to approximate long-haul corridors.
- **Validation**: Enforces distance and connectivity checks and flags atypical degree distributions.
- **Scenario preparation**: Adds risk groups and component blueprints, then emits a `NetGraph` scenario YAML with metadata for repeatable analysis.

## Quick start

```bash
python -m pip install -e .

# Inspect configuration and data availability
python -m topogen info config.yml

# Generate the integrated metro + highway graph
python -m topogen generate config.yml

# Build a NetGraph scenario YAML from the integrated graph
python -m topogen build config.yml -o config_scenario.yml
```

## CLI

```bash
python -m topogen --help
```

Key commands:

- `generate`: Build the integrated metro + highway graph.
- `build`: Emit a `NetGraph` scenario YAML from the integrated graph.
- `info`: Print configuration and input data status.

Configuration lives in `config.yml`. See sample configs in `topogen_configs_small/` and `topogen_configs/`. Requires Python 3.11+.
