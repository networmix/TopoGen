# TopoGen

TopoGen builds realistic backbone graphs for the continental US and prepares analysis-ready scenarios for `NetGraph`.

## What it does

- Generator: Constructs an integrated metro + highway graph from public datasets.
- Builder: Converts the integrated graph into a `NetGraph` scenario YAML for analysis.

## How it creates realistic graphs

- Uses Census Urban Areas (UAC20) to delineate metro regions and TIGER/Line primary roads for the backbone substrate.
- Projects to an equal-area CRS (EPSG:5070), snaps nearby nodes, removes invalid geometry, and keeps the largest connected component.
- Connects nearby metros via a k-nearest adjacency and computes highway-constrained paths to approximate long-haul corridors.
- Validates connectivity and distance constraints, assigns risk groups, and attaches component blueprints before emitting YAML.
- Validates connectivity and distance constraints, assigns risk groups, and attaches component blueprints before emitting YAML. YAML anchors can be disabled via `output.formatting.yaml_anchors: false` if you prefer expanded lists.

## Quick start

```bash
# Inspect configuration and data availability
python -m topogen info config.yml

# Generate the integrated metro + highway graph
python -m topogen generate config.yml

# Build a NetGraph scenario YAML from the integrated graph
python -m topogen build config.yml -o config_scenario.yml
```

See the [Installation](getting-started/installation.md) guide for setup and the [CLI Reference](reference/cli.md) for commands and options.
