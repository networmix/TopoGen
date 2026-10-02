# TopoGen

[![CI](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml)

TopoGen builds US backbone topologies from Census urban-area polygons and
TIGER/Line roads. It selects metros by land area and explicit overrides, finds
highway corridors, and emits [NetGraph](https://github.com/networmix/NetGraph)
scenarios with site blueprints, hardware, risk groups, traffic, and workflows.

## Install

Requires Python 3.11+. Dependencies include `ngraph>=0.23.1` and
`netgraph-core>=0.10.0`. TopoGen is not on PyPI; the `topogen` package there is
an unrelated project.

```bash
pip install git+https://github.com/networmix/TopoGen
```

For a source checkout:

```bash
git clone https://github.com/networmix/TopoGen
cd TopoGen
make dev
source venv/bin/activate
```

`make dev` installs development dependencies and Git hooks. For Superset
workspaces, use the setup command below instead.

## Generate and run a scenario

Copy an [example config](examples/) to `config.yml` and set its `data_sources`
paths to local Census urban areas, TIGER/Line primary roads, and state boundaries.
Paths are resolved from the working directory. State boundaries are used to
filter to the contiguous US and to draw maps.

```bash
topogen info config.yml
topogen generate config.yml -o output
topogen build config.yml -o output
ngraph inspect output/config_scenario.yml
ngraph run output/config_scenario.yml
```

`generate` writes `output/config_integrated_graph.json`, a metro-to-metro corridor
graph, and a preview JPEG. `build` reads the graph and writes
`output/config_scenario.yml`. File prefixes come from the config filename.

`build` checks the schema, NetGraph Scenario construction, topology, and hardware.
`ngraph run` executes the workflows. Add `--print` to `topogen build` to also print
the YAML and skip scenario validation.

To process a folder of configs, use `./build.sh examples output`. It reuses saved
graphs; pass `--force` to regenerate them or `--build-only` to rebuild scenarios.
See `./build.sh --help` for filename filters.

### Python

```python
from pathlib import Path
from topogen import TopologyConfig, build_integrated_graph, save_to_json

config = TopologyConfig.from_yaml(Path("config.yml"))
graph = build_integrated_graph(config)
save_to_json(
    graph,
    Path("output/config_integrated_graph.json"),
    config.projection.target_crs,
    config.output.formatting,
)
```

## Configuration

Configs are checked against [the JSON schema](topogen/schemas/topogen_config.json).
`make validate` checks all configs in `examples/`.

Files in `lib/` under the working directory override built-in definitions by name.
Each file is a direct mapping from names to definitions:

| File | Contents |
|------|----------|
| `blueprints.yml` | Site topology templates |
| `components.yml` | Routers and optics |
| `failure_policies.yml` | Failure modes and rules |
| `workflows.yml` | Lists of NetGraph workflow steps |

Set the demand-set name with `traffic.matrix_name` and routing preset names with
`traffic.flow_policy_config`. The built-in workflow follows the configured
demand-set name; custom workflows keep their explicit names. NetGraph validates
workflow arguments when the library is loaded.

Traffic models are `uniform`, `gravity`, and `hose`. With `build.tm_sizing.enabled`,
TopoGen routes traffic on a collapsed metro graph and sizes corridors for the
larger directional load, then applies headroom and capacity increments. Uniform
traffic includes same-metro DC pairs in the total split. Sizing models one demand
node per DC region and rejects unsupported endpoint patterns.

## Development

```bash
make check-ci   # Format checks, lint, types, config schemas, and tests with coverage
make check      # Also run pre-commit hooks; hooks may modify files
make test       # Tests with coverage
make lint       # Formatting, Ruff, and Pyright checks
make validate   # Example config schemas
```

### Superset workspaces

[`.superset/config.json`](.superset/config.json) defines three commands:

- **Setup** creates a local `venv`, installs `.[dev]`, and checks dependencies and
  imports. It selects Python 3.11–3.13, using `uv` to install 3.13 if needed.
  Set `SUPERSET_PYTHON` to choose an interpreter explicitly.
- **Run** executes `make check-ci`. TopoGen has no dev server and opens no ports.
- **Teardown** has nothing to stop: setup starts no background services.

Setup copies untracked root-level `.env` and `.env.*` files from
`$SUPERSET_ROOT_PATH`, preserving existing workspace files and skipping
`.example`, `.sample`, and `.template` files. It does not source these files,
copy datasets, or install shared Git hooks. Set dataset paths in your YAML.

For an existing workspace:

```bash
bash .superset/workspace.sh setup
source venv/bin/activate
bash .superset/workspace.sh check
```

Keep the root checkout's Superset configuration updated so new workspaces
receive it. Personal lifecycle commands go in the gitignored
`.superset/config.local.json`; see [Superset lifecycle scripts](https://docs.superset.sh/setup-teardown-scripts).

## License

[MIT](LICENSE)
