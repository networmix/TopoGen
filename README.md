# TopoGen

[![CI](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml)

TopoGen builds US backbone topologies from Census urban-area polygons and
TIGER/Line roads. It selects metros by land area and explicit overrides, finds
highway corridors, and emits [NetGraph](https://github.com/networmix/NetGraph)
scenarios with site blueprints, hardware, risk groups, traffic, and workflows.

## Install

Requires Python 3.11+. Dependencies include `ngraph>=0.24.0` and
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
multigraph. Preview JPEGs are written when their export settings are enabled. `build` reads the graph and writes
`output/config_scenario.yml`. File prefixes come from the config filename.

`build` checks the schema, NetGraph Scenario construction, topology, and hardware.
`ngraph run` executes the workflows. Add `--print` to `topogen build` to also print
the validated YAML. A failed build preserves the previous scenario YAML; successful
writes replace it atomically.

To process a folder of configs, use `./build.sh examples output`. It reuses saved
graphs; pass `--force` to regenerate them or `--build-only` to rebuild scenarios.
See `./build.sh --help` for filename filters.

### Python

```python
from pathlib import Path
from topogen import RunContext, TopologyConfig, build_integrated_graph, save_to_json
from topogen.scenario import build_scenario
from topogen.validation import validate_scenario_yaml

config = TopologyConfig.from_yaml(Path("config.yml"))
context = RunContext(Path("output"), stem="config")
graph = build_integrated_graph(config, context=context)
save_to_json(
    graph,
    context.path("integrated_graph.json"),
    config.projection.target_crs,
    config.output.formatting,
)
scenario_yaml = build_scenario(graph, config, context=context)
issues = validate_scenario_yaml(
    scenario_yaml,
    hw_component_map=config.components.hw_component,
    optics_map=config.components.optics,
)
assert not issues, issues
```

Without `context`, the Python generation/build functions return their results
without exporting files. Configuration holds topology and visualization settings;
`RunContext` holds output paths, the filename stem and the optional debug directory.
The CLI supplies this context automatically. `--debug-dir` exports the exact traffic
matrices used for sizing and emission, for all traffic models.

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
traffic uses `group_mode: group_pairwise`: each ordered pair of distinct DC sites
receives an equal share, including same-metro sites. Device counts within a DC do
not change its share. Sizing aggregates traffic by ordered metro pair before
routing and rejects unsupported endpoint patterns.

### Current contracts and migration

There is one supported format at each stage; obsolete inputs are rejected.

- Regenerate saved graphs with `topogen generate`: JSON requires
  `graph_type: corridors` and an explicit key for every parallel edge. The Python
  build API accepts an undirected `networkx.MultiGraph` of metro nodes only.
  Each metro supplies `name`, `name_orig`, `metro_id`, `x`, `y` and `radius_km`.
  Names and IDs must be unique nonempty strings; coordinates must be finite.
  The target CRS must be projected with metre units.
- Every discovered path survives extraction, JSON and site expansion. Chain
  contraction preserves alternate routes and both arcs of a ring. Risk tags are
  aggregated along each path, including risks shared with other paths.
- Import `build_scenario` from `topogen.scenario`; `topogen.scenario_builder` has
  been removed. Replace private configuration output attributes with `RunContext`
  and public `config.visualization` fields.
- Use `get_builtin_components()` and ordinary dictionary lookup/filtering for the
  effective component library. The unused singular/list/filter helpers,
  `TopologyConfig.summary()`, standalone point/bearing helpers, and unused
  `MetroCluster` distance/overlap/array helpers have been removed. `topogen info`
  remains the configuration summary command.
- Delete `highway_processing.min_cycle_nodes`, `validation_sample_size`,
  `build.tm_sizing.edge_select`, `output.scenario_metadata`,
  `components.assignments`, `components.library`, and failure/workflow
  `assignments.scenario_overrides`. These settings did not affect generation.
  Library definitions belong in the corresponding `lib/*.yml` file.
- Link `role_pairs` use strings such as `core|leaf`. Optics mappings explicitly
  name the **local and remote roles**, using `local->remote`. Specify both
  directions when both ends need optics; the module types may differ:

  ```yaml
  components:
    optics:
      leaf->spine: 800G-DR4
      spine->leaf: 1600G-2xDR4
  ```

  There is no implicit reverse mapping. Old `a-b` and `a|b` optics keys are rejected.
- Scenario hardware definitions must be present in the scenario's `components`.
  The builder includes components referenced by configured assignments and used
  blueprints. Audits never substitute the local built-in component library.
  Port audits count declared modules; optics audits check their capacity.
- Capacity/cost integer inputs must be finite and integral; booleans and fractional
  values are errors. Integer strings retain exact precision, and non-finite
  configuration numbers are rejected. Unknown routing policies and malformed library files fail
  explicitly. Optional libraries may be absent, but a present file must be a mapping.

Link `cost` is a minimum routing cost; the geometric distance remains separately
available as `distance_km`. Striping accepts only `width` (with optional
`mode: width`) or `mode: by_attr` plus `attribute`. It respects `one_to_one`,
endpoint matches and full device paths, including literal regex metacharacters.
Duplicate normalized priorities or metro overrides, unknown striping options and
unmatched forced metro patterns are errors. Nested blueprint dependencies are
included recursively; reference cycles are rejected.

Assembly generates traffic once and expands the complete device network once.
Capacity and optics are resolved per site edge against that expansion. Validation
constructs one full NetGraph scenario; coverage audits additionally expand tagged
rules and probe blueprint definitions, retaining node/link rules and variables.
Risk definitions use the length of their owning corridor path; references without
a corresponding path are errors.
DC capacity audits resolve NetGraph selectors and check each demand set separately
against external device-link capacity. `run_ngraph=False` checks metadata and
references only. Combined demand endpoints spanning multiple DCs require workflow
execution to determine their per-DC placement.
Hose generation rejects infeasible margins or a fit that does not converge.
With `rounding_gbps: 0`, gravity and hose preserve unquantized demand values.
Gravity jitter uses the scenario seed without changing Python's global PRNG.
Blueprint diagrams aggregate the already expanded devices and link capacities;
they do not implement a second version of the NetGraph DSL.
Map rendering uses network tiles through Contextily and may dominate wall time;
render failures propagate to the caller.

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
