# CLI Reference

TopoGen exposes subcommands for generating the integrated graph, building a `NetGraph` scenario, and inspecting configuration.

## Usage

```bash
python -m topogen [GLOBAL OPTIONS] <command> [COMMAND OPTIONS]
```

### Global options

- `-v, --verbose`: Enable debug logging.
- `--quiet`: Suppress console prints (log output only).

## Commands

### `info`

Show configuration values and input data availability.

```bash
python -m topogen info -c config.yml
```

Options:

- `-c, --config PATH` (default: `config.yml`)

### `generate`

Generate the integrated metro + highway graph to `output/integrated_graph.json`.

```bash
python -m topogen generate -c config.yml
```

Options:

- `-c, --config PATH` (default: `config.yml`)

### `build`

Build a `NetGraph` scenario YAML from the integrated graph.

```bash
python -m topogen build -c config.yml -o output/scenario.yaml
```

Options:

- `-c, --config PATH` (default: `config.yml`)
- `-o, --output PATH` (default: `output/scenario.yaml`)
- `--print` Print the YAML to stdout instead of writing to a file and validating

## Notes

- Configuration lives in `config.yml`. See the repository root for an example and `docs/` for details.
- YAML formatting: set `output.formatting.yaml_anchors: false` to emit fully expanded YAML without anchors/aliases.
