# TopoGen

[![CI](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/networmix/TopoGen/actions/workflows/python-test.yml)

TopoGen generates backbone network scenarios for the contiguous United States.
It selects metro locations from Census urban areas, uses highway paths as link
routes, and adds sites, routers, traffic and failure scenarios for
[NetGraph](https://github.com/networmix/NetGraph).

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/networmix/TopoGen
cd TopoGen
python3 -m venv venv
source venv/bin/activate
python -m pip install -e .
```

## Download data

Create a `data/` directory and save these Census files there. Keep the ZIP files
as downloaded; no need to unpack them.

- [Urban areas (2020)](https://www2.census.gov/geo/tiger/TIGER2020/UAC/tl_2020_us_uac20.zip)
- [Primary roads (2024)](https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip)
- [State boundaries (2024)](https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip)

The example configs already use these filenames. To store data elsewhere, change
`data_sources` in the config. Paths are relative to your working directory.

## Run

Start with the five-metro example. Run these commands from the repository root:

```bash
cp examples/small_baseline.yml config.yml
topogen info config.yml
topogen generate config.yml -o output
topogen build config.yml -o output
```

`info` shows the config and whether data files exist. `generate` finds highway
routes between metros. `build` adds the site topologies, hardware and traffic,
then validates the NetGraph scenario.

Files in `output/` use the config filename as their prefix:

| File | Contents |
| --- | --- |
| `config_integrated_graph.json` | Metro locations and highway routes |
| `config_scenario.yml` | NetGraph scenario, ready to run |
| `config_*.jpg` | Network maps and site diagrams |

Map backgrounds need an internet connection. To run the capacity, failure and
cost analyses defined in the scenario:

```bash
ngraph run output/config_scenario.yml --output output
```

## Configure

Edit `config.yml`. These are the main settings:

| Setting | Controls |
| --- | --- |
| `clustering.metro_clusters` | Number of metros |
| `clustering.override_metro_clusters` | Metros to include by name |
| `corridors` | Nearest neighbors, route count and distance limits |
| `build.build_defaults` | PoPs, data centers, site blueprints and link capacities |
| `build.build_overrides` | Different settings for selected metros |
| `build.tm_sizing` | Link sizing from traffic, including headroom |
| `traffic` | Traffic volume and model: uniform, gravity or hose |
| `components` | Router and optic assignments |

See [examples/](examples/) for baseline, Clos and Dragonfly configs, and the
[schema](topogen/schemas/topogen_config.json) for all settings.

Custom definitions go in `lib/blueprints.yml`, `lib/components.yml`,
`lib/failure_policies.yml` or `lib/workflows.yml`. Each file maps names to
definitions; matching names replace built-ins.

After changing geography, rerun `generate` and `build`. For site, hardware or
traffic changes, rerun `build`. To process all example configs:

```bash
./build.sh examples output
```

This writes each config's results to its own subdirectory. Use `--force` to
regenerate saved graphs or `--build-only` to rebuild scenarios from them.

## Development

```bash
python -m pip install -e '.[dev]'
make check-ci
```

In a Superset workspace, use `bash .superset/workspace.sh setup` to install the
development environment.

[Changes and migration notes](CHANGELOG.md) · [MIT license](LICENSE)
