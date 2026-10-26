# NetLab-TopoGen

[![CI](https://github.com/networmix/NetLab-TopoGen/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/networmix/NetLab-TopoGen/actions/workflows/python-test.yml)

NetLab-TopoGen creates backbone network scenarios for the contiguous United States.
It chooses metropolitan areas from Census data and follows highways to connect
them. It then adds network sites, routers, traffic demands and failure policies.
The result is a YAML scenario for [NetGraph](https://github.com/networmix/NetGraph).

There are two steps: `generate` chooses the locations and routes; `build` turns
those routes into a network. NetGraph runs the analysis of that network.

## Install

Use Python 3.11 or later. Run these commands in a terminal:

```bash
git clone https://github.com/networmix/NetLab-TopoGen
cd NetLab-TopoGen
python3 -m venv venv
source venv/bin/activate
python -m pip install -e .
```

This also installs NetGraph 0.25, which brings its computation engine,
NetGraph-Core. [pyproject.toml](pyproject.toml) lists the supported NetGraph
versions and the other dependencies.

## Download the geographic data

Create a `data/` directory in the repository and save these files there.
Keep the ZIP files as downloaded; do not unpack them.

- [Urban areas (2020)](https://www2.census.gov/geo/tiger/TIGER2020/UAC/tl_2020_us_uac20.zip)
- [Primary roads (2024)](https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip)
- [State boundaries (2024)](https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip)

The example configurations use these filenames. To store the files elsewhere,
change `data_sources` in your configuration. Relative paths start at the directory
where you run the command.

## Build a network

Run this five-metro example from the repository root:

```bash
cp examples/small_baseline.yml config.yml
netlab-topogen info config.yml
netlab-topogen generate config.yml -o output
netlab-topogen build config.yml -o output
```

`info` checks the configuration and shows whether its data files exist.
`generate` selects the metros: the urban areas you name, then the largest
remaining ones by land area. It then finds highway routes between nearby metros.
A route between two metros is called a **corridor**.

`build` adds points of presence (PoPs), where backbone routers are located, and
data-center sites. A **blueprint** describes the devices and links inside a site.
The build also assigns hardware, creates traffic demands and checks the scenario
with NetGraph. It writes files only after the check passes, so a scenario that
fails the check leaves the existing files unchanged.

Files in `output/` use the configuration filename as their prefix:

| File | Contents |
| --- | --- |
| `config_integrated_graph.json` | Metro locations and corridors from `generate`; `build` reads it |
| `config_scenario.yml` | The network, traffic, failure policies and analysis steps |
| `config_network_graph.json` | Sites and the links between them, with their capacities |
| `config_*.jpg` | Maps and site diagrams enabled in the configuration |

Map backgrounds need an internet connection; without one, `build` stops with an
error after writing the scenario and the site-network JSON. To run the analysis
steps saved in the scenario:

```bash
ngraph run output/config_scenario.yml --output output
```

## Change the design

Edit `config.yml`. Capacities and traffic volumes in the supplied examples use
Gbps; geographic distances use kilometers.

| Setting | What it controls |
| --- | --- |
| `clustering.metro_clusters` | Number of metro locations |
| `clustering.override_metro_clusters` | Name patterns of urban areas to include; they count toward `metro_clusters` |
| `corridors` | Nearby metros to connect, route count and distance limits |
| `build.build_defaults` | PoPs, data-center sites, blueprints, link capacities and costs |
| `build.build_overrides` | Different settings for selected metros |
| `build.tm_sizing` | Link capacities calculated from traffic, with spare capacity |
| `traffic` | Traffic volume and distribution model |
| `components` | Router types and optics for link endpoints |
| `failure_policies` | Failures to model in the scenario |
| `workflows` | Analysis steps for NetGraph to run |
| `visualization` | Maps and diagrams to export |

The traffic models are `uniform` (equal traffic between data-center sites),
`gravity` (traffic weighted by site size and distance), and `hose` (traffic fitted
to each site's send and receive totals). `traffic.flow_policy_config` gives each
priority class a NetGraph routing preset, spelled exactly as NetGraph names it,
such as `TE_MAXFLOW_UNLIM`.

Start with the [baseline, Clos and Dragonfly examples](examples/).
The [configuration schema](netlab_topogen/schemas/netlab_topogen_config.json) lists every
accepted field and its allowed values.

Your own sites, hardware, failures and analyses go in `lib/blueprints.yml`,
`lib/components.yml`, `lib/failure_policies.yml` and `lib/workflows.yml`.
The `netlab-topogen` command reads `lib/` in the directory where you run it. Each file
maps names to definitions; a name that matches a built-in definition replaces it.
Only the workflow the configuration selects is checked, so an unused workflow
cannot stop a build. The repository's `lib/` already adds failure policies and
workflows, such as `hose_samples_3` for three sampled traffic matrices. See
[how the pipeline works](dev/PIPELINE.md) for the build rules, Python entry
points and validation steps.

After changing locations or highway routing, run `generate` and `build` again.
After changing sites, hardware or traffic, run `build` again.

## Build several configurations

```bash
./build.sh examples output
```

Each configuration gets a subdirectory, such as `output/small_baseline/`.
The script reuses a saved corridor graph when one exists. Use `--force` to
regenerate the graphs, or `--build-only` to build scenarios from saved graphs.
A failed configuration makes the command return a failure status.

## Development

```bash
python -m pip install -e '.[dev]'
make check-ci
```

This checks formatting, lint, types, example configurations and tests.
In a Superset workspace, `bash .superset/workspace.sh setup` installs the
development environment; the Run button runs these checks.

[Changelog](CHANGELOG.md) · [MIT license](LICENSE)
