# How TopoGen builds a scenario

TopoGen first chooses metro locations and the highway routes between them.
It saves this geography so you can change the network design without processing
the road data again. A second step adds sites, devices and traffic, and produces
a NetGraph scenario.

Run commands from the repository root. Relative data and `lib/` paths start at
the working directory, even when the configuration file is in another directory.

## 1. Choose locations and corridors

```bash
topogen generate examples/small_baseline.yml -o output
```

`TopologyConfig.from_yaml` loads the settings and checks them against the
configuration schema. The geographic step then works in five stages:

1. It reads the Census urban areas, primary roads and state boundaries and keeps
   urban areas in the contiguous United States. Coordinates are projected to
   `projection.target_crs`, which must measure distance in meters.
2. It picks `clustering.metro_clusters` metros. Each pattern in
   `override_metro_clusters` selects the largest urban area whose name contains
   it, ignoring case. The remaining places go to the largest urban areas by land
   area. A metro is a point inside its urban area with a radius derived from its
   land area, capped at `max_uac_radius_km`.
3. It builds a graph of the selected highway classes and attaches each metro to
   its nearest highway point. That point must be within
   `validation.max_metro_highway_distance_km`.
4. It pairs each metro with its `corridors.k_nearest` nearest metros, counting
   only those within `max_edge_km` in a straight line. For each pair it keeps up
   to `k_paths` shortest highway routes of at most `max_corridor_distance_km`.
5. It gives each route a named risk group. A highway segment carries the risk
   group of every route that uses it, and a corridor collects the risk groups of
   all its segments. Corridors that share a highway segment therefore share a
   risk group, so a failure policy can fail them together. By default, segments
   inside a metro's radius are left out (`risk_groups.exclude_metro_radius_shared`).

The output is `output/small_baseline_integrated_graph.json`. Its graph is a
NetworkX `MultiGraph`: two metros can have several separate corridors. The edge
key is the route index, and key 0 is the shortest route. Corridor distances,
routes and risk groups belong to those edges. Do not convert it to a simple
graph, which would lose parallel corridors.

Python entry points are `topogen.build_integrated_graph`,
`topogen.save_to_json` and `topogen.load_from_json`. Serialization stores the
coordinate reference system with the graph. The build command checks it against
`projection.target_crs`; regenerate the graph if the projection changes.

## 2. Add sites, devices and traffic

```bash
topogen build examples/small_baseline.yml -o output
```

The build reads `output/small_baseline_integrated_graph.json`, the saved graph
with the same configuration filename prefix.

### Sites and links

Each metro gets `pop_per_metro` PoPs and `dc_regions_per_metro` data-center
sites, from `build.build_defaults` or from a `build_overrides` entry that names
the metro. Three kinds of site links connect them:

| Link type | Connects |
| --- | --- |
| `intra_metro` | Every pair of PoPs in a metro |
| `dc_to_pop` | Each data-center site to every PoP in its metro |
| `inter_metro_corridor` | The PoPs of two metros, once for each corridor between them |

A corridor connects every PoP pair of its two metros with `mode: mesh`. With
`mode: one_to_one` it connects PoP 1 to PoP 1, PoP 2 to PoP 2, and so on.

A link's routing `cost` is its length in whole kilometers, rounded up, unless
the configured link `cost` is larger. Corridors use their highway length; links
inside a metro use distances around a circle of the metro's radius. The rounded
length is also kept as `distance_km`. For example, a 412.3 km corridor with
configured cost 1 gets `cost: 413` and `distance_km: 413`.

### Traffic

Each data-center site has `traffic.mw_per_dc_region` megawatts of power and
offers `gbps_per_mw` Gbps per megawatt. The small baseline has three
data-center sites of 150 MW at 200 Gbps per MW. Each site offers 30,000 Gbps,
and the traffic matrix totals 90,000 Gbps. The model decides which site pairs
carry that traffic:

- `uniform` divides it equally among all ordered pairs of distinct sites.
- `gravity` weights a pair by the product of the two sites' power, divided by
  their distance; `alpha` and `beta` are the exponents of those terms.
- `hose` starts from random pair weights, optionally favoring short distances,
  and fits them so each site sends and receives its offered volume.

`traffic.gravity.rounding_gbps`, when above zero, rounds the emitted pair volumes
of both the gravity and the hose model to that step. Rounding can shift a site's
totals slightly away from its offered volume. `priority_ratios` divides the
volume into priority classes. The demand set is
named `traffic.matrix_name`. With the hose model and `samples` above 1, the build
writes one set per sample: `<matrix_name>_1`, `<matrix_name>_2` and so on. The
built-in workflow and traffic-based sizing use `<matrix_name>` itself, so select
a workflow that names the sample sets and set `build.tm_sizing.matrix_name` to
one of them. The repository's `lib/workflows.yml` has `hose_samples_3` and
`hose_samples_10` for `matrix_name: baseline_traffic_matrix`.

### Link sizing from traffic

When `build.tm_sizing.enabled` is true, TopoGen sizes links for the demand set
named by `tm_sizing.matrix_name`, which defaults to `traffic.matrix_name`:

1. It treats each metro as one node and routes every metro-to-metro demand over
   the corridor links by cost. `flow_placement` sets how traffic splits across
   equal-cost paths.
2. Each corridor link that carries traffic gets its peak load in either direction,
   multiplied by `headroom` and rounded up to a multiple of `quantum_gbps`.
3. A PoP's egress is the total capacity of its corridor links. A data-center link
   gets `alpha_dc_to_pop` times the egress of its PoP. A PoP-to-PoP link in a
   metro gets `beta_intra_pop` times the smaller egress of its two PoPs. Both are
   rounded up to a multiple of `quantum_gbps`.
4. With `respect_min_base_capacity`, no link falls below its configured capacity.

For example, take two metros with one PoP and one 25 MW data-center site each,
200 Gbps per MW and uniform traffic. Each direction carries 5,000 Gbps. With
headroom 1.3 and a 3,200 Gbps quantum, the corridor needs 6,500 Gbps, which
rounds up to 9,600 Gbps. Each PoP's egress is then 9,600 Gbps, so with
`alpha_dc_to_pop: 1.2` each data-center link needs 11,520 Gbps and gets 12,800.

### Devices and hardware

The build expands the site blueprints into devices and links. A site link's
capacity is a budget for that pair of sites; it is divided equally among the
device links that connect them. `components.hw_component` assigns a router
platform to each device role.

`components.optics` maps `local_role->remote_role` to the optic installed at the
local end of a link. Each end gets enough optics to carry its link: the link
capacity divided by the optic capacity, rounded up. In the example above, one
9,600 Gbps device link with `800G-ZR+` optics (800 Gbps each) needs 12 optics at
each end. When `optics` is set, every link end needs a mapping: a link between
`core` and `leaf` routers needs both `core->leaf` and `leaf->core`, or the build
stops with an error.

The scenario YAML contains `seed`, `blueprints`, `components`, the `network`,
`risk_groups`, failure policies under `failures`, traffic under `demands` and the
analysis `workflow`. Building the scenario does not execute that workflow.

`output.scenario_seed` controls the traffic generator and the scenario's random
seed. Built-in analysis steps derive their seeds from it. A custom workflow can
set a step's seed explicitly to override that behavior.

## 3. Validate and save

The CLI loads the scenario with NetGraph, which checks the YAML schema, the
workflow steps and the failure policies. It then checks the network for isolated
devices and audits hardware, ports, optics, roles and data-center capacity. For
each demand set, a data-center site's total sent traffic, and separately its total
received traffic, must fit within the capacity of the links that leave the site.
The CLI replaces the destination YAML only after validation succeeds, so a failed
build keeps the existing file.

`topogen.scenario.build_scenario` returns YAML without validating it. This script
builds, validates and writes a scenario from Python. It runs the same checks as
the CLI, but writes the file directly instead of replacing it atomically:

```python
from pathlib import Path

from topogen import RunContext, TopologyConfig, load_from_json
from topogen.scenario import build_scenario
from topogen.validation import validate_scenario_yaml

config = TopologyConfig.from_yaml(Path("examples/small_baseline.yml"))
context = RunContext(Path("output"), "small_baseline")
graph_path = context.path("integrated_graph.json")
graph, crs = load_from_json(graph_path)
assert crs == config.projection.target_crs

scenario_yaml = build_scenario(graph, config, context=context)
issues = validate_scenario_yaml(
    scenario_yaml,
    integrated_graph_path=graph_path,
    hw_component_map=config.components.hw_component,
    optics_map=config.components.optics,
)
if issues:
    raise SystemExit("\n".join(issues))
context.path("scenario.yml").write_text(scenario_yaml)
```

`validate_scenario_yaml` returns a list of problems; an empty list means the
scenario passed. The two component maps enable the hardware and optic coverage
checks that the CLI runs.

`topogen.RunContext` sets the output directory, filename prefix and optional
debug directory for one invocation. `RunContext(Path("output"), "small_baseline")`
names files with the `small_baseline_` prefix. Output paths are separate from
topology settings. With a context, `build_scenario` also writes
`small_baseline_network_graph.json` and the maps and diagrams enabled in the
configuration.

## Libraries and output

The `lib/` directory supplies custom blueprints, hardware components, failure
policies and workflows. Each YAML file maps names to definitions; custom names
are added and matching built-in names are replaced. Referenced site blueprints
and their dependencies are included in the scenario. Unknown references and
cyclic blueprint dependencies are errors.

`workflows.assignments.default` chooses a workflow. Each step uses `type` for its
analysis class and `demand_set` when it needs traffic. Built-in workflow steps use
`traffic.matrix_name`; workflows from `lib/workflows.yml` keep the names they list.
In the built-in workflow, `MaximumSupportedDemand` searches for the largest
multiplier of the traffic that the network can carry. `TrafficMatrixPlacement`
then places the traffic, scaled by that multiplier, under each sampled failure
of its `failure_policy`.

The `failures` section contains the policy named by
`failure_policies.assignments.default` and every policy that a workflow step
names. File prefixes come from the configuration filename, not the names inside
the YAML. The batch script adds one subdirectory per configuration.

## Working on the code

| Area | Main files |
| --- | --- |
| Settings and CLI | `topogen/config.py`, `topogen/cli.py`, `topogen/context.py` |
| Geographic processing | `topogen/metro_clusters.py`, `topogen/highway_graph.py`, `topogen/corridors.py`, `topogen/integrated_graph.py` |
| Scenario assembly | `topogen/scenario/assembly.py`, `sizing.py`, `expansion.py`, `striping.py` |
| Traffic models | `topogen/traffic_matrix.py` |
| Scenario checks | `topogen/validation/` |
| Maps and diagrams | `topogen/visualization.py`, `topogen/blueprint_viz.py` |

Run `make check-ci` after changes. It validates the example configurations and
runs formatting, lint, type checks and tests. Integration tests exercise scenario
building and NetGraph validation without downloading the national datasets.
To check geographic generation, download the files listed in the
[README](../README.md#download-the-geographic-data) and run both commands on this page.
