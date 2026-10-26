# How TopoGen builds a scenario

TopoGen first chooses metro locations and the highway routes between them.
It saves this geography so you can change the network design without processing
the road data again. A second step adds sites, devices and traffic, and produces
a NetGraph scenario.

Run commands from the repository root. Relative data paths and the command's
`lib/` directory start at the working directory, even when the configuration
file is in another directory.

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

### The stages

The build is a fixed sequence of stages. Each stage takes the results of earlier
stages and returns a new, read-only result; no stage changes its inputs, and the
build itself reads and writes no file. The first stage that finds a problem stops
the build with an error that says what is wrong and where.
NetGraph reads TopoGen's documents at three points: to list the devices of
striped blueprints, to expand site links into device links, and to validate the
finished scenario. Traffic generation and link sizing are TopoGen's own
computations.

| # | Stage | What it does | Function (module under `topogen/`) |
| --- | --- | --- | --- |
| 1 | Read corridors | Copies the metros and corridors out of the NetworkX graph | `read_corridors` (`scenario/sites.py`) |
| 2 | Resolve settings | Per metro: PoP and DC counts, blueprints, link settings, and each DC site's power | `resolve_metro_settings` (`scenario/sites.py`) |
| 3 | Finalize blueprints | The used blueprints and their dependencies, with each role's platform from `components.hw_component` | `finalize_blueprints` (`scenario/sites.py`) |
| 4 | Stripe devices | NetGraph expands every striped blueprint once; TopoGen groups their devices for striping | `stripe_groups` (`scenario/striping.py`) |
| 5 | Plan sites | Sites and the site links between them, in a fixed order | `plan_sites` (`scenario/sites.py`) |
| 6 | Expand sites | NetGraph expands the whole network once; TopoGen records the device links of each site link | `expand_sites` (`scenario/hardware.py`) |
| 7 | Generate traffic | Demands between DC sites | `generate_traffic` (`scenario/traffic.py`) |
| 8 | Size links | Capacity budgets from the traffic | `size_links` (`scenario/sizing.py`) |
| 9 | Assign hardware | Checks that every named component exists, then divides budgets among device links and counts optics | `check_components` (`scenario/document.py`), `assign_hardware` (`scenario/hardware.py`) |
| 10 | Write the scenario | The scenario document and its YAML text | `scenario_document`, `to_yaml` (`scenario/document.py`) |

`topogen.scenario.build_scenario(graph, config, library)` runs these stages and
returns a `Build`. The command then validates the text and publishes the files
(section 3).

Stage 6 refuses a site that expands to no devices, a site link that creates no
device links (a corridor from a metro to itself gives one), and a blueprint link
rule that creates no link. A site link whose ends select no device is refused
earlier, by NetGraph while it reads the network, with the link's paths in the
message.

### Numbers

Uniform traffic, link sizing and optic counts read each parsed number `x` as the
exact decimal that `repr(x)` writes, then calculate with fractions:
`headroom: 1.3` is exactly 13/10, not the nearest binary fraction. Spellings that
parse to the same float are indistinguishable (`1.0000000000000001` is read as
1). Demand volumes, budgets, egress sums and per-device shares are exact
fractions; a demand volume or capacity becomes its nearest float only when it is
written to the scenario.

### Sites and links

Each metro gets `pop_per_metro` PoPs and `dc_regions_per_metro` data-center
sites, from `build.build_defaults` or from a `build_overrides` entry that names
the metro. A site's id is `metro<index>/pop<n>` or `metro<index>/dc<n>`, where
the metro index follows the order of the saved graph. Three kinds of site links
connect the sites:

| Link type | Connects |
| --- | --- |
| `intra_metro` | Every pair of PoPs in a metro |
| `dc_to_pop` | Each data-center site to every PoP in its metro |
| `inter_metro_corridor` | The PoPs of two metros, once for each corridor between them |

A corridor connects every PoP pair of its two metros with `mode: mesh`. With
`mode: one_to_one` it connects PoP 1 to PoP 1, PoP 2 to PoP 2, and so on. A
corridor uses the `inter_metro_link` settings of the metro that comes first in
the saved graph.

A link's routing `cost` is its length in whole kilometers, rounded up, unless
the configured link `cost` is larger. Corridors use their highway length; links
inside a metro use distances around a circle of the metro's radius. The rounded
length is also kept as `distance_km`. For example, a 412.3 km corridor with
configured cost 1 gets `cost: 413` and `distance_km: 413`.

In the scenario, each site is one node group named by its site id, such as
`metro1/dc1`. Link rules are listed with all intra-metro links first, then
DC-to-PoP links, then corridors. Each rule runs from the endpoint that comes
first in site order: by metro, PoPs before DC sites, then by number.

### Data-center power

Each data-center site has a power in megawatts, resolved once per site:

1. `traffic.gravity.mw_per_dc_region_overrides` with the site's key, such as
   `columbus/dc2`;
2. otherwise the same overrides with the metro's name, such as `columbus`;
3. otherwise `traffic.mw_per_dc_region`.

Metro names are the lowercase names in the saved graph, such as `columbus` or
`salt-lake-city`. An override key that names neither a metro nor an
existing data-center site is an error, as is a negative power; both are refused
even when traffic is disabled. The site's devices carry the resolved power as
`mw_per_dc_region`, next to `gbps_per_mw`.

### Traffic

Each data-center site offers `gbps_per_mw` Gbps per megawatt of its power. The
small baseline has three data-center sites of 150 MW at 200 Gbps per MW. Each
site offers 30,000 Gbps, and the traffic matrix totals 90,000 Gbps. The model
decides which site pairs carry that traffic:

- `uniform` divides it equally among all ordered pairs of distinct sites. The
  scenario lists one demand per pair and priority class, with volume
  total × class ratio / (N × (N − 1)) for N sites, computed exactly. Three sites
  with a total of 600 Gbps get six demands of 100 Gbps. A positive total with
  fewer than two data-center sites is an error.
- `gravity` weights a pair by the product of the two sites' power, divided by
  their distance; `alpha` and `beta` are the exponents of those terms.
- `hose` starts from random pair weights, optionally favoring short distances,
  and fits them so each site sends and receives its offered volume.

Distances are between metro centers. For gravity and hose,
`traffic.gravity.rounding_gbps` rounds each unordered site pair's two-way total,
per priority class, under `rounding_policy`; that two-way total is then split
equally between the two directions. A 10 Gbps step therefore permits 15 Gbps in
each direction, a two-way total of 30. Zero disables rounding. Rounding and the
distribution of the remainder can change site totals. These models calculate in
floating point; sizing reads each emitted volume as the exact decimal that its
float's `repr` writes. Some of their sums use Python's built-in `sum`, which
adds floats more accurately from Python 3.12 on, so the same inputs can give
gravity and hose volumes that differ in the last digit between Python 3.11 and
later versions (a relative difference near 10^-16).

`priority_ratios` divides the volume into priority classes, and
`flow_policy_config` gives each class a NetGraph preset, written as the
demand's `preset`. A preset name must be spelled exactly as NetGraph names it,
such as `TE_MAXFLOW_UNLIM` or `SHORTEST_PATHS_ECMP`; any other spelling is an
error that lists the valid names.

The demand set is named `traffic.matrix_name`. With the hose model and `samples`
above 1, the build writes one set per sample: `<matrix_name>_1`,
`<matrix_name>_2` and so on. The built-in workflow and traffic-based sizing use
`<matrix_name>` itself, so select a workflow that names the sample sets and set
`build.tm_sizing.matrix_name` to one of them. The repository's
`lib/workflows.yml` has `hose_samples_3` and `hose_samples_10` for
`matrix_name: baseline_traffic_matrix`.

### Link sizing from traffic

When `build.tm_sizing.enabled` is true, TopoGen sizes links for the demand set
named by `tm_sizing.matrix_name`, which defaults to `traffic.matrix_name`:

1. It treats each metro as one node and each corridor site link as its own edge,
   with that link's cost. Every demand between two metros follows the shortest
   routes by cost. At each metro, the traffic that arrives is divided equally
   among the outgoing links that lie on a shortest route, so parallel links and
   equal-cost routes share it. Demands inside one metro load no corridor.
2. Each corridor link that carries traffic gets its peak load in either direction,
   multiplied by `headroom` and rounded up to a multiple of `quantum_gbps`
   (`quantum_gbps: 0` disables this rounding here and in step 3). A corridor link
   without traffic keeps its configured capacity.
3. A PoP's egress is the total capacity of its corridor links. A data-center link
   gets `alpha_dc_to_pop` times the egress of its PoP. A PoP-to-PoP link in a
   metro gets `beta_intra_pop` times the smaller egress of its two PoPs. Both are
   rounded up to a multiple of `quantum_gbps`.
4. With `respect_min_base_capacity`, no link falls below its configured capacity.

Sizing stops with an error when traffic is disabled, when there is no
data-center site, when the named demand set does not exist (the message lists
the sets) or is empty, when there are no corridor links, and when two metros
with traffic between them have no corridor route.

For example, take two metros 300 km apart with one PoP and one 25 MW data-center
site each, 200 Gbps per MW and uniform traffic. Each direction carries 5,000 Gbps.
With headroom 1.3 and a 3,200 Gbps quantum, the corridor needs 6,500 Gbps, which
rounds up to 9,600 Gbps. Each PoP's egress is then 9,600 Gbps, so with
`alpha_dc_to_pop: 1.2` each data-center link needs 11,520 Gbps and gets 12,800.

Because the arithmetic is exact, a capacity that lands exactly on a quantum step
stays there. With 1 MW per site instead, each direction carries 200 Gbps; with
headroom 1.1 and a 10 Gbps quantum the corridor needs exactly 220 Gbps and gets
220. TopoGen 0.5.2 computed 200 × 1.1 in floating point as 220.00000000000003
and gave the corridor 230. Capacities and optic counts can differ from 0.5.2
wherever its floating-point arithmetic disagreed with these rules, including
with `quantum_gbps: 0`: there 0.5.2 wrote this corridor as 220.00000000000003.

### Devices and hardware

The build expands the site blueprints into devices and links. A site link's
capacity is a budget for that pair of sites; it is divided equally among the
device links that connect them, and each device link carries the nearest float
to its exact share. `components.hw_component` assigns a router platform to each
device role.

`components.optics` maps `local_role->remote_role` to the optic installed at the
local end of a link. Each end gets the smallest number of optics whose total
capacity covers the device link's written capacity. Both capacities are read as
the exact decimals of their floats: the device-link capacity as written to the
scenario, and the optic capacity as the float NetGraph holds for the library
value. In the example above, one 9,600 Gbps device link with
`800G-ZR+` optics (800 Gbps each) needs 12 optics at each end, and each 12,800
Gbps data-center link needs 16. When `optics` is set, every link end needs a
mapping: a link between `core` and `leaf` routers needs both `core->leaf` and
`leaf->core`, or the build stops with an error. All device links of one site
link must map to the same optic at each end. An optic capacity that is not a
finite positive number, or a count too large to be written exactly as a float,
stops the build with an error that names the link, its end and the optic.

The scenario's `components` section holds the components that the
configuration names and those named inside the used blueprints. A name that the
library does not define stops the build; the error lists every unknown name.

### The scenario file

The scenario YAML contains `seed`, `blueprints`, `components`, the `network`,
`risk_groups`, failure policies under `failures`, traffic under `demands` and the
analysis `workflow`. Each demand names its source and target sites
(`^metro1/dc1/.*`), `mode: pairwise`, `priority`, `volume` and, when configured,
`preset`; gravity and hose demands also carry their distance as
`attrs.euclidean_km`. Building the scenario does not execute the workflow.

`output.scenario_seed` controls the traffic generator and the scenario's random
seed. Built-in analysis steps derive their seeds from it. A custom workflow can
set a step's seed explicitly to override that behavior.

## 3. Validate and save

`topogen.validation.validate_scenario(text, components)` checks the exact text
that would be written. It loads the scenario with NetGraph and prepares it for
analysis with every demand set bound, so NetGraph checks the schema, the
network, the failure policies, the demand sets and the selected workflow. When
NetGraph refuses the scenario, its message is the only issue. Otherwise
TopoGen's audits run on the prepared network and report, in this order:

1. Every demand set: no traffic that NetGraph would discard.
2. Data-center capacity: for each demand set, a data-center site's total sent
   traffic, and separately its total received traffic, must fit within the
   capacity of the enabled links that leave the site.
3. The network has devices, and every device has at least one link.
4. Every device has a role.
5. Platforms: when `components.hw_component` is set, every device role has a
   mapping key. A nonempty mapping requires a known platform on each device of
   that role; `dc: ""` permits a role without one. Every installed platform must
   exist and have a finite, positive count.
6. Optics: when `components.optics` is set, every link end has an optic; every
   named optic exists, and its count times its capacity covers the link
   capacity, by the same exact rule the build uses.
7. Each device's attached link capacity fits its platform's capacity, and the
   ports its optics use fit the platform's ports.

The result is a `Validation`: `issues`, a tuple of messages that is empty when
the scenario passed, and `prepared`, NetGraph's prepared scenario (absent when
NetGraph refused it). The drawings read their devices from `prepared`.

Nothing is written before validation passes, so a refused build leaves every
existing file unchanged. Then `write_scenario` replaces the scenario YAML and
`write_artifacts` writes the other files in this order: the traffic debug JSON
(only with `--debug-dir`), `<prefix>_network_graph.json`, the site map and the
blueprint diagrams enabled in the configuration. Each file is written to a
temporary file in its directory and then renamed over the destination, so a
reader sees the old file or the complete new one. New files get the permissions
of an ordinary file write (0666 minus the umask). If writing a file fails, the
command reports the error and keeps the files written before it.

### From Python

This is the command's sequence. It builds, validates and publishes the small
baseline from the repository root:

```python
from pathlib import Path

from topogen import RunContext, TopologyConfig, load_from_json
from topogen.library import load_library
from topogen.scenario import build_scenario, write_artifacts, write_scenario
from topogen.validation import validate_scenario

config = TopologyConfig.from_yaml(Path("examples/small_baseline.yml"))
graph, crs = load_from_json(Path("output/small_baseline_integrated_graph.json"))
if crs != config.projection.target_crs:
    raise ValueError(
        f"Graph CRS {crs} differs from configured {config.projection.target_crs}"
    )
build = build_scenario(graph, config, load_library(Path("lib")))
validation = validate_scenario(build.yaml, config.components)
if validation.issues:
    raise ValueError("Scenario validation failed:\n" + "\n".join(validation.issues))
write_scenario(Path("output/small_baseline_scenario.yml"), build.yaml)
write_artifacts(build, validation, config, RunContext(Path("output"), "small_baseline"))
```

`build_scenario` reads no file: the library is an argument. `load_library(path)`
returns the built-in definitions merged with the YAML files in that directory;
`load_library(None)` returns the built-ins alone. The `Build` it returns holds
the scenario text as `build.yaml` and the records the drawings and the
site-network JSON are made from: `traffic` (the demand sets, each demand with its
exact volume), `sizing` (each corridor link's peak load and each site link's
budget), `hardware` (each site link's device links, per-device capacity and
optics) and `site_plan` (the sites, the site links and the striping rules).
These records are read-only.

`write_artifacts` refuses a validation with issues. `topogen.RunContext` sets
the output directory, filename prefix and optional debug directory:
`RunContext(Path("output"), "small_baseline")` names files with the
`small_baseline_` prefix. The site map downloads its background tiles, so it
needs an internet connection.

## Libraries and output

The built-in library supplies blueprints, hardware components, failure policies
and workflows. The `topogen` command adds the files `blueprints.yml`,
`components.yml`, `failure_policies.yml` and `workflows.yml` from `lib/` in the
working directory; from Python, pass the directory to `load_library`. Each file
is a YAML mapping of names to definitions; custom names are added and matching
built-in names are replaced. A file that is not valid YAML or not a mapping is an
error when it is loaded.

Referenced site blueprints and their dependencies are included in the scenario.
Unknown references and cyclic blueprint dependencies are errors.

`workflows.assignments.default` chooses a workflow. Only the chosen workflow is
checked, when a scenario uses it: it must be a list of step mappings, and
validation checks its steps. Other entries in `lib/workflows.yml` do not affect
the build. Each step uses `type` for its analysis class and `demand_set` when it
needs traffic, with the step keys of NetGraph 0.25. A workflow written for
TopoGen 0.5.2 needs these renames: `resolution` to `alpha_tolerance`,
`parallelism` to `workers`, `store_failure_patterns` to `failure_trace`,
`include_flow_details` to `outputs.cost_distribution` and `include_used_edges` to
`outputs.edge_flows`; `alpha_from_field` is gone.

Built-in workflow steps use `traffic.matrix_name`; workflows from
`lib/workflows.yml` keep the names they list, even when they repeat a built-in
workflow exactly. In the built-in workflow, `MaximumSupportedDemand` searches for
the largest multiplier of the traffic that the network can carry. `TrafficMatrixPlacement` then places the
traffic, scaled by that multiplier, under each sampled failure of its
`failure_policy`.

The `failures` section contains the policy named by
`failure_policies.assignments.default` and every policy that a workflow step
names. File prefixes come from the configuration filename, not the names inside
the YAML. The batch script adds one subdirectory per configuration.

`import topogen` does not load Matplotlib or the drawing code. Import
`topogen.visualization` to draw maps and diagrams yourself.

## Working on the code

| Area | Main files |
| --- | --- |
| Settings and CLI | `topogen/config.py`, `topogen/cli.py`, `topogen/context.py` |
| Geographic processing | `topogen/metro_clusters.py`, `topogen/highway_graph.py`, `topogen/corridors.py`, `topogen/integrated_graph.py` |
| Libraries | `topogen/library.py` and the built-ins in `topogen/*_lib.py` |
| Build stages and records | `topogen/scenario/build.py`, `sites.py`, `striping.py`, `traffic.py`, `sizing.py`, `hardware.py`, `document.py`, `records.py` |
| Exact numbers and optic counts | `topogen/numeric.py` |
| Publication | `topogen/scenario/publication.py`, `topogen/scenario/artifacts.py` |
| Scenario checks | `topogen/validation/scenario.py`, `topogen/validation/audits/` |
| Maps and diagrams | `topogen/visualization.py`, `topogen/blueprint_viz.py` |

Run `make check-ci` after changes. It validates the example configurations and
runs formatting, lint, type checks and tests. Integration tests exercise scenario
building and NetGraph validation without downloading the national datasets.
To check geographic generation, download the files listed in the
[README](../README.md#download-the-geographic-data) and run both commands on this page.
