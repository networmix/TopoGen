# TopoGen streamlining

Objective: simplify the topology pipeline, make its configuration and graph
contracts explicit, fix correctness gaps, and improve measured performance.
This is one cleanup with implementation gates, not a sequence of releases.

Baseline: `68fd680`; `make check-ci` passes (315 tests). Official Census inputs
and reproducible baseline artifacts are in ignored `data/` and `output/`.
Before-change profiles and all four sample scenarios are saved in
`output/streamline/before/`. Dragonfly assembly expands NetGraph DSL 28 times
for 27 site adjacencies; capacity counting accounts for over half of profiled
assembly time.

## Gates

1. **One network expansion for assembly.** Generate demands once; retain their
   identity through sizing and emission. Expand the complete network once,
   including node rules; index concrete links by site edge. Use that index for
   capacity allocation and optics. Never group distinct capacity budgets by a
   shared adjacency family. Verify heterogeneous capacities, striping, custom
   blueprints, and semantic equivalence of the four examples.
2. **Explicit configuration and execution context.** Separate serializable
   settings from paths and export context. Remove hidden config attributes,
   no-op controls, duplicated link parsing, and fallback logic that conceals
   invalid state. Keep CLI errors actionable and only report success after
   validation. Document intentional contract changes.
3. **Coherent geographic graph contract.** Fix nearest-neighbor adjacency
   dependence on metro IDs; make path multiplicity honest throughout discovery,
   serialization, site construction, and risks. Replace whole-road-graph scans
   per corridor with path/index traversal; preserve geographic provenance.
   Verify on adversarial small graphs and the downloaded national datasets.
4. **Validation and delivery.** Preserve node rules in audits, avoid repeated
   expansion where the same network suffices, and keep construction, hardware,
   capacity and failure-policy checks. Run `make check-ci`, real generation and
   build for all example designs, before/after semantic comparisons and
   repeatable timings. Update README and Unreleased changelog with the final
   contracts, evidence and limitations. No push is part of this task.

Progress is recorded below as evidence becomes available. Existing behavior is
not a correctness oracle where a regression demonstrates a bug. Timing claims
must separate computation from optional map rendering and network tile access.


## Completed evidence

All four gates are complete. The package now has one explicit corridor graph
contract and one assembly path; removed APIs and formats have no compatibility
adapter. Required migration steps are in [README](../README.md#current-contracts-and-migration).

- Assembly generates traffic once and expands the complete device DSL once,
  including node rules. Capacities and endpoint optics use the actual concrete
  links for each site edge. Heterogeneous PoP budgets, asymmetric optics, missing
  reverse assignments and caller link attributes have regression coverage.
- Typed `RunContext` owns output paths; configuration has no hidden export state.
  Schema checks reject removed controls and formats. Library parsing is shared;
  malformed libraries and computation/render failures propagate. CLI validation
  precedes atomic YAML replacement, including `--print`.
- Corridor identity survives contraction, k-nearest pairing, k-path extraction,
  JSON and site expansion. Risks use the owning path's length and reject unknown
  owners. Tests cover parallel routes of different lengths, ID-order dependence,
  JSON round trips and shared risks. Road copies and scans were removed from
  contraction and corridor extraction where ownership/indexing suffices.
- `make check-ci`: **324 passed**, 87.61% coverage; Ruff formatting/lint, Pyright
  and all four example schemas pass. Pyright reports one existing dependency
  typing warning for Contextily (no type stub), with zero errors. Full log:
  `output/streamline/final-ci.log`. Real NetGraph parsing and reduced workflows
  execute in integration tests. Published changelog history is unchanged.

### Repeatable assembly measurement

Recorded for the initial cleanup, before the follow-up review below. These
timings were not rerun after the audit corrections.

Local Python 3.13.13, NetGraph 0.24.0, NetGraph-Core 0.11.0. Compare baseline
`68fd680` in an isolated checkout with this implementation, using the **same
frozen geographic graph**, corresponding example configs and one virtualenv.
One warmup, seven timed iterations; median milliseconds. Optional maps and
artifact I/O are excluded. This is an assembly measurement, not a claim about
national geographic generation or NetGraph simulation throughput.

| Example | Device nodes / links | Before, ms | After, ms | Build speedup |
| --- | ---: | ---: | ---: | ---: |
| small_baseline | 13 / 27 | 32.2 | 30.1 | 1.07x |
| small_clos | 127 / 756 | 55.3 | 37.4 | 1.48x |
| small_dragonfly | 133 / 739 | 83.5 | 43.9 | 1.90x |
| small_dragonfly_custom | 103 / 612 | 64.2 | 39.9 | 1.61x |

Evidence: `output/streamline/compare_assembly.py`, `benchmark-before/` and
`benchmark-after/` under `output/streamline/`. Every iteration passed scenario
validation. All four parsed scenario documents are equal after accounting for
explicit `:path0` adjacency identities and newly preserved `euclidean_km` /
`detour_ratio` attributes. Comparison script and result:
`verify_semantics.py`, `semantic-comparison.json` in the same directory.
Corrected geographic routing can intentionally change distances when regenerating
from Census inputs; the frozen-graph comparison isolates assembly changes.

### Real-data artifacts

Official Census ZIPs, provenance and integrity/schema/CRS checks are in `data/`
and `data/SOURCES.json`: 2,645 urban areas, 17,522 primary-road features and 56
state/territory boundaries. All four examples were generated from these data,
then built and validated. Each has 5 metros and 8 corridors. Current artifacts
are in `output/` and `output/streamline/final/`; old outputs are retained under
`output/streamline/before/`.

```bash
venv/bin/python -m topogen generate examples/small_baseline.yml -o output
venv/bin/python -m topogen build examples/small_baseline.yml -o output
```

The same commands succeed for `small_clos`, `small_dragonfly` and
`small_dragonfly_custom`. Maps and blueprint diagrams were rendered; the
baseline geographic/site maps and Clos diagram were visually inspected.
The real example builds with maps took about 12 seconds per CLI invocation;
Contextily tile access and rendering dominate compared with assembly alone.
Full Monte Carlo workflows on these real artifacts were not run. No remote CI,
commit, push or release was performed.


## Follow-up correctness review

The earlier passing suite did not establish absence of defects. Added behavioral
witnesses before fixes; failures and verification logs are in `output/recheck/`.
No compatibility path or format was added during this review.

- **DC demand accounting:** separate demand sets were summed, regex selectors
  bypassed checks, and DSL rules were used instead of actual external link
  capacity. The audit now resolves demands through NetGraph, checks every set
  independently, excludes internal/disabled links and reports unmatched demands.
  Dictionary-only validation now explicitly covers metadata/references, while
  capacity auditing belongs to the NetGraph-backed path.
- **Hose fitting:** impossible margins could silently become a different profile.
  Both fitting stages now share one vectorized fitter with feasibility and
  convergence checks; exactly saturated margins use their unique star solution.
- **Disabled rounding:** gravity and hose rounded to two decimals even with
  rounding disabled, erasing small demands. They now preserve unquantized values.
- **Numeric/config contracts:** integer strings above 2^53 lost precision through
  float conversion, and NaN/infinity could bypass schema bounds. Exact parsing
  and recursive finite-number checks reject these cases.
- **Geographic contracts:** projected metre units are now required; degree/foot
  CRS values previously reached metre-based calculations. Duplicate metro names
  could merge site identities; names/IDs and coordinate values are now checked.
  PoP coordinates are cross-checked even when a metro has no DC group.
- **Validation robustness:** malformed network/attribute/reference shapes report
  issues instead of aborting the validator.
- **Component inventory:** hardware overridden in a blueprint remained a library
  dependency. Inventory now scans finalized blueprints and reuses that section.
- **Cleanup:** removed an unused graph-filter argument, a silent role-map
  substitution and duplicated IPF loops; corrected library and audit API docs.

Final `make check-ci`: **341 passed**, **87.77% coverage**, Ruff and all four
example schemas pass, Pyright has zero errors and the existing Contextily stub
warning. Added 17 regression cases in `tests/test_review_regressions.py`.
`git diff --check` passes; published changelog history is preserved.

Additional behavioral evidence:

- Contraction preserves the first five shortest simple-path lengths on 163
  connected random graphs (`contraction-differential.txt`).
- Sixty seeded hose profiles at scales 1e-6, 1 and 1e6 preserve every DC total
  within 1e-8 relative error (`hose-numerics.txt`).
- All four real-data examples are rebuilt and validated with no change to their
  parsed scenario documents. All four workflow steps execute on each example
  with iteration counts reduced to 3; MSD alpha is positive (1.6875). Results and
  the reproducible runner are `real-workflow-results.json` and
  `run-real-scenarios.py` in `output/recheck/`.

This verifies local construction, validation and reduced workflow execution.
Full Monte Carlo runs and remote CI were not run. Changes remain uncommitted;
no push or release was performed.


## Full-source simplification and performance audit (completed)

Objective: inspect all package code and support scripts for straightforward
control flow, duplication, defects and unnecessary computation. Preserve no
legacy compatibility paths. Previous test success is a baseline, not completion.

Coverage gates:

1. Traffic models: shared pair selection/rounding/emission, deterministic PRNG,
   disabled-debug overhead, bounded fitting and conservation witnesses.
2. Site graph and sizing: endpoint/stripe selection, distinct corridor identities,
   repeated expansion/I/O, topology modes, capacity and serialization contracts.
3. Configuration and libraries: one validation boundary, exact conversion,
   overrides/references, coherent public loading API.
4. Geographic pipeline: metro selection, road anchoring/contraction, corridor
   discovery/risks, serialization and representative real-data execution.
5. Remaining modules: CLI/context, audits, visualization/helpers, entry points,
   packaging/dev/CI/workspace scripts and test adequacy.
6. Final evidence: full current checks, adversarial witnesses, before/after
   measurements on identical inputs, real examples and package build.

Inspection and performance artifacts are in `output/full-review/`; the 341-test
result in `output/recheck/` is the starting baseline. Current completion evidence
for every coverage gate is recorded below.

### Full-source audit completion

All six coverage gates are complete for the current local source. This is a
behavioral/code audit, not a proof that no possible defect remains. No legacy
adapter or compatibility re-export was introduced.

| Coverage gate | Inspected scope and outcome |
| --- | --- |
| Traffic | All three models, common inventory, partner selection, fitting, quantization, PRNG and output. Shared emission replaces duplicate branches; debug formatting is conditional. |
| Sites and sizing | All adjacency families, roles, stripe selection, expansion, capacity routing, risk ownership and serialization. Construction, sizing and network serialization have separate modules. Ordered metro-pair demands share one routing operation and one capacity update rule. |
| Config and libraries | Every dataclass/schema section and all four libraries. Schema owns scalar bounds; exact key conversion and cross-field checks remain explicit. Libraries include nested blueprint dependencies, detect cycles, and are finalized once before stripe selection. Workflow definitions load once per assembly. |
| Geography | UAC filtering/overrides, naming, coordinates, road loading/snapping, contraction, anchoring, nearest-neighbor pairs, corridor extraction/risk and JSON. The four national-data examples were regenerated and compared with saved graphs. |
| Remaining package and support | CLI/context/logging/public exports, all validation audits, maps/blueprint diagrams, Makefile, batch/analyzer/workspace scripts, packaging and CI configuration. Tests were migrated to current internal APIs; no compatibility wrappers. |
| Verification | Full local CI target, baseline-failing witnesses, traffic differential/performance checks, real generation/build/validation/reduced workflow runs, package build/metadata checks and installed-wheel smoke test. |

Additional defects isolated and fixed during this audit:

- Gravity top-K selection depended on alphabetical metro-name order; jitter used
  the global PRNG. Zero-share classes could also enter invalid normalization.
- Uniform traffic expanded across devices, introducing intra-DC traffic and
  changing site shares with blueprint size. Explicit `group_pairwise` emission
  now matches DC-level sizing.
- Striping ignored `one_to_one` and custom endpoint matches, collapsed equal
  device basenames from distinct groups and treated literal names as regexes.
  Stripe rules now address exact full paths and merge attributes per device.
  Disabled corridor risks no longer leak into emitted adjacencies.
- Config priority conversion truncated fractions; normalized duplicates and
  repeated metro overrides overwrote previous entries. Unknown stripe keys and
  unmatched forced metros were silently ignored. They now fail explicitly.
- Configured link cost was ignored for geometric links. It now acts as a minimum
  routing cost while `distance_km` retains the geographic value.
- Sizing accepted partial endpoint patterns and non-finite/negative volume.
  Those inputs are rejected before routing.
- Nested blueprint definitions omitted their dependencies. Both build and stripe
  selection now use the same finalized library with recursive dependencies.
- Blueprint diagrams implemented a second, permissive DSL parser; dictionary
  selectors created phantom groups. Both panels now use actual expanded devices,
  with group counts and summed link capacities.
- The UAC boundary mask assumed NAD83 regardless of source CRS. Metro coordinates
  were NumPy scalars, causing direct `generate -> build` Python calls to fail YAML
  emission even though a JSON round trip succeeded. The loader now returns native
  floats and uses the actual source CRS for filtering.
- Role audits accepted `None`, boolean and numeric roles as strings; chassis
  capacity audits accepted NaN/infinity. These malformed values now produce issues.
- Batch builds returned exit 0 after stage failures. The analyzer called k-core
  routines on unsupported MultiGraphs. Both have current-contract behavior;
  vertex analysis uses simple adjacency and preserves reported corridor counts.

Final evidence (all paths below are relative to `output/full-review/`):

- `final-check-ci.log`: **375 passed**, **89.52% coverage**; Ruff and four config
  validations pass; Pyright has **0 errors**, with one existing missing Contextily
  stub warning. `git diff --check` and shell syntax checks pass.
- The `*-before-tests.log` files retain failing witnesses; `real-examples-before.log`
  records the direct Python/YAML scalar failure before its fix.
- `traffic-parity.log`: **72 profiles** cover gravity/hose, multiple inventory
  sizes and all rounding policies. Non-volume fields match exactly; directional
  volumes match within 1e-12 relative / 1e-10 absolute tolerance.
- `traffic-before-bench.json` and `traffic-final-bench.json`: seven-run local
  medians on identical inputs, logging disabled, allocation peak measured
  separately. Gravity (80 DCs, 18,960 entries): **90.97 -> 6.50 ms** and
  **29,040,021 -> 9,586,183 bytes**. Hose (40 DCs, 3 samples, 14,040 entries):
  **12.56 -> 5.33 ms** and **9,911,842 -> 6,815,101 bytes**. These are generator
  microbenchmarks, not end-to-end or cross-machine performance claims.
- `real/results.json`: all four examples regenerate **5 metros / 8 corridors**;
  device/link counts are **13/27**, **127/756**, **133/739**, **103/612**.
  Four workflow steps run on each with Monte Carlo iterations reduced to **3**;
  MSD alpha remains **1.6875**. Full Monte Carlo runs were not performed.
- `network-parity.json`: all four expanded networks match the saved baseline
  exactly in device attrs and link capacities/costs/risks/hardware. The scenario
  text changes only by explicit `logic: or` on single-condition role selectors;
  the real examples use hose, so the uniform contract change is exercised by
  separate heterogeneous-DC regression tests.
- Fresh maps and blueprint diagrams are under `real/`; the final Clos diagram
  was visually inspected for counts, capacities, group layout and readable labels.
- `package-build.log`, `package-check.log`, `pip-check.log`, `wheel-smoke.log`:
  wheel/sdist build, Twine metadata checks and dependency checks pass. Importing
  the installed wheel (outside the source package path) loads its schema and
  builds/validates the real baseline.

Released changelog history remains byte-for-byte unchanged. Work is local on
`explain-topology-pipeline-2cd9ea05`, based on `68fd680`; changes remain uncommitted.
No push, remote CI, release or deployment was performed.

### Contextily typing follow-up (2026-10-04)

Added `typings/contextily/__init__.pyi` for the `add_basemap` arguments used by
TopoGen, checked against installed Contextily 1.7.1. This is an explicitly limited
static description, with no runtime wrapper, `Any` escape hatch or diagnostic
suppression. Pyright uses the explicit `typings` stub path.

`make lint` now reports **0 errors / 0 warnings**. A separate temporary type probe
accepts the real call and rejects incorrect alpha, attribution and Axes types
(`output/full-review/contextily-typing/probe-result.json`). No runtime source was
changed; the earlier 375-test result remains the latest full-suite run.

### Code and architecture recheck (2026-10-04)

This follow-up supersedes the earlier test count. It checks current production
modules, repository callers, tests, comments/docstrings, CLI/batch scripts and
example configuration. GeoPandas, Matplotlib, Contextily and its local stub remain.

Removed twelve helpers whose only repository consumers were their own tests:
the four point/transform/bearing functions, three MetroCluster convenience
methods, TopologyConfig.summary, and four singular/list/filter component APIs.
The effective component library remains available through get_builtin_components;
topogen info remains the configuration summary command. These are intentional
API removals with no compatibility aliases, documented in README and Unreleased.

Removed the unused corridor_path_ids edge tags and three redundant CorridorPath
fields. Scenario construction now reads the corridor MultiGraph directly instead
of copying every edge to an intermediate list of dictionaries. The shared site
edge identity lives in naming rather than forcing serialization to depend on
device expansion. Sizing resolves the Core flow-placement enum directly and
combines capacity adjustment, aggregate reporting and PoP egress accumulation.
Removed unused loggers, an empty TYPE_CHECKING block, obsolete type/noqa
suppression comments and dead batch-runner state. Production Python is smaller by
253 lines relative to this review's starting snapshot.

Two misleading diagnostics were reproduced before changes:

- A DC-to-PoP YAML rule received an “Inter-metro corridors” heading because the
  comment inserter searched the next 15 lines. Deleted this text heuristic;
  link_type attributes remain authoritative. Witness: before-build.log.
- Largest-component filtering legitimately removed a metro anchor, but raised
  a RuntimeError alleging a contraction bug. It now raises ValueError identifying
  highway filtering and the relevant settings. The new regression fails against
  the starting source (anchor-before.log) and passes after correction.

Corrected descriptions of full-mesh PoP adjacency with ring-arc costs,
representative points versus centroids, hose same-metro weighting, component
filtering and geometry cleanup counts. Reused the single dissolved CONUS geometry
without dissolving it again. Duplicate snapped road segments have the same length
by construction, so their unreachable shorter-edge branch was removed.

CLI tests no longer restore global print or mock sys.exit with another SystemExit
implementation. They now require the expected exceptions. The quiet-mode test
uses valid CLI syntax, checks successful dispatch, and demonstrates that the same
stub prints when quiet mode is absent. Previously it passed on argparse rejecting
the unsupported -c option. Ruff now checks unused noqa directives. The two SciPy
imports retain narrow missing-stub suppressions; removing them was checked and
does produce real missing-stub diagnostics. Contextily needs no suppression.

Evidence is under output/architecture-review/:

- final-check-ci.log: 348 tests pass, coverage 89.69%, all four configuration
  schemas validate. The count is 375 minus 28 cases belonging to deleted helpers
  plus one new anchor-filtering regression. No remaining behavior tests were
  removed. An optional unused-type-ignore check exposed a warning inside
  Pyright's bundled typeshed; that option was not retained. Final lint is recorded
  separately in final-lint.log.
- focused.log: 61 geographic, sizing and assembly checks pass.
- cli-check.log: 18 CLI and batch checks pass, including failure propagation.
- verify_examples.py, example-parity.json: actual national geographic inputs
  regenerate the exact saved graph (5 metros, 8 corridors). All four example
  geographic configurations were first checked equal, then each scenario rebuilt
  and validated. Parsed YAML and expanded node/link attributes, capacities, costs,
  risks and hardware exactly match the pre-review snapshots. Device/link counts:
  baseline 13/27, Clos 127/756, Dragonfly 133/739, custom Dragonfly 103/612.
- A production symbol-reference scan found no further unreferenced functions
  beyond the PyYAML ignore_aliases override, which is invoked by the serializer.
  This is repository-level evidence, not proof about unknown external consumers.

No new workflow simulation, tile download or performance benchmark was needed for
this pass. Shell syntax and git diff --check pass; released changelog history is
unchanged. This review was completed before committing and opening the PR.
