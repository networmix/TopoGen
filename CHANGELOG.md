# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Build**: Write the scenario and then each artifact atomically after validation; a refused build no longer replaces `*_network_graph.json`.
- **Scenario**: DC devices carry their own site's power as `mw_per_dc_region`; 0.5.2 wrote the global default even with overrides.
- **Sizing**: Use exact budgets and written-value optic ceilings; capacities and counts may change, including with `quantum_gbps: 0`.
- **Validation**: Check optic coverage with the same exact rule as the build; 0.5.2's float product could refuse a sufficient count.

### Changed

- **BREAKING**: **Package**: Install `netlab-topogen`, import `netlab_topogen`, and run `netlab-topogen`.

- **BREAKING**: **Dependencies**: Require `ngraph>=0.25.0,<0.26`; NetLab-TopoGen no longer depends on `netgraph-core` directly.
- **BREAKING**: **Traffic**: `traffic.flow_policy_config` takes NetGraph 0.25 preset names, case-sensitive; `TE_WCMP_UNLIM` is `TE_MAXFLOW_UNLIM`.
- **BREAKING**: **Traffic**: Uniform traffic is one demand per ordered pair of distinct DC sites; a single DC site with traffic is refused.
- **BREAKING**: **Traffic**: `mw_per_dc_region_overrides` refuses keys naming no metro or DC site, and negative powers, even with traffic off.
- **BREAKING**: **Workflows**: Use NetGraph 0.25 step keys (`workers`, `failure_trace`, `outputs`, `alpha_tolerance`); drop `alpha_from_field`.
- **BREAKING**: **Scenario**: Demands carry `preset` instead of `flow_policy`; workflow steps use NetGraph 0.25 keys.
- **BREAKING**: **Scenario**: One node group per site (`metro1/dc1`, not `metro1/dc[1-2]`); NetGraph expands the same network.
- **BREAKING**: **Python API**: `build_scenario(graph, config, library)` returns a `Build` and writes no files; pass `load_library(path)`.
- **BREAKING**: **Python API**: `validate_scenario(text, components)` returns a `Validation` with the issues and NetGraph's prepared scenario.
- **BREAKING**: **Python API**: `import netlab_topogen` no longer loads Matplotlib or drawing code; import `netlab_topogen.visualization` to draw.
- **BREAKING**: **Traffic**: Update `--debug-dir` readers for site demands with `source`, `target`, `volume` and `preset`.
- **Validation**: Run NetGraph's `prepare` with every demand set bound; refuse discarded traffic; check DC capacity on resolved demands.
- **Workflows**: Check a library workflow only when the configuration selects it; an unused broken entry no longer stops builds.
- **Build**: Report all unknown components in one error, after sizing errors; invalid optic capacities name the link end and optic.
- **Internal**: Build through read-only stage records in new `netlab_topogen.scenario` modules; sizing no longer calls NetGraph-Core.

### Added

- **Python API**: Add `netlab_topogen.library.load_library`, and `write_scenario` and `write_artifacts` in `netlab_topogen.scenario`.

### Removed

- **BREAKING**: **Config**: Remove `build.tm_sizing.flow_placement`; sizing always splits equally over equal-cost routes. Delete the key.
- **BREAKING**: **Python API**: Remove `validate_scenario_yaml` and `validate_scenario_dict`; use `netlab_topogen.validation.validate_scenario`.
- **BREAKING**: **Python API**: Remove the `get_builtin_*` library getters and `load_user_library`; use `netlab_topogen.library.load_library`.

## [0.5.2] - 2026-10-25

### Fixed

- **Workflows**: Derive failure-sampling seeds from the scenario seed instead of fixing the placement step at 42.

### Changed

- **Docs**: Explain installation, configuration and scenario generation in a standalone pipeline guide.

## [0.5.1] - 2026-10-25

### Changed

- **Dependencies**: Require `ngraph>=0.24.1,<0.25.0` and `netgraph-core>=0.10.0,<0.12` to stay on the supported NetGraph API.

## [0.5.0] - 2026-10-04

### Fixed

- **Geography**: Preserve alternate highway routes and their risks through generation and build; remove metro-order dependence from nearest-neighbor selection and fix coordinate-system handling.
- **Traffic**: Distribute uniform traffic between DC sites; make gravity partner selection independent of metro ordering and jitter reproducible. Reject impossible or unconverged hose fits and preserve small demands when rounding is disabled.
- **Capacity**: Size each corridor for peak directional load and match NetGraph's DC-pair traffic distribution. Check each demand set against actual external DC capacity, including regex-selected endpoints.
- **Routing**: Honor configured minimum link costs without changing geographic distances.
- **Hardware**: Apply striping and optic counts to the correct devices and link endpoints, respecting one-to-one site connections, hardware overrides and disabled risk groups.
- **Blueprints**: Restore missing Dragonfly links, include nested dependencies and reject cycles. Diagrams now show the expanded device network.
- **Workflows**: Honor `traffic.matrix_name`, use `design_analysis_brief` consistently, and allow `empty` as a no-failure policy.
- **Build**: Validate scenarios before replacing output YAML; preserve existing files on failure. Batch builds now return a failure status if any configuration fails.
- **Docs**: Correct installation instructions to use this repository on GitHub.

### Changed

- **BREAKING**: **NetGraph**: Require `ngraph>=0.24.0` and `netgraph-core>=0.10.0`. Generated YAML uses `demands` and `failures`; update tools that read these sections.
- **BREAKING**: **Workflows**: In `lib/workflows.yml`, replace `step_type` with `type` and `matrix_name` with `demand_set`; remove `placement_rounds`, `acceptance_rule`, `seeds_per_alpha` and `baseline`. Custom workflows are validated before output.
- **BREAKING**: **Failures**: Custom rules in `lib/failure_policies.yml` must use `scope`, `mode` and `match`.
- **BREAKING**: **Blueprints**: Replace `{var}` link selectors in `lib/blueprints.yml` with `$var` or `${var}`.
- **BREAKING**: **Traffic**: Use preset names in `traffic.flow_policy_config`. Capacity sizing now rejects unsupported demand endpoints instead of silently skipping their traffic.
- **BREAKING**: **Graphs**: Regenerate saved graphs with `topogen generate`; corridor JSON now requires a `MultiGraph` with explicit edge keys.
- **BREAKING**: **Config**: Remove ignored metadata, road-processing controls, `edge_select`, component assignments/library fields and scenario overrides. See the [current schema](topogen/schemas/topogen_config.json) for supported settings.
- **BREAKING**: **Optics**: Define each direction explicitly with `local->remote` mappings; reverse assignments are no longer inferred. Use `role|role` strings for link role selectors.
- **Validation**: Reject ambiguous or invalid configuration, including duplicate metro overrides, fractional integer fields, non-finite numbers and unmatched forced metros. Geographic calculations require a projected CRS in metres.
- **Build**: Reuse generated traffic and the expanded device network across capacity checks, hardware assignment and output.

### Added

- **Superset**: Add workspace setup and check commands.

### Removed

- **BREAKING**: **Python API**: Remove `topogen.scenario_builder`; use `topogen.scenario.build_scenario` and `RunContext` for output paths. Pass visualization settings explicitly.
- **BREAKING**: **Python API**: Remove point/bearing utilities, `MetroCluster` distance/overlap/array helpers, `TopologyConfig.summary()` and singular/list/filter component helpers. Use `get_builtin_components()` for components and `topogen info` for configuration summaries.

## [0.4.0] - 2026-03-15

### Changed

- Switch to the MIT license.

## [0.3.1] - 2025-12-07

### Fixed

- **TM Sizing**: Parallel edges (striped corridors) between metro pairs now sized independently. Previously, only one edge per metro pair was updated during TM-based capacity sizing, leaving striped links undersized.

## [0.3.0] - 2025-11-XX

### Changed

- Initial versioned release with TM-based capacity sizing.
