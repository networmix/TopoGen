# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
