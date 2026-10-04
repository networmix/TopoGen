# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Diagnostics**: Report metro anchors removed by highway filtering as a configuration/data error; correct ring-mesh, representative-point and geometry-cleanup descriptions. Stop inserting heuristic YAML section comments that could label the wrong links.
- **Traffic**: Keep top-K gravity partners independent of metro-name ordering; seed jitter locally and distribute uniform traffic between DC sites rather than individual devices.
- **Striping**: Respect one-to-one site mode and endpoint matches; identify devices by full literal paths and preserve disabled risk-group settings.
- **Config**: Reject fractional/duplicate priority keys, duplicate metro overrides, unknown striping fields and unmatched forced metro patterns. Apply configured link cost as a minimum without changing geographic distances.
- **Blueprints**: Include recursive dependencies and reject cycles; derive diagrams from resolved devices instead of a duplicate DSL interpreter.
- **Geography**: Filter urban areas using their actual CRS and return YAML-safe coordinates for direct Python generation/build calls.
- **Validation**: Reject non-string device roles and non-finite chassis capacities.
- **Tools**: Return failure from batch builds when any configuration fails; analyze vertex connectivity and k-cores on simple metro adjacency while preserving corridor counts.

- **Traffic**: Reject infeasible or unconverged hose fits; preserve small demands when rounding is disabled.
- **Validation**: Audit each demand set against expanded external DC capacity; resolve regex selectors, report unmatched demands, and handle malformed network structures without crashing.
- **Contracts**: Preserve exact integer strings; reject non-finite config values, non-metre/geographic CRS and duplicate metro identities; check PoP coordinates without DC groups.
- **Components**: Collect blueprint component dependencies after hardware overrides, removing replaced platform references.
- **TM Sizing**: Split uniform traffic over ordered DC pairs, including same-metro pairs, to match NetGraph demand expansion.
- **TM Sizing**: Size corridors for peak directional load even when `respect_min_base_capacity` is disabled.
- **TM Sizing**: Preserve separate corridors with repeated edge keys across different PoP pairs.
- **Blueprints**: Fix Dragonfly variable expansion to restore missing intra-group and inter-group links.
- **Metros**: Select the largest override match with current pandas types, including non-numeric indices.
- **Workflows**: Use `design_analysis_brief` as the default in both Python and YAML configurations.
- **Workflows**: Bind built-in demand references to `traffic.matrix_name`.
- **Failures**: Make the `empty` policy a valid no-failure mode.
- **Docs**: Install from GitHub; the `topogen` name on PyPI belongs to an unrelated project.
- **Geography**: Preserve alternate highway routes, isolated rings and all `k_paths` through corridor JSON and site expansion; form nearest-neighbor pairs independently of metro ID order. Resolve risk distance from its owning path and reject unknown risk references.
- **Hardware**: Resolve optic counts for each site edge and endpoint; preserve caller link attributes and geographic corridor metadata.
- **Validation**: Preserve node rules in expansion audits, require scenario-local component definitions, and audit declared optic port usage.
- **CLI**: Validate before atomically replacing scenario YAML, including `--print`; scope quiet output to the command.

### Changed

- **Internal**: Remove duplicate corridor metadata and intermediate edge copying; decouple site identifiers from device expansion and consolidate sizing capacity passes.
- **Checks**: Detect unused Ruff suppressions; make CLI tests verify successful quiet dispatch and required exception propagation.
- **Typing**: Describe the Contextily API used by TopoGen in a scoped local stub; keep missing-stub diagnostics enabled.
- **Internal**: Share traffic pair selection/rounding/emission, skip disabled debug formatting, route aggregated metro-pair demand once, reuse finalized blueprint libraries and build workflows once.
- **Internal**: Separate site construction, sizing and serialization; centralize schema bounds and metro naming. Remove the unused metro-loader formatting argument; configuration dataclasses reject undeclared attributes.

- **BREAKING**: **Scenarios**: Emit NetGraph `demands` and `failures` sections; update consumers of generated YAML.
- **BREAKING**: **Failures**: Custom rules must use `scope`, `mode` and `match`; migrate `lib/failure_policies.yml`.
- **BREAKING**: **Blueprints**: Link selectors in `lib/blueprints.yml` use NetGraph `$var`/`${var}` placeholders; `{var}` is no longer expanded.
- **BREAKING**: **Workflows**: Reject `placement_rounds`, `acceptance_rule`, `seeds_per_alpha`, `baseline`, `step_type`, `matrix_name`.
- **BREAKING**: **Traffic**: `traffic.flow_policy_config` requires preset names; replace integer values.
- **BREAKING**: **TM Sizing**: Reject unsupported demand endpoints instead of silently omitting their traffic.
- **Dependencies**: Require `ngraph>=0.24.0` and `netgraph-core>=0.10.0`.
- **Workflows**: Validate `lib/workflows.yml` with NetGraph before emission; steps use `type` and `demand_set`.
- **Workflows**: Use `parallelism: auto` in built-in and sample workflows.
- **Validation**: Construct the full NetGraph scenario to check workflow arguments and failure policies before topology audits.
- **Internal**: Remove redundant validation and scenario code, the PyPI publish workflow, the pre-commit Pyright hook and benchmark flags.
- **BREAKING**: **Contracts**: Require corridor `MultiGraph` JSON with explicit edge keys; regenerate saved graphs. Use `topogen.scenario.build_scenario`, `RunContext` and public visualization settings.
- **BREAKING**: **Config**: Remove ignored metadata, road-processing controls, `edge_select`, component assignments/library fields and scenario overrides; reject invalid integer quantities.
- **BREAKING**: **Optics**: Require explicit `local->remote` endpoint mappings; remove implicit reverse and legacy delimiter behavior. Link role selectors use `role|role` strings.
- **Assembly**: Generate traffic and expand the device network once per build; reuse the result for budgets, hardware and artifacts.
- **Internal**: Consolidate link parsing and user-library loading; remove format guessing, swallowed computation/render errors and duplicated graph copies.

### Added

- **Superset**: Add setup, teardown and run commands to install dev dependencies, copy local `.env` files and run checks.

### Removed

- **BREAKING**: Remove unused point/bearing utilities, `MetroCluster` distance/overlap/array helpers, `TopologyConfig.summary()` and singular/list/filter component helpers. Use the merged `get_builtin_components()` mapping and `topogen info`.
- **Dependencies**: Unused `seaborn`, `rich`, `nbformat`, `nbconvert`, `ipykernel`, `itables`, `rasterio` and `scikit-learn` requirements.

## [0.4.0] - 2026-03-15

### Changed

- Switch to the MIT license.

## [0.3.1] - 2025-12-07

### Fixed

- **TM Sizing**: Parallel edges (striped corridors) between metro pairs now sized independently. Previously, only one edge per metro pair was updated during TM-based capacity sizing, leaving striped links undersized.

## [0.3.0] - 2025-11-XX

### Changed

- Initial versioned release with TM-based capacity sizing.
