# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **TM Sizing**: Split uniform traffic over ordered DC pairs, including same-metro pairs, to match NetGraph demand expansion.
- **TM Sizing**: Size corridors for peak directional load even when `respect_min_base_capacity` is disabled.
- **TM Sizing**: Preserve separate corridors with repeated edge keys across different PoP pairs.
- **Blueprints**: Fix Dragonfly variable expansion to restore missing intra-group and inter-group links.
- **Metros**: Select the largest override match with current pandas types, including non-numeric indices.
- **Workflows**: Use `design_analysis_brief` as the default in both Python and YAML configurations.
- **Workflows**: Bind built-in demand references to `traffic.matrix_name`.
- **Failures**: Make the `empty` policy a valid no-failure mode.

### Changed

- **BREAKING**: **Scenarios**: Emit NetGraph `demands` and `failures` sections; update consumers of generated YAML.
- **BREAKING**: **Failures**: Custom rules must use `scope`, `mode` and `match`; migrate `lib/failure_policies.yml`.
- **BREAKING**: **Blueprints**: Link selectors in `lib/blueprints.yml` use NetGraph `$var`/`${var}` placeholders; `{var}` is no longer expanded.
- **BREAKING**: **Workflows**: Reject `placement_rounds`, `acceptance_rule`, `seeds_per_alpha`, `baseline`, `step_type`, `matrix_name`.
- **BREAKING**: **Traffic**: `traffic.flow_policy_config` requires preset names; replace integer values.
- **BREAKING**: **TM Sizing**: Reject unsupported demand endpoints instead of silently omitting their traffic.
- **Dependencies**: Require `ngraph>=0.23.1` and `netgraph-core>=0.10.0`.
- **Workflows**: Validate `lib/workflows.yml` with NetGraph before emission; steps use `type` and `demand_set`.
- **Workflows**: Use `parallelism: auto` in built-in and sample workflows.
- **Validation**: Construct the full NetGraph scenario to check workflow arguments and failure policies before topology audits.
- **Internal**: Remove redundant validation and scenario code; shorten comments and documentation.

### Added

- **Superset**: Add setup, teardown and run commands to install dev dependencies, copy local `.env` files and run checks.

## [0.4.0] - 2026-03-15

### Changed

- Switch to the MIT license.

## [0.3.1] - 2025-12-07

### Fixed

- **TM Sizing**: Parallel edges (striped corridors) between metro pairs now sized independently. Previously, only one edge per metro pair was updated during TM-based capacity sizing, leaving striped links undersized.

## [0.3.0] - 2025-11-XX

### Changed

- Initial versioned release with TM-based capacity sizing.
