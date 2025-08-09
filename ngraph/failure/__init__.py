"""Failure modeling package.

Provides primitives to define failure selection rules and to run Monte Carlo
failure analyses. The `policy` module defines data classes for expressing
selection logic over nodes, links, and risk groups. The `manager` subpackage
contains the engine that applies those policies to a `NetworkView` and runs
iterative analyses.

Public entry points:

- `ngraph.failure.policy` – failure selection rules and policy application
- `ngraph.failure.manager` – `FailureManager` for running analyses
"""
