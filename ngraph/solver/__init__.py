"""High-level solver interfaces binding models to algorithm implementations.

This package exposes problem-oriented APIs (e.g., max-flow between groups in a
`Network` or `NetworkView`) that wrap lower-level algorithm modules. These
wrappers avoid mutating the input model by constructing an internal graph with
pseudo source/sink nodes when required.
"""
