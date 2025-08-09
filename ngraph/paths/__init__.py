"""Path primitives for representing routing sequences and equal-cost bundles.

This package defines lightweight structures for path-centric operations:
- ``Path`` models a single node-and-parallel-edges sequence with a numeric cost.
- ``PathBundle`` groups one or more equal-cost paths compactly via a predecessor
  map, enabling enumeration of concrete paths on demand.
"""
