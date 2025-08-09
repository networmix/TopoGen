"""Profiling instrumentation and reporting for NetGraph.

This package exposes public profiling APIs for workflow execution:

- ``PerformanceProfiler``: CPU and wall-time profiling per workflow step.
- ``PerformanceReporter``: Text report generation from profiling results.
- ``ProfileResults`` and ``StepProfile``: Data structures for collected metrics.
"""

from .profiler import (
    PerformanceProfiler as PerformanceProfiler,
)
from .profiler import (
    PerformanceReporter as PerformanceReporter,
)
from .profiler import (
    ProfileResults as ProfileResults,
)
from .profiler import (
    StepProfile as StepProfile,
)

__all__ = [
    "PerformanceProfiler",
    "PerformanceReporter",
    "ProfileResults",
    "StepProfile",
]
