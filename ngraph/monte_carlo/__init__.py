"""Monte Carlo analysis helpers for FailureManager simulations.

Provides picklable analysis functions and structured result classes used by
``FailureManager.run_monte_carlo_analysis()`` when evaluating failure patterns.
Functions accept only hashable arguments to support multiprocessing and caching.
"""

from .functions import (
    demand_placement_analysis,
    max_flow_analysis,
    sensitivity_analysis,
)
from .results import (
    CapacityEnvelopeResults,
    DemandPlacementResults,
    SensitivityResults,
)

__all__ = [
    "max_flow_analysis",
    "demand_placement_analysis",
    "sensitivity_analysis",
    "CapacityEnvelopeResults",
    "DemandPlacementResults",
    "SensitivityResults",
]
