"""Types and data structures for algorithm analytics.

Defines immutable summary containers and aliases for algorithm outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from ngraph.algorithms.base import Cost

# Edge identifier tuple: (source_node, destination_node, edge_key)
Edge = Tuple[str, str, str]


@dataclass(frozen=True)
class FlowSummary:
    """Summary of max-flow computation results.

    Captures edge flows, residual capacities, reachable set, and min-cut.

    Attributes:
        total_flow: Maximum flow value achieved.
        edge_flow: Flow amount per edge, indexed by ``(src, dst, key)``.
        residual_cap: Remaining capacity per edge after placement.
        reachable: Nodes reachable from source in residual graph.
        min_cut: Saturated edges crossing the s-t cut.
        cost_distribution: Mapping of path cost to flow volume placed at that cost.
    """

    total_flow: float
    edge_flow: Dict[Edge, float]
    residual_cap: Dict[Edge, float]
    reachable: Set[str]
    min_cut: List[Edge]
    cost_distribution: Dict[Cost, float]
