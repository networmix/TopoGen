"""Base classes and enums for network analysis algorithms."""

from __future__ import annotations

from enum import IntEnum
from typing import Tuple, Union

from ngraph.graph.strict_multidigraph import EdgeID, NodeID

#: Represents numeric cost in the network (e.g. distance, latency, etc.).
Cost = Union[int, float]

#: A single path element is a tuple of:
#:   - The current node ID.
#:   - A tuple of zero or more parallel edge IDs from this node to the next node.
#: In a complete path, intermediate elements usually have a non-empty edge tuple,
#: while the final element has an empty tuple to indicate termination.
PathElement = Tuple[NodeID, Tuple[EdgeID, ...]]

#: A path is a tuple of PathElements forming a complete route from
#: a source node to a destination node.
PathTuple = Tuple[PathElement, ...]

#: Capacity threshold below which capacity values are treated as effectively zero.
MIN_CAP = 2**-12

#: Flow threshold below which flow values are treated as effectively zero.
MIN_FLOW = 2**-12


class PathAlg(IntEnum):
    """Path-finding algorithm types."""

    SPF = 1
    KSP_YENS = 2


class EdgeSelect(IntEnum):
    """Edge selection criteria.

    Determines which edges are considered for path-finding between a node and
    its neighbor(s).
    """

    #: Return all edges matching the minimum cost among the candidate edges.
    ALL_MIN_COST = 1
    #: Return all edges matching the minimum cost among edges with remaining capacity.
    ALL_MIN_COST_WITH_CAP_REMAINING = 2
    #: Return all edges that have remaining capacity, ignoring cost except for returning min_cost.
    ALL_ANY_COST_WITH_CAP_REMAINING = 3
    #: Return exactly one edge (the single lowest cost).
    SINGLE_MIN_COST = 4
    #: Return exactly one edge, the lowest-cost edge with remaining capacity.
    SINGLE_MIN_COST_WITH_CAP_REMAINING = 5
    #: Return exactly one edge factoring both cost and load:
    #: cost = (cost * 100) + round(flow / capacity * 10).
    SINGLE_MIN_COST_WITH_CAP_REMAINING_LOAD_FACTORED = 6
    #: Use a user-defined function for edge selection logic.
    USER_DEFINED = 99


class FlowPlacement(IntEnum):
    """Strategies to distribute flow across parallel equal-cost paths."""

    PROPORTIONAL = 1  # Flow is split proportional to capacity (Dinic-like approach)
    EQUAL_BALANCED = 2  # Flow is equally divided among parallel paths of equal cost
