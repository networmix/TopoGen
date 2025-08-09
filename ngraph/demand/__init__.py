"""Demand primitives for traffic placement.

Defines the `Demand` dataclass, which represents traffic volume between two
nodes. A `Demand` delegates flow realization and placement to a single
`FlowPolicy` instance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from ngraph.algorithms.base import MIN_FLOW
from ngraph.flows.policy import FlowPolicy
from ngraph.graph.strict_multidigraph import NodeID, StrictMultiDiGraph


@dataclass
class Demand:
    """Network demand between two nodes.

    A demand is realized via one or more flows created by a single
    `FlowPolicy`.

    Attributes:
        src_node: Source node identifier.
        dst_node: Destination node identifier.
        volume: Total demand volume to place.
        demand_class: Priority class; lower value indicates higher priority.
        flow_policy: Policy used to create and place flows for this demand.
        placed_demand: Volume successfully placed so far.
    """

    src_node: NodeID
    dst_node: NodeID
    volume: float
    demand_class: int = 0
    flow_policy: Optional[FlowPolicy] = None
    placed_demand: float = field(default=0.0, init=False)

    @staticmethod
    def _round_float(value: float) -> float:
        """Round ``value`` to avoid tiny floating point drift."""
        if math.isfinite(value):
            rounded = round(value, 12)
            if abs(rounded) < MIN_FLOW:
                return 0.0
            return rounded
        return value

    def __lt__(self, other: Demand) -> bool:
        """Return True if this demand should sort before ``other``.

        Demands sort by ``demand_class`` ascending (lower value = higher priority).

        Args:
            other: Demand to compare against.

        Returns:
            True if this instance has a lower ``demand_class`` than ``other``.
        """
        return self.demand_class < other.demand_class

    def __str__(self) -> str:
        """Return a concise representation with src, dst, volume, priority, placed."""
        return (
            f"Demand(src_node={self.src_node}, dst_node={self.dst_node}, "
            f"volume={self.volume}, demand_class={self.demand_class}, "
            f"placed_demand={self.placed_demand})"
        )

    def place(
        self,
        flow_graph: StrictMultiDiGraph,
        max_fraction: float = 1.0,
        max_placement: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Places demand volume onto the network via self.flow_policy.

        Args:
            flow_graph: Graph to place flows onto.
            max_fraction: Fraction of the remaining demand to place now.
            max_placement: Absolute upper bound on the volume to place now.

        Returns:
            A tuple ``(placed_now, remaining)`` where:
            - ``placed_now`` is the volume placed in this call.
            - ``remaining`` is the volume that could not be placed in this call.

        Raises:
            RuntimeError: If no FlowPolicy is set on this Demand.
            ValueError: If max_fraction is outside [0, 1].
        """
        if self.flow_policy is None:
            raise RuntimeError("No FlowPolicy set on this Demand.")

        if not (0 <= max_fraction <= 1):
            raise ValueError("max_fraction must be in the range [0, 1].")

        to_place = self.volume - self.placed_demand
        if max_placement is not None:
            to_place = min(to_place, max_placement)

        if max_fraction > 0:
            to_place = min(to_place, self.volume * max_fraction)
        else:
            # If max_fraction <= 0, do not place any new volume (unless volume is infinite).
            to_place = self.volume if self.volume == float("inf") else 0.0

        # Delegate flow placement
        self.flow_policy.place_demand(
            flow_graph,
            self.src_node,
            self.dst_node,
            self.demand_class,
            to_place,
        )

        # placed_now is the difference from the old placed_demand
        placed_now = self.flow_policy.placed_demand - self.placed_demand
        self.placed_demand = self._round_float(self.flow_policy.placed_demand)
        remaining = to_place - placed_now

        placed_now = self._round_float(placed_now)
        remaining = self._round_float(remaining)

        return placed_now, remaining
