"""Flow and FlowIndex classes for traffic flow representation."""

from __future__ import annotations

from typing import Hashable, NamedTuple, Optional, Set, Tuple

from ngraph.algorithms.base import MIN_FLOW
from ngraph.algorithms.placement import (
    FlowPlacement,
    place_flow_on_graph,
    remove_flow_from_graph,
)
from ngraph.graph.strict_multidigraph import EdgeID, NodeID, StrictMultiDiGraph
from ngraph.paths.bundle import PathBundle


class FlowIndex(NamedTuple):
    """Unique identifier for a flow.

    Attributes:
        src_node: Source node.
        dst_node: Destination node.
        flow_class: Flow class label (hashable).
        flow_id: Monotonic integer id for this flow.
    """

    src_node: NodeID
    dst_node: NodeID
    flow_class: Hashable
    flow_id: int


class Flow:
    """Represents a fraction of demand routed along a given PathBundle.

    In traffic-engineering scenarios, a `Flow` object can model:
      - MPLS LSPs/tunnels with explicit paths,
      - IP forwarding behavior (with ECMP or UCMP),
      - Or anything that follows a specific set of paths.
    """

    def __init__(
        self,
        path_bundle: PathBundle,
        flow_index: FlowIndex,
        excluded_edges: Optional[Set[EdgeID]] = None,
        excluded_nodes: Optional[Set[NodeID]] = None,
    ) -> None:
        """Initialize a flow.

        Args:
            path_bundle: Paths this flow uses.
            flow_index: Identifier for this flow.
            excluded_edges: Edges to exclude from usage.
            excluded_nodes: Nodes to exclude from usage.
        """
        self.path_bundle: PathBundle = path_bundle
        self.flow_index: FlowIndex = flow_index
        self.excluded_edges: Set[EdgeID] = excluded_edges or set()
        self.excluded_nodes: Set[NodeID] = excluded_nodes or set()

        # Convenience references for flow endpoints
        self.src_node: NodeID = path_bundle.src_node
        self.dst_node: NodeID = path_bundle.dst_node

        # Track how much flow has been successfully placed
        self.placed_flow: float = 0.0

    def __str__(self) -> str:
        """Return a concise string for this flow."""
        return f"Flow(flow_index={self.flow_index}, placed_flow={self.placed_flow})"

    def place_flow(
        self,
        flow_graph: StrictMultiDiGraph,
        to_place: float,
        flow_placement: FlowPlacement,
    ) -> Tuple[float, float]:
        """Place or update this flow on the graph.

        Args:
            flow_graph: Graph tracking capacities and usage.
            to_place: Amount of flow requested to be placed.
            flow_placement: Strategy for distributing flow among equal-cost paths.

        Returns:
            tuple[float, float]: (placed_flow, remaining_flow).
        """
        placed_flow = 0.0

        # Only place flow if above the minimum threshold
        if to_place >= MIN_FLOW:
            flow_placement_meta = place_flow_on_graph(
                flow_graph=flow_graph,
                src_node=self.src_node,
                dst_node=self.dst_node,
                pred=self.path_bundle.pred,
                flow=to_place,
                flow_index=self.flow_index,
                flow_placement=flow_placement,
            )
            placed_flow = flow_placement_meta.placed_flow
            to_place = flow_placement_meta.remaining_flow
            self.placed_flow += placed_flow

        return placed_flow, to_place

    def remove_flow(self, flow_graph: StrictMultiDiGraph) -> None:
        """Remove this flow from the graph."""
        remove_flow_from_graph(flow_graph, flow_index=self.flow_index)
        self.placed_flow = 0.0
