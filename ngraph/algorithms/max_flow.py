"""Maximum-flow computation via iterative shortest-path augmentation.

Implements a practical Edmonds–Karp-like procedure using SPF with capacity
constraints and configurable flow-splitting across equal-cost parallel edges.
Provides helpers for saturated-edge detection and simple sensitivity analysis.
"""

from typing import Dict, Literal, Union, overload

from ngraph.algorithms.base import EdgeSelect, FlowPlacement
from ngraph.algorithms.flow_init import init_flow_graph
from ngraph.algorithms.placement import place_flow_on_graph
from ngraph.algorithms.spf import Cost, spf
from ngraph.algorithms.types import FlowSummary
from ngraph.graph.strict_multidigraph import NodeID, StrictMultiDiGraph


# Use @overload to provide precise static type safety for conditional return types.
# The function returns different types based on boolean flags: float, tuple[float, FlowSummary],
# tuple[float, StrictMultiDiGraph], or tuple[float, FlowSummary, StrictMultiDiGraph].
@overload
def calc_max_flow(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    dst_node: NodeID,
    *,
    return_summary: Literal[False] = False,
    return_graph: Literal[False] = False,
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
    shortest_path: bool = False,
    reset_flow_graph: bool = False,
    capacity_attr: str = "capacity",
    flow_attr: str = "flow",
    flows_attr: str = "flows",
    copy_graph: bool = True,
    tolerance: float = 1e-10,
) -> float: ...


@overload
def calc_max_flow(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    dst_node: NodeID,
    *,
    return_summary: Literal[True],
    return_graph: Literal[False] = False,
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
    shortest_path: bool = False,
    reset_flow_graph: bool = False,
    capacity_attr: str = "capacity",
    flow_attr: str = "flow",
    flows_attr: str = "flows",
    copy_graph: bool = True,
    tolerance: float = 1e-10,
) -> tuple[float, FlowSummary]: ...


@overload
def calc_max_flow(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    dst_node: NodeID,
    *,
    return_summary: Literal[False] = False,
    return_graph: Literal[True],
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
    shortest_path: bool = False,
    reset_flow_graph: bool = False,
    capacity_attr: str = "capacity",
    flow_attr: str = "flow",
    flows_attr: str = "flows",
    copy_graph: bool = True,
    tolerance: float = 1e-10,
) -> tuple[float, StrictMultiDiGraph]: ...


@overload
def calc_max_flow(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    dst_node: NodeID,
    *,
    return_summary: Literal[True],
    return_graph: Literal[True],
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
    shortest_path: bool = False,
    reset_flow_graph: bool = False,
    capacity_attr: str = "capacity",
    flow_attr: str = "flow",
    flows_attr: str = "flows",
    copy_graph: bool = True,
    tolerance: float = 1e-10,
) -> tuple[float, FlowSummary, StrictMultiDiGraph]: ...


def calc_max_flow(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    dst_node: NodeID,
    *,
    return_summary: bool = False,
    return_graph: bool = False,
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
    shortest_path: bool = False,
    reset_flow_graph: bool = False,
    capacity_attr: str = "capacity",
    flow_attr: str = "flow",
    flows_attr: str = "flows",
    copy_graph: bool = True,
    tolerance: float = 1e-10,
) -> Union[float, tuple]:
    """Compute max flow between two nodes in a directed multi-graph.

    Uses iterative shortest-path augmentation with capacity-aware SPF and
    configurable flow placement.

    By default, this function:
      1. Creates or re-initializes a flow-aware copy of the graph (via ``init_flow_graph``).
      2. Repeatedly finds a path from ``src_node`` to ``dst_node`` using ``spf`` with
         capacity constraints (``EdgeSelect.ALL_MIN_COST_WITH_CAP_REMAINING``).
      3. Places flow along that path (via ``place_flow_on_graph``) until no augmenting path
         remains or the capacities are exhausted.

    If ``shortest_path=True``, the function performs only one iteration (single augmentation)
    and returns the flow placed along that single path (not the true max flow).

    Args:
        graph (StrictMultiDiGraph):
            The original graph containing capacity/flow attributes on each edge.
        src_node (NodeID):
            The source node for flow.
        dst_node (NodeID):
            The destination node for flow.
        return_summary (bool):
            If True, return a FlowSummary with detailed flow analytics.
            Defaults to False.
        return_graph (bool):
            If True, return the mutated flow graph along with other results.
            Defaults to False.
        flow_placement (FlowPlacement):
            Determines how flow is split among parallel edges of equal cost.
            Defaults to ``FlowPlacement.PROPORTIONAL``.
        shortest_path (bool):
            If True, place flow only once along the first shortest path found and return
            immediately, rather than iterating for the true max flow.
        reset_flow_graph (bool):
            If True, reset any existing flow data (e.g., ``flow_attr``, ``flows_attr``).
            Defaults to False.
        capacity_attr (str):
            The name of the capacity attribute on edges. Defaults to "capacity".
        flow_attr (str):
            The name of the aggregated flow attribute on edges. Defaults to "flow".
        flows_attr (str):
            The name of the per-flow dictionary attribute on edges. Defaults to "flows".
        copy_graph (bool):
            If True, work on a copy of the original graph so it remains unmodified.
            Defaults to True.
        tolerance (float):
            Tolerance for floating-point comparisons when determining saturated edges
            and residual capacity. Defaults to 1e-10.

    Returns:
        Union[float, tuple]:
            - If neither flag: ``float`` total flow.
            - If return_summary only: ``tuple[float, FlowSummary]``.
            - If both flags: ``tuple[float, FlowSummary, StrictMultiDiGraph]``.

    Notes:
        - When using return_summary or return_graph, the return value is a tuple.

    Examples:
        >>> g = StrictMultiDiGraph()
        >>> g.add_node('A')
        >>> g.add_node('B')
        >>> g.add_node('C')
        >>> g.add_edge('A', 'B', capacity=10.0, flow=0.0, flows={}, cost=1)
        >>> g.add_edge('B', 'C', capacity=5.0, flow=0.0, flows={}, cost=1)
        >>>
        >>> # Basic usage (scalar return)
        >>> max_flow_value = calc_max_flow(g, 'A', 'C')
        >>> print(max_flow_value)
        5.0
        >>>
        >>> # With flow summary analytics
        >>> flow, summary = calc_max_flow(g, 'A', 'C', return_summary=True)
        >>> print(f"Min-cut edges: {summary.min_cut}")
        >>>
        >>> # With both summary and mutated graph
        >>> flow, summary, flow_graph = calc_max_flow(
        ...     g, 'A', 'C', return_summary=True, return_graph=True
        ... )
        >>> # flow_graph contains the flow assignments
    """
    # Handle self-loop case: when source equals destination, max flow is always 0
    # Degenerate case (s == t):
    # Flow value |f| is the net surplus at the vertex.
    # Conservation forces that surplus to zero, so the
    # only feasible (and thus maximum) flow value is 0.
    if src_node == dst_node:
        if return_summary or return_graph:
            # For consistency, we need to create a minimal flow graph for summary/graph returns
            flow_graph = init_flow_graph(
                graph.copy() if copy_graph else graph,
                flow_attr,
                flows_attr,
                reset_flow_graph,
            )
            return _build_return_value(
                0.0,
                flow_graph,
                src_node,
                return_summary,
                return_graph,
                capacity_attr,
                flow_attr,
                tolerance,
                {},  # Empty cost distribution for self-loop case
            )
        else:
            return 0.0

    # Initialize a flow-aware graph (copy or in-place).
    flow_graph = init_flow_graph(
        graph.copy() if copy_graph else graph,
        flow_attr,
        flows_attr,
        reset_flow_graph,
    )

    # Initialize cost distribution tracking
    cost_distribution: Dict[Cost, float] = {}

    # First path-finding iteration.
    costs, pred = spf(
        flow_graph, src_node, edge_select=EdgeSelect.ALL_MIN_COST_WITH_CAP_REMAINING
    )
    flow_meta = place_flow_on_graph(
        flow_graph,
        src_node,
        dst_node,
        pred,
        flow_placement=flow_placement,
        capacity_attr=capacity_attr,
        flow_attr=flow_attr,
        flows_attr=flows_attr,
    )
    max_flow = flow_meta.placed_flow

    # Track cost distribution for first iteration
    if dst_node in costs and flow_meta.placed_flow > 0:
        path_cost = costs[dst_node]
        cost_distribution[path_cost] = (
            cost_distribution.get(path_cost, 0.0) + flow_meta.placed_flow
        )

    # If only one path (single augmentation) is desired, return early.
    if shortest_path:
        return _build_return_value(
            max_flow,
            flow_graph,
            src_node,
            return_summary,
            return_graph,
            capacity_attr,
            flow_attr,
            tolerance,
            cost_distribution,
        )

    # Otherwise, repeatedly find augmenting paths until no new flow can be placed.
    while True:
        costs, pred = spf(
            flow_graph, src_node, edge_select=EdgeSelect.ALL_MIN_COST_WITH_CAP_REMAINING
        )
        if dst_node not in pred:
            # No path found; we've reached max flow.
            break

        flow_meta = place_flow_on_graph(
            flow_graph,
            src_node,
            dst_node,
            pred,
            flow_placement=flow_placement,
            capacity_attr=capacity_attr,
            flow_attr=flow_attr,
            flows_attr=flows_attr,
        )
        if flow_meta.placed_flow <= tolerance:
            # No significant additional flow could be placed; at capacity.
            break

        max_flow += flow_meta.placed_flow

        # Track cost distribution for this iteration
        if dst_node in costs and flow_meta.placed_flow > 0:
            path_cost = costs[dst_node]
            cost_distribution[path_cost] = (
                cost_distribution.get(path_cost, 0.0) + flow_meta.placed_flow
            )

    return _build_return_value(
        max_flow,
        flow_graph,
        src_node,
        return_summary,
        return_graph,
        capacity_attr,
        flow_attr,
        tolerance,
        cost_distribution,
    )


def _build_return_value(
    max_flow: float,
    flow_graph: StrictMultiDiGraph,
    src_node: NodeID,
    return_summary: bool,
    return_graph: bool,
    capacity_attr: str,
    flow_attr: str,
    tolerance: float,
    cost_distribution: Dict[Cost, float],
) -> Union[float, tuple]:
    """Build the appropriate return value based on the requested flags."""
    if not (return_summary or return_graph):
        return max_flow

    summary = None
    if return_summary:
        summary = _build_flow_summary(
            max_flow,
            flow_graph,
            src_node,
            capacity_attr,
            flow_attr,
            tolerance,
            cost_distribution,
        )

    ret: list = [max_flow]
    if return_summary:
        ret.append(summary)
    if return_graph:
        ret.append(flow_graph)

    return tuple(ret) if len(ret) > 1 else ret[0]


def _build_flow_summary(
    total_flow: float,
    flow_graph: StrictMultiDiGraph,
    src_node: NodeID,
    capacity_attr: str,
    flow_attr: str,
    tolerance: float,
    cost_distribution: Dict[Cost, float],
) -> FlowSummary:
    """Construct a ``FlowSummary`` from the flow-graph state."""
    edge_flow = {}
    residual_cap = {}

    # Extract flow and residual capacity for each edge
    for u, v, k, d in flow_graph.edges(data=True, keys=True):
        edge = (u, v, k)
        f = d.get(flow_attr, 0.0)
        edge_flow[edge] = f
        residual_cap[edge] = d[capacity_attr] - f

    # BFS in residual graph to find reachable nodes from source
    reachable = set()
    stack = [src_node]
    while stack:
        n = stack.pop()
        if n in reachable:
            continue
        reachable.add(n)
        for _, nbr, _, d in flow_graph.out_edges(n, data=True, keys=True):
            if (
                d[capacity_attr] - d.get(flow_attr, 0.0) > tolerance
                and nbr not in reachable
            ):
                stack.append(nbr)

    # Find min-cut edges (saturated edges crossing the cut)
    min_cut = [
        (u, v, k)
        for u, v, k, d in flow_graph.edges(data=True, keys=True)
        if u in reachable
        and v not in reachable
        and d[capacity_attr] - d.get(flow_attr, 0.0) <= tolerance
    ]

    return FlowSummary(
        total_flow=total_flow,
        edge_flow=edge_flow,
        residual_cap=residual_cap,
        reachable=reachable,
        min_cut=min_cut,
        cost_distribution=cost_distribution,
    )


def saturated_edges(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    dst_node: NodeID,
    *,
    capacity_attr: str = "capacity",
    flow_attr: str = "flow",
    tolerance: float = 1e-10,
    **kwargs,
) -> list[tuple]:
    """Identify saturated edges in the max-flow solution.

    Args:
        graph: The graph to analyze
        src_node: Source node
        dst_node: Destination node
        capacity_attr: Name of capacity attribute
        flow_attr: Name of flow attribute
        tolerance: Tolerance for considering an edge saturated
        **kwargs: Additional arguments passed to calc_max_flow

    Returns:
        list[tuple]: Edges ``(u, v, k)`` with residual capacity <= ``tolerance``.
    """
    result = calc_max_flow(
        graph,
        src_node,
        dst_node,
        return_summary=True,
        capacity_attr=capacity_attr,
        flow_attr=flow_attr,
        **kwargs,
    )
    # Ensure we have a tuple to unpack
    if isinstance(result, tuple) and len(result) >= 2:
        # Handle tuple unpacking - could be 2 or 3 elements
        if len(result) == 2:
            _, summary = result
        else:
            _, summary, _ = result
    else:
        raise ValueError(
            "Expected tuple return from calc_max_flow with return_summary=True"
        )

    return [
        edge for edge, residual in summary.residual_cap.items() if residual <= tolerance
    ]


def run_sensitivity(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    dst_node: NodeID,
    *,
    capacity_attr: str = "capacity",
    flow_attr: str = "flow",
    change_amount: float = 1.0,
    **kwargs,
) -> dict[tuple, float]:
    """Simple sensitivity analysis for per-edge capacity changes.

    Tests changing each saturated edge capacity by change_amount and measures
    the resulting change in total flow. Positive values increase capacity,
    negative values decrease capacity (with validation to prevent negative capacities).

    Args:
        graph: The graph to analyze
        src_node: Source node
        dst_node: Destination node
        capacity_attr: Name of capacity attribute
        flow_attr: Name of flow attribute
        change_amount: Amount to change capacity for testing (positive=increase, negative=decrease)
        **kwargs: Additional arguments passed to calc_max_flow

    Returns:
        dict[tuple, float]: Flow delta per modified edge.
    """
    # Get baseline flow and identify saturated edges - ensure scalar return
    baseline_flow = calc_max_flow(
        graph,
        src_node,
        dst_node,
        return_summary=False,
        return_graph=False,
        capacity_attr=capacity_attr,
        flow_attr=flow_attr,
        **kwargs,
    )
    assert isinstance(baseline_flow, (int, float))

    saturated = saturated_edges(
        graph,
        src_node,
        dst_node,
        capacity_attr=capacity_attr,
        flow_attr=flow_attr,
        **kwargs,
    )

    sensitivity = {}

    for edge in saturated:
        u, v, k = edge

        # Create modified graph with changed edge capacity
        test_graph = graph.copy()
        edge_data = test_graph.get_edge_data(u, v, k)
        if edge_data is not None:
            # Create a mutable copy of the edge data
            edge_data = dict(edge_data)
            original_capacity = edge_data[capacity_attr]
            new_capacity = original_capacity + change_amount

            # If the change would result in negative capacity, set to 0
            if new_capacity < 0:
                new_capacity = 0

            edge_data[capacity_attr] = new_capacity
            test_graph.remove_edge(u, v, k)
            test_graph.add_edge(u, v, k, **edge_data)

            # Calculate new max flow - ensure scalar return
            new_flow = calc_max_flow(
                test_graph,
                src_node,
                dst_node,
                return_summary=False,
                return_graph=False,
                capacity_attr=capacity_attr,
                flow_attr=flow_attr,
                **kwargs,
            )
            assert isinstance(new_flow, (int, float))

            # Record flow change
            sensitivity[edge] = new_flow - baseline_flow

    return sensitivity
