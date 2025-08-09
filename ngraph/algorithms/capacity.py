"""Capacity calculation algorithms for network analysis.

This module computes feasible flow given a predecessor DAG from a shortest-path
routine and supports two placement strategies: proportional and equal-balanced
in reversed orientation. Functions follow a Dinic-like blocking-flow approach
for proportional placement.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Set, Tuple

from ngraph.algorithms.base import MIN_CAP, MIN_FLOW, FlowPlacement
from ngraph.graph.strict_multidigraph import EdgeID, NodeID, StrictMultiDiGraph


def _init_graph_data(
    flow_graph: StrictMultiDiGraph,
    pred: Dict[NodeID, Dict[NodeID, List[EdgeID]]],
    init_node: NodeID,
    flow_placement: FlowPlacement,
    capacity_attr: str,
    flow_attr: str,
) -> Tuple[
    Dict[NodeID, Dict[NodeID, Tuple[EdgeID, ...]]],
    Dict[NodeID, int],
    Dict[NodeID, Dict[NodeID, float]],
    Dict[NodeID, Dict[NodeID, float]],
]:
    """Build reversed residual structures used by the flow routine.

    Constructs reversed adjacency and residual capacity for the subset of the
    graph that can reach ``init_node`` in forward orientation.

    Args:
        flow_graph: The multigraph with capacity and flow attributes on edges.
        pred: Forward adjacency mapping: node -> (adjacent node -> list of EdgeIDs).
              This is a DAG typically produced by a shortest-path routine from the
              source in the forward direction.
        init_node: The node from which we perform the reversed BFS (generally the
                   destination in forward flow).
        flow_placement: Strategy for distributing flow (PROPORTIONAL or EQUAL_BALANCED).
        capacity_attr: Name of the capacity attribute on edges.
        flow_attr: Name of the flow attribute on edges.

    Returns:
        A tuple ``(succ, levels, residual_cap, flow_dict)`` where:
          - ``succ``: Reversed adjacency mapping (u->v in reversed sense).
          - ``levels``: Node -> level mapping for Dinic; initialized to -1.
          - ``residual_cap``: Residual capacities in reversed orientation.
          - ``flow_dict``: Net flow in reversed orientation (initialized to 0).
    """
    edges = flow_graph.get_edges()

    # Reversed adjacency: For each forward edge u->v in pred, store v->u in succ.
    succ: Dict[NodeID, Dict[NodeID, Tuple[EdgeID, ...]]] = defaultdict(dict)

    # Will store BFS levels (set to -1 here, updated later in _set_levels_bfs).
    levels: Dict[NodeID, int] = {}

    # Residual capacities in the reversed orientation
    residual_cap: Dict[NodeID, Dict[NodeID, float]] = defaultdict(dict)

    # Net flow (updated during flow pushes)
    flow_dict: Dict[NodeID, Dict[NodeID, float]] = defaultdict(dict)

    # Standard BFS to collect only the portion of pred reachable from init_node (in reverse)
    visited: Set[NodeID] = set()
    queue: Deque[NodeID] = deque()

    visited.add(init_node)
    levels[init_node] = -1
    queue.append(init_node)

    while queue:
        node = queue.popleft()

        # Check each forward adjacency from node in pred, so we can form reversed edges.
        for adj_node, edge_list in pred.get(node, {}).items():
            # Build reversed adjacency once
            if node not in succ[adj_node]:
                succ[adj_node][node] = tuple(edge_list)

            # Calculate available capacities of the forward edges
            capacities = []
            for eid in edge_list:
                e_attrs = edges[eid][3]  # Slightly faster repeated access
                cap_val = e_attrs[capacity_attr]
                flow_val = e_attrs[flow_attr]
                c = cap_val - flow_val
                if c < 0.0:
                    c = 0.0
                capacities.append(c)

            # Set reversed and forward capacities in the residual_cap structure
            if flow_placement == FlowPlacement.PROPORTIONAL:
                # Sum capacities of parallel edges for the reversed edge
                fwd_capacity = sum(capacities)
                residual_cap[node][adj_node] = (
                    fwd_capacity if fwd_capacity >= MIN_CAP else 0.0
                )
                # Reverse edge in the BFS sense starts with 0 capacity
                residual_cap[adj_node][node] = 0.0

            elif flow_placement == FlowPlacement.EQUAL_BALANCED:
                # min(...) * number_of_parallel_edges
                if capacities:
                    rev_cap = min(capacities) * len(capacities)
                    residual_cap[adj_node][node] = (
                        rev_cap if rev_cap >= MIN_CAP else 0.0
                    )
                else:
                    residual_cap[adj_node][node] = 0.0
                # The forward edge in reversed orientation starts at 0 capacity
                residual_cap[node][adj_node] = 0.0

            else:
                raise ValueError(f"Unsupported flow placement: {flow_placement}")

            # Initialize net flow for both orientations to 0
            flow_dict[node][adj_node] = 0.0
            flow_dict[adj_node][node] = 0.0

            # Enqueue adj_node if not visited
            if adj_node not in visited:
                visited.add(adj_node)
                levels[adj_node] = -1
                queue.append(adj_node)

    # Ensure every node in the entire graph has at least an empty adjacency dict in succ
    # (some nodes might be outside the reversed BFS component).
    for n in flow_graph.nodes():
        succ.setdefault(n, {})

    return succ, levels, residual_cap, flow_dict


def _set_levels_bfs(
    start_node: NodeID,
    levels: Dict[NodeID, int],
    residual_cap: Dict[NodeID, Dict[NodeID, float]],
) -> None:
    """Assign BFS levels on the reversed residual graph.

    Considers edges with residual capacity >= ``MIN_CAP``.

    Args:
        start_node: The starting node for the BFS (acts as the 'source' in reversed graph).
        levels: The dict from node -> BFS level (modified in-place).
        residual_cap: The dict of reversed residual capacities for edges.
    """
    # Reset all node levels to -1 (unvisited)
    for nd in levels:
        levels[nd] = -1
    levels[start_node] = 0

    queue: Deque[NodeID] = deque([start_node])
    while queue:
        u = queue.popleft()
        # Explore all neighbors of u in the reversed graph
        for v, cap_uv in residual_cap[u].items():
            # Only traverse edges with sufficient capacity and unvisited nodes
            if cap_uv >= MIN_CAP and levels[v] < 0:
                levels[v] = levels[u] + 1
                queue.append(v)


def _push_flow_dfs(
    current: NodeID,
    sink: NodeID,
    flow_in: float,
    residual_cap: Dict[NodeID, Dict[NodeID, float]],
    flow_dict: Dict[NodeID, Dict[NodeID, float]],
    levels: Dict[NodeID, int],
) -> float:
    """Push flow via DFS on the reversed residual graph.

    Only edges that respect the level structure are considered, i.e.
    ``levels[nxt] == levels[current] + 1``.

    Args:
        current: The current node in the DFS.
        sink: The target node in the reversed orientation.
        flow_in: The amount of flow available to push from the current node.
        residual_cap: Residual capacities of edges in the reversed graph.
        flow_dict: Tracks the net flow pushed along edges in the reversed graph.
        levels: BFS levels in the reversed graph (from `_set_levels_bfs`).

    Returns:
        The total amount of flow successfully pushed from `current` to `sink`.
    """
    # Base case: reached sink
    if current == sink:
        return flow_in

    total_pushed = 0.0
    neighbors = list(
        residual_cap[current].items()
    )  # snapshot to avoid iteration changes

    for nxt, capacity_uv in neighbors:
        if capacity_uv < MIN_CAP:
            continue
        if levels.get(nxt, -1) != levels[current] + 1:
            continue

        flow_to_push = min(flow_in, capacity_uv)
        if flow_to_push < MIN_FLOW:
            continue

        pushed = _push_flow_dfs(
            nxt, sink, flow_to_push, residual_cap, flow_dict, levels
        )
        if pushed >= MIN_FLOW:
            # Update residual capacities
            residual_cap[current][nxt] -= pushed
            residual_cap[nxt][current] += pushed

            # Update net flow (remember, we're in reversed orientation)
            flow_dict[current][nxt] += pushed
            flow_dict[nxt][current] -= pushed

            flow_in -= pushed
            total_pushed += pushed

            if flow_in < MIN_FLOW:
                break

    return total_pushed


def _equal_balance_bfs(
    src_node: NodeID,
    succ: Dict[NodeID, Dict[NodeID, Tuple[EdgeID, ...]]],
    flow_dict: Dict[NodeID, Dict[NodeID, float]],
) -> None:
    """Distribute nominal unit flow equally over reversed adjacency.

    Performs a BFS-like pass from ``src_node`` over ``succ`` to assign fractional
    flows, later scaled to respect capacities.

    Args:
        src_node: The node from which a nominal flow of 1.0 is injected (in reversed orientation).
        succ: The reversed adjacency dict, where succ[u][v] is a tuple of edges (u->v in reversed sense).
        flow_dict: The net flow dictionary to be updated with the BFS distribution.
    """
    # Count total parallel edges leaving each node
    node_split: Dict[NodeID, int] = {}
    for node, neighbors in succ.items():
        node_split[node] = sum(len(edge_tuple) for edge_tuple in neighbors.values())

    queue: Deque[Tuple[NodeID, float]] = deque([(src_node, 1.0)])
    visited: Set[NodeID] = set()

    while queue:
        node, incoming_flow = queue.popleft()
        visited.add(node)

        # If no edges or negligible incoming flow, skip
        split_count = node_split.get(node, 0)
        if split_count <= 0 or incoming_flow < MIN_FLOW:
            continue

        # Distribute the incoming_flow among outgoing edges, proportional to the count of parallel edges
        for nxt, edge_tuple in succ[node].items():
            if not edge_tuple:
                continue
            push_flow = (incoming_flow * len(edge_tuple)) / float(split_count)
            if push_flow < MIN_FLOW:
                continue

            flow_dict[node][nxt] += push_flow
            flow_dict[nxt][node] -= push_flow

            if nxt not in visited:
                # Note: we queue each node only once in this scheme.
                # If a node can be reached from multiple parents before being popped,
                # the BFS will handle the first discovered flow.
                # This behavior matches the existing tests and usage expectations.
                queue.append((nxt, push_flow))


def calc_graph_capacity(
    flow_graph: StrictMultiDiGraph,
    src_node: NodeID,
    dst_node: NodeID,
    pred: Dict[NodeID, Dict[NodeID, List[EdgeID]]],
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
    capacity_attr: str = "capacity",
    flow_attr: str = "flow",
) -> Tuple[float, Dict[NodeID, Dict[NodeID, float]]]:
    """Calculate feasible flow and flow fractions between two nodes.

    In PROPORTIONAL mode (similar to Dinic in reversed orientation):
      1. Build the reversed residual graph from dst_node (via `_init_graph_data`).
      2. Use BFS (in `_set_levels_bfs`) to build a level graph and DFS (`_push_flow_dfs`)
         to push blocking flows, repeating until no more flow can be pushed.
      3. The net flow found is stored in reversed orientation. Convert final flows
         to forward orientation by negating and normalizing by the total.

    In EQUAL_BALANCED mode:
      1. Build reversed adjacency from dst_node (also via `_init_graph_data`),
         ignoring capacity checks in that BFS.
      2. Perform a BFS pass from src_node (`_equal_balance_bfs`) to distribute a
         nominal flow of 1.0 equally among parallel edges.
      3. Determine the scaling ratio so that no edge capacity is exceeded.
         Scale the flow assignments accordingly, then normalize to the forward sense.

    Args:
        flow_graph: The multigraph with capacity and flow attributes.
        src_node: The source node in the forward graph.
        dst_node: The destination node in the forward graph.
        pred: Forward adjacency mapping (node -> (adjacent node -> list of EdgeIDs)),
              typically produced by `spf(..., multipath=True)`. Must be a DAG.
        flow_placement: The flow distribution strategy (PROPORTIONAL or EQUAL_BALANCED).
        capacity_attr: Name of the capacity attribute on edges.
        flow_attr: Name of the flow attribute on edges.

    Returns:
        tuple[float, dict[NodeID, dict[NodeID, float]]]:
            - Total feasible flow from ``src_node`` to ``dst_node``.
            - Normalized flow fractions in forward orientation (``[u][v]`` >= 0).

    Raises:
        ValueError: If src_node or dst_node is not in the graph, or the flow_placement
                    is unsupported.
    """
    if src_node not in flow_graph or dst_node not in flow_graph:
        raise ValueError(
            f"Source node {src_node} or destination node {dst_node} not found in the graph."
        )

    # Handle self-loop case: when source equals destination, max flow is always 0
    # Degenerate case (s == t):
    # Flow value |f| is the net surplus at the vertex.
    # Conservation forces that surplus to zero, so the
    # only feasible (and thus maximum) flow value is 0.
    if src_node == dst_node:
        return 0.0, defaultdict(dict)

    # Build reversed data structures from dst_node
    succ, levels, residual_cap, flow_dict = _init_graph_data(
        flow_graph=flow_graph,
        pred=pred,
        init_node=dst_node,
        flow_placement=flow_placement,
        capacity_attr=capacity_attr,
        flow_attr=flow_attr,
    )

    total_flow = 0.0

    if flow_placement == FlowPlacement.PROPORTIONAL:
        # Repeatedly build the level graph and push blocking flows
        while True:
            _set_levels_bfs(dst_node, levels, residual_cap)
            # If src_node is unreachable (level <= 0), no more flow
            if levels.get(src_node, -1) <= 0:
                break

            pushed = _push_flow_dfs(
                current=dst_node,
                sink=src_node,
                flow_in=float("inf"),
                residual_cap=residual_cap,
                flow_dict=flow_dict,
                levels=levels,
            )
            if pushed < MIN_FLOW:
                break
            total_flow += pushed

        # If no flow found, reset flows to zero
        if total_flow < MIN_FLOW:
            total_flow = 0.0
            for u in flow_dict:
                for v in flow_dict[u]:
                    flow_dict[u][v] = 0.0
        else:
            # Convert reversed flows to forward sense
            for u in flow_dict:
                for v in flow_dict[u]:
                    # Negative and normalized
                    flow_dict[u][v] = -(flow_dict[u][v] / total_flow)

    elif flow_placement == FlowPlacement.EQUAL_BALANCED:
        # 1. Distribute nominal flow of 1.0 from src_node
        _equal_balance_bfs(src_node, succ, flow_dict)

        # 2. Determine the scaling ratio so that no edge in reversed orientation exceeds capacity
        min_ratio = float("inf")
        for u, neighbors in succ.items():
            for v in neighbors:
                assigned_flow = flow_dict[u][v]
                if assigned_flow >= MIN_FLOW:
                    cap_uv = residual_cap[u].get(v, 0.0)
                    if assigned_flow > 0.0:
                        ratio = cap_uv / assigned_flow
                        if ratio < min_ratio:
                            min_ratio = ratio

        if min_ratio == float("inf") or min_ratio < MIN_FLOW:
            # No feasible flow
            total_flow = 0.0
        else:
            total_flow = min_ratio
            # Scale flows to fit capacities
            for u in flow_dict:
                for v in flow_dict[u]:
                    val = flow_dict[u][v] * total_flow
                    flow_dict[u][v] = val if abs(val) >= MIN_FLOW else 0.0

            # Normalize flows to forward direction
            for u in flow_dict:
                for v in flow_dict[u]:
                    if abs(flow_dict[u][v]) > 0.0:
                        flow_dict[u][v] /= total_flow

    else:
        raise ValueError(f"Unsupported flow placement: {flow_placement}")

    # Clamp small flows to zero
    for u in flow_dict:
        for v in flow_dict[u]:
            if abs(flow_dict[u][v]) < MIN_FLOW:
                flow_dict[u][v] = 0.0

    return total_flow, flow_dict
