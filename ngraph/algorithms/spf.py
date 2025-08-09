"""Shortest-path-first (SPF) algorithms.

Implements Dijkstra-like SPF with pluggable edge-selection policies and a
Yen-like KSP generator. Specialized fast paths exist for common selection
strategies without exclusions.
"""

from heapq import heappop, heappush
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from ngraph.algorithms.base import (
    MIN_CAP,
    Cost,
    EdgeSelect,
)
from ngraph.algorithms.edge_select import edge_select_fabric
from ngraph.algorithms.paths import resolve_to_paths
from ngraph.graph.strict_multidigraph import (
    AttrDict,
    EdgeID,
    NodeID,
    StrictMultiDiGraph,
)


def _spf_fast_all_min_cost_dijkstra(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    multipath: bool,
) -> Tuple[Dict[NodeID, Cost], Dict[NodeID, Dict[NodeID, List[EdgeID]]]]:
    """Specialized Dijkstra SPF for:
      - EdgeSelect.ALL_MIN_COST
      - No excluded edges/nodes.

    Finds all edges with the same minimal cost between two nodes if multipath=True.
    If multipath=False, new minimal-cost paths overwrite old ones, though edges
    are still collected together for immediate neighbor expansion.

    Args:
        graph: Directed graph (StrictMultiDiGraph).
        src_node: Source node for SPF.
        multipath: Whether to record multiple equal-cost paths.

    Returns:
        A tuple of (costs, pred):
          - costs: Maps each reachable node to the minimal cost from src_node.
          - pred: For each reachable node, a dict of predecessor -> list of edges
            from the predecessor to that node. If multipath=True, there may be
            multiple predecessors for the same node.
    """
    outgoing_adjacencies = graph._adj  # type: ignore[attr-defined]
    if src_node not in outgoing_adjacencies:
        raise KeyError(f"Source node '{src_node}' is not in the graph.")

    costs: Dict[NodeID, Cost] = {src_node: 0.0}
    pred: Dict[NodeID, Dict[NodeID, List[EdgeID]]] = {src_node: {}}
    min_pq: List[Tuple[Cost, NodeID]] = [(0.0, src_node)]

    while min_pq:
        current_cost, node_id = heappop(min_pq)
        if current_cost > costs[node_id]:
            continue

        # Explore neighbors
        for neighbor_id, edges_map in outgoing_adjacencies[node_id].items():
            min_edge_cost: Optional[Cost] = None
            selected_edges: List[EdgeID] = []

            # Gather the minimal cost edge(s)
            for e_id, e_attr in edges_map.items():
                edge_cost = e_attr["cost"]
                if min_edge_cost is None or edge_cost < min_edge_cost:
                    min_edge_cost = edge_cost
                    selected_edges = [e_id]
                elif multipath and edge_cost == min_edge_cost:
                    selected_edges.append(e_id)

            if min_edge_cost is None:
                continue

            new_cost = current_cost + min_edge_cost
            if (neighbor_id not in costs) or (new_cost < costs[neighbor_id]):
                costs[neighbor_id] = new_cost
                pred[neighbor_id] = {node_id: selected_edges}
                heappush(min_pq, (new_cost, neighbor_id))
            elif multipath and new_cost == costs[neighbor_id]:
                pred[neighbor_id][node_id] = selected_edges

    return costs, pred


def _spf_fast_all_min_cost_with_cap_remaining_dijkstra(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    multipath: bool,
) -> Tuple[Dict[NodeID, Cost], Dict[NodeID, Dict[NodeID, List[EdgeID]]]]:
    """Specialized Dijkstra SPF for:
      - EdgeSelect.ALL_MIN_COST_WITH_CAP_REMAINING
      - No excluded edges/nodes

    Only considers edges whose (capacity - flow) >= MIN_CAP. Among those edges,
    finds all edges with the same minimal cost if multipath=True.

    Args:
        graph: Directed graph (StrictMultiDiGraph).
        src_node: Source node for SPF.
        multipath: Whether to record multiple equal-cost paths.

    Returns:
        A tuple of (costs, pred):
          - costs: Maps each reachable node to the minimal cost from src_node.
          - pred: For each reachable node, a dict of predecessor -> list of edges
            from the predecessor to that node.
    """
    outgoing_adjacencies = graph._adj  # type: ignore[attr-defined]
    if src_node not in outgoing_adjacencies:
        raise KeyError(f"Source node '{src_node}' is not in the graph.")

    costs: Dict[NodeID, Cost] = {src_node: 0.0}
    pred: Dict[NodeID, Dict[NodeID, List[EdgeID]]] = {src_node: {}}
    min_pq: List[Tuple[Cost, NodeID]] = [(0.0, src_node)]

    while min_pq:
        current_cost, node_id = heappop(min_pq)
        if current_cost > costs[node_id]:
            continue

        # Explore neighbors; skip edges without enough remaining capacity
        for neighbor_id, edges_map in outgoing_adjacencies[node_id].items():
            min_edge_cost: Optional[Cost] = None
            selected_edges: List[EdgeID] = []

            for e_id, e_attr in edges_map.items():
                if (e_attr["capacity"] - e_attr["flow"]) >= MIN_CAP:
                    edge_cost = e_attr["cost"]
                    if min_edge_cost is None or edge_cost < min_edge_cost:
                        min_edge_cost = edge_cost
                        selected_edges = [e_id]
                    elif multipath and edge_cost == min_edge_cost:
                        selected_edges.append(e_id)

            if min_edge_cost is None:
                continue

            new_cost = current_cost + min_edge_cost
            if (neighbor_id not in costs) or (new_cost < costs[neighbor_id]):
                costs[neighbor_id] = new_cost
                pred[neighbor_id] = {node_id: selected_edges}
                heappush(min_pq, (new_cost, neighbor_id))
            elif multipath and new_cost == costs[neighbor_id]:
                pred[neighbor_id][node_id] = selected_edges

    return costs, pred


def spf(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    edge_select: EdgeSelect = EdgeSelect.ALL_MIN_COST,
    edge_select_func: Optional[
        Callable[
            [
                StrictMultiDiGraph,
                NodeID,
                NodeID,
                Dict[EdgeID, AttrDict],
                Set[EdgeID],
                Set[NodeID],
            ],
            Tuple[Cost, List[EdgeID]],
        ]
    ] = None,
    multipath: bool = True,
    excluded_edges: Optional[Set[EdgeID]] = None,
    excluded_nodes: Optional[Set[NodeID]] = None,
) -> Tuple[Dict[NodeID, Cost], Dict[NodeID, Dict[NodeID, List[EdgeID]]]]:
    """Compute shortest paths from a source node.

    By default, uses EdgeSelect.ALL_MIN_COST. If multipath=True, multiple equal-cost
    paths to the same node will be recorded in the predecessor structure. If no
    excluded edges/nodes are given and edge_select is one of the specialized
    (ALL_MIN_COST or ALL_MIN_COST_WITH_CAP_REMAINING), it uses a fast specialized
    routine.

    Args:
        graph: The directed graph (StrictMultiDiGraph).
        src_node: The source node from which to compute shortest paths.
        edge_select: The edge selection strategy. Defaults to ALL_MIN_COST.
        edge_select_func: If provided, overrides the default edge selection function.
            Must return (cost, list_of_edges) for the given node->neighbor adjacency.
        multipath: Whether to record multiple same-cost paths.
        excluded_edges: A set of edge IDs to ignore in the graph.
        excluded_nodes: A set of node IDs to ignore in the graph.

    Returns:
        tuple[dict[NodeID, Cost], dict[NodeID, dict[NodeID, list[EdgeID]]]]:
            Costs and predecessor mapping.

    Raises:
        KeyError: If src_node does not exist in graph.
    """
    if excluded_edges is None:
        excluded_edges = set()
    if excluded_nodes is None:
        excluded_nodes = set()

    # Use specialized fast code if applicable
    if edge_select_func is None:
        if not excluded_edges and not excluded_nodes:
            if edge_select == EdgeSelect.ALL_MIN_COST:
                return _spf_fast_all_min_cost_dijkstra(graph, src_node, multipath)
            elif edge_select == EdgeSelect.ALL_MIN_COST_WITH_CAP_REMAINING:
                return _spf_fast_all_min_cost_with_cap_remaining_dijkstra(
                    graph, src_node, multipath
                )
        else:
            edge_select_func = edge_select_fabric(edge_select)

    # Ensure edge_select_func is set at this point
    if edge_select_func is None:
        edge_select_func = edge_select_fabric(edge_select)

    outgoing_adjacencies = graph._adj  # type: ignore[attr-defined]
    if src_node not in outgoing_adjacencies:
        raise KeyError(f"Source node '{src_node}' is not in the graph.")

    costs: Dict[NodeID, Cost] = {src_node: 0.0}
    pred: Dict[NodeID, Dict[NodeID, List[EdgeID]]] = {src_node: {}}
    min_pq: List[Tuple[Cost, NodeID]] = [(0.0, src_node)]

    while min_pq:
        current_cost, node_id = heappop(min_pq)
        if current_cost > costs[node_id]:
            continue
        if node_id in excluded_nodes:
            continue

        # Evaluate each neighbor using the provided edge_select_func
        for neighbor_id, edges_dict in outgoing_adjacencies[node_id].items():
            if neighbor_id in excluded_nodes:
                continue

            edge_cost, selected_edges = edge_select_func(
                graph,
                node_id,
                neighbor_id,
                edges_dict,
                excluded_edges,
                excluded_nodes,
            )
            if not selected_edges:
                continue

            new_cost = current_cost + edge_cost
            if (neighbor_id not in costs) or (new_cost < costs[neighbor_id]):
                costs[neighbor_id] = new_cost
                pred[neighbor_id] = {node_id: selected_edges}
                heappush(min_pq, (new_cost, neighbor_id))
            elif multipath and new_cost == costs[neighbor_id]:
                pred[neighbor_id][node_id] = selected_edges

    return costs, pred


def ksp(
    graph: StrictMultiDiGraph,
    src_node: NodeID,
    dst_node: NodeID,
    edge_select: EdgeSelect = EdgeSelect.ALL_MIN_COST,
    edge_select_func: Optional[
        Callable[
            [
                StrictMultiDiGraph,
                NodeID,
                NodeID,
                Dict[EdgeID, AttrDict],
                Set[EdgeID],
                Set[NodeID],
            ],
            Tuple[Cost, List[EdgeID]],
        ]
    ] = None,
    max_k: Optional[int] = None,
    max_path_cost: Cost = float("inf"),
    max_path_cost_factor: Optional[float] = None,
    multipath: bool = True,
    excluded_edges: Optional[Set[EdgeID]] = None,
    excluded_nodes: Optional[Set[NodeID]] = None,
) -> Iterator[Tuple[Dict[NodeID, Cost], Dict[NodeID, Dict[NodeID, List[EdgeID]]]]]:
    """Yield up to k shortest paths using a Yen-like algorithm.

    The initial SPF (shortest path) is computed; subsequent paths are found by systematically
    excluding edges/nodes used by previously generated paths. Each iteration yields a
    (costs, pred) describing one path. Stops if there are no more valid paths or if max_k
    is reached.

    Args:
        graph: The directed graph (StrictMultiDiGraph).
        src_node: The source node.
        dst_node: The destination node.
        edge_select: The edge selection strategy. Defaults to ALL_MIN_COST.
        edge_select_func: Optional override of the default edge selection function.
        max_k: If set, yields at most k distinct paths.
        max_path_cost: If set, do not yield any path whose total cost > max_path_cost.
        max_path_cost_factor: If set, updates max_path_cost to:
            min(max_path_cost, best_path_cost * max_path_cost_factor).
        multipath: Whether to consider multiple same-cost expansions in SPF.
        excluded_edges: Set of edge IDs to exclude globally.
        excluded_nodes: Set of node IDs to exclude globally.

    Yields:
        Tuple of ``(costs, pred)`` per discovered path in ascending cost order.
    """
    if edge_select_func is None:
        edge_select_func = edge_select_fabric(edge_select)

    excluded_edges = excluded_edges or set()
    excluded_nodes = excluded_nodes or set()

    shortest_paths = []  # Stores paths found so far: (costs, pred, excl_e, excl_n)
    candidates: List[
        Tuple[
            Cost,
            int,
            Dict[NodeID, Cost],
            Dict[NodeID, Dict[NodeID, List[EdgeID]]],
            Set[EdgeID],
            Set[NodeID],
        ]
    ] = []
    visited = set()  # Tracks path signatures to avoid duplicates

    # 1) Compute the initial shortest path
    costs_init, pred_init = spf(
        graph,
        src_node,
        edge_select,
        edge_select_func,
        multipath,
        excluded_edges,
        excluded_nodes,
    )
    if dst_node not in pred_init:
        return  # No path exists from src_node to dst_node

    best_path_cost = costs_init[dst_node]
    if max_path_cost_factor:
        max_path_cost = min(max_path_cost, best_path_cost * max_path_cost_factor)

    if best_path_cost > max_path_cost:
        return

    shortest_paths.append(
        (costs_init, pred_init, excluded_edges.copy(), excluded_nodes.copy())
    )
    yield costs_init, pred_init

    candidate_id = 0

    while True:
        if max_k and len(shortest_paths) >= max_k:
            break

        root_costs, root_pred, root_excl_e, root_excl_n = shortest_paths[-1]
        # For each realized path from src->dst in the last SPF
        for path in resolve_to_paths(src_node, dst_node, root_pred):
            # Spur node iteration
            for idx, (spur_node, _edges_list) in enumerate(path[:-1]):
                # The path up to but not including spur_node
                root_path = path[:idx]

                # Copy the excluded sets
                excl_e = root_excl_e.copy()
                excl_n = root_excl_n.copy()

                # Remove edges (and possibly nodes) used in previous shortest paths that
                # share the same root_path
                for _sp_costs, sp_pred, sp_ex_e, sp_ex_n in shortest_paths:
                    for p in resolve_to_paths(src_node, dst_node, sp_pred):
                        if p[:idx] == root_path:
                            excl_e.update(sp_ex_e)
                            # Exclude the next edge in that path to force a different route
                            excl_e.update(p[idx][1])
                            excl_n.update(sp_ex_n)
                            excl_n.update(n_e[0] for n_e in p[:idx])

                spur_costs, spur_pred = spf(
                    graph,
                    spur_node,
                    edge_select,
                    edge_select_func,
                    multipath,
                    excl_e,
                    excl_n,
                )
                if dst_node not in spur_pred:
                    continue

                # Shift all spur_costs relative to the cost from src->spur_node
                spur_base_cost = root_costs[spur_node]
                for node_key, node_val in spur_costs.items():
                    spur_costs[node_key] = node_val + spur_base_cost

                # Combine root + spur costs and preds
                total_costs = dict(root_costs)
                total_costs.update(spur_costs)

                total_pred = dict(root_pred)
                for node_key, node_pred in spur_pred.items():
                    # Replace spur_node's chain, but keep root_path info
                    if node_key != spur_node:
                        total_pred[node_key] = node_pred

                path_edge_ids = tuple(
                    sorted(
                        edge_id
                        for nbrs in total_pred.values()
                        for edge_list_ids in nbrs.values()
                        for edge_id in edge_list_ids
                    )
                )
                if path_edge_ids not in visited:
                    if total_costs[dst_node] <= max_path_cost:
                        heappush(
                            candidates,
                            (
                                total_costs[dst_node],
                                candidate_id,
                                total_costs,
                                total_pred,
                                excl_e,
                                excl_n,
                            ),
                        )
                        visited.add(path_edge_ids)
                        candidate_id += 1

        if not candidates:
            break

        # Pop the best candidate path from the min-heap
        _, _, costs_cand, pred_cand, excl_e_cand, excl_n_cand = heappop(candidates)
        shortest_paths.append((costs_cand, pred_cand, excl_e_cand, excl_n_cand))
        yield costs_cand, pred_cand
