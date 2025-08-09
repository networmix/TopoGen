"""Edge selection algorithms for routing.

Provides selection routines used by SPF to choose candidate edges between
neighbors according to cost and capacity constraints.
"""

from math import isclose
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ngraph.algorithms.base import MIN_CAP, Cost, EdgeSelect
from ngraph.graph.strict_multidigraph import (
    AttrDict,
    EdgeID,
    NodeID,
    StrictMultiDiGraph,
)


def edge_select_fabric(
    edge_select: EdgeSelect,
    select_value: Optional[Any] = None,
    edge_select_func: Optional[
        Callable[
            [
                StrictMultiDiGraph,
                NodeID,
                NodeID,
                Dict[EdgeID, AttrDict],
                Optional[Set[EdgeID]],
                Optional[Set[NodeID]],
            ],
            Tuple[Cost, List[EdgeID]],
        ]
    ] = None,
    excluded_edges: Optional[Set[EdgeID]] = None,
    excluded_nodes: Optional[Set[NodeID]] = None,
    cost_attr: str = "cost",
    capacity_attr: str = "capacity",
    flow_attr: str = "flow",
) -> Callable[
    [
        StrictMultiDiGraph,
        NodeID,
        NodeID,
        Dict[EdgeID, AttrDict],
        Optional[Set[EdgeID]],
        Optional[Set[NodeID]],
    ],
    Tuple[Cost, List[EdgeID]],
]:
    """Create an edge-selection callable for SPF.

    Args:
        edge_select: An EdgeSelect enum specifying the selection strategy.
        select_value: An optional numeric threshold or scaling factor for capacity checks.
        edge_select_func: A user-supplied function if edge_select=USER_DEFINED.
        excluded_edges: A set of edges to ignore entirely.
        excluded_nodes: A set of nodes to skip (if the destination node is in this set).
        cost_attr: The edge attribute name representing cost.
        capacity_attr: The edge attribute name representing capacity.
        flow_attr: The edge attribute name representing current flow.

    Returns:
        Callable: Function with signature
            ``(graph, src, dst, edges_dict, excluded_edges, excluded_nodes) ->
            (selected_cost, [edge_ids])``.
    """

    def get_all_min_cost_edges(
        graph: StrictMultiDiGraph,
        src_node: NodeID,
        dst_node: NodeID,
        edges_map: Dict[EdgeID, AttrDict],
        excluded_edges: Optional[Set[EdgeID]] = None,
        excluded_nodes: Optional[Set[NodeID]] = None,
    ) -> Tuple[Cost, List[EdgeID]]:
        """Return all edges whose cost is the minimum among available edges.
        If the destination node is excluded, returns (inf, []).
        """
        if excluded_nodes and dst_node in excluded_nodes:
            return float("inf"), []

        edge_list: List[EdgeID] = []
        min_cost = float("inf")

        for edge_id, attr in edges_map.items():
            if excluded_edges and edge_id in excluded_edges:
                continue

            cost_val = attr[cost_attr]
            if cost_val < min_cost:
                min_cost = cost_val
                edge_list = [edge_id]
            elif isclose(cost_val, min_cost, abs_tol=1e-12):
                edge_list.append(edge_id)

        return min_cost, edge_list

    def get_single_min_cost_edge(
        graph: StrictMultiDiGraph,
        src_node: NodeID,
        dst_node: NodeID,
        edges_map: Dict[EdgeID, AttrDict],
        excluded_edges: Optional[Set[EdgeID]] = None,
        excluded_nodes: Optional[Set[NodeID]] = None,
    ) -> Tuple[Cost, List[EdgeID]]:
        """Return exactly one edge: the single lowest-cost edge.
        If the destination node is excluded, returns (inf, []).
        """
        if excluded_nodes and dst_node in excluded_nodes:
            return float("inf"), []

        chosen_edge: List[EdgeID] = []
        min_cost = float("inf")

        for edge_id, attr in edges_map.items():
            if excluded_edges and edge_id in excluded_edges:
                continue

            cost_val = attr[cost_attr]
            if cost_val < min_cost:
                min_cost = cost_val
                chosen_edge = [edge_id]

        return min_cost, chosen_edge

    def get_all_edges_with_cap_remaining(
        graph: StrictMultiDiGraph,
        src_node: NodeID,
        dst_node: NodeID,
        edges_map: Dict[EdgeID, AttrDict],
        excluded_edges: Optional[Set[EdgeID]] = None,
        excluded_nodes: Optional[Set[NodeID]] = None,
    ) -> Tuple[Cost, List[EdgeID]]:
        """Return all edges that have remaining capacity >= min_cap,
        ignoring cost differences (though return the minimal cost found).
        """
        if excluded_nodes and dst_node in excluded_nodes:
            return float("inf"), []

        edge_list: List[EdgeID] = []
        min_cost = float("inf")
        min_cap = select_value if select_value is not None else MIN_CAP

        for edge_id, attr in edges_map.items():
            if excluded_edges and edge_id in excluded_edges:
                continue

            capacity_val = attr[capacity_attr]
            flow_val = attr[flow_attr]
            remaining_cap = capacity_val - flow_val

            if remaining_cap >= min_cap:
                cost_val = attr[cost_attr]
                if cost_val < min_cost:
                    min_cost = cost_val
                edge_list.append(edge_id)

        return min_cost, edge_list

    def get_all_min_cost_edges_with_cap_remaining(
        graph: StrictMultiDiGraph,
        src_node: NodeID,
        dst_node: NodeID,
        edges_map: Dict[EdgeID, AttrDict],
        excluded_edges: Optional[Set[EdgeID]] = None,
        excluded_nodes: Optional[Set[NodeID]] = None,
    ) -> Tuple[Cost, List[EdgeID]]:
        """Return all edges that have remaining capacity >= min_cap
        among those with the minimum cost.
        """
        if excluded_nodes and dst_node in excluded_nodes:
            return float("inf"), []

        edge_list: List[EdgeID] = []
        min_cost = float("inf")
        min_cap = select_value if select_value is not None else MIN_CAP

        for edge_id, attr in edges_map.items():
            if excluded_edges and edge_id in excluded_edges:
                continue

            capacity_val = attr[capacity_attr]
            flow_val = attr[flow_attr]
            remaining_cap = capacity_val - flow_val

            if remaining_cap >= min_cap:
                cost_val = attr[cost_attr]
                if cost_val < min_cost:
                    min_cost = cost_val
                    edge_list = [edge_id]
                elif isclose(cost_val, min_cost, abs_tol=1e-12):
                    edge_list.append(edge_id)

        return min_cost, edge_list

    def get_single_min_cost_edge_with_cap_remaining(
        graph: StrictMultiDiGraph,
        src_node: NodeID,
        dst_node: NodeID,
        edges_map: Dict[EdgeID, AttrDict],
        excluded_edges: Optional[Set[EdgeID]] = None,
        excluded_nodes: Optional[Set[NodeID]] = None,
    ) -> Tuple[Cost, List[EdgeID]]:
        """Return exactly one edge with the minimal cost among those
        that have remaining capacity >= min_cap.
        """
        if excluded_nodes and dst_node in excluded_nodes:
            return float("inf"), []

        chosen_edge: List[EdgeID] = []
        min_cost = float("inf")
        min_cap = select_value if select_value is not None else MIN_CAP

        for edge_id, attr in edges_map.items():
            if excluded_edges and edge_id in excluded_edges:
                continue

            capacity_val = attr[capacity_attr]
            flow_val = attr[flow_attr]
            remaining_cap = capacity_val - flow_val

            if remaining_cap >= min_cap:
                cost_val = attr[cost_attr]
                if cost_val < min_cost:
                    min_cost = cost_val
                    chosen_edge = [edge_id]

        return min_cost, chosen_edge

    def get_single_min_cost_edge_with_cap_remaining_load_factored(
        graph: StrictMultiDiGraph,
        src_node: NodeID,
        dst_node: NodeID,
        edges_map: Dict[EdgeID, AttrDict],
        excluded_edges: Optional[Set[EdgeID]] = None,
        excluded_nodes: Optional[Set[NodeID]] = None,
    ) -> Tuple[Cost, List[EdgeID]]:
        """Return exactly one edge, factoring both cost and load level
        into a combined cost: cost_factor = (cost * 100) + round((flow/capacity)*10).
        Only edges with remaining capacity >= min_cap are considered.
        """
        if excluded_nodes and dst_node in excluded_nodes:
            return float("inf"), []

        chosen_edge: List[EdgeID] = []
        min_cost_factor = float("inf")
        min_cap = select_value if select_value is not None else MIN_CAP

        for edge_id, attr in edges_map.items():
            if excluded_edges and edge_id in excluded_edges:
                continue

            capacity_val = attr[capacity_attr]
            flow_val = attr[flow_attr]
            remaining_cap = capacity_val - flow_val

            if remaining_cap >= min_cap:
                base_cost = attr[cost_attr] * 100
                # Avoid division by zero if capacity_val == 0
                load_factor = (
                    round((flow_val / capacity_val) * 10) if capacity_val else 999999
                )
                cost_val = base_cost + load_factor

                if cost_val < min_cost_factor:
                    min_cost_factor = cost_val
                    chosen_edge = [edge_id]

        return float(min_cost_factor), chosen_edge

    # --------------------------------------------------------------------------
    # Map the EdgeSelect enum to the appropriate inner function.
    # --------------------------------------------------------------------------
    if edge_select == EdgeSelect.ALL_MIN_COST:
        return get_all_min_cost_edges
    elif edge_select == EdgeSelect.SINGLE_MIN_COST:
        return get_single_min_cost_edge
    elif edge_select == EdgeSelect.ALL_MIN_COST_WITH_CAP_REMAINING:
        return get_all_min_cost_edges_with_cap_remaining
    elif edge_select == EdgeSelect.ALL_ANY_COST_WITH_CAP_REMAINING:
        return get_all_edges_with_cap_remaining
    elif edge_select == EdgeSelect.SINGLE_MIN_COST_WITH_CAP_REMAINING:
        return get_single_min_cost_edge_with_cap_remaining
    elif edge_select == EdgeSelect.SINGLE_MIN_COST_WITH_CAP_REMAINING_LOAD_FACTORED:
        return get_single_min_cost_edge_with_cap_remaining_load_factored
    elif edge_select == EdgeSelect.USER_DEFINED:
        if edge_select_func is None:
            raise ValueError(
                "edge_select=USER_DEFINED requires 'edge_select_func' to be provided."
            )
        return edge_select_func
    else:
        raise ValueError(f"Unknown edge_select value {edge_select}")
