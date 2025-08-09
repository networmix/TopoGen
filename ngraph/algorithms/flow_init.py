"""Flow graph initialization utilities.

Ensures nodes and edges carry aggregate (``flow_attr``) and per-flow
(``flows_attr``) attributes, optionally resetting existing values.
"""

from __future__ import annotations

from ngraph.graph.strict_multidigraph import StrictMultiDiGraph


def init_flow_graph(
    flow_graph: StrictMultiDiGraph,
    flow_attr: str = "flow",
    flows_attr: str = "flows",
    reset_flow_graph: bool = True,
) -> StrictMultiDiGraph:
    """Ensure that nodes and edges expose flow-related attributes.

    For each node and edge:

    - The attribute named `flow_attr` (default: "flow") is set to 0.
    - The attribute named `flows_attr` (default: "flows") is set to an empty dict.

    If `reset_flow_graph` is True, any existing flow values in these attributes
    are overwritten; otherwise they are only created if missing.

    Args:
        flow_graph: The StrictMultiDiGraph whose nodes and edges should be
            prepared for flow assignment.
        flow_attr: The attribute name to track a numeric flow value per node/edge.
        flows_attr: The attribute name to track multiple flow identifiers (and flows).
        reset_flow_graph: If True, reset existing flows (set to 0). If False, do not overwrite.

    Returns:
        StrictMultiDiGraph: The same graph instance after attribute checks.
    """
    # Initialize or reset edge attributes
    for edge_data in flow_graph.get_edges().values():
        attr_dict = edge_data[3]  # The fourth element is the edge attribute dict
        attr_dict.setdefault(flow_attr, 0)
        attr_dict.setdefault(flows_attr, {})
        if reset_flow_graph:
            attr_dict[flow_attr] = 0
            attr_dict[flows_attr] = {}

    # Initialize or reset node attributes
    for node_data in flow_graph.get_nodes().values():
        node_data.setdefault(flow_attr, 0)
        node_data.setdefault(flows_attr, {})
        if reset_flow_graph:
            node_data[flow_attr] = 0
            node_data[flows_attr] = {}

    return flow_graph
