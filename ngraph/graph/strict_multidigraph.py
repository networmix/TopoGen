"""Strict multi-directed graph with validation and convenience APIs.

`StrictMultiDiGraph` extends `networkx.MultiDiGraph` to enforce explicit node
management, unique edge identifiers, and predictable error handling. It exposes
helpers to access nodes/edges as dictionaries and to serialize in node-link
format via `to_dict()`.
"""

from __future__ import annotations

from pickle import dumps, loads
from typing import Any, Dict, Hashable, List, Optional, Tuple

import networkx as nx

from ngraph.utils.ids import new_base64_uuid

NodeID = Hashable
EdgeID = Hashable
AttrDict = Dict[str, Any]
EdgeTuple = Tuple[NodeID, NodeID, EdgeID, AttrDict]


class StrictMultiDiGraph(nx.MultiDiGraph):
    """A custom multi-directed graph with strict rules and unique edge IDs.

    This class enforces:
      - No automatic creation of missing nodes when adding an edge.
      - No duplicate nodes (raises ValueError on duplicates).
      - No duplicate edges by key (raises ValueError on duplicates).
      - Removing non-existent nodes or edges raises ValueError.
      - Each edge key must be unique; by default, a Base64-UUID is generated
        if none is provided.
      - ``copy()`` can perform a pickle-based deep copy that may be faster
        than the NetworkX default.

    Inherits from:
        networkx.MultiDiGraph
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a StrictMultiDiGraph.

        Args:
            *args: Positional arguments forwarded to the MultiDiGraph constructor.
            **kwargs: Keyword arguments forwarded to the MultiDiGraph constructor.

        Attributes:
            _edges: Map edge key to ``(source_node, target_node, edge_key, attribute_dict)``.
        """
        super().__init__(*args, **kwargs)
        self._edges: Dict[EdgeID, EdgeTuple] = {}

    def copy(self, as_view: bool = False, pickle: bool = True) -> StrictMultiDiGraph:
        """Create a copy of this graph.

        By default, use pickle-based deep copying. If ``pickle=False``,
        call the parent class's copy, which supports views.

        Args:
            as_view: If True, return a view instead of a full copy; only used
                if ``pickle=False``.
            pickle: If True, perform a pickle-based deep copy.

        Returns:
            StrictMultiDiGraph: A new instance (or view) of the graph.
        """
        if not pickle:
            return super().copy(as_view=as_view)  # type: ignore[return-value]
        return loads(dumps(self))

    #
    # Node management
    #
    def add_node(self, node_for_adding: NodeID, **attr: Any) -> None:
        """Add a single node, disallowing duplicates.

        Args:
            node_for_adding: The node to add.
            **attr: Arbitrary attributes for this node.

        Raises:
            ValueError: If the node already exists in the graph.
        """
        if node_for_adding in self:
            raise ValueError(f"Node '{node_for_adding}' already exists in this graph.")
        super().add_node(node_for_adding, **attr)

    def remove_node(self, n: NodeID) -> None:
        """Remove a single node and all incident edges.

        Args:
            n: The node to remove.

        Raises:
            ValueError: If the node does not exist in the graph.
        """
        if n not in self:
            raise ValueError(f"Node '{n}' does not exist.")
        # Remove any edges that reference this node
        to_delete = [
            e_id for e_id, (s, t, _, _) in self._edges.items() if s == n or t == n
        ]
        for e_id in to_delete:
            del self._edges[e_id]

        super().remove_node(n)

    #
    # Edge management
    #
    def add_edge(
        self,
        u_for_edge: NodeID,
        v_for_edge: NodeID,
        key: Optional[EdgeID] = None,
        **attr: Any,
    ) -> EdgeID:
        """Add a directed edge from u_for_edge to v_for_edge.

        If no key is provided, a unique Base64-UUID is generated. This method
        does not create nodes automatically; both u_for_edge and v_for_edge
        must already exist in the graph.

        Args:
            u_for_edge: The source node. Must exist in the graph.
            v_for_edge: The target node. Must exist in the graph.
            key: The unique edge key. If None, a new key is generated. Must not
                already be in use if provided.
            **attr: Arbitrary edge attributes.

        Returns:
            EdgeID: The key associated with this new edge.

        Raises:
            ValueError: If either node does not exist, or if the key is already in use.
        """
        if u_for_edge not in self:
            raise ValueError(f"Source node '{u_for_edge}' does not exist.")
        if v_for_edge not in self:
            raise ValueError(f"Target node '{v_for_edge}' does not exist.")

        if key is None:
            key = new_base64_uuid()
        else:
            if key in self._edges:
                raise ValueError(f"Edge with id '{key}' already exists.")

        super().add_edge(u_for_edge, v_for_edge, key=key, **attr)
        # At this point, key is guaranteed to be non-None (either provided or generated)
        assert key is not None
        self._edges[key] = (
            u_for_edge,
            v_for_edge,
            key,
            self[u_for_edge][v_for_edge][key],  # pyright: ignore[reportArgumentType]
        )
        return key

    def remove_edge(
        self,
        u: NodeID,
        v: NodeID,
        key: Optional[EdgeID] = None,
    ) -> None:
        """Remove an edge (or edges) between nodes u and v.

        If key is provided, remove only that edge. Otherwise, remove all edges
        from u to v.

        Args:
            u: The source node of the edge(s). Must exist in the graph.
            v: The target node of the edge(s). Must exist in the graph.
            key: If provided, remove the edge with this key. Otherwise, remove
                all edges from u to v.

        Raises:
            ValueError: If the nodes do not exist, or if the specified edge key
                does not exist, or if no edges are found from u to v.
        """
        if u not in self:
            raise ValueError(f"Source node '{u}' does not exist.")
        if v not in self:
            raise ValueError(f"Target node '{v}' does not exist.")

        if key is not None:
            if key not in self._edges:
                raise ValueError(f"No edge with id='{key}' found from {u} to {v}.")
            src_node, dst_node, _, _ = self._edges[key]
            if src_node != u or dst_node != v:
                raise ValueError(
                    f"Edge with id='{key}' is actually from {src_node} to {dst_node}, "
                    f"not from {u} to {v}."
                )
            self.remove_edge_by_id(key)
        else:
            if v not in self.succ[u]:
                raise ValueError(f"No edges from '{u}' to '{v}' to remove.")
            edge_ids = tuple(self.succ[u][v])
            if not edge_ids:
                raise ValueError(f"No edges from '{u}' to '{v}' to remove.")
            for e_id in edge_ids:
                self.remove_edge_by_id(e_id)

    def remove_edge_by_id(self, key: EdgeID) -> None:
        """Remove a directed edge by its unique key.

        Args:
            key: The key identifying the edge to remove.

        Raises:
            ValueError: If no edge with this key exists in the graph.
        """
        if key not in self._edges:
            raise ValueError(f"Edge with id='{key}' not found.")
        src_node, dst_node, _, _ = self._edges.pop(key)
        super().remove_edge(src_node, dst_node, key=key)

    #
    # Convenience methods
    #
    def get_nodes(self) -> Dict[NodeID, AttrDict]:
        """Retrieve all nodes and their attributes as a dictionary.

        Returns:
            Dict[NodeID, AttrDict]: A mapping of node ID to its attributes.
        """
        return dict(self.nodes(data=True))

    def get_edges(self) -> Dict[EdgeID, EdgeTuple]:
        """Retrieve a dictionary of all edges by their keys.

        Returns:
            Dict[EdgeID, EdgeTuple]: A mapping of edge key to
                ``(source_node, target_node, edge_key, edge_attributes)``.
        """
        return self._edges

    def get_edge_attr(self, key: EdgeID) -> AttrDict:
        """Retrieve the attribute dictionary of a specific edge.

        Args:
            key: The unique edge key.

        Returns:
            AttrDict: The attribute dictionary for the edge.

        Raises:
            ValueError: If no edge with this key is found.
        """
        if key not in self._edges:
            raise ValueError(f"Edge with id='{key}' not found.")
        return self._edges[key][3]

    def has_edge_by_id(self, key: EdgeID) -> bool:
        """Check whether an edge with the given key exists.

        Args:
            key: The unique edge key to check.

        Returns:
            bool: True if the edge key exists, otherwise False.
        """
        return key in self._edges

    def edges_between(self, u: NodeID, v: NodeID) -> List[EdgeID]:
        """List all edge keys from node u to node v.

        Args:
            u: The source node.
            v: The target node.

        Returns:
            List[EdgeID]: List of edge keys from u to v, or an empty list if none exist.
        """
        if u not in self.succ or v not in self.succ[u]:
            return []
        return list(self.succ[u][v].keys())

    def update_edge_attr(self, key: EdgeID, **attr: Any) -> None:
        """Update attributes on an existing edge by key.

        Args:
            key: The unique edge key to update.
            **attr: Arbitrary edge attributes to add or modify.

        Raises:
            ValueError: If the edge with the given key does not exist.
        """
        if key not in self._edges:
            raise ValueError(f"Edge with id='{key}' not found.")
        self._edges[key][3].update(attr)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the graph to a dictionary representation suitable for JSON serialization.

        Return a node-link format dictionary with graph attributes, nodes, and edges.
        The format is compatible with visualization libraries like D3.js.

        Returns:
            Dict[str, Any]: Dictionary containing 'graph', 'nodes', and 'links' keys.
        """
        # Import here to avoid circular import
        from ngraph.graph.io import graph_to_node_link

        return graph_to_node_link(self)
