"""Lightweight representation of a single routing path.

The ``Path`` dataclass stores a node-and-parallel-edges sequence and a numeric
cost. Cached properties expose derived sequences for nodes and edges, and
helpers provide equality, ordering by cost, and sub-path extraction with cost
recalculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterator, Set, Tuple

from ngraph.algorithms.base import Cost, PathTuple
from ngraph.graph.strict_multidigraph import EdgeID, NodeID, StrictMultiDiGraph


@dataclass
class Path:
    """Represents a single path in the network.

    Attributes:
        path_tuple (PathTuple):
            A sequence of path elements. Each element is a tuple of the form
            (node_id, (edge_id_1, edge_id_2, ...)), where the final element typically has an empty tuple.
        cost (Cost):
            The total numeric cost (e.g., distance or metric) of the path.
        edges (Set[EdgeID]):
            A set of all edge IDs encountered in the path.
        nodes (Set[NodeID]):
            A set of all node IDs encountered in the path.
        edge_tuples (Set[Tuple[EdgeID, ...]]):
            A set of all tuples of parallel edges from each path element (including the final empty tuple).
    """

    path_tuple: PathTuple
    cost: Cost
    edges: Set[EdgeID] = field(init=False, default_factory=set, repr=False)
    nodes: Set[NodeID] = field(init=False, default_factory=set, repr=False)
    edge_tuples: Set[Tuple[EdgeID, ...]] = field(
        init=False, default_factory=set, repr=False
    )

    def __post_init__(self) -> None:
        """Populate `edges`, `nodes`, and `edge_tuples` from `path_tuple`."""
        for node, parallel_edges in self.path_tuple:
            self.nodes.add(node)
            self.edges.update(parallel_edges)
            self.edge_tuples.add(parallel_edges)

    def __getitem__(self, idx: int) -> Tuple[NodeID, Tuple[EdgeID, ...]]:
        """Return the (node, parallel_edges) tuple at the specified index.

        Args:
            idx: The index of the desired path element.

        Returns:
            A tuple containing the node ID and its associated parallel edges.
        """
        return self.path_tuple[idx]

    def __iter__(self) -> Iterator[Tuple[NodeID, Tuple[EdgeID, ...]]]:
        """Iterate over each (node, parallel_edges) element in the path.

        Yields:
            Each element from `path_tuple` in order.
        """
        return iter(self.path_tuple)

    def __len__(self) -> int:
        """Return the number of elements in the path.

        Returns:
            The length of `path_tuple`.
        """
        return len(self.path_tuple)

    @property
    def src_node(self) -> NodeID:
        """Return the first node in the path (the source node)."""
        return self.path_tuple[0][0]

    @property
    def dst_node(self) -> NodeID:
        """Return the last node in the path (the destination node)."""
        return self.path_tuple[-1][0]

    def __lt__(self, other: Any) -> bool:
        """Compare two paths based on their cost.

        Args:
            other: Another Path instance.

        Returns:
            True if this path's cost is less than the other's cost; otherwise, False.
            Returns NotImplemented if `other` is not a Path.
        """
        if not isinstance(other, Path):
            return NotImplemented
        return self.cost < other.cost

    def __eq__(self, other: Any) -> bool:
        """Check equality by comparing path structure and cost.

        Args:
            other: Another Path instance.

        Returns:
            True if both the `path_tuple` and `cost` are equal; otherwise, False.
            Returns NotImplemented if `other` is not a Path.
        """
        if not isinstance(other, Path):
            return NotImplemented
        return (self.path_tuple == other.path_tuple) and (self.cost == other.cost)

    def __hash__(self) -> int:
        """Compute a hash based on the (path_tuple, cost) tuple.

        Returns:
            The hash value of this Path.
        """
        return hash((self.path_tuple, self.cost))

    def __repr__(self) -> str:
        """Return a string representation of the path including its tuple and cost.

        Returns:
            A debug-friendly string representation.
        """
        return f"Path({self.path_tuple}, cost={self.cost})"

    @cached_property
    def edges_seq(self) -> Tuple[Tuple[EdgeID, ...], ...]:
        """Return a tuple containing the sequence of parallel-edge tuples for each path element except the last.

        Returns:
            A tuple of parallel-edge tuples; returns an empty tuple if the path has 1 or fewer elements.
        """
        if len(self.path_tuple) <= 1:
            return ()
        return tuple(parallel_edges for _, parallel_edges in self.path_tuple[:-1])

    @cached_property
    def nodes_seq(self) -> Tuple[NodeID, ...]:
        """Return a tuple of node IDs in order along the path.

        Returns:
            A tuple containing the ordered sequence of nodes from source to destination.
        """
        return tuple(node for node, _ in self.path_tuple)

    def get_sub_path(
        self,
        dst_node: NodeID,
        graph: StrictMultiDiGraph,
        cost_attr: str = "cost",
    ) -> Path:
        """Create a sub-path ending at the specified destination node, recalculating the cost.

        The sub-path is formed by truncating the original path at the first occurrence
        of `dst_node` and ensuring that the final element has an empty tuple of edges.
        The cost is recalculated as the sum of the minimum cost (based on `cost_attr`)
        among parallel edges for each step leading up to (but not including) the target.

        Args:
            dst_node: The node at which to truncate the path.
            graph: The graph containing edge attributes.
            cost_attr: The edge attribute name to use for cost (default is "cost").

        Returns:
            A new Path instance representing the sub-path from the original source to `dst_node`.

        Raises:
            ValueError: If `dst_node` is not found in the current path.
        """
        edges_map = graph.get_edges()
        new_elements = []
        new_cost = 0.0
        found = False

        for node, parallel_edges in self.path_tuple:
            if node == dst_node:
                found = True
                # Append the target node with an empty edge tuple.
                new_elements.append((node, ()))
                break

            new_elements.append((node, parallel_edges))
            if parallel_edges:
                # Accumulate cost using the minimum cost among parallel edges.
                new_cost += min(
                    edges_map[e_id][3][cost_attr] for e_id in parallel_edges
                )

        if not found:
            raise ValueError(f"Node '{dst_node}' not found in path.")

        return Path(tuple(new_elements), new_cost)
