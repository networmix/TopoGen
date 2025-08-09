"""Read-only view of a ``Network`` with temporary exclusions.

This module defines a view over ``Network`` objects that can exclude nodes and
links for analysis without mutating the base network. It supports what-if
analysis, including failure simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from ngraph.solver.maxflow import (
    max_flow as _solver_max_flow,
)
from ngraph.solver.maxflow import (
    max_flow_detailed as _solver_max_flow_detailed,
)
from ngraph.solver.maxflow import (
    max_flow_with_graph as _solver_max_flow_with_graph,
)
from ngraph.solver.maxflow import (
    max_flow_with_summary as _solver_max_flow_with_summary,
)
from ngraph.solver.maxflow import (
    saturated_edges as _solver_saturated_edges,
)
from ngraph.solver.maxflow import (
    sensitivity_analysis as _solver_sensitivity_analysis,
)
from ngraph.solver.paths import (
    k_shortest_paths as _solver_k_shortest_paths,
)
from ngraph.solver.paths import (
    shortest_path_costs as _solver_shortest_path_costs,
)
from ngraph.solver.paths import (
    shortest_paths as _solver_shortest_paths,
)

if TYPE_CHECKING:
    from ngraph.algorithms.base import FlowPlacement
    from ngraph.algorithms.types import FlowSummary
    from ngraph.graph.strict_multidigraph import StrictMultiDiGraph
    from ngraph.model.network import Link, Network, Node, RiskGroup
    from ngraph.paths.path import Path as _NGPath

__all__ = ["NetworkView"]


@dataclass(frozen=True)
class NetworkView:
    """Read-only overlay that hides selected nodes/links from a base Network.

    NetworkView provides filtered access to a Network where both scenario-disabled
    elements (Node.disabled, Link.disabled) and analysis-excluded elements are
    hidden from algorithms. This enables failure simulation and what-if analysis
    without mutating the base Network.

    Multiple NetworkView instances can safely operate on the same base Network
    concurrently, each with different exclusion sets.

    Example:
        ```python
        # Create view excluding specific nodes for failure analysis
        view = NetworkView.from_excluded_sets(
            base_network,
            excluded_nodes=["node1", "node2"],
            excluded_links=["link1"]
        )

        # Run analysis on filtered topology
        flows = view.max_flow("source.*", "sink.*")
        ```

    Attributes:
        _base: The underlying Network object.
        _excluded_nodes: Frozen set of node names to exclude from analysis.
        _excluded_links: Frozen set of link IDs to exclude from analysis.
    """

    _base: "Network"
    _excluded_nodes: frozenset[str] = frozenset()
    _excluded_links: frozenset[str] = frozenset()

    def is_node_hidden(self, name: str) -> bool:
        """Check if a node is hidden in this view.

        Args:
            name: Name of the node to check.

        Returns:
            True if the node is hidden (disabled or excluded), False otherwise.
        """
        node = self._base.nodes.get(name)
        if node is None:
            return True  # Node doesn't exist, treat as hidden
        return node.disabled or name in self._excluded_nodes

    def is_link_hidden(self, link_id: str) -> bool:
        """Check if a link is hidden in this view.

        Args:
            link_id: ID of the link to check.

        Returns:
            True if the link is hidden (disabled or excluded), False otherwise.
        """
        link = self._base.links.get(link_id)
        if link is None:
            return True  # Link doesn't exist, treat as hidden
        return (
            link.disabled
            or link_id in self._excluded_links
            or self.is_node_hidden(link.source)
            or self.is_node_hidden(link.target)
        )

    @property
    def nodes(self) -> Dict[str, "Node"]:
        """Get visible nodes in this view.

        Returns:
            Dictionary mapping node names to Node objects for all visible nodes.
        """
        return {
            name: node
            for name, node in self._base.nodes.items()
            if not self.is_node_hidden(name)
        }

    @property
    def links(self) -> Dict[str, "Link"]:
        """Get visible links in this view.

        Returns:
            Dictionary mapping link IDs to Link objects for all visible links.
        """
        return {
            link_id: link
            for link_id, link in self._base.links.items()
            if not self.is_link_hidden(link_id)
        }

    @property
    def risk_groups(self) -> Dict[str, "RiskGroup"]:
        """Get all risk groups from the base network.

        Returns:
            Dictionary mapping risk group names to RiskGroup objects.
        """
        return self._base.risk_groups

    @property
    def attrs(self) -> Dict[str, Any]:
        """Get network attributes from the base network.

        Returns:
            Dictionary of network attributes.
        """
        return self._base.attrs

    def to_strict_multidigraph(self, add_reverse: bool = True) -> "StrictMultiDiGraph":
        """Create a StrictMultiDiGraph representation of this view.

        Creates a filtered graph excluding disabled nodes/links and analysis exclusions.
        Results are cached for performance when multiple flow operations are called.

        Args:
            add_reverse: If True, add reverse edges for each link.

        Returns:
            StrictMultiDiGraph with scenario-disabled and analysis-excluded
            elements filtered out.
        """
        # Get or initialize cache (handle frozen dataclass)
        cache = getattr(self, "_graph_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_graph_cache", cache)

        # Use simple cache based on add_reverse parameter
        if add_reverse not in cache:
            cache[add_reverse] = self._base._build_graph(
                add_reverse=add_reverse,
                excluded_nodes=self._excluded_nodes,
                excluded_links=self._excluded_links,
            )
        return cache[add_reverse]

    def select_node_groups_by_path(self, path: str) -> Dict[str, List["Node"]]:
        """Select and group visible nodes by regex or attribute directive.

        Follows the same semantics as ``Network.select_node_groups_by_path`` but
        filters out nodes hidden in this view (disabled or excluded).

        Args:
            path: Regex pattern on node.name, or strict attribute directive ``attr:<name>``.

        Returns:
            Dictionary mapping group labels to lists of matching visible nodes.
        """
        # Get groups from base network, then filter to visible nodes
        base_groups = self._base.select_node_groups_by_path(path)
        filtered_groups = {}

        for label, nodes in base_groups.items():
            visible_nodes = [
                node for node in nodes if not self.is_node_hidden(node.name)
            ]
            if visible_nodes:  # Only include groups with visible nodes
                filtered_groups[label] = visible_nodes

        return filtered_groups

    def max_flow(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
        shortest_path: bool = False,
        flow_placement: Optional["FlowPlacement"] = None,
    ) -> Dict[Tuple[str, str], float]:
        """Compute maximum flow between node groups in this view.

        Args:
            source_path: Regex pattern for selecting source nodes.
            sink_path: Regex pattern for selecting sink nodes.
            mode: Either "combine" or "pairwise".
            shortest_path: If True, flows are constrained to shortest paths.
            flow_placement: Flow placement strategy.

        Returns:
            Dictionary mapping (source_label, sink_label) to flow values.
        """
        return _solver_max_flow(
            self,
            source_path,
            sink_path,
            mode=mode,
            shortest_path=shortest_path,
            flow_placement=(
                flow_placement
                or __import__(
                    "ngraph.algorithms.base", fromlist=["FlowPlacement"]
                ).FlowPlacement.PROPORTIONAL
            ),
        )

    def max_flow_with_summary(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
        shortest_path: bool = False,
        flow_placement: Optional["FlowPlacement"] = None,
    ) -> Dict[Tuple[str, str], Tuple[float, "FlowSummary"]]:
        """Compute maximum flow with detailed analytics summary.

        Args:
            source_path: Regex pattern for selecting source nodes.
            sink_path: Regex pattern for selecting sink nodes.
            mode: Either "combine" or "pairwise".
            shortest_path: If True, flows are constrained to shortest paths.
            flow_placement: Flow placement strategy.

        Returns:
            Dictionary mapping (source_label, sink_label) to (flow_value, summary) tuples.
        """
        return _solver_max_flow_with_summary(
            self,
            source_path,
            sink_path,
            mode=mode,
            shortest_path=shortest_path,
            flow_placement=(
                flow_placement
                or __import__(
                    "ngraph.algorithms.base", fromlist=["FlowPlacement"]
                ).FlowPlacement.PROPORTIONAL
            ),
        )

    def max_flow_with_graph(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
        shortest_path: bool = False,
        flow_placement: Optional["FlowPlacement"] = None,
    ) -> Dict[Tuple[str, str], Tuple[float, "StrictMultiDiGraph"]]:
        """Compute maximum flow and return flow-assigned graph.

        Args:
            source_path: Regex pattern for selecting source nodes.
            sink_path: Regex pattern for selecting sink nodes.
            mode: Either "combine" or "pairwise".
            shortest_path: If True, flows are constrained to shortest paths.
            flow_placement: Flow placement strategy.

        Returns:
            Dictionary mapping (source_label, sink_label) to (flow_value, flow_graph) tuples.
        """
        return _solver_max_flow_with_graph(
            self,
            source_path,
            sink_path,
            mode=mode,
            shortest_path=shortest_path,
            flow_placement=(
                flow_placement
                or __import__(
                    "ngraph.algorithms.base", fromlist=["FlowPlacement"]
                ).FlowPlacement.PROPORTIONAL
            ),
        )

    def max_flow_detailed(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
        shortest_path: bool = False,
        flow_placement: Optional["FlowPlacement"] = None,
    ) -> Dict[Tuple[str, str], Tuple[float, "FlowSummary", "StrictMultiDiGraph"]]:
        """Compute maximum flow with complete analytics and graph.

        Args:
            source_path: Regex pattern for selecting source nodes.
            sink_path: Regex pattern for selecting sink nodes.
            mode: Either "combine" or "pairwise".
            shortest_path: If True, flows are constrained to shortest paths.
            flow_placement: Flow placement strategy.

        Returns:
            Dictionary mapping (source_label, sink_label) to
            (flow_value, summary, flow_graph) tuples.
        """
        return _solver_max_flow_detailed(
            self,
            source_path,
            sink_path,
            mode=mode,
            shortest_path=shortest_path,
            flow_placement=(
                flow_placement
                or __import__(
                    "ngraph.algorithms.base", fromlist=["FlowPlacement"]
                ).FlowPlacement.PROPORTIONAL
            ),
        )

    def saturated_edges(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
        tolerance: float = 1e-10,
        shortest_path: bool = False,
        flow_placement: Optional["FlowPlacement"] = None,
    ) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
        """Identify saturated edges in max flow solutions.

        Args:
            source_path: Regex pattern for selecting source nodes.
            sink_path: Regex pattern for selecting sink nodes.
            mode: Either "combine" or "pairwise".
            tolerance: Tolerance for considering an edge saturated.
            shortest_path: If True, flows are constrained to shortest paths.
            flow_placement: Flow placement strategy.

        Returns:
            Dictionary mapping (source_label, sink_label) to lists of
            saturated edge tuples (u, v, key).
        """
        return _solver_saturated_edges(
            self,
            source_path,
            sink_path,
            mode=mode,
            tolerance=tolerance,
            shortest_path=shortest_path,
            flow_placement=(
                flow_placement
                or __import__(
                    "ngraph.algorithms.base", fromlist=["FlowPlacement"]
                ).FlowPlacement.PROPORTIONAL
            ),
        )

    def sensitivity_analysis(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
        change_amount: float = 1.0,
        shortest_path: bool = False,
        flow_placement: Optional["FlowPlacement"] = None,
    ) -> Dict[Tuple[str, str], Dict[Tuple[str, str, str], float]]:
        """Perform sensitivity analysis on capacity changes.

        Args:
            source_path: Regex pattern for selecting source nodes.
            sink_path: Regex pattern for selecting sink nodes.
            mode: Either "combine" or "pairwise".
            change_amount: Amount to change capacity for testing.
            shortest_path: If True, flows are constrained to shortest paths.
            flow_placement: Flow placement strategy.

        Returns:
            Dictionary mapping (source_label, sink_label) to dictionaries
            of edge sensitivity values.
        """
        return _solver_sensitivity_analysis(
            self,
            source_path,
            sink_path,
            mode=mode,
            change_amount=change_amount,
            shortest_path=shortest_path,
            flow_placement=(
                flow_placement
                or __import__(
                    "ngraph.algorithms.base", fromlist=["FlowPlacement"]
                ).FlowPlacement.PROPORTIONAL
            ),
        )

    def shortest_path_costs(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
    ) -> Dict[Tuple[str, str], float]:
        """Return minimal path costs between node groups in this view."""
        return _solver_shortest_path_costs(self, source_path, sink_path, mode=mode)

    def shortest_paths(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "combine",
        *,
        split_parallel_edges: bool = False,
    ) -> Dict[Tuple[str, str], List[_NGPath]]:
        """Return concrete shortest path(s) between selected node groups."""
        return _solver_shortest_paths(
            self,
            source_path,
            sink_path,
            mode=mode,
            split_parallel_edges=split_parallel_edges,
        )

    def k_shortest_paths(
        self,
        source_path: str,
        sink_path: str,
        mode: str = "pairwise",
        *,
        max_k: int = 3,
        max_path_cost: float = float("inf"),
        max_path_cost_factor: Optional[float] = None,
        split_parallel_edges: bool = False,
    ) -> Dict[Tuple[str, str], List[_NGPath]]:
        """Return up to K shortest paths per group pair."""
        return _solver_k_shortest_paths(
            self,
            source_path,
            sink_path,
            mode=mode,
            max_k=max_k,
            max_path_cost=max_path_cost,
            max_path_cost_factor=max_path_cost_factor,
            split_parallel_edges=split_parallel_edges,
        )

    @classmethod
    def from_excluded_sets(
        cls,
        base: "Network",
        excluded_nodes: Iterable[str] = (),
        excluded_links: Iterable[str] = (),
    ) -> "NetworkView":
        """Create a NetworkView with specified exclusions.

        Args:
            base: Base Network to create view over.
            excluded_nodes: Node names to exclude from analysis.
            excluded_links: Link IDs to exclude from analysis.

        Returns:
            NetworkView with specified exclusions applied.
        """
        return cls(
            _base=base,
            _excluded_nodes=frozenset(excluded_nodes),
            _excluded_links=frozenset(excluded_links),
        )
