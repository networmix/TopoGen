"""Workflow step for basic node and link statistics.

Computes and stores network statistics including node/link counts,
capacity distributions, cost distributions, and degree distributions. Supports
optional exclusion simulation and disabled entity handling.

YAML Configuration Example:
    ```yaml
    workflow:
      - step_type: NetworkStats
        name: "network_statistics"           # Optional: Custom name for this step
        include_disabled: false              # Include disabled nodes/links in stats
        excluded_nodes: ["node1", "node2"]   # Optional: Temporary node exclusions
        excluded_links: ["link1", "link3"]   # Optional: Temporary link exclusions
    ```

Results stored in `scenario.results`:
    - Node statistics: node_count
    - Link statistics: link_count, total_capacity, mean_capacity, median_capacity,
      min_capacity, max_capacity, mean_cost, median_cost, min_cost, max_cost
    - Degree statistics: mean_degree, median_degree, min_degree, max_degree
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import TYPE_CHECKING, Dict, Iterable, List

from ngraph.model.view import NetworkView
from ngraph.workflow.base import WorkflowStep, register_workflow_step

if TYPE_CHECKING:
    from ngraph.scenario import Scenario


@dataclass
class NetworkStats(WorkflowStep):
    """Compute basic node and link statistics for the network.

    Supports optional exclusion simulation using NetworkView without modifying the base network.

    Attributes:
        include_disabled: If True, include disabled nodes and links in statistics.
            If False, only consider enabled entities.
        excluded_nodes: Optional list of node names to exclude (temporary exclusion).
        excluded_links: Optional list of link IDs to exclude (temporary exclusion).
    """

    include_disabled: bool = False
    excluded_nodes: Iterable[str] = ()
    excluded_links: Iterable[str] = ()

    def run(self, scenario: Scenario) -> None:
        """Compute and store network statistics.

        If `excluded_nodes` or `excluded_links` are specified, uses `NetworkView` to
        simulate exclusions without modifying the base network.

        Args:
            scenario: The scenario containing the network to analyze.

        Returns:
            None
        """
        # Create view if we have exclusions, otherwise use base network
        if self.excluded_nodes or self.excluded_links:
            network_or_view = NetworkView.from_excluded_sets(
                scenario.network,
                excluded_nodes=self.excluded_nodes,
                excluded_links=self.excluded_links,
            )
            nodes = network_or_view.nodes
            links = network_or_view.links
        else:
            # Use base network, optionally filtering disabled
            if self.include_disabled:
                nodes = scenario.network.nodes
                links = scenario.network.links
            else:
                nodes = {
                    name: node
                    for name, node in scenario.network.nodes.items()
                    if not node.disabled
                }
                links = {
                    link_id: link
                    for link_id, link in scenario.network.links.items()
                    if not link.disabled
                    and link.source in nodes  # Source node must be enabled
                    and link.target in nodes  # Target node must be enabled
                }

        # Compute node statistics
        node_count = len(nodes)
        scenario.results.put(self.name, "node_count", node_count)

        # Compute link statistics
        link_count = len(links)
        scenario.results.put(self.name, "link_count", link_count)

        if links:
            capacities = [link.capacity for link in links.values()]
            costs = [link.cost for link in links.values()]

            scenario.results.put(self.name, "total_capacity", sum(capacities))
            scenario.results.put(self.name, "mean_capacity", mean(capacities))
            scenario.results.put(self.name, "median_capacity", median(capacities))
            scenario.results.put(self.name, "min_capacity", min(capacities))
            scenario.results.put(self.name, "max_capacity", max(capacities))

            scenario.results.put(self.name, "mean_cost", mean(costs))
            scenario.results.put(self.name, "median_cost", median(costs))
            scenario.results.put(self.name, "min_cost", min(costs))
            scenario.results.put(self.name, "max_cost", max(costs))

        # Compute degree statistics (only for enabled nodes)
        if nodes:
            degrees: Dict[str, int] = {name: 0 for name in nodes}

            for link in links.values():
                if link.source in degrees:
                    degrees[link.source] += 1
                if link.target in degrees:
                    degrees[link.target] += 1

            degree_values: List[int] = list(degrees.values())
            scenario.results.put(self.name, "mean_degree", mean(degree_values))
            scenario.results.put(self.name, "median_degree", median(degree_values))
            scenario.results.put(self.name, "min_degree", min(degree_values))
            scenario.results.put(self.name, "max_degree", max(degree_values))


# Register the class after definition to avoid decorator ordering issues
register_workflow_step("NetworkStats")(NetworkStats)
