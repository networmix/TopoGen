"""Graph building workflow component.

Converts scenario network definitions into StrictMultiDiGraph structures suitable
for analysis algorithms. No additional parameters required beyond basic workflow step options.

YAML Configuration Example:
    ```yaml
    workflow:
      - step_type: BuildGraph
        name: "build_network_graph"  # Optional: Custom name for this step
    ```

Results stored in `scenario.results`:
    - graph: `StrictMultiDiGraph` object with bidirectional links
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ngraph.workflow.base import WorkflowStep, register_workflow_step

if TYPE_CHECKING:
    from ngraph.scenario import Scenario


@dataclass
class BuildGraph(WorkflowStep):
    """A workflow step that builds a StrictMultiDiGraph from scenario.network.

    This step converts the scenario's network definition into a graph structure
    suitable for analysis algorithms. No additional parameters are required.
    """

    def run(self, scenario: Scenario) -> None:
        """Build the network graph and store it in results.

        Args:
            scenario: Scenario containing the network model.

        Returns:
            None
        """
        graph = scenario.network.to_strict_multidigraph(add_reverse=True)
        scenario.results.put(self.name, "graph", graph)


# Register the class after definition to avoid decorator ordering issues
register_workflow_step("BuildGraph")(BuildGraph)
