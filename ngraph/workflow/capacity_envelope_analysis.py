"""Capacity envelope analysis workflow component.

Monte Carlo analysis of network capacity under random failures using FailureManager.
Generates statistical distributions (envelopes) of maximum flow capacity between
node groups across failure scenarios. Supports parallel processing, baseline analysis,
and configurable failure policies.

This component uses the `FailureManager` convenience method to perform the analysis,
ensuring consistency with the programmatic API while providing workflow integration.

YAML Configuration Example:
    ```yaml
    workflow:
      - step_type: CapacityEnvelopeAnalysis
        name: "capacity_envelope_monte_carlo"  # Optional: Custom name for this step
        source_path: ".*(dc.+)"                # Regex pattern for DC source node groups
        sink_path: ".*(dc.+)"                  # Regex pattern for DC sink node groups
         mode: "combine"                        # "combine" or "pairwise" flow analysis
        failure_policy: "random_failures"      # Optional: Named failure policy to use
        iterations: 1000                       # Number of Monte-Carlo trials
        parallelism: 4                         # Number of parallel worker processes
        shortest_path: false                   # Use shortest paths only
        flow_placement: "PROPORTIONAL"         # Flow placement strategy
        baseline: true                         # Optional: Run first iteration without failures
        seed: 42                               # Optional: Seed for reproducible results
        store_failure_patterns: false          # Optional: Store failure patterns in results
        include_flow_summary: false            # Optional: Collect detailed flow summary statistics
    ```

Results stored in `scenario.results`:
    - capacity_envelopes: Mapping of flow keys to capacity envelope data (serializable)
    - failure_pattern_results: Frequency map of failure patterns (if `store_failure_patterns=True`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ngraph.algorithms.base import FlowPlacement
from ngraph.failure.manager.manager import FailureManager
from ngraph.logging import get_logger
from ngraph.workflow.base import WorkflowStep, register_workflow_step

if TYPE_CHECKING:
    from ngraph.scenario import Scenario

logger = get_logger(__name__)


@dataclass
class CapacityEnvelopeAnalysis(WorkflowStep):
    """Capacity envelope analysis workflow step using FailureManager convenience method.

    This workflow step uses the FailureManager.run_max_flow_monte_carlo() convenience method
    to perform analysis, ensuring consistency with the programmatic API while providing
    workflow integration and result storage.

    Attributes:
        source_path: Regex pattern for source node groups.
        sink_path: Regex pattern for sink node groups.
        mode: Flow analysis mode ("combine" or "pairwise").
        failure_policy: Name of failure policy in scenario.failure_policy_set.
        iterations: Number of Monte-Carlo trials.
        parallelism: Number of parallel worker processes.
        shortest_path: Whether to use shortest paths only.
        flow_placement: Flow placement strategy.
        baseline: Whether to run first iteration without failures as baseline.
        seed: Optional seed for reproducible results.
        store_failure_patterns: Whether to store failure patterns in results.
        include_flow_summary: Whether to collect detailed flow summary statistics (cost distribution, min-cut edges).
    """

    source_path: str = ""
    sink_path: str = ""
    mode: str = "combine"
    failure_policy: str | None = None
    iterations: int = 1
    parallelism: int = 1
    shortest_path: bool = False
    flow_placement: FlowPlacement | str = FlowPlacement.PROPORTIONAL
    baseline: bool = False
    seed: int | None = None
    store_failure_patterns: bool = False
    include_flow_summary: bool = False

    def __post_init__(self):
        """Validate parameters and convert string `flow_placement` to enum.

        Raises:
            ValueError: If parameters are outside accepted ranges or invalid.
        """
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.parallelism < 1:
            raise ValueError("parallelism must be >= 1")
        if self.mode not in {"combine", "pairwise"}:
            raise ValueError("mode must be 'combine' or 'pairwise'")
        if self.baseline and self.iterations < 2:
            raise ValueError(
                "baseline=True requires iterations >= 2 "
                "(first iteration is baseline, remaining are with failures)"
            )

        # Convert string flow_placement to enum if needed
        if isinstance(self.flow_placement, str):
            try:
                self.flow_placement = FlowPlacement[self.flow_placement.upper()]
            except KeyError:
                valid_values = ", ".join([e.name for e in FlowPlacement])
                raise ValueError(
                    f"Invalid flow_placement '{self.flow_placement}'. "
                    f"Valid values are: {valid_values}"
                ) from None

    def run(self, scenario: "Scenario") -> None:
        """Execute capacity envelope analysis using `FailureManager`.

        Args:
            scenario: The scenario containing network, failure policies, and results.

        Returns:
            None
        """
        logger.info(f"Starting capacity envelope analysis: {self.name}")
        logger.debug(
            f"Analysis parameters: source_path={self.source_path}, sink_path={self.sink_path}, "
            f"mode={self.mode}, iterations={self.iterations}, parallelism={self.parallelism}, "
            f"failure_policy={self.failure_policy}, baseline={self.baseline}, "
            f"include_flow_summary={self.include_flow_summary}"
        )

        # Create FailureManager instance
        failure_manager = FailureManager(
            network=scenario.network,
            failure_policy_set=scenario.failure_policy_set,
            policy_name=self.failure_policy,
        )

        # Use the convenience method to get results
        logger.debug(
            f"Running {self.iterations} iterations with parallelism={self.parallelism}"
        )
        envelope_results = failure_manager.run_max_flow_monte_carlo(
            source_path=self.source_path,
            sink_path=self.sink_path,
            mode=self.mode,
            iterations=self.iterations,
            parallelism=self.parallelism,
            shortest_path=self.shortest_path,
            flow_placement=self.flow_placement,
            baseline=self.baseline,
            seed=self.seed,
            store_failure_patterns=self.store_failure_patterns,
            include_flow_summary=self.include_flow_summary,
        )

        logger.info(f"Generated {len(envelope_results.envelopes)} capacity envelopes")

        # Convert envelope objects to serializable format for scenario storage
        envelopes_dict = {
            flow_key: envelope.to_dict()
            for flow_key, envelope in envelope_results.envelopes.items()
        }

        # Store results in scenario
        scenario.results.put(self.name, "capacity_envelopes", envelopes_dict)

        # Store failure patterns if requested
        if self.store_failure_patterns and envelope_results.failure_patterns:
            pattern_results_dict = {
                pattern_key: pattern.to_dict()
                for pattern_key, pattern in envelope_results.failure_patterns.items()
            }
            scenario.results.put(
                self.name, "failure_pattern_results", pattern_results_dict
            )

        logger.info(f"Capacity envelope analysis completed: {self.name}")


# Register the workflow step
register_workflow_step("CapacityEnvelopeAnalysis")(CapacityEnvelopeAnalysis)
