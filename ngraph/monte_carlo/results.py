"""Structured result objects for FailureManager analysis functions.

These classes provide interfaces for accessing Monte Carlo analysis
results from FailureManager convenience methods. Visualization is handled by
specialized analyzer classes in the workflow.analysis module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ngraph.results.artifacts import CapacityEnvelope, FailurePatternResult


@dataclass
class CapacityEnvelopeResults:
    """Results from capacity envelope Monte Carlo analysis.

    This class provides data access for capacity envelope analysis results.
    For visualization, use CapacityMatrixAnalyzer from ngraph.workflow.analysis.

    Attributes:
        envelopes: Dictionary mapping flow keys to CapacityEnvelope objects.
        failure_patterns: Dictionary mapping pattern keys to FailurePatternResult objects.
        source_pattern: Source node regex pattern used in analysis.
        sink_pattern: Sink node regex pattern used in analysis.
        mode: Flow analysis mode ("combine" or "pairwise").
        iterations: Number of Monte Carlo iterations performed.
        metadata: Additional analysis metadata from FailureManager.
    """

    envelopes: Dict[str, CapacityEnvelope]
    failure_patterns: Dict[str, FailurePatternResult]
    source_pattern: str
    sink_pattern: str
    mode: str
    iterations: int
    metadata: Dict[str, Any]

    def flow_keys(self) -> List[str]:
        """Get list of all flow keys in results.

        Returns:
            List of flow keys (e.g., ["datacenter->edge", "edge->datacenter"]).
        """
        return list(self.envelopes.keys())

    def get_envelope(self, flow_key: str) -> CapacityEnvelope:
        """Get CapacityEnvelope for a specific flow.

        Args:
            flow_key: Flow key (e.g., "datacenter->edge").

        Returns:
            CapacityEnvelope object with frequency-based statistics

        Raises:
            KeyError: If flow_key not found in results.
        """
        if flow_key not in self.envelopes:
            available = ", ".join(self.envelopes.keys())
            raise KeyError(f"Flow key '{flow_key}' not found. Available: {available}")
        return self.envelopes[flow_key]

    def summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all flow pairs.

        Returns:
            Dictionary mapping flow keys to statistics (mean, std, percentiles, etc.)
        """
        stats = {}
        for flow_key, envelope in self.envelopes.items():
            stats[flow_key] = {
                "mean": envelope.mean_capacity,
                "std": envelope.stdev_capacity,
                "min": envelope.min_capacity,
                "max": envelope.max_capacity,
                "samples": envelope.total_samples,
                "p5": envelope.get_percentile(5),
                "p25": envelope.get_percentile(25),
                "p50": envelope.get_percentile(50),
                "p75": envelope.get_percentile(75),
                "p95": envelope.get_percentile(95),
            }
        return stats

    def to_dataframe(self) -> pd.DataFrame:
        """Convert capacity envelopes to DataFrame for analysis.

        Returns:
            DataFrame with flow statistics for each flow pair
        """
        stats = self.summary_statistics()
        return pd.DataFrame.from_dict(stats, orient="index")

    def get_failure_pattern_summary(self) -> pd.DataFrame:
        """Get summary of failure patterns if available.

        Returns:
            DataFrame with failure pattern frequencies and impact
        """
        if not self.failure_patterns:
            return pd.DataFrame()

        data = []
        for pattern_key, pattern in self.failure_patterns.items():
            row = {
                "pattern_key": pattern_key,
                "count": pattern.count,
                "is_baseline": pattern.is_baseline,
                "failed_nodes": len(pattern.excluded_nodes),
                "failed_links": len(pattern.excluded_links),
                "total_failures": len(pattern.excluded_nodes)
                + len(pattern.excluded_links),
            }

            # Add capacity impact for each flow
            for flow_key, capacity in pattern.capacity_matrix.items():
                row[f"capacity_{flow_key}"] = capacity

            data.append(row)

        return pd.DataFrame(data)

    def get_cost_distribution(self, flow_key: str) -> Dict[float, Dict[str, float]]:
        """Get cost distribution statistics for a specific flow.

        Args:
            flow_key: Flow key (e.g., "datacenter->edge").

        Returns:
            Dictionary mapping cost values to their statistics
            (mean, min, max, total_samples, frequencies)

        Raises:
            KeyError: If flow_key not found in results.
        """
        envelope = self.get_envelope(flow_key)
        return envelope.flow_summary_stats.get("cost_distribution_stats", {})

    def get_min_cut_frequencies(self, flow_key: str) -> Dict[str, int]:
        """Get min-cut edge frequencies for a specific flow.

        Args:
            flow_key: Flow key (e.g., "datacenter->edge").

        Returns:
            Dictionary mapping edge identifiers to occurrence frequencies

        Raises:
            KeyError: If flow_key not found in results.
        """
        envelope = self.get_envelope(flow_key)
        return envelope.flow_summary_stats.get("min_cut_frequencies", {})

    def cost_distribution_summary(self) -> pd.DataFrame:
        """Get cost distribution summary across all flows.

        Returns:
            DataFrame with cost distribution statistics for all flows
        """
        data = []
        for flow_key, envelope in self.envelopes.items():
            cost_stats = envelope.flow_summary_stats.get("cost_distribution_stats", {})
            for cost, stats in cost_stats.items():
                row = {
                    "flow_key": flow_key,
                    "cost": cost,
                    "mean_flow": stats.get("mean", 0.0),
                    "min_flow": stats.get("min", 0.0),
                    "max_flow": stats.get("max", 0.0),
                    "total_samples": stats.get("total_samples", 0),
                    "unique_values": len(stats.get("frequencies", {})),
                }
                data.append(row)

        return pd.DataFrame(data)

    def export_summary(self) -> Dict[str, Any]:
        """Export summary for serialization.

        Returns:
            Dictionary with all results data in serializable format
        """
        return {
            "source_pattern": self.source_pattern,
            "sink_pattern": self.sink_pattern,
            "mode": self.mode,
            "iterations": self.iterations,
            "metadata": self.metadata,
            "envelopes": {key: env.to_dict() for key, env in self.envelopes.items()},
            "failure_patterns": {
                key: fp.to_dict() for key, fp in self.failure_patterns.items()
            },
            "summary_statistics": self.summary_statistics(),
            "cost_distribution_summary": self.cost_distribution_summary().to_dict(
                "records"
            )
            if not self.cost_distribution_summary().empty
            else [],
        }


@dataclass
class DemandPlacementResults:
    """Results from demand placement Monte Carlo analysis.

    Attributes:
        raw_results: Raw results from FailureManager
        iterations: Number of Monte Carlo iterations
        baseline: Optional baseline result (no failures)
        failure_patterns: Dictionary mapping pattern keys to failure pattern results
        metadata: Additional analysis metadata from FailureManager
    """

    raw_results: dict[str, Any]
    iterations: int
    baseline: Optional[dict[str, Any]] = None
    failure_patterns: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.failure_patterns is None:
            self.failure_patterns = {}
        if self.metadata is None:
            self.metadata = {}

    def success_rate_distribution(self) -> pd.DataFrame:
        """Get demand placement success rate distribution as DataFrame.

        Returns:
            DataFrame with success rates across iterations.
        """
        results = []
        for i, result in enumerate(self.raw_results["results"]):
            success_rate = result.get("overall_placement_ratio", 0.0)
            results.append({"iteration": i, "success_rate": success_rate})
        return pd.DataFrame(results)

    def summary_statistics(self) -> dict[str, float]:
        """Get summary statistics for success rates.

        Returns:
            Dictionary with success rate statistics.
        """
        df = self.success_rate_distribution()
        success_rates = df["success_rate"]
        return {
            "mean": float(success_rates.mean()),
            "std": float(success_rates.std()),
            "min": float(success_rates.min()),
            "max": float(success_rates.max()),
            "p5": float(success_rates.quantile(0.05)),
            "p25": float(success_rates.quantile(0.25)),
            "p50": float(success_rates.quantile(0.50)),
            "p75": float(success_rates.quantile(0.75)),
            "p95": float(success_rates.quantile(0.95)),
        }


@dataclass
class SensitivityResults:
    """Results from sensitivity Monte Carlo analysis.

    Attributes:
        raw_results: Raw results from FailureManager
        iterations: Number of Monte Carlo iterations
        baseline: Optional baseline result (no failures)
        component_scores: Aggregated component impact scores by flow
        failure_patterns: Dictionary mapping pattern keys to failure pattern results
        source_pattern: Source node regex pattern used in analysis
        sink_pattern: Sink node regex pattern used in analysis
        mode: Flow analysis mode ("combine" or "pairwise")
        metadata: Additional analysis metadata from FailureManager
    """

    raw_results: dict[str, Any]
    iterations: int
    baseline: Optional[dict[str, Any]] = None
    component_scores: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
    failure_patterns: Optional[Dict[str, Any]] = None
    source_pattern: Optional[str] = None
    sink_pattern: Optional[str] = None
    mode: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.component_scores is None:
            self.component_scores = {}
        if self.failure_patterns is None:
            self.failure_patterns = {}
        if self.metadata is None:
            self.metadata = {}

    def component_impact_distribution(self) -> pd.DataFrame:
        """Get component impact distribution as DataFrame.

        Returns:
            DataFrame with component criticality scores.
        """
        if not self.component_scores:
            return pd.DataFrame()

        # Flatten component scores across all flows
        data = []
        for flow_key, components in self.component_scores.items():
            for component_key, stats in components.items():
                row = {
                    "flow_key": flow_key,
                    "component": component_key,
                    "mean_impact": stats.get("mean", 0.0),
                    "max_impact": stats.get("max", 0.0),
                    "min_impact": stats.get("min", 0.0),
                    "sample_count": stats.get("count", 0),
                }
                data.append(row)

        return pd.DataFrame(data)

    def flow_keys(self) -> List[str]:
        """Get list of all flow keys in results.

        Returns:
            List of flow keys (e.g., ["datacenter->edge", "edge->datacenter"]).
        """
        return list(self.component_scores.keys()) if self.component_scores else []

    def get_flow_sensitivity(self, flow_key: str) -> Dict[str, Dict[str, float]]:
        """Get component sensitivity scores for a specific flow.

        Args:
            flow_key: Flow key (e.g., "datacenter->edge").

        Returns:
            Dictionary mapping component IDs to impact statistics

        Raises:
            KeyError: If flow_key not found in results.
        """
        if not self.component_scores or flow_key not in self.component_scores:
            available = (
                ", ".join(self.component_scores.keys())
                if self.component_scores
                else "none"
            )
            raise KeyError(f"Flow key '{flow_key}' not found. Available: {available}")
        return self.component_scores[flow_key]

    def summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for component impact across all flows.

        Returns:
            Dictionary mapping component IDs to aggregated impact statistics
        """
        from collections import defaultdict

        if not self.component_scores:
            return {}

        # Aggregate across flows for each component
        component_aggregates = defaultdict(list)
        for _flow_key, components in self.component_scores.items():
            for component_key, stats in components.items():
                component_aggregates[component_key].append(stats.get("mean", 0.0))

        # Calculate overall statistics
        summary = {}
        for component_key, impact_values in component_aggregates.items():
            if impact_values:
                summary[component_key] = {
                    "mean_impact": sum(impact_values) / len(impact_values),
                    "max_impact": max(impact_values),
                    "min_impact": min(impact_values),
                    "flow_count": len(impact_values),
                }

        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Convert sensitivity results to DataFrame for analysis.

        Returns:
            DataFrame with component impact statistics
        """
        return self.component_impact_distribution()

    def get_failure_pattern_summary(self) -> pd.DataFrame:
        """Get summary of failure patterns if available.

        Returns:
            DataFrame with failure pattern frequencies and sensitivity impact
        """
        if not self.failure_patterns:
            return pd.DataFrame()

        data = []
        for pattern_key, pattern in self.failure_patterns.items():
            row = {
                "pattern_key": pattern_key,
                "count": pattern.get("count", 0),
                "is_baseline": pattern.get("is_baseline", False),
                "failed_nodes": len(pattern.get("excluded_nodes", [])),
                "failed_links": len(pattern.get("excluded_links", [])),
                "total_failures": len(pattern.get("excluded_nodes", []))
                + len(pattern.get("excluded_links", [])),
            }

            # Add sensitivity results for each flow
            sensitivity_result = pattern.get("sensitivity_result", {})
            for flow_key, components in sensitivity_result.items():
                # Average sensitivity across components for this pattern
                if components:
                    avg_sensitivity = sum(components.values()) / len(components)
                    row[f"avg_sensitivity_{flow_key}"] = avg_sensitivity

            data.append(row)

        return pd.DataFrame(data)

    def export_summary(self) -> Dict[str, Any]:
        """Export summary for serialization.

        Returns:
            Dictionary with all results data in serializable format
        """
        return {
            "source_pattern": self.source_pattern,
            "sink_pattern": self.sink_pattern,
            "mode": self.mode,
            "iterations": self.iterations,
            "metadata": self.metadata or {},
            "component_scores": self.component_scores or {},
            "failure_patterns": self.failure_patterns or {},
            "summary_statistics": self.summary_statistics(),
        }
