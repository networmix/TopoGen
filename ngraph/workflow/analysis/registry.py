"""Analysis registry for mapping workflow steps to analysis modules.

This module provides the central registry that defines which analysis modules
should be executed for each workflow step type, eliminating fragile data-based
parsing and creating a clear, maintainable mapping system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .base import NotebookAnalyzer

__all__ = ["AnalysisConfig", "AnalysisRegistry", "get_default_registry"]


@dataclass
class AnalysisConfig:
    """Configuration for a single analysis module execution.

    Attributes:
        analyzer_class: The analyzer class to instantiate.
        method_name: The method to call on the analyzer (default: 'analyze_and_display').
        kwargs: Additional keyword arguments to pass to the method.
        section_title: Title for the notebook section (auto-generated if None).
        enabled: Whether this analysis is enabled (default: True).
    """

    analyzer_class: Type[NotebookAnalyzer]
    method_name: str = "analyze_and_display"
    kwargs: Dict[str, Any] = field(default_factory=dict)
    section_title: Optional[str] = None
    enabled: bool = True


@dataclass
class AnalysisRegistry:
    """Registry mapping workflow step types to their analysis configurations.

    The registry defines which analysis modules should run for each workflow step,
    providing a clear and maintainable mapping that replaces fragile data parsing.
    """

    _mappings: Dict[str, List[AnalysisConfig]] = field(default_factory=dict)

    def register(
        self,
        step_type: str,
        analyzer_class: Type[NotebookAnalyzer],
        method_name: str = "analyze_and_display",
        section_title: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Register an analysis module for a workflow step type.

        Args:
            step_type: The workflow step type (e.g., 'CapacityEnvelopeAnalysis').
            analyzer_class: The analyzer class to use.
            method_name: Method to call on the analyzer.
            section_title: Title for the notebook section.
            **kwargs: Additional arguments to pass to the analysis method.
        """
        if step_type not in self._mappings:
            self._mappings[step_type] = []

        config = AnalysisConfig(
            analyzer_class=analyzer_class,
            method_name=method_name,
            kwargs=kwargs,
            section_title=section_title or f"{analyzer_class.__name__} Analysis",
        )

        self._mappings[step_type].append(config)

    def get_analyses(self, step_type: str) -> List[AnalysisConfig]:
        """Get all analysis configurations for a workflow step type.

        Args:
            step_type: The workflow step type.

        Returns:
            List of analysis configurations for this step type.
        """
        return [
            config for config in self._mappings.get(step_type, []) if config.enabled
        ]

    def has_analyses(self, step_type: str) -> bool:
        """Return True if any analyses are registered for a workflow step type.

        Args:
            step_type: The workflow step type.

        Returns:
            True if analyses are registered and enabled for this step type.
        """
        return len(self.get_analyses(step_type)) > 0

    def get_all_step_types(self) -> List[str]:
        """Return all registered workflow step types.

        Returns:
            List of all workflow step types with registered analyses.
        """
        return list(self._mappings.keys())


def get_default_registry() -> AnalysisRegistry:
    """Create and return the default analysis registry with standard mappings.

    Returns:
        Configured registry with standard workflow step -> analysis mappings.
    """
    from .capacity_matrix import CapacityMatrixAnalyzer
    from .summary import SummaryAnalyzer

    registry = AnalysisRegistry()

    # Network statistics analysis
    registry.register(
        "NetworkStats",
        SummaryAnalyzer,
        method_name="analyze_network_stats",
        section_title="Network Statistics",
    )

    # Capacity envelope analysis - capacity matrix
    registry.register(
        "CapacityEnvelopeAnalysis",
        CapacityMatrixAnalyzer,
        method_name="analyze_and_display_step",
        section_title="Capacity Matrix Analysis",
    )

    # Capacity envelope analysis - flow availability curves
    registry.register(
        "CapacityEnvelopeAnalysis",
        CapacityMatrixAnalyzer,
        method_name="analyze_and_display_flow_availability",
        section_title="Flow Availability Analysis",
    )

    # Build graph analysis
    registry.register(
        "BuildGraph",
        SummaryAnalyzer,
        method_name="analyze_build_graph",
        section_title="Graph Construction",
    )

    return registry
