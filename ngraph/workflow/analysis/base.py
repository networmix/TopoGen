"""Base classes for notebook analysis components.

Defines a simple interface for notebook-oriented analyzers that both compute
results and render them. Concrete analyzers implement `analyze()`,
`display_analysis()`, and `get_description()`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


class NotebookAnalyzer(ABC):
    """Base class for notebook analysis components."""

    @abstractmethod
    def analyze(self, results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform the analysis and return results.

        Args:
            results: Input results dictionary to analyze.
            **kwargs: Analyzer-specific options.

        Returns:
            Dictionary containing analysis artifacts.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a concise description of the analyzer purpose."""
        pass

    def analyze_and_display(self, results: Dict[str, Any], **kwargs) -> None:
        """Analyze results and display them in notebook format.

        Args:
            results: Input results dictionary to analyze.
            **kwargs: Analyzer-specific options.
        """
        analysis = self.analyze(results, **kwargs)
        self.display_analysis(analysis, **kwargs)

    @abstractmethod
    def display_analysis(self, analysis: Dict[str, Any], **kwargs) -> None:
        """Display analysis results in notebook format.

        Args:
            analysis: Analysis artifacts returned by `analyze()`.
            **kwargs: Display options.
        """
        pass


@dataclass
class AnalysisContext:
    """Context information for analysis execution."""

    step_name: str
    results: Dict[str, Any]
    config: Dict[str, Any]
