"""Summary analysis for workflow results.

This module contains `SummaryAnalyzer`, which processes workflow step results
to generate high-level summaries, counts step types, and provides overview
statistics for network construction and analysis results.
"""

from typing import Any, Dict

from .base import NotebookAnalyzer


class SummaryAnalyzer(NotebookAnalyzer):
    """Generates summary statistics and overviews of workflow results.

    Counts and categorizes workflow steps by type (capacity, flow, other),
    displays network statistics for graph construction steps, and provides
    high-level summaries for analysis overview.
    """

    def analyze(self, results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Analyze and summarize all results.

        Args:
            results: Dictionary containing all workflow step results.
            **kwargs: Unused.

        Returns:
            Summary statistics including total steps and category counts.
        """
        total_steps = len(results)
        capacity_steps = len(
            [
                s
                for s, data in results.items()
                if isinstance(data, dict) and "capacity_envelopes" in data
            ]
        )
        flow_steps = len(
            [
                s
                for s, data in results.items()
                if isinstance(data, dict)
                and any(k.startswith("max_flow:") for k in data.keys())
            ]
        )
        other_steps = total_steps - capacity_steps - flow_steps

        return {
            "status": "success",
            "total_steps": total_steps,
            "capacity_steps": capacity_steps,
            "flow_steps": flow_steps,
            "other_steps": other_steps,
        }

    def get_description(self) -> str:
        return "Generates summary statistics and overviews of workflow results"

    def display_analysis(self, analysis: Dict[str, Any], **kwargs) -> None:
        """Display summary analysis.

        Args:
            analysis: Summary statistics returned by `analyze()`.
            **kwargs: Unused.
        """
        print("📊 NetGraph Analysis Summary")
        print("=" * 40)

        stats = analysis
        print(f"Total Analysis Steps: {stats['total_steps']:,}")
        print(f"Capacity Envelope Steps: {stats['capacity_steps']:,}")
        print(f"Flow Analysis Steps: {stats['flow_steps']:,}")
        print(f"Other Data Steps: {stats['other_steps']:,}")

        if stats["total_steps"] > 0:
            print(
                f"\n✅ Analysis complete. Processed {stats['total_steps']:,} workflow steps."
            )
        else:
            print("\n❌ No analysis results found.")

    def analyze_network_stats(self, results: Dict[str, Any], **kwargs) -> None:
        """Analyze and display network statistics for a specific step.

        Args:
            results: Dictionary containing all workflow step results.
            **kwargs: Additional arguments including step_name.

        Raises:
            ValueError: If step_name is missing or no data found for the step.
        """
        step_name = kwargs.get("step_name", "")
        if not step_name:
            raise ValueError("No step name provided for network stats analysis")

        step_data = results.get(step_name, {})
        if not step_data:
            raise ValueError(f"No data found for step: {step_name}")

        print(f"📊 Network Statistics: {step_name}")
        print("=" * 50)

        # Display node and link counts
        node_count = step_data.get("node_count")
        link_count = step_data.get("link_count")

        if node_count is not None:
            print(f"Nodes: {node_count:,}")
        if link_count is not None:
            print(f"Links: {link_count:,}")

        # Display capacity statistics
        capacity_stats = [
            "total_capacity",
            "mean_capacity",
            "median_capacity",
            "min_capacity",
            "max_capacity",
        ]
        capacity_data = {
            stat: step_data.get(stat)
            for stat in capacity_stats
            if step_data.get(stat) is not None
        }

        if capacity_data:
            print("\nCapacity Statistics:")
            for stat, value in capacity_data.items():
                label = stat.replace("_", " ").title()
                print(f"  {label}: {value:,.2f}")

        # Display cost statistics
        cost_stats = ["mean_cost", "median_cost", "min_cost", "max_cost"]
        cost_data = {
            stat: step_data.get(stat)
            for stat in cost_stats
            if step_data.get(stat) is not None
        }

        if cost_data:
            print("\nCost Statistics:")
            for stat, value in cost_data.items():
                label = stat.replace("_", " ").title()
                print(f"  {label}: {value:,.2f}")

        # Display degree statistics
        degree_stats = ["mean_degree", "median_degree", "min_degree", "max_degree"]
        degree_data = {
            stat: step_data.get(stat)
            for stat in degree_stats
            if step_data.get(stat) is not None
        }

        if degree_data:
            print("\nNode Degree Statistics:")
            for stat, value in degree_data.items():
                label = stat.replace("_", " ").title()
                print(f"  {label}: {value:.1f}")

    def analyze_build_graph(self, results: Dict[str, Any], **kwargs) -> None:
        """Analyze and display graph construction results.

        Args:
            results: Dictionary containing all workflow step results.
            **kwargs: Additional arguments including step_name.

        Raises:
            ValueError: If step_name is missing or no data found for the step.
        """
        step_name = kwargs.get("step_name", "")
        if not step_name:
            raise ValueError("No step name provided for graph analysis")

        step_data = results.get(step_name, {})
        if not step_data:
            raise ValueError(f"No data found for step: {step_name}")

        print(f"🔗 Graph Construction: {step_name}")
        print("=" * 50)

        graph = step_data.get("graph")
        if graph:
            print("✅ Graph successfully constructed")
            # Could add more details about the graph if needed
        else:
            print("❌ No graph data found")
