"""Capacity envelope analysis utilities.

This module contains `CapacityMatrixAnalyzer`, responsible for processing capacity
envelope results, computing statistics, and generating notebook visualizations.
Works with both CapacityEnvelopeResults objects and workflow step data.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .base import NotebookAnalyzer

if TYPE_CHECKING:
    from ngraph.monte_carlo.results import CapacityEnvelopeResults

__all__ = ["CapacityMatrixAnalyzer"]


class CapacityMatrixAnalyzer(NotebookAnalyzer):
    """Processes capacity envelope data into matrices and flow availability analysis.

    Transforms capacity envelope results from CapacityEnvelopeAnalysis workflow steps
    or CapacityEnvelopeResults objects into matrices, statistical summaries, and
    flow availability distributions. Provides visualization methods for notebook output
    including capacity matrices, flow CDFs, and reliability curves.

    Can be used in two modes:
    1. Workflow mode: analyze() with workflow step results dictionary
    2. Direct mode: analyze_results() with CapacityEnvelopeResults object
    """

    def analyze_results(
        self, results: "CapacityEnvelopeResults", **kwargs
    ) -> Dict[str, Any]:
        """Analyze a `CapacityEnvelopeResults` object directly.

        Args:
            results: Results object from the failure manager.
            **kwargs: Additional arguments (unused).

        Returns:
            Analysis dictionary with capacity matrix, statistics, and viz data.

        Raises:
            ValueError: If no valid envelope data is present.
            RuntimeError: If analysis computation fails.
        """
        try:
            # Convert CapacityEnvelopeResults to workflow-compatible format
            envelopes = {key: env.to_dict() for key, env in results.envelopes.items()}

            matrix_data = self._extract_matrix_data(envelopes)
            if not matrix_data:
                raise ValueError("No valid capacity envelope data in results object")

            df_matrix = pd.DataFrame(matrix_data)
            capacity_matrix = self._create_capacity_matrix(df_matrix)
            statistics = self._calculate_statistics(capacity_matrix)

            return {
                "status": "success",
                "step_name": f"{results.source_pattern}->{results.sink_pattern}",
                "matrix_data": matrix_data,
                "capacity_matrix": capacity_matrix,
                "statistics": statistics,
                "visualization_data": self._prepare_visualization_data(capacity_matrix),
                "envelope_results": results,  # Keep reference to original object
            }

        except Exception as exc:
            raise RuntimeError(
                f"Error analyzing capacity envelope results: {exc}"
            ) from exc

    def display_capacity_distributions(
        self,
        results: "CapacityEnvelopeResults",
        flow_key: Optional[str] = None,
        bins: int = 30,
    ) -> None:
        """Display capacity distribution plots for `CapacityEnvelopeResults`.

        Args:
            results: Results object to visualize.
            flow_key: Specific flow to plot (default: all flows).
            bins: Number of histogram bins.
        """
        import seaborn as sns

        print("📊 Capacity Distribution Analysis")
        print(f"Source pattern: {results.source_pattern}")
        print(f"Sink pattern: {results.sink_pattern}")
        print(f"Iterations: {results.iterations:,}")
        print(f"Flow pairs: {len(results.envelopes):,}\n")

        try:
            if flow_key:
                # Plot single flow
                envelope = results.get_envelope(flow_key)
                values = envelope.expand_to_values()

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(
                    values,
                    bins=bins,
                    alpha=0.7,
                    edgecolor="black",
                    color=sns.color_palette()[0],
                )
                ax.set_title(f"Capacity Distribution: {flow_key}")
                ax.set_xlabel("Capacity")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

                # Add statistics
                mean_val = envelope.mean_capacity
                ax.axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    label=f"Mean: {mean_val:.2f}",
                )
                ax.legend()

            else:
                # Plot all flows
                n_flows = len(results.envelopes)
                colors = sns.color_palette("husl", n_flows)

                fig, ax = plt.subplots(figsize=(12, 8))
                for i, (fkey, envelope) in enumerate(results.envelopes.items()):
                    values = envelope.expand_to_values()
                    ax.hist(values, bins=bins, alpha=0.6, label=fkey, color=colors[i])

                ax.set_title("Capacity Distributions (All Flows)")
                ax.set_xlabel("Capacity")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()
            plt.show()

        except Exception as exc:
            print(f"⚠️  Visualization error: {exc}")

    def display_percentile_comparison(self, results: "CapacityEnvelopeResults") -> None:
        """Display percentile comparison plots for `CapacityEnvelopeResults`.

        Args:
            results: Results object to visualize.
        """
        import seaborn as sns

        print("📈 Capacity Percentile Comparison")

        try:
            percentiles = [5, 25, 50, 75, 95]
            flow_keys = results.flow_keys()

            data = []
            for fkey in flow_keys:
                envelope = results.envelopes[fkey]
                row = [envelope.get_percentile(p) for p in percentiles]
                data.append(row)

            df = pd.DataFrame(
                data, index=flow_keys, columns=[f"p{p}" for p in percentiles]
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            df.plot(
                kind="bar", ax=ax, color=sns.color_palette("viridis", len(percentiles))
            )
            ax.set_title("Capacity Percentiles by Flow")
            ax.set_xlabel("Flow")
            ax.set_ylabel("Capacity")
            ax.legend(title="Percentile")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        except Exception as exc:
            print(f"⚠️  Visualization error: {exc}")

    def analyze_and_display_envelope_results(
        self, results: "CapacityEnvelopeResults", **kwargs
    ) -> None:
        """Complete analysis and display for CapacityEnvelopeResults object.

        Args:
            results: Results object to analyze and display.
            **kwargs: Additional arguments.
        """
        # Perform analysis
        analysis = self.analyze_results(results, **kwargs)

        # Display capacity matrix
        self.display_analysis(analysis, **kwargs)

        # Display distribution plots
        self.display_capacity_distributions(results)

        # Display percentile comparison
        self.display_percentile_comparison(results)

        # Display flow availability if we have frequency data
        try:
            # Convert to workflow format for flow availability analysis
            step_data = {
                "capacity_envelopes": {
                    key: env.to_dict() for key, env in results.envelopes.items()
                }
            }
            workflow_results = {"envelope_analysis": step_data}

            self.analyze_flow_availability(
                workflow_results, step_name="envelope_analysis"
            )
            self.analyze_and_display_flow_availability(
                workflow_results, step_name="envelope_analysis"
            )
        except Exception as exc:
            print(f"ℹ️  Flow availability analysis skipped: {exc}")

    def analyze(self, results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Analyze capacity envelopes and create matrix visualization.

        Args:
            results: Dictionary containing all workflow step results.
            **kwargs: Additional arguments including `step_name`.

        Returns:
            Dictionary containing analysis results with capacity matrix and statistics.

        Raises:
            ValueError: If `step_name` is missing or no valid envelope data found.
            RuntimeError: If analysis computation fails.
        """
        step_name: Optional[str] = kwargs.get("step_name")
        if not step_name:
            raise ValueError("step_name required for capacity matrix analysis")

        step_data = results.get(step_name, {})
        envelopes = step_data.get("capacity_envelopes", {})

        if not envelopes:
            raise ValueError(f"No capacity envelope data found for step: {step_name}")

        try:
            matrix_data = self._extract_matrix_data(envelopes)
            if not matrix_data:
                raise ValueError(
                    f"No valid capacity envelope data in step: {step_name}"
                )

            df_matrix = pd.DataFrame(matrix_data)
            capacity_matrix = self._create_capacity_matrix(df_matrix)
            statistics = self._calculate_statistics(capacity_matrix)

            return {
                "status": "success",
                "step_name": step_name,
                "matrix_data": matrix_data,
                "capacity_matrix": capacity_matrix,
                "statistics": statistics,
                "visualization_data": self._prepare_visualization_data(capacity_matrix),
            }

        except Exception as exc:
            raise RuntimeError(
                f"Error analyzing capacity matrix for {step_name}: {exc}"
            ) from exc

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _extract_matrix_data(self, envelopes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract flattened matrix data from envelope dictionary."""
        matrix_data: List[Dict[str, Any]] = []

        for flow_path, envelope_data in envelopes.items():
            parsed_flow = self._parse_flow_path(flow_path)
            capacity = self._extract_capacity_value(envelope_data)

            if parsed_flow and capacity is not None:
                matrix_data.append(
                    {
                        "source": parsed_flow["source"],
                        "destination": parsed_flow["destination"],
                        "capacity": capacity,
                        "flow_path": flow_path,
                        "direction": parsed_flow["direction"],
                    }
                )

        return matrix_data

    @staticmethod
    def _parse_flow_path(flow_path: str) -> Optional[Dict[str, str]]:
        """Parse flow path string into components.

        Supported formats are "src->dst" and "src<->dst".
        """
        if "<->" in flow_path:
            source, destination = flow_path.split("<->", 1)
            return {
                "source": source.strip(),
                "destination": destination.strip(),
                "direction": "bidirectional",
            }
        if "->" in flow_path:
            source, destination = flow_path.split("->", 1)
            return {
                "source": source.strip(),
                "destination": destination.strip(),
                "direction": "directed",
            }
        return None

    @staticmethod
    def _extract_capacity_value(envelope_data: Any) -> Optional[float]:
        """Return numeric capacity from envelope data.

        Accepts plain numbers or the canonical dict format with a "max" key.
        """
        if isinstance(envelope_data, (int, float)):
            return float(envelope_data)

        if isinstance(envelope_data, dict):
            # Extract capacity from canonical format
            if "max" in envelope_data:
                cap_val = envelope_data["max"]
                if isinstance(cap_val, (int, float)):
                    return float(cap_val)
        return None

    @staticmethod
    def _create_capacity_matrix(df_matrix: pd.DataFrame) -> pd.DataFrame:
        """Create a pivot table suitable for matrix display."""
        return df_matrix.pivot_table(
            index="source",
            columns="destination",
            values="capacity",
            aggfunc="max",
            fill_value=0,
        )

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_statistics(capacity_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic statistics for the capacity matrix."""
        non_zero_values = capacity_matrix.values[capacity_matrix.values > 0]
        if len(non_zero_values) == 0:
            return {"has_data": False}

        non_self_loop_flows = 0
        for source in capacity_matrix.index:
            for dest in capacity_matrix.columns:
                if source == dest:
                    continue  # skip self-loops
                capacity_val = capacity_matrix.loc[source, dest]
                try:
                    numeric_val = pd.to_numeric(capacity_val, errors="coerce")
                    if pd.notna(numeric_val):
                        non_self_loop_flows += 1
                except (ValueError, TypeError):
                    continue

        num_nodes = len(capacity_matrix.index)
        total_possible_flows = num_nodes * (num_nodes - 1)
        flow_density = (
            non_self_loop_flows / total_possible_flows * 100
            if total_possible_flows
            else 0
        )

        return {
            "has_data": True,
            "total_flows": non_self_loop_flows,
            "total_possible": total_possible_flows,
            "flow_density": flow_density,
            "capacity_min": float(non_zero_values.min()),
            "capacity_max": float(non_zero_values.max()),
            "capacity_mean": float(non_zero_values.mean()),
            "capacity_p25": float(pd.Series(non_zero_values).quantile(0.25)),
            "capacity_p50": float(pd.Series(non_zero_values).quantile(0.50)),
            "capacity_p75": float(pd.Series(non_zero_values).quantile(0.75)),
            "num_sources": num_nodes,
            "num_destinations": len(capacity_matrix.columns),
        }

    @staticmethod
    def _format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[name-match]
        """Return a copy of the DataFrame with thousands separators applied."""
        if df.empty:
            return df

        df_formatted = df.copy()
        for col in df_formatted.select_dtypes(include=["number"]):
            df_formatted[col] = df_formatted[col].map(
                lambda x: f"{x:,.0f}"
                if pd.notna(x) and x == int(x)
                else f"{x:,.1f}"
                if pd.notna(x)
                else x
            )
        return df_formatted

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _prepare_visualization_data(
        self, capacity_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Prepare auxiliary data structures for visualisation/widgets."""
        capacity_ranking: List[Dict[str, Any]] = []
        for source in capacity_matrix.index:
            for dest in capacity_matrix.columns:
                if source == dest:
                    continue
                capacity_val = capacity_matrix.loc[source, dest]
                try:
                    numeric_val = pd.to_numeric(capacity_val, errors="coerce")
                    if pd.notna(numeric_val):
                        capacity_ranking.append(
                            {
                                "Source": source,
                                "Destination": dest,
                                "Capacity": float(numeric_val),
                                "Flow Path": f"{source} -> {dest}",
                            }
                        )
                except (ValueError, TypeError):
                    continue

        capacity_ranking.sort(key=lambda x: x["Capacity"], reverse=True)
        capacity_ranking_df = pd.DataFrame(capacity_ranking)

        # Create matrix display with source as index and destinations as columns
        matrix_display = capacity_matrix.copy()
        matrix_display.index.name = "Source"
        matrix_display.columns.name = "Destination"

        return {
            "matrix_display": matrix_display,
            "capacity_ranking": capacity_ranking_df,
            "has_data": capacity_matrix.sum().sum() > 0,
            "has_ranking_data": bool(capacity_ranking),
        }

    # ------------------------------------------------------------------
    # Public display helpers
    # ------------------------------------------------------------------

    def get_description(self) -> str:  # noqa: D401 – simple return
        return "Processes capacity envelope data into matrices and flow availability analysis"

    # ----------------------------- display ------------------------------

    def display_analysis(self, analysis: Dict[str, Any], **kwargs) -> None:  # noqa: C901 – large but fine
        """Pretty-print analysis results to the notebook/stdout.

        Args:
            analysis: Analysis results dictionary from the analyze method.
            **kwargs: Additional arguments (unused).
        """
        step_name = analysis.get("step_name", "Unknown")
        print(f"✅ Analyzing capacity matrix for {step_name}")

        stats = analysis["statistics"]
        if not stats["has_data"]:
            print("No capacity data available")
            return

        print("Matrix Statistics:")
        print(f"  Sources: {stats['num_sources']:,} nodes")
        print(f"  Destinations: {stats['num_destinations']:,} nodes")
        print(
            f"  Flows: {stats['total_flows']:,}/{stats['total_possible']:,} ({stats['flow_density']:.1f}%)"
        )
        print(
            f"  Capacity range: {stats['capacity_min']:,.2f} - {stats['capacity_max']:,.2f}"
        )
        print("  Capacity statistics:")
        print(f"    Mean: {stats['capacity_mean']:,.2f}")
        print(f"    P25: {stats['capacity_p25']:,.2f}")
        print(f"    P50 (median): {stats['capacity_p50']:,.2f}")
        print(f"    P75: {stats['capacity_p75']:,.2f}")

        viz_data = analysis["visualization_data"]
        if viz_data["has_data"]:
            matrix_display = viz_data["matrix_display"]
            matrix_display_formatted = self._format_dataframe_for_display(
                matrix_display
            )
            print("\n🔢 Full Capacity Matrix:")
            _get_show()(  # pylint: disable=not-callable
                matrix_display_formatted,
                caption=f"Capacity Matrix - {step_name}",
                scrollY="400px",
                scrollX=True,
                scrollCollapse=True,
                paging=False,
            )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def analyze_and_display_all_steps(self, results: Dict[str, Any]) -> None:  # noqa: D401
        """Run analyze/display on every step containing capacity_envelopes."""
        found_data = False
        for step_name, step_data in results.items():
            if isinstance(step_data, dict) and "capacity_envelopes" in step_data:
                found_data = True
                self.display_analysis(self.analyze(results, step_name=step_name))
                print()  # spacing between steps
        if not found_data:
            print("No capacity envelope data found in results")

    def analyze_and_display_step(self, results: Dict[str, Any], **kwargs) -> None:
        """Analyze and display results for a specific step.

        Args:
            results: Dictionary containing all workflow step results.
            **kwargs: Additional arguments including step_name.
        """
        step_name = kwargs.get("step_name")
        if not step_name:
            print("❌ No step name provided for capacity matrix analysis")
            return

        try:
            analysis = self.analyze(results, step_name=step_name)
            self.display_analysis(analysis)
        except Exception as e:
            print(f"❌ Capacity matrix analysis failed: {e}")
            raise

    def analyze_and_display_flow_availability(
        self, results: Dict[str, Any], **kwargs
    ) -> None:
        """Analyze and display flow availability for a specific step.

        Args:
            results: Dictionary containing all workflow step results.
            **kwargs: Additional arguments including step_name.

        Raises:
            ValueError: If step_name is missing or no capacity envelope data found.
        """
        step_name = kwargs.get("step_name")
        if not step_name:
            raise ValueError("No step name provided for flow availability analysis")

        # Check if the step has capacity_envelopes data for flow availability analysis
        step_data = results.get(step_name, {})
        if "capacity_envelopes" not in step_data:
            raise ValueError(
                f"❌ No capacity envelope data found for step: {step_name}. "
                "Flow availability analysis requires capacity envelope data from CapacityEnvelopeAnalysis."
            )

        envelopes = step_data["capacity_envelopes"]
        if not envelopes:
            raise ValueError(f"❌ Empty capacity envelopes found for step: {step_name}")

        # Call the flow availability analysis method
        try:
            result = self.analyze_flow_availability(results, step_name=step_name)
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise

        stats = result["statistics"]
        viz_data = result["visualization_data"]
        maximum_flow = result["maximum_flow"]
        total_samples = result["total_samples"]
        aggregated_flows = result["aggregated_flows"]
        skipped_self_loops = result["skipped_self_loops"]
        total_envelopes = result["total_envelopes"]

        # Summary statistics with filtering info
        print(
            f"🔢 Sample Statistics (n={total_samples} from {aggregated_flows} flows, "
            f"skipped {skipped_self_loops} self-loops, {total_envelopes} total):"
        )
        print(f"   Maximum Flow: {maximum_flow:.2f}")
        print(
            f"   Mean Flow:    {stats['mean_flow']:.2f} ({stats['relative_mean']:.1f}%)"
        )
        print(
            f"   Median Flow:  {stats['median_flow']:.2f} ({stats['flow_percentiles']['p50']['relative']:.1f}%)"
        )
        print(
            f"   Std Dev:      {stats['flow_std']:.2f} ({stats['relative_std']:.1f}%) "
            f"→ Flow dispersion magnitude relative to mean"
        )
        print(
            f"   CV:           {stats['coefficient_of_variation']:.1f}% "
            f"→ Normalized variability metric: <30% stable, >50% high variance\n"
        )

        print("📈 Flow Distribution Percentiles:")
        for p_name in ["p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
            if p_name in stats["flow_percentiles"]:
                p_data = stats["flow_percentiles"][p_name]
                percentile_num = p_name[1:]
                print(
                    f"   {percentile_num:>2}th percentile: {p_data['absolute']:8.2f} ({p_data['relative']:5.1f}%)"
                )
        print()

        print("🎯 Network Reliability Analysis:")
        for reliability in ["99.99%", "99.9%", "99%", "95%", "90%", "80%"]:
            flow_fraction = viz_data["reliability_thresholds"].get(reliability, 0)
            flow_pct = flow_fraction * 100
            print(f"   {reliability} reliability: ≥{flow_pct:5.1f}% of maximum flow")
        print()

        print("📐 Distribution Characteristics:")
        dist_metrics = viz_data["distribution_metrics"]
        gini = dist_metrics["gini_coefficient"]
        quartile = dist_metrics["quartile_coefficient"]
        range_ratio = dist_metrics["flow_range_ratio"]

        print(
            f"   Gini Coefficient:     {gini:.3f} "
            f"→ Flow inequality: 0=uniform, 1=maximum inequality"
        )
        print(
            f"   Quartile Coefficient: {quartile:.3f} "
            f"→ Interquartile spread: (Q3-Q1)/(Q3+Q1), measures distribution skew"
        )
        print(
            f"   Range Ratio:          {range_ratio:.3f} "
            f"→ Total variation span: (max-min)/max, failure impact magnitude\n"
        )

        # Render plots for flow availability analysis
        try:
            cdf_data = viz_data["cdf_data"]
            percentile_data = viz_data["percentile_data"]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            ax1.plot(
                cdf_data["flow_values"],
                cdf_data["cumulative_probabilities"],
                "b-",
                linewidth=2,
                label="Empirical CDF",
            )
            ax1.set_xlabel("Relative flow f")
            ax1.set_ylabel("Cumulative probability P(Flow ≤ f)")
            ax1.set_title("Empirical CDF of Delivered Flow")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(
                percentile_data["percentiles"],
                percentile_data["flow_at_percentiles"],
                "r-",
                linewidth=2,
                label="Flow Reliability Curve",
            )
            # Flow Reliability Curve (F(p)): shows the flow that can be
            # delivered with probability ≥ p.
            ax2.set_xlabel("Reliability level p")
            ax2.set_ylabel("Guaranteed flow F(p)")
            ax2.set_title("Flow Reliability Curve (F(p))")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.show()
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  Visualisation error: {exc}")

    # ------------------------------------------------------------------
    # Flow-availability analysis
    # ------------------------------------------------------------------

    def analyze_flow_availability(
        self, results: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Create CDF/availability distribution from capacity envelope frequencies.

        Args:
            results: Dictionary containing all workflow step results.
            **kwargs: Additional arguments including step_name.

        Returns:
            Dictionary containing flow availability analysis results.

        Raises:
            ValueError: If step_name is missing or no valid envelope data found.
            RuntimeError: If analysis computation fails.
        """
        step_name: Optional[str] = kwargs.get("step_name")
        if not step_name:
            raise ValueError("step_name required for flow availability analysis")

        step_data = results.get(step_name, {})
        envelopes = step_data.get("capacity_envelopes", {})

        if not envelopes:
            raise ValueError(f"No capacity envelopes found for step: {step_name}")

        # Aggregate frequencies from all capacity envelopes, excluding self-loops
        total_capacity_frequencies: Dict[float, int] = {}
        skipped_self_loops = 0
        processed_flows = 0

        for flow_key, envelope_data in envelopes.items():
            if not isinstance(envelope_data, dict):
                raise ValueError(f"Invalid envelope data format for flow {flow_key}")

            # Check if this is a self-loop (source == destination)
            flow_parts = flow_key.split("->")
            if len(flow_parts) == 2 and flow_parts[0] == flow_parts[1]:
                skipped_self_loops += 1
                continue  # Skip self-loops (source == destination)

            frequencies = envelope_data.get("frequencies", {})
            if not frequencies:
                continue  # Skip empty envelopes

            processed_flows += 1
            # Aggregate frequencies into total distribution
            for capacity_str, count in frequencies.items():
                try:
                    capacity_value = float(capacity_str)
                    count_value = int(count)
                    total_capacity_frequencies[capacity_value] = (
                        total_capacity_frequencies.get(capacity_value, 0) + count_value
                    )
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid capacity frequency data in {flow_key}: {capacity_str}={count}, error: {e}"
                    ) from e

        if not total_capacity_frequencies:
            if skipped_self_loops > 0 and processed_flows == 0:
                raise ValueError(
                    f"All {skipped_self_loops} flows in step {step_name} are self-loops. "
                    "Flow availability analysis requires non-self-loop flows with capacity data."
                )
            else:
                raise ValueError(
                    f"No valid frequency data found in capacity envelopes for step: {step_name}. "
                    f"Processed {processed_flows} flows, skipped {skipped_self_loops} self-loops."
                )

        # Convert aggregated frequencies to samples for analysis
        total_flow_samples = []
        for capacity, count in total_capacity_frequencies.items():
            total_flow_samples.extend([capacity] * count)

        if not total_flow_samples:
            raise ValueError(
                f"No flow samples generated from frequency data for step: {step_name}"
            )

        try:
            sorted_samples = sorted(total_flow_samples)
            n_samples = len(sorted_samples)
            maximum_flow = max(sorted_samples)

            if maximum_flow == 0:
                raise ValueError(
                    "All aggregated flow samples are zero - cannot compute availability metrics"
                )

            flow_cdf: List[tuple[float, float]] = []
            for i, flow in enumerate(sorted_samples):
                cumulative_prob = (i + 1) / n_samples
                relative_flow = flow / maximum_flow
                flow_cdf.append((relative_flow, cumulative_prob))

            availability_curve = [
                (rel_flow, 1 - cum_prob) for rel_flow, cum_prob in flow_cdf
            ]
            statistics = self._calculate_flow_statistics(
                total_flow_samples, maximum_flow
            )
            viz_data = self._prepare_flow_cdf_visualization_data(
                flow_cdf, availability_curve, maximum_flow
            )

            return {
                "status": "success",
                "step_name": step_name,
                "flow_cdf": flow_cdf,
                "availability_curve": availability_curve,
                "statistics": statistics,
                "maximum_flow": maximum_flow,
                "total_samples": n_samples,
                "aggregated_flows": processed_flows,
                "skipped_self_loops": skipped_self_loops,
                "total_envelopes": len(envelopes),
                "visualization_data": viz_data,
            }
        except Exception as exc:
            raise RuntimeError(
                f"Error analyzing flow availability for {step_name}: {exc}"
            ) from exc

    # Helper methods for flow-availability analysis

    @staticmethod
    def _calculate_flow_statistics(
        samples: List[float], maximum_flow: float
    ) -> Dict[str, Any]:
        """Calculate statistical metrics for flow samples.

        Args:
            samples: Flow sample values.
            maximum_flow: Maximum flow value.

        Returns:
            Dictionary containing statistical metrics.
        """
        if not samples or maximum_flow == 0:
            return {"has_data": False}

        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        flow_percentiles: Dict[str, Dict[str, float]] = {}
        sorted_samples = sorted(samples)
        n_samples = len(samples)
        for p in percentiles:
            idx = min(max(int((p / 100) * n_samples), 0), n_samples - 1)
            flow_at_percentile = sorted_samples[idx]
            flow_percentiles[f"p{p}"] = {
                "absolute": flow_at_percentile,
                "relative": (flow_at_percentile / maximum_flow) * 100,
            }

        mean_flow = sum(samples) / len(samples)
        std_flow = pd.Series(samples).std()

        return {
            "has_data": True,
            "maximum_flow": maximum_flow,
            "minimum_flow": min(samples),
            "mean_flow": mean_flow,
            "median_flow": flow_percentiles["p50"]["absolute"],
            "flow_range": maximum_flow - min(samples),
            "flow_std": std_flow,
            "relative_mean": (mean_flow / maximum_flow) * 100,
            "relative_min": (min(samples) / maximum_flow) * 100,
            "relative_std": (std_flow / maximum_flow) * 100,
            "flow_percentiles": flow_percentiles,
            "total_samples": len(samples),
            "coefficient_of_variation": (std_flow / mean_flow) * 100
            if mean_flow
            else 0,
        }

    @staticmethod
    def _prepare_flow_cdf_visualization_data(
        flow_cdf: List[tuple[float, float]],
        availability_curve: List[tuple[float, float]],
        maximum_flow: float,
    ) -> Dict[str, Any]:
        """Prepare flow CDF data for visualization.

        Args:
            flow_cdf: Pairs of relative flow and cumulative probability.
            availability_curve: Pairs of relative flow and availability probability.
            maximum_flow: Maximum flow value.

        Returns:
            Dictionary containing visualization data.
        """
        if not flow_cdf or not availability_curve:
            return {"has_data": False}

        flow_values = [v for v, _ in flow_cdf]
        cumulative_probs = [p for _, p in flow_cdf]

        percentiles: List[float] = []
        flow_at_percentiles: List[float] = []
        for rel_flow, avail_prob in availability_curve:
            percentiles.append(avail_prob)
            flow_at_percentiles.append(rel_flow)

        reliability_thresholds = [99.99, 99.9, 99, 95, 90, 80, 70, 50]
        threshold_flows: Dict[str, float] = {}
        for threshold in reliability_thresholds:
            target_avail = threshold / 100
            flow_at_threshold = next(
                (
                    rel_flow
                    for rel_flow, avail_prob in availability_curve
                    if avail_prob >= target_avail
                ),
                0,
            )
            threshold_flows[f"{threshold}%"] = flow_at_threshold

        sorted_flows = sorted(flow_values)
        n = len(sorted_flows)
        cumsum = sum((i + 1) * flow for i, flow in enumerate(sorted_flows))
        total_sum = sum(sorted_flows)
        gini = (2 * cumsum) / (n * total_sum) - (n + 1) / n if total_sum else 0

        return {
            "has_data": True,
            "cdf_data": {
                "flow_values": flow_values,
                "cumulative_probabilities": cumulative_probs,
            },
            "percentile_data": {
                "percentiles": percentiles,
                "flow_at_percentiles": flow_at_percentiles,
            },
            "reliability_thresholds": threshold_flows,
            "distribution_metrics": {
                "gini_coefficient": gini,
                "flow_range_ratio": max(flow_values) - min(flow_values),
                "quartile_coefficient": CapacityMatrixAnalyzer._calculate_quartile_coefficient(
                    sorted_flows
                ),
            },
        }

    @staticmethod
    def _calculate_quartile_coefficient(sorted_values: List[float]) -> float:
        """Calculate quartile coefficient for flow distribution.

        Args:
            sorted_values: List of sorted flow values.

        Returns:
            Quartile coefficient value.
        """
        if len(sorted_values) < 4:
            return 0.0
        n = len(sorted_values)
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        return (q3 - q1) / (q3 + q1) if (q3 + q1) else 0.0


# Helper to get the show function from the analysis module


def _get_show():  # noqa: D401
    wrapper = importlib.import_module("ngraph.workflow.analysis")
    return wrapper.show
