"""Problem-level max-flow API bound to the model layer.

Functions here operate on a model context that provides:

- to_strict_multidigraph(add_reverse: bool = True) -> StrictMultiDiGraph
- select_node_groups_by_path(path: str) -> dict[str, list[Node]]

They accept either a `Network` or a `NetworkView`. The input context is not
mutated. Pseudo source and sink nodes are attached on a working graph when
computing flows between groups.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ngraph.algorithms.base import FlowPlacement
from ngraph.algorithms.max_flow import (
    calc_max_flow,
    run_sensitivity,
)
from ngraph.algorithms.max_flow import (
    saturated_edges as _algo_saturated_edges,
)
from ngraph.algorithms.types import FlowSummary

try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:  # pragma: no cover - typing only
        from ngraph.graph.strict_multidigraph import StrictMultiDiGraph  # noqa: F401
        from ngraph.model.network import Network, Node  # noqa: F401
except Exception:  # pragma: no cover - safety in unusual environments
    pass


def max_flow(
    context: Any,
    source_path: str,
    sink_path: str,
    *,
    mode: str = "combine",
    shortest_path: bool = False,
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
) -> Dict[Tuple[str, str], float]:
    """Compute max flow between groups selected from the context.

    Creates a working graph from the context, adds a pseudo source attached to
    the selected source nodes and a pseudo sink attached to the selected sink
    nodes, then runs the max-flow routine.

    Args:
        context: `Network` or `NetworkView` providing selection and graph APIs.
        source_path: Selection expression for source groups.
        sink_path: Selection expression for sink groups.
        mode: Aggregation strategy. "combine" considers all sources as one
            group and all sinks as one group. "pairwise" evaluates each
            source-label and sink-label pair separately.
        shortest_path: If True, perform a single augmentation along the first
            shortest path instead of the full max-flow.
        flow_placement: Strategy for splitting flow among equal-cost parallel
            edges.

    Returns:
        Dict[Tuple[str, str], float]: Total flow per (source_label, sink_label).

    Raises:
        ValueError: If no matching sources or sinks are found, or if ``mode``
            is not one of {"combine", "pairwise"}.
    """
    src_groups = context.select_node_groups_by_path(source_path)
    snk_groups = context.select_node_groups_by_path(sink_path)

    if not src_groups:
        raise ValueError(f"No source nodes found matching '{source_path}'.")
    if not snk_groups:
        raise ValueError(f"No sink nodes found matching '{sink_path}'.")

    base_graph = context.to_strict_multidigraph().copy()

    if mode == "combine":
        combined_src_nodes: list = []
        combined_snk_nodes: list = []
        combined_src_label = "|".join(sorted(src_groups.keys()))
        combined_snk_label = "|".join(sorted(snk_groups.keys()))

        for group_nodes in src_groups.values():
            combined_src_nodes.extend(group_nodes)
        for group_nodes in snk_groups.values():
            combined_snk_nodes.extend(group_nodes)

        if not combined_src_nodes or not combined_snk_nodes:
            return {(combined_src_label, combined_snk_label): 0.0}

        # Overlap -> zero flow
        if {n.name for n in combined_src_nodes} & {n.name for n in combined_snk_nodes}:
            flow_val = 0.0
        else:
            flow_val = _compute_flow_single_group(
                context,
                combined_src_nodes,
                combined_snk_nodes,
                shortest_path,
                flow_placement,
                prebuilt_graph=base_graph,
            )
        return {(combined_src_label, combined_snk_label): flow_val}

    if mode == "pairwise":
        results: Dict[Tuple[str, str], float] = {}
        for src_label, src_nodes in src_groups.items():
            for snk_label, snk_nodes in snk_groups.items():
                if src_nodes and snk_nodes:
                    if {n.name for n in src_nodes} & {n.name for n in snk_nodes}:
                        flow_val = 0.0
                    else:
                        flow_val = _compute_flow_single_group(
                            context,
                            src_nodes,
                            snk_nodes,
                            shortest_path,
                            flow_placement,
                            prebuilt_graph=base_graph,
                        )
                else:
                    flow_val = 0.0
                results[(src_label, snk_label)] = flow_val
        return results

    raise ValueError(f"Invalid mode '{mode}'. Must be 'combine' or 'pairwise'.")


def max_flow_with_summary(
    context: Any,
    source_path: str,
    sink_path: str,
    *,
    mode: str = "combine",
    shortest_path: bool = False,
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
) -> Dict[Tuple[str, str], Tuple[float, FlowSummary]]:
    """Compute max flow and return a summary for each group pair.

    The summary includes total flow, per-edge flow, residual capacity,
    reachable set from the source in the residual graph, min-cut edges, and a
    cost distribution over augmentation steps.

    Args:
        context: `Network` or `NetworkView` providing selection and graph APIs.
        source_path: Selection expression for source groups.
        sink_path: Selection expression for sink groups.
        mode: "combine" or "pairwise". See ``max_flow``.
        shortest_path: If True, perform only one augmentation step.
        flow_placement: Strategy for splitting among equal-cost parallel edges.

    Returns:
        Dict[Tuple[str, str], Tuple[float, FlowSummary]]: For each
        (source_label, sink_label), the total flow and the associated summary.

    Raises:
        ValueError: If no matching sources or sinks are found, or if ``mode``
            is invalid.
    """
    src_groups = context.select_node_groups_by_path(source_path)
    snk_groups = context.select_node_groups_by_path(sink_path)

    if not src_groups:
        raise ValueError(f"No source nodes found matching '{source_path}'.")
    if not snk_groups:
        raise ValueError(f"No sink nodes found matching '{sink_path}'.")

    if mode == "combine":
        combined_src_nodes: list = []
        combined_snk_nodes: list = []
        combined_src_label = "|".join(sorted(src_groups.keys()))
        combined_snk_label = "|".join(sorted(snk_groups.keys()))
        for group_nodes in src_groups.values():
            combined_src_nodes.extend(group_nodes)
        for group_nodes in snk_groups.values():
            combined_snk_nodes.extend(group_nodes)
        if not combined_src_nodes or not combined_snk_nodes:
            empty = _empty_summary()
            return {(combined_src_label, combined_snk_label): (0.0, empty)}
        if {n.name for n in combined_src_nodes} & {n.name for n in combined_snk_nodes}:
            empty = _empty_summary()
            return {(combined_src_label, combined_snk_label): (0.0, empty)}
        flow_val, summary = _compute_flow_with_summary_single_group(
            context,
            combined_src_nodes,
            combined_snk_nodes,
            shortest_path,
            flow_placement,
        )
        return {(combined_src_label, combined_snk_label): (flow_val, summary)}

    if mode == "pairwise":
        results: Dict[Tuple[str, str], Tuple[float, FlowSummary]] = {}
        for src_label, src_nodes in src_groups.items():
            for snk_label, snk_nodes in snk_groups.items():
                if src_nodes and snk_nodes:
                    if {n.name for n in src_nodes} & {n.name for n in snk_nodes}:
                        results[(src_label, snk_label)] = (0.0, _empty_summary())
                    else:
                        results[(src_label, snk_label)] = (
                            _compute_flow_with_summary_single_group(
                                context,
                                src_nodes,
                                snk_nodes,
                                shortest_path,
                                flow_placement,
                            )
                        )
                else:
                    results[(src_label, snk_label)] = (0.0, _empty_summary())
        return results

    raise ValueError(f"Invalid mode '{mode}'. Must be 'combine' or 'pairwise'.")


def max_flow_with_graph(
    context: Any,
    source_path: str,
    sink_path: str,
    *,
    mode: str = "combine",
    shortest_path: bool = False,
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
) -> Dict[Tuple[str, str], Tuple[float, "StrictMultiDiGraph"]]:
    """Compute max flow and return the mutated flow graph for each pair.

    Args:
        context: `Network` or `NetworkView` providing selection and graph APIs.
        source_path: Selection expression for source groups.
        sink_path: Selection expression for sink groups.
        mode: "combine" or "pairwise". See ``max_flow``.
        shortest_path: If True, perform only one augmentation step.
        flow_placement: Strategy for splitting among equal-cost parallel edges.

    Returns:
        Dict[Tuple[str, str], Tuple[float, StrictMultiDiGraph]]: For each
        (source_label, sink_label), the total flow and the flow-assigned graph.

    Raises:
        ValueError: If no matching sources or sinks are found, or if ``mode``
            is invalid.
    """
    src_groups = context.select_node_groups_by_path(source_path)
    snk_groups = context.select_node_groups_by_path(sink_path)

    if not src_groups:
        raise ValueError(f"No source nodes found matching '{source_path}'.")
    if not snk_groups:
        raise ValueError(f"No sink nodes found matching '{sink_path}'.")

    if mode == "combine":
        combined_src_nodes: list = []
        combined_snk_nodes: list = []
        combined_src_label = "|".join(sorted(src_groups.keys()))
        combined_snk_label = "|".join(sorted(snk_groups.keys()))
        for group_nodes in src_groups.values():
            combined_src_nodes.extend(group_nodes)
        for group_nodes in snk_groups.values():
            combined_snk_nodes.extend(group_nodes)
        if not combined_src_nodes or not combined_snk_nodes:
            base_graph = context.to_strict_multidigraph().copy()
            return {(combined_src_label, combined_snk_label): (0.0, base_graph)}
        if {n.name for n in combined_src_nodes} & {n.name for n in combined_snk_nodes}:
            base_graph = context.to_strict_multidigraph().copy()
            return {(combined_src_label, combined_snk_label): (0.0, base_graph)}
        flow_val, flow_graph = _compute_flow_with_graph_single_group(
            context,
            combined_src_nodes,
            combined_snk_nodes,
            shortest_path,
            flow_placement,
        )
        return {(combined_src_label, combined_snk_label): (flow_val, flow_graph)}

    if mode == "pairwise":
        results: Dict[Tuple[str, str], Tuple[float, "StrictMultiDiGraph"]] = {}
        for src_label, src_nodes in src_groups.items():
            for snk_label, snk_nodes in snk_groups.items():
                if src_nodes and snk_nodes:
                    if {n.name for n in src_nodes} & {n.name for n in snk_nodes}:
                        base_graph = context.to_strict_multidigraph().copy()
                        results[(src_label, snk_label)] = (0.0, base_graph)
                    else:
                        results[(src_label, snk_label)] = (
                            _compute_flow_with_graph_single_group(
                                context,
                                src_nodes,
                                snk_nodes,
                                shortest_path,
                                flow_placement,
                            )
                        )
                else:
                    base_graph = context.to_strict_multidigraph().copy()
                    results[(src_label, snk_label)] = (0.0, base_graph)
        return results

    raise ValueError(f"Invalid mode '{mode}'. Must be 'combine' or 'pairwise'.")


def max_flow_detailed(
    context: Any,
    source_path: str,
    sink_path: str,
    *,
    mode: str = "combine",
    shortest_path: bool = False,
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
) -> Dict[Tuple[str, str], Tuple[float, FlowSummary, "StrictMultiDiGraph"]]:
    """Compute max flow, return summary and flow graph for each pair.

    Args:
        context: `Network` or `NetworkView` providing selection and graph APIs.
        source_path: Selection expression for source groups.
        sink_path: Selection expression for sink groups.
        mode: "combine" or "pairwise". See ``max_flow``.
        shortest_path: If True, perform only one augmentation step.
        flow_placement: Strategy for splitting among equal-cost parallel edges.

    Returns:
        Dict[Tuple[str, str], Tuple[float, FlowSummary, StrictMultiDiGraph]]:
        For each (source_label, sink_label), the total flow, a summary, and the
        flow-assigned graph.

    Raises:
        ValueError: If no matching sources or sinks are found, or if ``mode``
            is invalid.
    """
    src_groups = context.select_node_groups_by_path(source_path)
    snk_groups = context.select_node_groups_by_path(sink_path)

    if not src_groups:
        raise ValueError(f"No source nodes found matching '{source_path}'.")
    if not snk_groups:
        raise ValueError(f"No sink nodes found matching '{sink_path}'.")

    if mode == "combine":
        combined_src_nodes: list = []
        combined_snk_nodes: list = []
        combined_src_label = "|".join(sorted(src_groups.keys()))
        combined_snk_label = "|".join(sorted(snk_groups.keys()))
        for group_nodes in src_groups.values():
            combined_src_nodes.extend(group_nodes)
        for group_nodes in snk_groups.values():
            combined_snk_nodes.extend(group_nodes)
        if not combined_src_nodes or not combined_snk_nodes:
            base_graph = context.to_strict_multidigraph().copy()
            return {
                (combined_src_label, combined_snk_label): (
                    0.0,
                    _empty_summary(),
                    base_graph,
                )
            }
        if {n.name for n in combined_src_nodes} & {n.name for n in combined_snk_nodes}:
            base_graph = context.to_strict_multidigraph().copy()
            return {
                (combined_src_label, combined_snk_label): (
                    0.0,
                    _empty_summary(),
                    base_graph,
                )
            }
        flow_val, summary, flow_graph = _compute_flow_detailed_single_group(
            context,
            combined_src_nodes,
            combined_snk_nodes,
            shortest_path,
            flow_placement,
        )
        return {
            (combined_src_label, combined_snk_label): (flow_val, summary, flow_graph)
        }

    if mode == "pairwise":
        results: Dict[
            Tuple[str, str], Tuple[float, FlowSummary, "StrictMultiDiGraph"]
        ] = {}
        for src_label, src_nodes in src_groups.items():
            for snk_label, snk_nodes in snk_groups.items():
                if src_nodes and snk_nodes:
                    if {n.name for n in src_nodes} & {n.name for n in snk_nodes}:
                        base_graph = context.to_strict_multidigraph().copy()
                        results[(src_label, snk_label)] = (
                            0.0,
                            _empty_summary(),
                            base_graph,
                        )
                    else:
                        results[(src_label, snk_label)] = (
                            _compute_flow_detailed_single_group(
                                context,
                                src_nodes,
                                snk_nodes,
                                shortest_path,
                                flow_placement,
                            )
                        )
                else:
                    base_graph = context.to_strict_multidigraph().copy()
                    results[(src_label, snk_label)] = (
                        0.0,
                        _empty_summary(),
                        base_graph,
                    )
        return results

    raise ValueError(f"Invalid mode '{mode}'. Must be 'combine' or 'pairwise'.")


def saturated_edges(
    context: Any,
    source_path: str,
    sink_path: str,
    *,
    mode: str = "combine",
    tolerance: float = 1e-10,
    shortest_path: bool = False,
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
    """Identify saturated edges for each selected group pair.

    Args:
        context: `Network` or `NetworkView` providing selection and graph APIs.
        source_path: Selection expression for source groups.
        sink_path: Selection expression for sink groups.
        mode: "combine" or "pairwise". See ``max_flow``.
        tolerance: Residual capacity threshold to consider an edge saturated.
        shortest_path: If True, perform only one augmentation step.
        flow_placement: Strategy for splitting among equal-cost parallel edges.

    Returns:
        Dict[Tuple[str, str], list[tuple[str, str, str]]]: For each
        (source_label, sink_label), a list of saturated edges ``(u, v, k)``.

    Raises:
        ValueError: If no matching sources or sinks are found, or if ``mode``
            is invalid.
    """
    src_groups = context.select_node_groups_by_path(source_path)
    snk_groups = context.select_node_groups_by_path(sink_path)

    if not src_groups:
        raise ValueError(f"No source nodes found matching '{source_path}'.")
    if not snk_groups:
        raise ValueError(f"No sink nodes found matching '{sink_path}'.")

    if mode == "combine":
        combined_src_nodes: list = []
        combined_snk_nodes: list = []
        combined_src_label = "|".join(sorted(src_groups.keys()))
        combined_snk_label = "|".join(sorted(snk_groups.keys()))
        for group_nodes in src_groups.values():
            combined_src_nodes.extend(group_nodes)
        for group_nodes in snk_groups.values():
            combined_snk_nodes.extend(group_nodes)
        if not combined_src_nodes or not combined_snk_nodes:
            return {(combined_src_label, combined_snk_label): []}
        if {n.name for n in combined_src_nodes} & {n.name for n in combined_snk_nodes}:
            saturated_list: List[Tuple[str, str, str]] = []
        else:
            saturated_list = _compute_saturated_edges_single_group(
                context,
                combined_src_nodes,
                combined_snk_nodes,
                tolerance,
                shortest_path,
                flow_placement,
            )
        return {(combined_src_label, combined_snk_label): saturated_list}

    if mode == "pairwise":
        results: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
        for src_label, src_nodes in src_groups.items():
            for snk_label, snk_nodes in snk_groups.items():
                if src_nodes and snk_nodes:
                    if {n.name for n in src_nodes} & {n.name for n in snk_nodes}:
                        saturated_list = []
                    else:
                        saturated_list = _compute_saturated_edges_single_group(
                            context,
                            src_nodes,
                            snk_nodes,
                            tolerance,
                            shortest_path,
                            flow_placement,
                        )
                else:
                    saturated_list = []
                results[(src_label, snk_label)] = saturated_list
        return results

    raise ValueError(f"Invalid mode '{mode}'. Must be 'combine' or 'pairwise'.")


def sensitivity_analysis(
    context: Any,
    source_path: str,
    sink_path: str,
    *,
    mode: str = "combine",
    change_amount: float = 1.0,
    shortest_path: bool = False,
    flow_placement: FlowPlacement = FlowPlacement.PROPORTIONAL,
) -> Dict[Tuple[str, str], Dict[Tuple[str, str, str], float]]:
    """Perform a simple sensitivity analysis per saturated edge.

    For each saturated edge, test a capacity change of ``change_amount`` and
    report the change in total flow. Positive amounts increase capacity; negative
    amounts decrease capacity (with lower bound at zero).

    Args:
        context: `Network` or `NetworkView` providing selection and graph APIs.
        source_path: Selection expression for source groups.
        sink_path: Selection expression for sink groups.
        mode: "combine" or "pairwise". See ``max_flow``.
        change_amount: Capacity delta to apply when testing each saturated edge.
        shortest_path: If True, perform only one augmentation step.
        flow_placement: Strategy for splitting among equal-cost parallel edges.

    Returns:
        Dict[Tuple[str, str], Dict[Tuple[str, str, str], float]]: For each
        (source_label, sink_label), a mapping from saturated edge ``(u, v, k)``
        to the change in total flow after applying the capacity delta.

    Raises:
        ValueError: If no matching sources or sinks are found, or if ``mode``
            is invalid.
    """
    src_groups = context.select_node_groups_by_path(source_path)
    snk_groups = context.select_node_groups_by_path(sink_path)

    if not src_groups:
        raise ValueError(f"No source nodes found matching '{source_path}'.")
    if not snk_groups:
        raise ValueError(f"No sink nodes found matching '{sink_path}'.")

    if mode == "combine":
        combined_src_nodes: list = []
        combined_snk_nodes: list = []
        combined_src_label = "|".join(sorted(src_groups.keys()))
        combined_snk_label = "|".join(sorted(snk_groups.keys()))
        for group_nodes in src_groups.values():
            combined_src_nodes.extend(group_nodes)
        for group_nodes in snk_groups.values():
            combined_snk_nodes.extend(group_nodes)
        if not combined_src_nodes or not combined_snk_nodes:
            return {(combined_src_label, combined_snk_label): {}}
        if {n.name for n in combined_src_nodes} & {n.name for n in combined_snk_nodes}:
            sensitivity_dict: Dict[Tuple[str, str, str], float] = {}
        else:
            sensitivity_dict = _compute_sensitivity_single_group(
                context,
                combined_src_nodes,
                combined_snk_nodes,
                change_amount,
                shortest_path,
                flow_placement,
            )
        return {(combined_src_label, combined_snk_label): sensitivity_dict}

    if mode == "pairwise":
        results: Dict[Tuple[str, str], Dict[Tuple[str, str, str], float]] = {}
        for src_label, src_nodes in src_groups.items():
            for snk_label, snk_nodes in snk_groups.items():
                if src_nodes and snk_nodes:
                    if {n.name for n in src_nodes} & {n.name for n in snk_nodes}:
                        sensitivity_dict = {}
                    else:
                        sensitivity_dict = _compute_sensitivity_single_group(
                            context,
                            src_nodes,
                            snk_nodes,
                            change_amount,
                            shortest_path,
                            flow_placement,
                        )
                else:
                    sensitivity_dict = {}
                results[(src_label, snk_label)] = sensitivity_dict
        return results

    raise ValueError(f"Invalid mode '{mode}'. Must be 'combine' or 'pairwise'.")


# --- Single-group helpers ---------------------------------------------------


def _compute_flow_single_group(
    context: Any,
    sources: list,
    sinks: list,
    shortest_path: bool,
    flow_placement: FlowPlacement,
    *,
    prebuilt_graph: Optional["StrictMultiDiGraph"] = None,
) -> float:
    active_sources = [s for s in sources if not s.disabled]
    active_sinks = [s for s in sinks if not s.disabled]
    if not active_sources or not active_sinks:
        return 0.0
    graph = (
        prebuilt_graph.copy()
        if prebuilt_graph is not None
        else context.to_strict_multidigraph()
    )
    graph.add_node("source")
    graph.add_node("sink")
    for s_node in active_sources:
        graph.add_edge("source", s_node.name, capacity=float("inf"), cost=0)
    for t_node in active_sinks:
        graph.add_edge(t_node.name, "sink", capacity=float("inf"), cost=0)
    return calc_max_flow(
        graph,
        "source",
        "sink",
        flow_placement=flow_placement,
        shortest_path=shortest_path,
        copy_graph=False,
    )


def _compute_flow_with_summary_single_group(
    context: Any,
    sources: list,
    sinks: list,
    shortest_path: bool,
    flow_placement: FlowPlacement,
) -> Tuple[float, FlowSummary]:
    active_sources = [s for s in sources if not s.disabled]
    active_sinks = [s for s in sinks if not s.disabled]
    if not active_sources or not active_sinks:
        return 0.0, _empty_summary()
    graph = context.to_strict_multidigraph().copy()
    graph.add_node("source")
    graph.add_node("sink")
    for s_node in active_sources:
        graph.add_edge("source", s_node.name, capacity=float("inf"), cost=0)
    for t_node in active_sinks:
        graph.add_edge(t_node.name, "sink", capacity=float("inf"), cost=0)
    flow_val, summary = calc_max_flow(
        graph,
        "source",
        "sink",
        return_summary=True,
        flow_placement=flow_placement,
        shortest_path=shortest_path,
        copy_graph=False,
    )
    return flow_val, summary


def _compute_flow_with_graph_single_group(
    context: Any,
    sources: list,
    sinks: list,
    shortest_path: bool,
    flow_placement: FlowPlacement,
) -> Tuple[float, "StrictMultiDiGraph"]:
    active_sources = [s for s in sources if not s.disabled]
    active_sinks = [s for s in sinks if not s.disabled]
    if not active_sources or not active_sinks:
        base_graph = context.to_strict_multidigraph().copy()
        return 0.0, base_graph
    graph = context.to_strict_multidigraph().copy()
    graph.add_node("source")
    graph.add_node("sink")
    for s_node in active_sources:
        graph.add_edge("source", s_node.name, capacity=float("inf"), cost=0)
    for t_node in active_sinks:
        graph.add_edge(t_node.name, "sink", capacity=float("inf"), cost=0)
    flow_val, flow_graph = calc_max_flow(
        graph,
        "source",
        "sink",
        return_graph=True,
        flow_placement=flow_placement,
        shortest_path=shortest_path,
        copy_graph=False,
    )
    return flow_val, flow_graph


def _compute_flow_detailed_single_group(
    context: Any,
    sources: list,
    sinks: list,
    shortest_path: bool,
    flow_placement: FlowPlacement,
) -> Tuple[float, FlowSummary, "StrictMultiDiGraph"]:
    active_sources = [s for s in sources if not s.disabled]
    active_sinks = [s for s in sinks if not s.disabled]
    if not active_sources or not active_sinks:
        base_graph = context.to_strict_multidigraph().copy()
        return 0.0, _empty_summary(), base_graph
    graph = context.to_strict_multidigraph().copy()
    graph.add_node("source")
    graph.add_node("sink")
    for s_node in active_sources:
        graph.add_edge("source", s_node.name, capacity=float("inf"), cost=0)
    for t_node in active_sinks:
        graph.add_edge(t_node.name, "sink", capacity=float("inf"), cost=0)
    flow_val, summary, flow_graph = calc_max_flow(
        graph,
        "source",
        "sink",
        return_summary=True,
        return_graph=True,
        flow_placement=flow_placement,
        shortest_path=shortest_path,
        copy_graph=False,
    )
    return flow_val, summary, flow_graph


def _compute_saturated_edges_single_group(
    context: Any,
    sources: list,
    sinks: list,
    tolerance: float,
    shortest_path: bool,
    flow_placement: FlowPlacement,
) -> List[Tuple[str, str, str]]:
    active_sources = [s for s in sources if not s.disabled]
    active_sinks = [s for s in sinks if not s.disabled]
    if not active_sources or not active_sinks:
        return []
    graph = context.to_strict_multidigraph().copy()
    graph.add_node("source")
    graph.add_node("sink")
    for s_node in active_sources:
        graph.add_edge("source", s_node.name, capacity=float("inf"), cost=0)
    for t_node in active_sinks:
        graph.add_edge(t_node.name, "sink", capacity=float("inf"), cost=0)
    return _algo_saturated_edges(
        graph,
        "source",
        "sink",
        tolerance=tolerance,
        flow_placement=flow_placement,
        shortest_path=shortest_path,
        copy_graph=False,
    )


def _compute_sensitivity_single_group(
    context: Any,
    sources: list,
    sinks: list,
    change_amount: float,
    shortest_path: bool,
    flow_placement: FlowPlacement,
) -> Dict[Tuple[str, str, str], float]:
    active_sources = [s for s in sources if not s.disabled]
    active_sinks = [s for s in sinks if not s.disabled]
    if not active_sources or not active_sinks:
        return {}
    graph = context.to_strict_multidigraph().copy()
    graph.add_node("source")
    graph.add_node("sink")
    for s_node in active_sources:
        graph.add_edge("source", s_node.name, capacity=float("inf"), cost=0)
    for t_node in active_sinks:
        graph.add_edge(t_node.name, "sink", capacity=float("inf"), cost=0)
    return run_sensitivity(
        graph,
        "source",
        "sink",
        change_amount=change_amount,
        flow_placement=flow_placement,
        shortest_path=shortest_path,
        copy_graph=False,
    )


def _empty_summary() -> FlowSummary:
    return FlowSummary(
        total_flow=0.0,
        edge_flow={},
        residual_cap={},
        reachable=set(),
        min_cut=[],
        cost_distribution={},
    )
