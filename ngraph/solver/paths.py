"""Shortest-path solver wrappers bound to the model layer.

Expose convenience functions for computing shortest paths between node groups
selected from a ``Network`` or ``NetworkView`` context. Selection semantics
mirror the max-flow wrappers with ``mode`` in {"combine", "pairwise"}.

Functions return minimal costs or concrete ``Path`` objects built from SPF
predecessor maps. Parallel equal-cost edges can be expanded into distinct
paths.

All functions fail fast on invalid selection inputs and do not mutate the
input context.

Note:
    For path queries, overlapping source/sink membership is treated as
    unreachable.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ngraph.algorithms.base import EdgeSelect
from ngraph.algorithms.spf import ksp, spf
from ngraph.paths.path import Path


def shortest_path_costs(
    context: Any,
    source_path: str,
    sink_path: str,
    *,
    mode: str = "combine",
    edge_select: EdgeSelect = EdgeSelect.ALL_MIN_COST,
) -> Dict[Tuple[str, str], float]:
    """Return minimal path cost(s) between selected node groups.

    Args:
        context: Network or NetworkView.
        source_path: Selection expression for source groups.
        sink_path: Selection expression for sink groups.
        mode: "combine" or "pairwise".
        edge_select: SPF edge selection strategy.

    Returns:
        Mapping from (source_label, sink_label) to minimal cost; ``inf`` if no
        path.

    Raises:
        ValueError: If no source nodes match ``source_path``.
        ValueError: If no sink nodes match ``sink_path``.
        ValueError: If ``mode`` is not "combine" or "pairwise".
    """
    src_groups = context.select_node_groups_by_path(source_path)
    snk_groups = context.select_node_groups_by_path(sink_path)

    if not src_groups:
        raise ValueError(f"No source nodes found matching '{source_path}'.")
    if not snk_groups:
        raise ValueError(f"No sink nodes found matching '{sink_path}'.")

    graph = context.to_strict_multidigraph().copy()

    def _active(nodes: Iterable[Any]) -> List[Any]:
        return [n for n in nodes if not getattr(n, "disabled", False)]

    if mode == "combine":
        combined_src_nodes: List[Any] = []
        combined_snk_nodes: List[Any] = []
        combined_src_label = "|".join(sorted(src_groups.keys()))
        combined_snk_label = "|".join(sorted(snk_groups.keys()))

        for group_nodes in src_groups.values():
            combined_src_nodes.extend(group_nodes)
        for group_nodes in snk_groups.values():
            combined_snk_nodes.extend(group_nodes)

        active_sources = _active(combined_src_nodes)
        active_sinks = _active(combined_snk_nodes)
        if not active_sources or not active_sinks:
            return {(combined_src_label, combined_snk_label): float("inf")}
        if {n.name for n in active_sources} & {n.name for n in active_sinks}:
            return {(combined_src_label, combined_snk_label): float("inf")}

        best_cost = float("inf")
        for s in active_sources:
            costs, _ = spf(graph, s.name, edge_select=edge_select, multipath=True)
            for t in active_sinks:
                c = costs.get(t.name)
                if c is not None and c < best_cost:
                    best_cost = c
        return {(combined_src_label, combined_snk_label): best_cost}

    if mode == "pairwise":
        results: Dict[Tuple[str, str], float] = {}
        for src_label, src_nodes in src_groups.items():
            for snk_label, snk_nodes in snk_groups.items():
                active_sources = _active(src_nodes)
                active_sinks = _active(snk_nodes)
                if not active_sources or not active_sinks:
                    results[(src_label, snk_label)] = float("inf")
                    continue
                if {n.name for n in active_sources} & {n.name for n in active_sinks}:
                    results[(src_label, snk_label)] = float("inf")
                    continue
                best_cost = float("inf")
                for s in active_sources:
                    costs, _ = spf(
                        graph, s.name, edge_select=edge_select, multipath=True
                    )
                    for t in active_sinks:
                        c = costs.get(t.name)
                        if c is not None and c < best_cost:
                            best_cost = c
                results[(src_label, snk_label)] = best_cost
        return results

    raise ValueError(f"Invalid mode '{mode}'. Must be 'combine' or 'pairwise'.")


def shortest_paths(
    context: Any,
    source_path: str,
    sink_path: str,
    *,
    mode: str = "combine",
    edge_select: EdgeSelect = EdgeSelect.ALL_MIN_COST,
    split_parallel_edges: bool = False,
) -> Dict[Tuple[str, str], List[Path]]:
    """Return concrete shortest path(s) between selected node groups.

    Args:
        context: Network or NetworkView.
        source_path: Selection expression for source groups.
        sink_path: Selection expression for sink groups.
        mode: "combine" or "pairwise".
        edge_select: SPF edge selection strategy.
        split_parallel_edges: Expand parallel edges into distinct paths when True.

    Returns:
        Mapping from (source_label, sink_label) to list of Path. Empty if
        unreachable.

    Raises:
        ValueError: If no source nodes match ``source_path``.
        ValueError: If no sink nodes match ``sink_path``.
        ValueError: If ``mode`` is not "combine" or "pairwise".
    """
    src_groups = context.select_node_groups_by_path(source_path)
    snk_groups = context.select_node_groups_by_path(sink_path)

    if not src_groups:
        raise ValueError(f"No source nodes found matching '{source_path}'.")
    if not snk_groups:
        raise ValueError(f"No sink nodes found matching '{sink_path}'.")

    graph = context.to_strict_multidigraph().copy()

    def _active(nodes: Iterable[Any]) -> List[Any]:
        return [n for n in nodes if not getattr(n, "disabled", False)]

    def _best_paths_for_groups(
        src_nodes: List[Any], snk_nodes: List[Any]
    ) -> List[Path]:
        active_sources = _active(src_nodes)
        active_sinks = _active(snk_nodes)
        if not active_sources or not active_sinks:
            return []
        if {n.name for n in active_sources} & {n.name for n in active_sinks}:
            return []

        best_cost = float("inf")
        best_paths: List[Path] = []

        from ngraph.algorithms.paths import resolve_to_paths as _resolve

        for s in active_sources:
            costs, pred = spf(graph, s.name, edge_select=edge_select, multipath=True)
            for t in active_sinks:
                if t.name not in pred:
                    continue
                cost_to_t = costs.get(t.name, float("inf"))
                if cost_to_t < best_cost:
                    best_cost = cost_to_t
                    best_paths = [
                        Path(path_tuple, cost_to_t)
                        for path_tuple in _resolve(
                            s.name, t.name, pred, split_parallel_edges
                        )
                    ]
                elif cost_to_t == best_cost:
                    best_paths.extend(
                        Path(path_tuple, cost_to_t)
                        for path_tuple in _resolve(
                            s.name, t.name, pred, split_parallel_edges
                        )
                    )

        if best_paths:
            best_paths = sorted(set(best_paths))
        return best_paths

    if mode == "combine":
        combined_src_nodes: List[Any] = []
        combined_snk_nodes: List[Any] = []
        combined_src_label = "|".join(sorted(src_groups.keys()))
        combined_snk_label = "|".join(sorted(snk_groups.keys()))

        for group_nodes in src_groups.values():
            combined_src_nodes.extend(group_nodes)
        for group_nodes in snk_groups.values():
            combined_snk_nodes.extend(group_nodes)

        paths_list = _best_paths_for_groups(combined_src_nodes, combined_snk_nodes)
        return {(combined_src_label, combined_snk_label): paths_list}

    if mode == "pairwise":
        results: Dict[Tuple[str, str], List[Path]] = {}
        for src_label, src_nodes in src_groups.items():
            for snk_label, snk_nodes in snk_groups.items():
                results[(src_label, snk_label)] = _best_paths_for_groups(
                    src_nodes, snk_nodes
                )
        return results

    raise ValueError(f"Invalid mode '{mode}'. Must be 'combine' or 'pairwise'.")


def k_shortest_paths(
    context: Any,
    source_path: str,
    sink_path: str,
    *,
    mode: str = "pairwise",
    max_k: int = 3,
    edge_select: EdgeSelect = EdgeSelect.ALL_MIN_COST,
    max_path_cost: float = float("inf"),
    max_path_cost_factor: Optional[float] = None,
    split_parallel_edges: bool = False,
) -> Dict[Tuple[str, str], List[Path]]:
    """Return up to K shortest paths per group pair.

    Args:
        context: Network or NetworkView.
        source_path: Selection expression for source groups.
        sink_path: Selection expression for sink groups.
        mode: "pairwise" (default) or "combine".
        max_k: Max paths per pair.
        edge_select: SPF/KSP edge selection strategy.
        max_path_cost: Absolute cost threshold.
        max_path_cost_factor: Relative threshold versus best path.
        split_parallel_edges: Expand parallel edges into distinct paths when True.

    Returns:
        Mapping from (source_label, sink_label) to list of Path (<= max_k).

    Raises:
        ValueError: If no source nodes match ``source_path``.
        ValueError: If no sink nodes match ``sink_path``.
        ValueError: If ``mode`` is not "combine" or "pairwise".
    """
    src_groups = context.select_node_groups_by_path(source_path)
    snk_groups = context.select_node_groups_by_path(sink_path)

    if not src_groups:
        raise ValueError(f"No source nodes found matching '{source_path}'.")
    if not snk_groups:
        raise ValueError(f"No sink nodes found matching '{sink_path}'.")

    graph = context.to_strict_multidigraph().copy()

    def _active(nodes: Iterable[Any]) -> List[Any]:
        return [n for n in nodes if not getattr(n, "disabled", False)]

    def _ksp_for_groups(src_nodes: List[Any], snk_nodes: List[Any]) -> List[Path]:
        active_sources = _active(src_nodes)
        active_sinks = _active(snk_nodes)
        if not active_sources or not active_sinks:
            return []
        if {n.name for n in active_sources} & {n.name for n in active_sinks}:
            return []

        # Choose best pair to seed thresholds
        best_pair: Optional[Tuple[str, str]] = None
        best_cost = float("inf")
        for s in active_sources:
            costs, pred = spf(graph, s.name, edge_select=edge_select, multipath=True)
            for t in active_sinks:
                if t.name not in pred:
                    continue
                c = costs.get(t.name, float("inf"))
                if c < best_cost:
                    best_cost = c
                    best_pair = (s.name, t.name)

        if best_pair is None:
            return []

        results: List[Path] = []
        s_name, t_name = best_pair
        count = 0
        from ngraph.algorithms.paths import resolve_to_paths as _resolve

        for costs_i, pred_i in ksp(
            graph,
            s_name,
            t_name,
            edge_select=edge_select,
            max_k=max_k,
            max_path_cost=max_path_cost,
            max_path_cost_factor=max_path_cost_factor,
            multipath=True,
        ):
            if t_name not in pred_i:
                continue
            cost_val = costs_i[t_name]
            for path_tuple in _resolve(s_name, t_name, pred_i, split_parallel_edges):
                results.append(Path(path_tuple, cost_val))
                count += 1
                if count >= max_k:
                    break
            if count >= max_k:
                break

        if results:
            results = sorted(set(results))[:max_k]
        return results

    if mode == "combine":
        combined_src_nodes: List[Any] = []
        combined_snk_nodes: List[Any] = []
        combined_src_label = "|".join(sorted(src_groups.keys()))
        combined_snk_label = "|".join(sorted(snk_groups.keys()))

        for group_nodes in src_groups.values():
            combined_src_nodes.extend(group_nodes)
        for group_nodes in snk_groups.values():
            combined_snk_nodes.extend(group_nodes)

        return {
            (combined_src_label, combined_snk_label): _ksp_for_groups(
                combined_src_nodes, combined_snk_nodes
            )
        }

    if mode == "pairwise":
        results: Dict[Tuple[str, str], List[Path]] = {}
        for src_label, src_nodes in src_groups.items():
            for snk_label, snk_nodes in snk_groups.items():
                results[(src_label, snk_label)] = _ksp_for_groups(src_nodes, snk_nodes)
        return results

    raise ValueError(f"Invalid mode '{mode}'. Must be 'combine' or 'pairwise'.")
