"""Path manipulation utilities.

Provides helpers to enumerate realized paths from a predecessor map produced by
SPF/KSP, with optional expansion of parallel edges into distinct paths.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterator, List

from ngraph.algorithms.base import PathTuple
from ngraph.graph.strict_multidigraph import EdgeID, NodeID


def resolve_to_paths(
    src_node: NodeID,
    dst_node: NodeID,
    pred: Dict[NodeID, Dict[NodeID, List[EdgeID]]],
    split_parallel_edges: bool = False,
) -> Iterator[PathTuple]:
    """Enumerate all paths from a predecessor map.

    Args:
        src_node: Source node ID.
        dst_node: Destination node ID.
        pred: Predecessor map from SPF or KSP.
        split_parallel_edges: If True, expand parallel edges into distinct paths.

    Yields:
        PathTuple: Sequence of ``(node_id, (edge_ids,))`` pairs from source to dest.
    """
    # If dst_node not in pred, no paths exist
    if dst_node not in pred:
        return

    seen = {dst_node}
    # Each stack entry: [(current_node, tuple_of_edgeIDs), predecessor_index]
    stack: List[List[Any]] = [[(dst_node, ()), 0]]
    top = 0

    while top >= 0:
        node_edges, nbr_idx = stack[top]
        current_node, _ = node_edges

        if current_node == src_node:
            # Rebuild the path by slicing stack up to top, then reversing
            full_path_reversed = [frame[0] for frame in stack[: top + 1]]
            path_tuple = tuple(reversed(full_path_reversed))

            if not split_parallel_edges:
                yield path_tuple
            else:
                # Expand parallel edges for each segment except the final destination
                ranges = [range(len(seg[1])) for seg in path_tuple[:-1]]
                for combo in product(*ranges):
                    expanded = []
                    for i, seg in enumerate(path_tuple):
                        if i < len(combo):
                            # pick a single edge from seg[1]
                            chosen_edge = (seg[1][combo[i]],)
                            expanded.append((seg[0], chosen_edge))
                        else:
                            # last node has an empty edges tuple
                            expanded.append((seg[0], ()))
                    yield tuple(expanded)

        # Try next predecessor of current_node
        current_pred_map = pred[current_node]
        keys = list(current_pred_map.keys())
        if nbr_idx < len(keys):
            stack[top][1] = nbr_idx + 1
            next_pred = keys[nbr_idx]
            edge_list = current_pred_map[next_pred]

            if next_pred in seen:
                # cycle detected, skip
                continue
            seen.add(next_pred)

            top += 1
            next_node_edges = (next_pred, tuple(edge_list))
            if top == len(stack):
                stack.append([next_node_edges, 0])
            else:
                stack[top] = [next_node_edges, 0]
        else:
            # backtrack
            seen.discard(current_node)
            top -= 1
