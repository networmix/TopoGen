"""Integrated graph analysis helper.

This script loads an integrated (corridor-level) graph JSON and prints:

- Metros ordered by number of adjacencies (degree)
- Global node connectivity and a minimum vertex cut
- A high-degree vertex cut of the same cardinality when possible
- Auxiliary metrics (k-core size, articulation points if any)

Usage:
    python dev/analyze_integrated_graph.py baseline_integrated_graph.json
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

from topogen.integrated_graph import load_from_json

# -----------------------------
# Data structures
# -----------------------------


@dataclass(frozen=True, slots=True)
class MetroInfo:
    """Basic metro information used for reporting.

    Attributes:
        node_key: NetworkX node key (coordinate tuple).
        metro_id: Metro ID string if available.
        name: Sanitized metro name if available, else metro_id or coord string.
        degree: Unweighted degree in the corridor graph.
    """

    node_key: tuple[float, float]
    metro_id: str
    name: str
    degree: int


# -----------------------------
# Utilities
# -----------------------------


def _format_int(x: int) -> str:
    return f"{x:,}"


def _format_float(x: float, decimals: int = 2) -> str:
    return f"{x:,.{decimals}f}"


def _is_disconnected_after_removal(graph: nx.Graph, nodes_to_remove: Iterable) -> bool:
    """Return True if removing nodes disconnects the graph.

    Args:
        graph: Input graph assumed connected initially.
        nodes_to_remove: Nodes to remove and test connectivity.

    Returns:
        True if the remaining graph has more than one connected component.
    """

    remaining = graph.copy()
    remaining.remove_nodes_from(nodes_to_remove)
    if remaining.number_of_nodes() == 0:
        return True
    return not nx.is_connected(remaining)


def _find_high_degree_cut(
    graph: nx.Graph,
    k: int,
    search_pool_size: int = 20,
) -> list[tuple[float, float]]:
    """Find a size-k vertex cut among high-degree nodes if one exists.

    Strategy:
        - Sort nodes by degree (desc) and take the top `search_pool_size`.
        - Try all k-combinations in that pool (lexicographic order) and
          return the first that disconnects the graph.

    Complexity:
        O(C(pool, k)) connectivity checks; works well for small k (<= 5).

    Args:
        graph: Connected corridor-level graph.
        k: Target cut cardinality (global node connectivity).
        search_pool_size: Number of highest-degree nodes to consider.

    Returns:
        A list of nodes if found, otherwise an empty list.
    """

    if k <= 0:
        return []

    # Prepare candidate pool
    degree_pairs: list[tuple[tuple[float, float], int]] = [
        (n, d)
        for n, d in list(graph.degree())  # type: ignore[misc]
    ]
    degree_pairs_sorted = sorted(degree_pairs, key=lambda t: t[1], reverse=True)
    pool_limit = min(search_pool_size, len(degree_pairs_sorted))
    pool_nodes = [n for n, _d in degree_pairs_sorted[:pool_limit]]

    # Brute-force combinations within the pool
    for combo in combinations(pool_nodes, k):
        if _is_disconnected_after_removal(graph, combo):
            return list(combo)
    return []


def _greedy_disconnect(
    graph: nx.Graph, order: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Greedy removal by given order until the graph disconnects.

    Useful as a fallback when a size-k high-degree cut is not found.
    Attempts a simple post-shrink step to reduce the set.
    """

    removed: list[tuple[float, float]] = []
    for node in order:
        removed.append(node)
        if _is_disconnected_after_removal(graph, removed):
            break

    # Post-shrink: try to drop redundant nodes while preserving disconnection
    changed = True
    while changed:
        changed = False
        for n in list(removed):
            test = [x for x in removed if x != n]
            if _is_disconnected_after_removal(graph, test):
                removed = test
                changed = True
    return removed


def _collect_metros(graph: nx.Graph) -> list[MetroInfo]:
    """Convert graph nodes to MetroInfo records.

    Expects nodes to have `metro_id` and `name` attributes. Falls back to
    coordinate string if attributes are missing.
    """

    infos: list[MetroInfo] = []
    for node, data in graph.nodes(data=True):
        metro_id = str(data.get("metro_id", ""))
        name_attr = data.get("name")
        if isinstance(name_attr, str) and name_attr:
            name = name_attr
        elif metro_id:
            name = metro_id
        else:
            name = f"({node[0]:.0f},{node[1]:.0f})"
        deg_val: int = graph.degree(node)  # type: ignore[assignment]
        infos.append(
            MetroInfo(
                node_key=(float(node[0]), float(node[1])),
                metro_id=metro_id,
                name=name,
                degree=deg_val,
            )
        )
    return infos


def _build_degree_frame(metros: list[MetroInfo]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "node_key": [m.node_key for m in metros],
            "metro_id": [m.metro_id for m in metros],
            "name": [m.name for m in metros],
            "degree": [m.degree for m in metros],
        }
    )
    df = df.sort_values(["degree", "name"], ascending=[False, True]).reset_index(
        drop=True
    )
    return df


# -----------------------------
# Main analysis routine
# -----------------------------


def analyze_graph(json_path: Path, top_n: int | None = None) -> None:
    """Run analysis and print findings.

    Args:
        json_path: Path to integrated graph JSON (corridor-level).
        top_n: Optional limit when printing rankings.
    """

    graph, crs = load_from_json(json_path)

    # Basic shape and connectivity
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    is_conn = nx.is_connected(graph)
    kconn = nx.node_connectivity(graph) if num_nodes > 1 else 0

    print(f"File: {json_path}")
    print(f"CRS: {crs}")
    print(
        f"Graph: nodes={_format_int(num_nodes)}, edges={_format_int(num_edges)}, connected={is_conn}"
    )
    print(f"Global node connectivity (k): {kconn}")

    # Degree ranking
    metros = _collect_metros(graph)
    df = _build_degree_frame(metros)

    limit = top_n if isinstance(top_n, int) and top_n > 0 else len(df)
    print("\nMetros ordered by number of adjacencies (degree):")
    for rank, (_idx, row) in enumerate(df.head(limit).iterrows(), start=1):
        name = row["name"]
        degree = row["degree"]
        mid = row["metro_id"]
        print(f"  {rank:>3}. {name:35s}  degree={degree:2d}  id={mid}")

    # Minimum vertex cut (size k)
    min_cut_nodes: list[tuple[float, float]] = []
    if kconn > 0:
        # Using exact minimum node cut
        from networkx.algorithms.connectivity import cuts as nx_cuts

        min_cut_set = nx_cuts.minimum_node_cut(graph)  # global min cut
        min_cut_nodes = list(min_cut_set)
        # Order by degree desc for readability
        deg_map: dict[tuple[float, float], int] = {
            n: graph.degree(n)
            for n in graph.nodes  # type: ignore[dict-item]
        }
        min_cut_nodes = sorted(
            min_cut_nodes, key=lambda n: deg_map.get(n, 0), reverse=True
        )

    if min_cut_nodes:
        print(f"\nMinimum vertex cut (size {len(min_cut_nodes)}):")
        deg_map_print: dict[tuple[float, float], int] = {
            n: graph.degree(n)
            for n in graph.nodes  # type: ignore[dict-item]
        }
        for n in min_cut_nodes:
            data = graph.nodes[n]
            name = str(data.get("name", data.get("metro_id", str(n))))
            print(f"  - {name}  (degree={deg_map_print.get(n, 0)})")
    else:
        print("\nMinimum vertex cut: none (graph disconnected or trivial)")

    # High-degree cut search of size k
    if kconn > 0:
        hd_cut = _find_high_degree_cut(graph, k=kconn, search_pool_size=20)
        if hd_cut:
            print(
                f"\nHigh-degree vertex cut of size {kconn} (within top-20 by degree):"
            )
            deg_map_hd: dict[tuple[float, float], int] = {
                n: graph.degree(n)
                for n in graph.nodes  # type: ignore[dict-item]
            }
            for n in hd_cut:
                data = graph.nodes[n]
                name = str(data.get("name", data.get("metro_id", str(n))))
                print(f"  - {name}  (degree={deg_map_hd.get(n, 0)})")
        else:
            # Fallback: greedy removal until disconnected
            degree_pairs2: list[tuple[tuple[float, float], int]] = [
                (n, d)
                for n, d in list(graph.degree())  # type: ignore[misc]
            ]
            ordered_nodes = [
                n for n, _d in sorted(degree_pairs2, key=lambda t: t[1], reverse=True)
            ]
            greedy = _greedy_disconnect(graph, ordered_nodes)
            print(
                f"\nNo size-{kconn} cut found among top-20 degrees. Greedy disconnection set size={len(greedy)}:"
            )
            deg_map_hd2: dict[tuple[float, float], int] = {
                n: graph.degree(n)
                for n in graph.nodes  # type: ignore[dict-item]
            }
            for n in greedy:
                data = graph.nodes[n]
                name = str(data.get("name", data.get("metro_id", str(n))))
                print(f"  - {name}  (degree={deg_map_hd2.get(n, 0)})")

    # Articulation points and k-core for additional context
    arts = list(nx.articulation_points(graph)) if num_nodes > 0 else []
    if arts:
        print(f"\nArticulation points (cut vertices): {len(arts)}")
        deg_map2: dict[tuple[float, float], int] = {
            n: graph.degree(n)
            for n in graph.nodes  # type: ignore[dict-item]
        }
        arts_sorted = sorted(arts, key=lambda n: deg_map2.get(n, 0), reverse=True)
        show_ap = min(20, len(arts_sorted))
        for n in arts_sorted[:show_ap]:
            data = graph.nodes[n]
            name = str(data.get("name", data.get("metro_id", str(n))))
            print(f"  - {name}  (degree={deg_map2.get(n, 0)})")

    core_num = nx.core_number(graph) if num_nodes > 0 else {}
    if core_num:
        max_core_k = max(core_num.values())
        in_max_core = [n for n, k in core_num.items() if k == max_core_k]
        print(f"\nMax k-core: k={max_core_k}, size={len(in_max_core)}")
        # Show up to 20 members sorted by degree
        deg_map3: dict[tuple[float, float], int] = {
            n: graph.degree(n)
            for n in graph.nodes  # type: ignore[dict-item]
        }
        in_max_core_sorted = sorted(
            in_max_core, key=lambda n: deg_map3.get(n, 0), reverse=True
        )
        show_core = min(20, len(in_max_core_sorted))
        for n in in_max_core_sorted[:show_core]:
            data = graph.nodes[n]
            name = str(data.get("name", data.get("metro_id", str(n))))
            print(f"  - {name}  (degree={deg_map3.get(n, 0)})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze an integrated corridor graph JSON"
    )
    parser.add_argument("json_path", type=Path, help="Path to integrated graph JSON")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only print top-N metros by degree (default: all)",
    )
    args = parser.parse_args()

    analyze_graph(args.json_path, top_n=args.top)


if __name__ == "__main__":
    main()
