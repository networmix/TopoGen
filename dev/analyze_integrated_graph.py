"""Report corridor counts and metro vertex connectivity from current graph JSON.

Usage: python dev/analyze_integrated_graph.py output/small_baseline_integrated_graph.json
Parallel corridors count toward reported degree. Vertex cuts and k-cores operate
on the simple adjacency graph, since parallel paths do not add metro neighbors.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx

from topogen.integrated_graph import load_from_json


def _find_high_degree_cut(graph: nx.Graph, k: int, pool_size: int = 20) -> list:
    """Search up to C(pool_size, k) cuts, testing views without copying the graph."""
    pool = sorted(graph, key=lambda node: (-graph.degree(node), node))[:pool_size]
    for cut in combinations(pool, k):
        remaining = graph.subgraph(graph.nodes - set(cut))
        if len(remaining) <= 1 or not nx.is_connected(remaining):
            return list(cut)
    return []


def analyze_graph(json_path: Path, top_n: int | None = None) -> None:
    corridors, crs = load_from_json(json_path)
    if not corridors:
        raise ValueError("Cannot analyze an empty corridor graph")
    graph = nx.Graph(corridors)
    degrees = dict(corridors.degree())
    order = sorted(graph, key=lambda node: (-degrees[node], graph.nodes[node]["name"]))
    connectivity = nx.node_connectivity(graph) if len(graph) > 1 else 0
    print(f"File: {json_path}\nCRS: {crs}")
    print(
        f"Graph: nodes={len(graph):,}, corridors={corridors.number_of_edges():,}, connected={nx.is_connected(graph)}"
    )
    print(f"Global node connectivity (k): {connectivity}")

    def report(title: str, nodes: list) -> None:
        print(f"\n{title}:")
        for node in nodes:
            attrs = graph.nodes[node]
            print(
                f"  {attrs['name']:35s} degree={degrees[node]:2d} id={attrs['metro_id']}"
            )

    report("Metros ordered by corridor degree", order[:top_n])
    if connectivity:
        cut = nx.minimum_node_cut(graph)
        report(
            f"Minimum vertex cut (size {len(cut)})",
            [node for node in order if node in cut],
        )
        high_degree = _find_high_degree_cut(graph, connectivity)
        if high_degree:
            report(
                f"High-degree vertex cut (size {connectivity}, top-20 pool)",
                high_degree,
            )
        else:
            print(f"\nNo size-{connectivity} cut found in the top-20 pool.")
    else:
        print("\nMinimum vertex cut: none (graph disconnected or trivial)")
    articulation = set(nx.articulation_points(graph))
    if articulation:
        report(
            "Articulation points", [node for node in order if node in articulation][:20]
        )
    cores = nx.core_number(graph)
    maximum = max(cores.values())
    members = [node for node in order if cores[node] == maximum]
    report(f"Max k-core (k={maximum}, size={len(members)})", members[:20])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()
    if args.top is not None and args.top <= 0:
        parser.error("--top must be positive")
    analyze_graph(args.json_path, args.top)


if __name__ == "__main__":
    main()
