"""Render current corridor, site, and blueprint contracts without format guessing."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import contextily as cx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle
from ngraph import Network

from topogen.metro_clusters import MetroCluster


@contextmanager
def _figure(output_path: Path, *, dpi: int, figsize=(12, 8), panels=1):
    """Close figures on every path; propagate drawing and output failures."""
    fig, axes = plt.subplots(1, panels, figsize=figsize)
    try:
        yield fig, axes
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, format="jpeg", bbox_inches="tight")
    finally:
        plt.close(fig)


def _boundary(ax, path: Path, target_crs: str) -> None:
    from topogen.geo_utils import create_conus_mask

    create_conus_mask(path, target_crs).plot(
        ax=ax, edgecolor="#777777", facecolor="#f6f7f9", linewidth=0.4
    )


def export_cluster_map(
    centroids: np.ndarray, output_path: Path, conus_boundary_path: Path, target_crs: str
) -> None:
    """Draw projected metro points over the configured CONUS boundary and basemap."""
    if len(centroids) == 0:
        raise ValueError("Cannot create map: no centroids provided")
    with _figure(output_path, dpi=150) as (_, ax):
        _boundary(ax, conus_boundary_path, target_crs)
        ax.scatter(centroids[:, 0], centroids[:, 1], s=24, color="#c42c41")
        cx.add_basemap(ax, crs=target_crs, attribution=False, alpha=0.3)
        ax.set_title(f"Metro clusters: {len(centroids)}")
        ax.set_aspect("equal")
        ax.set_axis_off()


def export_integrated_graph_map(
    metros: list[MetroCluster],
    graph: nx.MultiGraph,
    output_path: Path,
    conus_boundary_path: Path,
    target_crs: str,
    *,
    use_real_geometry: bool = False,
    dpi: int = 150,
) -> None:
    """Draw each corridor using road geometry or its metro endpoints."""
    if not metros:
        raise ValueError("Cannot create map: no metros provided")
    if not graph.is_multigraph():
        raise TypeError("Expected a corridor MultiGraph")
    by_id = {metro.metro_id: metro for metro in metros}
    with _figure(output_path, dpi=dpi) as (_, ax):
        _boundary(ax, conus_boundary_path, target_crs)
        for _, _, data in graph.edges(data=True):
            points = (
                data["geometry"]
                if use_real_geometry
                else [
                    by_id[data["metro_a"]].coordinates,
                    by_id[data["metro_b"]].coordinates,
                ]
            )
            xs, ys = zip(*points, strict=False)
            ax.plot(xs, ys, color="#397eb5", linewidth=1.1, alpha=0.75)
        for metro in metros:
            x, y = metro.coordinates
            ax.add_patch(
                Circle(
                    (x, y),
                    metro.radius_km * 1000,
                    fill=False,
                    color="#999999",
                    linewidth=0.6,
                )
            )
            ax.scatter([x], [y], s=25, color="#c42c41", zorder=3)
            ax.annotate(
                metro.name,
                (x, y),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
            )
        cx.add_basemap(ax, crs=target_crs, attribution=False, alpha=0.3)
        ax.set_title(f"{len(metros)} metros · {graph.number_of_edges()} corridors")
        ax.set_aspect("equal")
        ax.set_axis_off()


def export_site_graph_map(
    G: nx.MultiGraph,
    output_path: Path,
    *,
    figure_size: tuple[int, int] = (14, 10),
    metro_scale: float = 1.0,
    dpi: int = 300,
    target_crs: str | None = None,
) -> None:
    """Draw positioned sites and label total site-edge capacities in Gbps."""
    if not G:
        raise ValueError("Cannot create site graph map: graph is empty")
    positions = {
        node: (data["pos_x"], data["pos_y"]) for node, data in G.nodes(data=True)
    }
    metros = {}
    for _, data in G.nodes(data=True):
        metros[data["metro_idx"]] = (
            data["center_x"],
            data["center_y"],
            data["radius_m"],
            data["metro_name"],
        )
    with _figure(output_path, dpi=dpi, figsize=figure_size) as (_, ax):
        for _idx, (x, y, radius, name) in sorted(metros.items()):
            ax.add_patch(
                Circle(
                    (x, y),
                    radius * metro_scale,
                    fill=False,
                    color="#999999",
                    linewidth=0.7,
                )
            )
            ax.annotate(
                name,
                (x, y + radius * metro_scale),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        capacities = {}
        for u, v, data in G.edges(data=True):
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            ax.plot([x0, x1], [y0, y1], color="#397eb5", linewidth=0.8, alpha=0.7)
            if data["link_type"] == "inter_metro_corridor":
                pair = tuple(sorted((G.nodes[u]["metro_idx"], G.nodes[v]["metro_idx"])))
                capacities[pair] = capacities.get(pair, 0.0) + data["target_capacity"]
        for (u, v), capacity in capacities.items():
            x0, y0 = metros[u][:2]
            x1, y1 = metros[v][:2]
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                f"{capacity:,.0f}",
                fontsize=7,
                ha="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
        for node, data in G.nodes(data=True):
            x, y = positions[node]
            ax.scatter(
                [x],
                [y],
                marker="s" if data["site_kind"] == "dc" else "o",
                color="#c42c41",
                s=30,
                zorder=3,
            )
        if target_crs is not None:
            cx.add_basemap(ax, crs=target_crs, attribution=False, alpha=0.3)
        ax.set_title(
            f"{len(G)} sites · {G.number_of_edges()} adjacencies · corridor totals in Gbps"
        )
        ax.set_aspect("equal")
        ax.set_axis_off()


def export_blueprint_diagram(
    blueprint_name: str,
    net: Network,
    selected_site_path: str,
    output_path: Path,
    *,
    dpi: int = 300,
    figure_size: tuple[int, int] = (14, 6),
    seed: int = 7,
) -> None:
    """Draw blueprint groups and the already expanded devices of one site."""
    from .blueprint_viz import build_abstract_view, collect_concrete_site

    view = build_abstract_view(net, selected_site_path, include_self_loops=True)
    nodes, positions, links = collect_concrete_site(net, selected_site_path)
    if not nodes:
        raise ValueError(f"No expanded nodes found under site '{selected_site_path}'")
    with _figure(output_path, dpi=dpi, figsize=figure_size, panels=2) as (fig, axes):
        abstract, concrete = axes
        layout = nx.spring_layout(view.graph, seed=seed)
        nx.draw_networkx_nodes(
            view.graph, layout, node_size=900, node_color="#f0f0ff", ax=abstract
        )
        nx.draw_networkx_labels(
            view.graph, layout, labels=view.node_labels, font_size=8, ax=abstract
        )
        nx.draw_networkx_edges(
            view.graph, layout, width=1.2, edge_color="#666666", ax=abstract
        )
        nx.draw_networkx_edge_labels(
            view.graph, layout, edge_labels=view.edge_labels, font_size=7, ax=abstract
        )
        for node, label in view.self_loops:
            x, y = layout[node]
            abstract.add_patch(Circle((x, y), 0.15, fill=False, color="#888888"))
            abstract.text(x + 0.16, y + 0.16, label, fontsize=7)
        for source, target, _ in links:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            concrete.plot([x0, x1], [y0, y1], color="#397eb5", linewidth=0.8)
        for node in nodes:
            x, y = positions[node]
            concrete.scatter([x], [y], s=60, color="#c42c41")
            concrete.text(
                x, y, node.split("/", 2)[-1], fontsize=6, ha="center", va="bottom"
            )
        abstract.set_title(f"Abstract: {blueprint_name}")
        concrete.set_title(f"Concrete: {selected_site_path}")
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"Blueprint {blueprint_name}")
