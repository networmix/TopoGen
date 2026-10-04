"""Optional artifacts from the already resolved site and device networks."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import networkx as nx
from ngraph import Network

from topogen.config import TopologyConfig
from topogen.context import RunContext

from .network import save_site_graph_json


def export_artifacts(
    graph: nx.MultiGraph,
    network: Network,
    config: TopologyConfig,
    context: RunContext,
    traffic_matrices: dict[str, Any],
) -> None:
    """Export requested files; errors propagate to the caller."""
    output_dir = context.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = context.stem
    if context.debug_dir is not None:
        context.debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = context.debug_dir / f"{stem}_traffic_debug.json"
        debug_path.write_text(
            json.dumps(
                {"model": config.traffic.model, "matrices": traffic_matrices}, indent=2
            )
            + "\n"
        )
    save_site_graph_json(
        graph,
        output_dir / f"{stem}_network_graph.json",
        json_indent=config.output.formatting.json_indent,
    )
    if config.visualization.export_site_graph:
        from topogen.visualization import export_site_graph_map

        export_site_graph_map(
            graph,
            output_dir / f"{stem}_site_graph.jpg",
            dpi=config.visualization.dpi,
            target_crs=config.projection.target_crs,
        )
    if not config.visualization.export_blueprint_diagrams:
        return

    from topogen.visualization import export_blueprint_diagram

    attached: dict[str, float] = defaultdict(float)
    for link in network.links.values():
        for endpoint in (link.source, link.target):
            attached["/".join(endpoint.split("/")[:2])] += link.capacity
    representatives: dict[str, str] = {}
    for site, attrs in graph.nodes(data=True):
        blueprint = attrs["site_blueprint"]
        previous = representatives.get(blueprint)
        if previous is None or attached[site] > attached[previous]:
            representatives[blueprint] = site
    for blueprint, site in sorted(representatives.items()):
        export_blueprint_diagram(
            blueprint,
            network,
            site,
            output_dir / f"{stem}_blueprint_{blueprint}.jpg",
            dpi=config.visualization.dpi,
        )
