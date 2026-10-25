"""Publish artifacts after validation, one complete file at a time."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING

import networkx as nx

from topogen.config import TopologyConfig
from topogen.context import RunContext
from topogen.library import thaw
from topogen.naming import site_edge_id, site_of

from .network import save_site_graph_json
from .publication import write_atomic
from .records import Build

if TYPE_CHECKING:
    from topogen.validation import Validation


def write_artifacts(
    build: Build, validation: Validation, config: TopologyConfig, context: RunContext
) -> None:
    """Publish artifacts in order; an error retains earlier completed files."""
    if validation.issues or validation.prepared is None:
        raise ValueError("Cannot publish artifacts for an invalid scenario")
    network = validation.prepared.scenario.network
    graph = nx.MultiGraph()
    for site, attrs in build.site_plan.nodes(data=True):
        graph.add_node(site, **thaw(attrs))
    for u, v, key, attrs in build.site_plan.edges(keys=True, data=True):
        edge = site_edge_id(u, v, key)
        data = thaw(attrs)
        data["base_capacity"] = data["target_capacity"] = float(
            build.sizing.capacity[edge]
        )
        data["capacity"] = float(build.hardware[edge].capacity)
        assigned = build.hardware[edge]
        if assigned.source_optic is not None:
            data["hardware"] = {
                "source": {
                    "component": assigned.source_optic,
                    "count": float(assigned.source_count),
                },
                "target": {
                    "component": assigned.target_optic,
                    "count": float(assigned.target_count),
                },
            }
        graph.add_edge(u, v, key=key, **data)
    output_dir, stem = context.output_dir, context.stem
    if context.debug_dir is not None:
        matrices = {
            m.name: [{**asdict(d), "volume": float(d.volume)} for d in m.demands]
            for m in build.traffic
        }
        payload = (
            json.dumps(
                {"model": config.traffic.model, "matrices": matrices},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        write_atomic(
            context.debug_dir / f"{stem}_traffic_debug.json",
            lambda path: path.write_text(payload) and None,
        )
    write_atomic(
        output_dir / f"{stem}_network_graph.json",
        lambda path: save_site_graph_json(
            graph, path, json_indent=config.output.formatting.json_indent
        ),
    )
    if config.visualization.export_site_graph:
        from topogen.visualization import export_site_graph_map

        write_atomic(
            output_dir / f"{stem}_site_graph.jpg",
            lambda path: export_site_graph_map(
                graph,
                path,
                dpi=config.visualization.dpi,
                target_crs=config.projection.target_crs,
            ),
        )
    if not config.visualization.export_blueprint_diagrams:
        return
    from topogen.visualization import export_blueprint_diagram

    attached: dict[str, float] = defaultdict(float)
    for link in network.links.values():
        for endpoint in (link.source, link.target):
            attached[site_of(endpoint)] += link.capacity
    representatives = {}
    for site, attrs in graph.nodes(data=True):
        blueprint = attrs["site_blueprint"]
        previous = representatives.get(blueprint)
        if previous is None or attached[site] > attached[previous]:
            representatives[blueprint] = site
    for blueprint, site in sorted(representatives.items()):
        write_atomic(
            output_dir / f"{stem}_blueprint_{blueprint}.jpg",
            lambda path, bp=blueprint, selected=site: export_blueprint_diagram(
                bp, network, selected, path, dpi=config.visualization.dpi
            ),
        )
