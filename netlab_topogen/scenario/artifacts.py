"""Publish artifacts after validation, one complete file at a time."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

from netlab_topogen.config import TopologyConfig
from netlab_topogen.context import RunContext
from netlab_topogen.library import thaw
from netlab_topogen.naming import site_of

from .publication import write_atomic
from .records import Build

if TYPE_CHECKING:
    from netlab_topogen.validation import Validation


def write_artifacts(
    build: Build, validation: Validation, config: TopologyConfig, context: RunContext
) -> None:
    """Publish artifacts in order; an error retains earlier completed files."""
    if validation.issues or validation.prepared is None:
        raise ValueError("Cannot publish artifacts for an invalid scenario")
    network = validation.prepared.scenario.network
    graph = nx.MultiGraph()
    sites = {site.id: site for site in build.site_plan.sites}
    for site in build.site_plan.sites:
        metro = site.metro
        graph.add_node(
            site.id,
            metro_idx=metro.index,
            metro_name=metro.name,
            site_kind=site.kind,
            site_ordinal=site.ordinal,
            site_blueprint=site.blueprint,
            pos_x=site.position[0],
            pos_y=site.position[1],
            center_x=metro.x,
            center_y=metro.y,
            radius_m=1000.0 * metro.radius_km,
        )
    for link in build.site_plan.links:
        data = {
            "attrs": thaw(link.attrs),
            "link_type": link.kind,
            "base_capacity": float(build.sizing.capacity[link.id]),
            "target_capacity": float(build.sizing.capacity[link.id]),
            "cost": link.cost,
            "adjacency_id": link.adjacency_id,
            "distance_km": link.distance_km,
            "source_metro": sites[link.source].metro.name,
            "target_metro": sites[link.target].metro.name,
            "match": thaw(link.match),
            "role_pairs": list(link.role_pairs),
        }
        if link.kind == "inter_metro_corridor":
            data.update(
                risk_groups=list(link.risk_groups),
                euclidean_km=link.euclidean_km,
                detour_ratio=link.detour_ratio,
            )
        assigned = build.hardware[link.id]
        data["capacity"] = float(assigned.capacity)
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
        graph.add_edge(link.source, link.target, key=link.id.split("|", 2)[2], **data)
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
        from netlab_topogen.visualization import export_site_graph_map

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
    from netlab_topogen.visualization import export_blueprint_diagram

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


def save_site_graph_json(
    graph: nx.MultiGraph, path: Path, *, json_indent: int = 2
) -> None:
    """Serialize the local artifact projection, preserving parallel edge keys."""
    payload = {
        "graph_type": "site_network",
        "nodes": [
            {"id": str(name), **thaw(attrs)} for name, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": str(source),
                "target": str(target),
                "key": str(key),
                **thaw(attrs),
            }
            for source, target, key, attrs in graph.edges(keys=True, data=True)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=json_indent, allow_nan=False))
