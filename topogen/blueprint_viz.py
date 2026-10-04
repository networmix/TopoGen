"""Aggregate and position the devices already expanded by NetGraph."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
from ngraph import Network


@dataclass(frozen=True, slots=True)
class AbstractView:
    graph: nx.MultiDiGraph
    node_labels: dict[str, str]
    edge_labels: dict[tuple[str, str, int], str]
    self_loops: list[tuple[str, str]]


def build_abstract_view(
    net: Network, selected_site_path: str, *, include_self_loops: bool = True
) -> AbstractView:
    """Aggregate actual devices by blueprint parent, and links by group pair.

    Counts and capacity totals come from the same resolved network as the concrete
    panel. Nested blueprints, selector dictionaries, and variable expansion need
    no second DSL interpreter here. Self-loop labels show internal capacity.
    """
    nodes, _, links = collect_concrete_site(net, selected_site_path)
    prefix = selected_site_path + "/"
    parents = {
        node: node.removeprefix(prefix).rsplit("/", 1)[0]
        if "/" in node.removeprefix(prefix)
        else "devices"
        for node in nodes
    }
    members: dict[str, list[str]] = defaultdict(list)
    for node, parent in parents.items():
        members[parent].append(node)
    graph = nx.MultiDiGraph()
    labels = {}
    for group, names in sorted(members.items()):
        roles = sorted(
            {
                net.nodes[name].attrs["role"]
                for name in names
                if "role" in net.nodes[name].attrs
            }
        )
        graph.add_node(group)
        labels[group] = f"{group}\nN={len(names)}" + (
            f"\nrole={','.join(roles)}" if roles else ""
        )
    capacities: dict[tuple[str, str], float] = defaultdict(float)
    for source, target, capacity in links:
        a, b = sorted((parents[source], parents[target]))
        capacities[a, b] += capacity
    edges = {}
    loops = []
    for (a, b), capacity in sorted(capacities.items()):
        text = f"{capacity:,.0f} Gbps"
        if a == b:
            labels[a] += f"\ninternal: {text}"
            if include_self_loops:
                loops.append((a, text))
        else:
            key = graph.add_edge(a, b)
            edges[a, b, key] = text
    return AbstractView(graph, labels, edges, loops)


def collect_concrete_site(
    net: Network, selected_site_path: str
) -> tuple[list[str], dict[str, tuple[float, float]], list[tuple[str, str, float]]]:
    """Return node names, layout positions, and internal links for one site.

    Positions use evenly spaced columns for actual blueprint parent groups.
    Links are ``(source, target, capacity)`` tuples.
    """

    def _site_head(name: str) -> str:
        parts = str(name).split("/", 2)
        return "/".join(parts[:2]) if len(parts) >= 2 else str(name)

    internal_nodes: list[str] = []
    for node in net.nodes.values():
        nname = node.name
        if _site_head(nname) == selected_site_path:
            internal_nodes.append(nname)

    groups_concrete: dict[str, list[str]] = {}
    for name in internal_nodes:
        parent = name.rsplit("/", 1)[0]
        groups_concrete.setdefault(parent, []).append(name)
    node_pos: dict[str, tuple[float, float]] = {}
    for column, group in enumerate(sorted(groups_concrete)):
        members = sorted(groups_concrete[group])
        for row, name in enumerate(members):
            node_pos[name] = (float(column), row - (len(members) - 1) / 2)

    internal_links: list[tuple[str, str, float]] = []
    for link in net.links.values():
        s, t, cap = link.source, link.target, link.capacity
        if _site_head(s) == selected_site_path and _site_head(t) == selected_site_path:
            internal_links.append((s, t, cap))

    return internal_nodes, node_pos, internal_links
