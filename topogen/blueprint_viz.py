"""Blueprint visualization helpers.

This module provides small, focused helpers to construct visualization-ready
data for abstract (group-level) topologies and to extract a concrete view for a
single site from the expanded network graph. Keeping this logic separate from
map rendering makes it easier to maintain and test.

The abstract builder expands group selectors that use variable placeholders
like ``G{gu}/G{gu}_r{ru}`` using the adjacency ``expand_vars`` and
``expansion_mode`` fields from the blueprint. Edges are aggregated at the
group level. Intra-group meshes are summarized into per-node labels instead of
rendering self-loops that are often not visible in standard layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx


@dataclass(frozen=True, slots=True)
class AbstractView:
    """Container for the abstract view.

    Attributes:
        graph: MultiDiGraph with group nodes and inter-group edges.
        node_labels: Mapping of node -> label text to draw.
        edge_labels: Mapping of (u, v, key) -> label text to draw.
        self_loops: List of (group, label) describing intra-group connectivity
            to be rendered as self-loop markers in the abstract view.
    """

    graph: nx.MultiDiGraph
    node_labels: dict[str, str]
    edge_labels: dict[tuple[str, str, int], str]
    self_loops: list[tuple[str, str]]


def _first_path_component(selector: str) -> str:
    """Return the first path component of a selector.

    Examples:
        "/G{g}" -> "G{g}"; "G1/G1_r1" -> "G1".
    """

    s = str(selector or "").strip()
    if s.startswith("/"):
        s = s[1:]
    return s.split("/", 1)[0]


def _extract_vars(template: str) -> list[str]:
    """Extract variable names inside ``{}`` from ``template`` in order."""

    import re as _re

    return _re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template)


def _iter_assignments(
    expand_vars: dict[str, Iterable[Any]] | None,
    var_names: list[str],
    expansion_mode: str | None,
) -> list[dict[str, Any]]:
    """Create a list of variable assignments.

    For ``zip`` mode, align by index across the specified ``var_names``.
    For ``product`` (or anything else), produce the cartesian product.

    Empty ``var_names`` yields a single empty assignment.
    """

    if not var_names:
        return [{}]

    ev: dict[str, list[Any]] = {
        name: list((expand_vars or {}).get(name, [])) for name in var_names
    }

    # If any variable lacks values, nothing can be expanded → single empty assignment
    if any(len(v) == 0 for v in ev.values()):
        return [{}]

    if (expansion_mode or "zip").lower() == "zip":
        # Align by the minimum available length to avoid IndexError if mismatched
        n = min(len(v) for v in ev.values())
        result: list[dict[str, Any]] = []
        for i in range(n):
            result.append({name: ev[name][i] for name in var_names})
        return result

    # Cartesian product
    from itertools import product

    keys = list(var_names)
    vals = [ev[k] for k in keys]
    return [dict(zip(keys, comb, strict=False)) for comb in product(*vals)]


def _subst(template: str, values: dict[str, Any]) -> str:
    """Substitute ``{var}`` placeholders in ``template`` with ``values``.

    Unknown variables are left untouched to avoid accidental collapsing to
    empty strings.
    """

    out = template
    for name in _extract_vars(template):
        val = values.get(name)
        if val is not None:
            out = out.replace("{" + name + "}", str(val))
    return out


def build_abstract_view(
    blueprint_def: dict[str, Any], *, include_self_loops: bool = True
) -> AbstractView:
    """Construct the abstract group-level view from a blueprint definition.

    The returned graph contains a node for each group in ``groups`` and an
    inter-group edge for every adjacency expansion. Intra-group 'mesh' entries
    are summarized into node labels to avoid invisible self-loops.

    Args:
        blueprint_def: Mapping with ``groups`` and ``adjacency``.

    Returns:
        AbstractView with graph and labels ready for drawing.

    Raises:
        ValueError: If the input mapping is missing required keys.
    """

    if not isinstance(blueprint_def, dict):
        raise ValueError("blueprint_def must be a mapping")
    groups = blueprint_def.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("blueprint_def must include a 'groups' mapping")

    abstract = nx.MultiDiGraph()

    # Create nodes with base labels
    node_labels: dict[str, str] = {}
    for gname, gdef in groups.items():
        count = int((gdef or {}).get("node_count", 0))
        role = str(((gdef or {}).get("attrs", {}) or {}).get("role", ""))
        lbl = f"{gname}\nN={count}" + (f"\nrole={role}" if role else "")
        abstract.add_node(gname)
        node_labels[gname] = lbl

    # Helper to attach a small extra line under node label
    def _append_node_note(group_name: str, text: str) -> None:
        base = node_labels.get(group_name, group_name)
        if text and text not in base:
            node_labels[group_name] = base + f"\n{text}"

    edge_labels: dict[tuple[str, str, int], str] = {}
    self_loops: list[tuple[str, str]] = []

    for adj in blueprint_def.get("adjacency", []) or []:
        src_sel = str(adj.get("source", ""))
        dst_sel = str(adj.get("target", ""))
        pattern = str(adj.get("pattern", ""))
        expand_vars = adj.get("expand_vars") or {}
        expansion_mode = str(adj.get("expansion_mode", "zip") or "zip")

        lp = adj.get("link_params", {}) or {}
        attrs = lp.get("attrs", {}) or {}
        cap_val = attrs.get("target_capacity", lp.get("capacity"))
        cap_str = f"{float(cap_val):,.0f}" if cap_val is not None else ""
        edge_label_text = (pattern if pattern else "") + (
            f"\n{cap_str}" if cap_str else ""
        )

        # Extract only the group part of selectors and expand variables jointly
        src_group_tpl = _first_path_component(src_sel)
        dst_group_tpl = _first_path_component(dst_sel)

        involved_vars = list(
            dict.fromkeys(_extract_vars(src_group_tpl) + _extract_vars(dst_group_tpl))
        )

        assignments = _iter_assignments(expand_vars, involved_vars, expansion_mode)

        # If there are no variables, a single assignment will be produced
        # producing a single (possibly identical) group pair.
        pairs: list[tuple[str, str]] = []
        for assign in assignments:
            su = _subst(src_group_tpl, assign)
            sv = _subst(dst_group_tpl, assign)
            pairs.append((su, sv))

        # If this is a same-group mesh, surface it as a node note
        if all(u == v for u, v in pairs):
            # Intra-group adjacency. Keep label note and optionally emit self-loop.
            for u, _v in pairs:
                _append_node_note(
                    u, (pattern if pattern else "") + (f" {cap_str}" if cap_str else "")
                )
                if include_self_loops:
                    # One loop per group is sufficient visually; deduplicate.
                    if (u, edge_label_text) not in self_loops:
                        self_loops.append((u, edge_label_text))
            continue

        # Add inter-group edges, deduplicated by unordered pair while preserving
        # a clean label. We still add as MultiDiGraph to keep potential
        # multiplicity if future blueprints require it.
        seen: set[tuple[str, str]] = set()
        for u, v in pairs:
            if u == v:
                # Skip self-loops in abstract view
                continue
            _key_pair = (u, v) if (u, v) not in seen else None
            # For directionality, keep as given. Also add the reverse marker to
            # the set so we do not repeat the same undirected pair when inputs
            # contain both directions.
            if (u, v) in seen or (v, u) in seen:
                continue
            seen.add((u, v))
            seen.add((v, u))
            k = abstract.add_edge(u, v)
            edge_labels[(u, v, k)] = edge_label_text

    return AbstractView(
        graph=abstract,
        node_labels=node_labels,
        edge_labels=edge_labels,
        self_loops=self_loops,
    )


def collect_concrete_site(
    net: Any, selected_site_path: str
) -> tuple[list[str], dict[str, tuple[float, float]], list[tuple[str, str, float]]]:
    """Collect concrete intra-site nodes, simple positions, and links.

    This mirrors existing logic in ``visualization.export_blueprint_diagram`` but
    is provided as a small re-usable unit for future rendering backends.

    Returns:
        node_names: List of node names under the site.
        node_positions: Deterministic circular layout positions in unit circle.
        internal_links: List of (src, dst, capacity) restricted to the site.
    """

    def _site_head(name: str) -> str:
        parts = str(name).split("/", 2)
        return "/".join(parts[:2]) if len(parts) >= 2 else str(name)

    # Collect internal nodes strictly under the selected site
    internal_nodes: list[str] = []
    for node in getattr(net, "nodes", {}).values():  # type: ignore[union-attr]
        try:
            nname = str(node.name)
        except Exception:
            nname = str(node)
        if _site_head(nname) == selected_site_path:
            internal_nodes.append(nname)

    # Group heuristic based on local suffix before first digit
    import math as _m
    import re as _re

    import numpy as _np

    def _group_of(node_name: str) -> str:
        tail = node_name.split("/", 2)[-1]
        m = _re.match(r"([A-Za-z_]+)", tail)
        return m.group(1) if m else tail

    groups_concrete: dict[str, list[str]] = {}
    for n in internal_nodes:
        g = _group_of(n)
        groups_concrete.setdefault(g, []).append(n)

    rng = _np.random.default_rng(7)
    K = max(1, len(groups_concrete))
    R_group = 1.0
    r_node = 0.25
    group_angles = {
        g: (2.0 * _m.pi * i) / float(K) for i, g in enumerate(sorted(groups_concrete))
    }
    node_pos: dict[str, tuple[float, float]] = {}
    for g, angle in group_angles.items():
        gx = R_group * _m.cos(angle)
        gy = R_group * _m.sin(angle)
        members = groups_concrete[g]
        n = len(members)
        if n == 1:
            node_pos[members[0]] = (gx, gy)
        else:
            for j, nn in enumerate(sorted(members)):
                theta = (2.0 * _m.pi * j) / float(n)
                rr = r_node * (0.85 + 0.3 * float(rng.random()))
                node_pos[nn] = (gx + rr * _m.cos(theta), gy + rr * _m.sin(theta))

    # Collect internal links only (strictly intra-site visualization)
    internal_links: list[tuple[str, str, float]] = []
    for link in getattr(net, "links", {}).values():  # type: ignore[union-attr]
        try:
            s_obj = link.source
            t_obj = link.target
            s = str(getattr(s_obj, "name", s_obj))
            t = str(getattr(t_obj, "name", t_obj))
            cap = float(getattr(link, "capacity", 0.0) or 0.0)
        except Exception:
            continue
        if _site_head(s) == selected_site_path and _site_head(t) == selected_site_path:
            internal_links.append((s, t, cap))

    return internal_nodes, node_pos, internal_links
