"""Build abstract blueprint views and concrete site layouts for rendering.

Abstract views use blueprint ``nodes``, ``links``, and ``expand.vars``/``expand.mode``
fields. They aggregate links by group and record intra-group links as labels
and optional self-loop markers.
"""

from __future__ import annotations

import re
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
        "/G${g}" -> "G${g}"; "G1/G1_r1" -> "G1".
    """

    s = str(selector or "").strip()
    if s.startswith("/"):
        s = s[1:]
    return s.split("/", 1)[0]


# NetGraph placeholder syntax: ``$var`` or ``${var}``.
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _extract_vars(template: str) -> list[str]:
    """Return ``$var``/``${var}`` names in ``template`` in order."""

    return [m.group(1) or m.group(2) for m in _VAR_PATTERN.finditer(template)]


def _iter_assignments(
    expand_vars: dict[str, Iterable[Any]] | None,
    var_names: list[str],
    expansion_mode: str | None,
) -> list[dict[str, Any]]:
    """Expand variable values by index (default ``zip``) or Cartesian product.

    Zip stops at the shortest input. No variables, or a variable without values,
    returns one empty assignment.
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

    from itertools import product

    keys = list(var_names)
    vals = [ev[k] for k in keys]
    return [dict(zip(keys, comb, strict=False)) for comb in product(*vals)]


def _subst(template: str, values: dict[str, Any]) -> str:
    """Replace known ``$var``/``${var}`` placeholders, leaving unknown ones intact."""

    def _replace(match: re.Match[str]) -> str:
        val = values.get(match.group(1) or match.group(2))
        return match.group(0) if val is None else str(val)

    return _VAR_PATTERN.sub(_replace, template)


def build_abstract_view(
    blueprint_def: dict[str, Any], *, include_self_loops: bool = True
) -> AbstractView:
    """Build a group-level view from blueprint ``nodes`` and ``links``.

    Within each link rule, inter-group edges are deduplicated by unordered pair.
    Same-group rules become node notes and, if requested, self-loop markers.
    Raises ValueError when ``nodes`` is missing or is not a mapping.
    """

    if not isinstance(blueprint_def, dict):
        raise ValueError("blueprint_def must be a mapping")
    groups = blueprint_def.get("nodes")
    if not isinstance(groups, dict):
        raise ValueError("blueprint_def must include a 'nodes' mapping")

    abstract = nx.MultiDiGraph()

    node_labels: dict[str, str] = {}
    for gname, gdef in groups.items():
        count = int((gdef or {}).get("count", 0))
        role = str(((gdef or {}).get("attrs", {}) or {}).get("role", ""))
        lbl = f"{gname}\nN={count}" + (f"\nrole={role}" if role else "")
        abstract.add_node(gname)
        node_labels[gname] = lbl

    def _append_node_note(group_name: str, text: str) -> None:
        base = node_labels.get(group_name, group_name)
        if text and text not in base:
            node_labels[group_name] = base + f"\n{text}"

    edge_labels: dict[tuple[str, str, int], str] = {}
    self_loops: list[tuple[str, str]] = []

    for adj in blueprint_def.get("links", []) or []:
        src_sel = str(adj.get("source", ""))
        dst_sel = str(adj.get("target", ""))
        pattern = str(adj.get("pattern", ""))
        expand_block = adj.get("expand") or {}
        expand_vars = expand_block.get("vars") if isinstance(expand_block, dict) else {}
        expand_vars = expand_vars or {}
        expansion_mode = str(
            (expand_block.get("mode") if isinstance(expand_block, dict) else None)
            or "zip"
        )

        attrs = adj.get("attrs", {}) or {}
        cap_val = attrs.get("target_capacity", adj.get("capacity"))
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

        pairs: list[tuple[str, str]] = []
        for assign in assignments:
            su = _subst(src_group_tpl, assign)
            sv = _subst(dst_group_tpl, assign)
            pairs.append((su, sv))

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

        seen: set[tuple[str, str]] = set()
        for u, v in pairs:
            if u == v:
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
    """Return node names, layout positions, and internal links for one site.

    Positions cluster nodes by local name prefix around a unit circle, using
    seeded jitter. Links are ``(source, target, capacity)`` tuples.
    """

    def _site_head(name: str) -> str:
        parts = str(name).split("/", 2)
        return "/".join(parts[:2]) if len(parts) >= 2 else str(name)

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
