"""Graph serialization functions for node-link and edge-list formats."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ngraph.graph.strict_multidigraph import NodeID, StrictMultiDiGraph


def graph_to_node_link(graph: StrictMultiDiGraph) -> Dict[str, Any]:
    """Convert a StrictMultiDiGraph into a node-link dict representation.

    This representation is suitable for JSON serialization (e.g., for D3.js or Nx formats).

    The returned dict has the following structure:
        {
            "graph": { ... top-level graph attributes ... },
            "nodes": [
                {"id": node_id, "attr": { ... node attributes ... }},
                ...
            ],
            "links": [
                {
                    "source": <indexed_node>,
                    "target": <indexed_node>,
                    "key": <edge_id>,
                    "attr": { ... edge attributes ... }
                },
                ...
            ]
        }

    Args:
        graph: The StrictMultiDiGraph to convert.

    Returns:
        A dict containing the 'graph' attributes, list of 'nodes', and list of 'links'.
    """
    # Get nodes with their attributes and enforce a stable ordering.
    node_dict = graph.get_nodes()
    node_list = list(node_dict.keys())
    node_map = {node_id: i for i, node_id in enumerate(node_list)}

    return {
        "graph": dict(graph.graph),
        "nodes": [
            {"id": node_id, "attr": dict(node_dict[node_id])} for node_id in node_list
        ],
        "links": [
            {
                "source": node_map[src],
                "target": node_map[dst],
                "key": edge_id,
                "attr": dict(edge_attrs),
            }
            for edge_id, (src, dst, _, edge_attrs) in graph.get_edges().items()
        ],
    }


def node_link_to_graph(data: Dict[str, Any]) -> StrictMultiDiGraph:
    """Reconstruct a StrictMultiDiGraph from its node-link dict representation.

    Expected input format:
        {
            "graph": { ... graph attributes ... },
            "nodes": [
                {"id": <node_id>, "attr": { ... node attributes ... }},
                ...
            ],
            "links": [
                {
                    "source": <indexed_node>,
                    "target": <indexed_node>,
                    "key": <edge_id>,
                    "attr": { ... edge attributes ... }
                },
                ...
            ]
        }

    Args:
        data: A dict representing the node-link structure.

    Returns:
        A StrictMultiDiGraph reconstructed from the provided data.

    Raises:
        KeyError: If required keys (e.g., "id" or "attr" on nodes) are missing.
    """
    # Create the graph with the top-level attributes.
    graph_attrs = data.get("graph", {})
    graph = StrictMultiDiGraph(**graph_attrs)

    # Build a mapping from integer indices to original node IDs.
    node_map: Dict[int, NodeID] = {}
    for idx, node_obj in enumerate(data.get("nodes", [])):
        node_id = node_obj["id"]
        graph.add_node(node_id, **node_obj["attr"])
        node_map[idx] = node_id

    # Add edges using the index mapping.
    for edge_obj in data.get("links", []):
        src_id = node_map[edge_obj["source"]]
        dst_id = node_map[edge_obj["target"]]
        edge_key = edge_obj.get("key", None)
        edge_attr = edge_obj.get("attr", {})
        graph.add_edge(src_id, dst_id, key=edge_key, **edge_attr)

    return graph


def edgelist_to_graph(
    lines: Iterable[str],
    columns: List[str],
    separator: str = " ",
    graph: Optional[StrictMultiDiGraph] = None,
    source: str = "src",
    target: str = "dst",
    key: str = "key",
) -> StrictMultiDiGraph:
    """Build or update a StrictMultiDiGraph from an edge list.

    Each line in the input is split by the specified separator into tokens. These tokens
    are mapped to column names provided in `columns`. The tokens corresponding to `source`
    and `target` become the node IDs. If a `key` column exists, its token is used as the edge
    ID; remaining tokens are added as edge attributes.

    Args:
        lines: An iterable of strings, each representing one edge.
        columns: A list of column names, e.g. ["src", "dst", "cost"].
        separator: The separator used to split each line (default is a space).
        graph: An existing StrictMultiDiGraph to update; if None, a new graph is created.
        source: The column name for the source node ID.
        target: The column name for the target node ID.
        key: The column name for a custom edge ID (if present).

    Returns:
        The updated (or newly created) StrictMultiDiGraph.

    Raises:
        RuntimeError: If a line does not match the expected number of columns.
    """
    if graph is None:
        graph = StrictMultiDiGraph()

    for line in lines:
        # Remove only newline characters.
        line = line.rstrip("\r\n")
        tokens = line.split(separator)
        if len(tokens) != len(columns):
            raise RuntimeError(
                f"Line '{line}' does not match expected columns {columns} (token count mismatch)."
            )

        line_dict = dict(zip(columns, tokens, strict=False))
        src_id = line_dict[source]
        dst_id = line_dict[target]
        edge_key = line_dict.get(key, None)

        # All tokens not corresponding to source, target, or key become edge attributes.
        attr_dict = {
            k: v for k, v in line_dict.items() if k not in (source, target, key)
        }

        # Ensure nodes exist since StrictMultiDiGraph does not auto-create nodes.
        if src_id not in graph:
            graph.add_node(src_id)
        if dst_id not in graph:
            graph.add_node(dst_id)

        graph.add_edge(src_id, dst_id, key=edge_key, **attr_dict)

    return graph


def graph_to_edgelist(
    graph: StrictMultiDiGraph,
    columns: Optional[List[str]] = None,
    separator: str = " ",
    source_col: str = "src",
    target_col: str = "dst",
    key_col: str = "key",
) -> List[str]:
    """Convert a StrictMultiDiGraph into an edge-list text representation.

    Each line in the output represents one edge with tokens joined by the given separator.
    By default, the output columns are:
        [source_col, target_col, key_col] + sorted(edge_attribute_names)

    If an explicit list of columns is provided, those columns (in that order) are used,
    and any missing values are output as an empty string.

    Args:
        graph: The StrictMultiDiGraph to export.
        columns: Optional list of column names. If None, they are auto-generated.
        separator: The string used to join tokens (default is a space).
        source_col: The column name for the source node (default "src").
        target_col: The column name for the target node (default "dst").
        key_col: The column name for the edge key (default "key").

    Returns:
        A list of strings, each representing one edge in the specified column format.
    """
    edge_dicts: List[Dict[str, str]] = []
    all_attr_keys = set()

    # Build a list of dicts for each edge.
    for edge_id, (src, dst, _, edge_attrs) in graph.get_edges().items():
        # Use "is not None" to correctly handle edge keys such as 0.
        key_val = str(edge_id) if edge_id is not None else ""
        row = {
            source_col: str(src),
            target_col: str(dst),
            key_col: key_val,
        }
        for attr_key, attr_val in edge_attrs.items():
            row[attr_key] = str(attr_val)
            all_attr_keys.add(attr_key)
        edge_dicts.append(row)

    # Auto-generate columns if not provided.
    if columns is None:
        sorted_attr_keys = sorted(all_attr_keys)
        columns = [source_col, target_col, key_col] + sorted_attr_keys

    lines: List[str] = []
    for row_dict in edge_dicts:
        # For each specified column, output the corresponding value or an empty string if absent.
        tokens = [row_dict.get(col, "") for col in columns]
        lines.append(separator.join(tokens))

    return lines
