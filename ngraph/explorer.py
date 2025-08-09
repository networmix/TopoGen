"""NetworkExplorer class for analyzing network hierarchy and structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ngraph.components import ComponentsLibrary
from ngraph.logging import get_logger
from ngraph.model.network import Network, Node

logger = get_logger(__name__)


@dataclass
class ExternalLinkBreakdown:
    """Holds stats for external links to a particular other subtree.

    Attributes:
        link_count (int): Number of links to that other subtree.
        link_capacity (float): Sum of capacities for those links.
    """

    link_count: int = 0
    link_capacity: float = 0.0


@dataclass
class TreeStats:
    """Aggregated statistics for a single tree node (subtree).

    Attributes:
        node_count (int): Total number of nodes in this subtree.
        internal_link_count (int): Number of internal links in this subtree.
        internal_link_capacity (float): Sum of capacities for those internal links.
        external_link_count (int): Number of external links from this subtree to another.
        external_link_capacity (float): Sum of capacities for those external links.
        external_link_details (Dict[str, ExternalLinkBreakdown]): Breakdown by other subtree path.
        total_cost (float): Cumulative cost (nodes + links).
        total_power (float): Cumulative power (nodes + links).
    """

    node_count: int = 0

    internal_link_count: int = 0
    internal_link_capacity: float = 0.0

    external_link_count: int = 0
    external_link_capacity: float = 0.0

    external_link_details: Dict[str, ExternalLinkBreakdown] = field(
        default_factory=dict
    )

    total_cost: float = 0.0
    total_power: float = 0.0


@dataclass(eq=False)
class TreeNode:
    """Represents a node in the hierarchical tree.

    Attributes:
        name (str): Name/label of this node.
        parent (Optional[TreeNode]): Pointer to the parent tree node.
        children (Dict[str, TreeNode]): Mapping of child name -> child TreeNode.
        subtree_nodes (Set[str]): Node names in the subtree (all nodes, ignoring disabled).
        active_subtree_nodes (Set[str]): Node names in the subtree (only enabled).
        stats (TreeStats): Aggregated stats for "all" view.
        active_stats (TreeStats): Aggregated stats for "active" (only enabled) view.
        raw_nodes (List[Node]): Direct Node objects at this hierarchy level.
    """

    name: str
    parent: Optional[TreeNode] = None

    children: Dict[str, TreeNode] = field(default_factory=dict)

    # "All" includes disabled nodes; "Active" excludes them.
    subtree_nodes: Set[str] = field(default_factory=set)
    active_subtree_nodes: Set[str] = field(default_factory=set)

    stats: TreeStats = field(default_factory=TreeStats)
    active_stats: TreeStats = field(default_factory=TreeStats)

    raw_nodes: List[Node] = field(default_factory=list)

    def __hash__(self) -> int:
        # Keep identity-based hashing so each node is unique in sets/dicts.
        return id(self)

    def add_child(self, child_name: str) -> TreeNode:
        """Ensure a child node named 'child_name' exists and return it."""
        if child_name not in self.children:
            child_node = TreeNode(name=child_name, parent=self)
            self.children[child_name] = child_node
        return self.children[child_name]

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0


class NetworkExplorer:
    """Provides hierarchical exploration of a Network, computing statistics in two modes:
    'all' (ignores disabled) and 'active' (only enabled).
    """

    def __init__(
        self,
        network: Network,
        components_library: Optional[ComponentsLibrary] = None,
    ) -> None:
        self.network = network
        self.components_library = components_library or ComponentsLibrary()

        self.root_node: Optional[TreeNode] = None

        # For quick lookups:
        self._node_map: Dict[str, TreeNode] = {}  # node_name -> deepest TreeNode
        self._path_map: Dict[str, TreeNode] = {}  # path -> TreeNode

        # Cache for ancestor sets:
        self._ancestors_cache: Dict[TreeNode, Set[TreeNode]] = {}

    @classmethod
    def explore_network(
        cls,
        network: Network,
        components_library: Optional[ComponentsLibrary] = None,
    ) -> NetworkExplorer:
        """Build a NetworkExplorer, constructing a tree plus 'all' and 'active' stats."""
        instance = cls(network, components_library)

        # 1) Build hierarchy
        instance.root_node = instance._build_hierarchy_tree()

        # 2) Compute subtree sets for "all" (ignoring disabled state)
        instance._compute_subtree_sets_all(instance.root_node)

        # 3) Compute subtree sets for "active" (excluding disabled)
        instance._compute_subtree_sets_active(instance.root_node)

        # 4) Build node & path maps
        instance._build_node_map(instance.root_node)
        instance._build_path_map(instance.root_node)

        # 5) Aggregate statistics (both 'all' and 'active')
        instance._compute_statistics()

        return instance

    def _build_hierarchy_tree(self) -> TreeNode:
        """Build a multi-level tree by splitting node names on '/'.
        Example: "dc1/plane1/ssw/ssw-1" => root/dc1/plane1/ssw/ssw-1
        """
        root = TreeNode(name="root")
        for nd in self.network.nodes.values():
            path_parts = nd.name.split("/")
            current = root
            for part in path_parts:
                current = current.add_child(part)
            current.raw_nodes.append(nd)
        return root

    def _compute_subtree_sets_all(self, node: TreeNode) -> Set[str]:
        """Recursively collect all node names (regardless of disabled) into subtree_nodes."""
        collected = set()
        for child in node.children.values():
            collected |= self._compute_subtree_sets_all(child)
        for nd in node.raw_nodes:
            collected.add(nd.name)
        node.subtree_nodes = collected
        return collected

    def _compute_subtree_sets_active(self, node: TreeNode) -> Set[str]:
        """Recursively collect enabled node names into active_subtree_nodes.
        A node is considered enabled if nd.attrs.get("disabled") is not truthy.
        """
        collected = set()
        for child in node.children.values():
            collected |= self._compute_subtree_sets_active(child)
        for nd in node.raw_nodes:
            if not nd.attrs.get("disabled"):
                collected.add(nd.name)
        node.active_subtree_nodes = collected
        return collected

    def _build_node_map(self, node: TreeNode) -> None:
        """Assign each node's name to the *deepest* TreeNode that actually holds it.
        We do a parent-first approach so children override if needed.
        """
        # Map the raw_nodes at this level
        for nd in node.raw_nodes:
            self._node_map[nd.name] = node

        # Then recurse, letting children override deeper nodes
        for child in node.children.values():
            self._build_node_map(child)

    def _build_path_map(self, node: TreeNode) -> None:
        """Build a path->TreeNode map for easy lookups. Skips "root" in path strings."""
        path_str = self._compute_full_path(node)
        self._path_map[path_str] = node
        for child in node.children.values():
            self._build_path_map(child)

    def _compute_full_path(self, node: TreeNode) -> str:
        """Return a '/'-joined path, omitting "root"."""
        parts = []
        current = node
        while current and current.name != "root":
            parts.append(current.name)
            current = current.parent
        return "/".join(reversed(parts))

    def _get_ancestors(self, node: TreeNode) -> Set[TreeNode]:
        """Return a cached set of this node's ancestors (including itself)."""
        if node in self._ancestors_cache:
            return self._ancestors_cache[node]

        ancestors = set()
        current = node
        while current is not None:
            ancestors.add(current)
            current = current.parent
        self._ancestors_cache[node] = ancestors
        return ancestors

    def _compute_statistics(self) -> None:
        """Populates two stats sets for each TreeNode:
        - node.stats (all, ignoring disabled)
        - node.active_stats (only enabled nodes/links)
        """

        # First, zero them out
        def reset_stats(n: TreeNode):
            n.stats = TreeStats()
            n.active_stats = TreeStats()
            for c in n.children.values():
                reset_stats(c)

        if self.root_node:
            reset_stats(self.root_node)

        # 1) Node counts from subtree sets
        def set_node_counts(n: TreeNode):
            n.stats.node_count = len(n.subtree_nodes)
            n.active_stats.node_count = len(n.active_subtree_nodes)
            for c in n.children.values():
                set_node_counts(c)

        if self.root_node:
            set_node_counts(self.root_node)

        # 2) Accumulate node cost/power
        for nd in self.network.nodes.values():
            hw_comp_name = nd.attrs.get("hw_component")
            comp = None
            if hw_comp_name:
                comp = self.components_library.get(hw_comp_name)
                if comp is None:
                    logger.warning(
                        "Node '%s' references unknown hw_component '%s'.",
                        nd.name,
                        hw_comp_name,
                    )
            cost_val = comp.total_cost() if comp else 0.0
            power_val = comp.total_power() if comp else 0.0

            tree_node = self._node_map[nd.name]
            # "All" includes disabled
            for an in self._get_ancestors(tree_node):
                an.stats.total_cost += cost_val
                an.stats.total_power += power_val

            # "Active" excludes disabled
            if not nd.attrs.get("disabled"):
                for an in self._get_ancestors(tree_node):
                    an.active_stats.total_cost += cost_val
                    an.active_stats.total_power += power_val

        # 3) Accumulate link stats (internal/external + cost/power)
        for link in self.network.links.values():
            src = link.source
            dst = link.target

            link_comp_name = link.attrs.get("hw_component")
            link_comp = None
            if link_comp_name:
                link_comp = self.components_library.get(link_comp_name)
                if link_comp is None:
                    logger.warning(
                        "Link '%s->%s' references unknown hw_component '%s'.",
                        src,
                        dst,
                        link_comp_name,
                    )
            link_cost = link_comp.total_cost() if link_comp else 0.0
            link_power = link_comp.total_power() if link_comp else 0.0
            cap = link.capacity

            src_node = self._node_map[src]
            dst_node = self._node_map[dst]
            A_src = self._get_ancestors(src_node)
            A_dst = self._get_ancestors(dst_node)

            inter_anc = A_src & A_dst  # sees link as "internal"
            xor_anc = A_src ^ A_dst  # sees link as "external"

            # ----- "ALL" stats -----
            for an in inter_anc:
                an.stats.internal_link_count += 1
                an.stats.internal_link_capacity += cap
                an.stats.total_cost += link_cost
                an.stats.total_power += link_power
            for an in xor_anc:
                an.stats.external_link_count += 1
                an.stats.external_link_capacity += cap
                an.stats.total_cost += link_cost
                an.stats.total_power += link_power

                if an in A_src:
                    other_path = self._compute_full_path(dst_node)
                else:
                    other_path = self._compute_full_path(src_node)
                bd = an.stats.external_link_details.setdefault(
                    other_path, ExternalLinkBreakdown()
                )
                bd.link_count += 1
                bd.link_capacity += cap

            # ----- "ACTIVE" stats -----
            # If link or either endpoint is disabled, skip
            if link.attrs.get("disabled"):
                continue
            if self.network.nodes[src].attrs.get("disabled"):
                continue
            if self.network.nodes[dst].attrs.get("disabled"):
                continue

            for an in inter_anc:
                an.active_stats.internal_link_count += 1
                an.active_stats.internal_link_capacity += cap
                an.active_stats.total_cost += link_cost
                an.active_stats.total_power += link_power
            for an in xor_anc:
                an.active_stats.external_link_count += 1
                an.active_stats.external_link_capacity += cap
                an.active_stats.total_cost += link_cost
                an.active_stats.total_power += link_power

                if an in A_src:
                    other_path = self._compute_full_path(dst_node)
                else:
                    other_path = self._compute_full_path(src_node)
                bd = an.active_stats.external_link_details.setdefault(
                    other_path, ExternalLinkBreakdown()
                )
                bd.link_count += 1
                bd.link_capacity += cap

    def print_tree(
        self,
        node: Optional[TreeNode] = None,
        indent: int = 0,
        max_depth: Optional[int] = None,
        skip_leaves: bool = False,
        detailed: bool = False,
        include_disabled: bool = True,
    ) -> None:
        """Print the hierarchy from 'node' down (default: root).

        Args:
            node (TreeNode): subtree to print, or root if None
            indent (int): indentation level
            max_depth (int): if set, limit display depth
            skip_leaves (bool): if True, skip leaf subtrees
            detailed (bool): if True, print link capacity breakdowns
            include_disabled (bool): If False, show stats only for enabled nodes/links.
                                     Subtrees with zero active nodes are omitted.
        """
        if node is None:
            node = self.root_node
            if node is None:
                print("No hierarchy built yet.")
                return

        if max_depth is not None and indent > max_depth:
            return

        # Pick which stats to display
        stats = node.stats if include_disabled else node.active_stats

        # If 'active' mode and this node has 0 nodes, omit it (unless it's the root)
        if not include_disabled and stats.node_count == 0 and node.parent is not None:
            return

        # Possibly skip leaves
        if skip_leaves and node.is_leaf() and node.parent is not None:
            return

        total_links = stats.internal_link_count + stats.external_link_count
        line = (
            f"{'  ' * indent}- {node.name or 'root'} | "
            f"Nodes={stats.node_count}, Links={total_links}, "
            f"Cost={stats.total_cost}, Power={stats.total_power}"
        )
        if detailed:
            line += (
                f" | IntLinkCap={stats.internal_link_capacity}, "
                f"ExtLinkCap={stats.external_link_capacity}"
            )

        print(line)

        # If detailed, show external link breakdown
        if detailed and stats.external_link_details:
            rolled_map: Dict[str, ExternalLinkBreakdown] = {}
            for other_path, info in stats.external_link_details.items():
                rolled_path = other_path
                if skip_leaves:
                    # If that path is a leaf, roll up
                    rolled_path = self._roll_up_if_leaf(rolled_path)
                accum = rolled_map.setdefault(rolled_path, ExternalLinkBreakdown())
                accum.link_count += info.link_count
                accum.link_capacity += info.link_capacity

            for path_str in sorted(rolled_map.keys()):
                ext_info = rolled_map[path_str]
                if path_str == "":
                    path_str = "[root]"
                print(
                    f"{'  ' * indent}   -> External to [{path_str}]: "
                    f"{ext_info.link_count} links, cap={ext_info.link_capacity}"
                )

        # Recurse on children
        for child in node.children.values():
            self.print_tree(
                node=child,
                indent=indent + 1,
                max_depth=max_depth,
                skip_leaves=skip_leaves,
                detailed=detailed,
                include_disabled=include_disabled,
            )

    def _roll_up_if_leaf(self, path: str) -> str:
        """If 'path' is a leaf node's path, climb up until a non-leaf or root is found."""
        node = self._path_map.get(path)
        if not node:
            return path
        while node.parent and node.parent.name != "root" and node.is_leaf():
            node = node.parent
        return self._compute_full_path(node)
