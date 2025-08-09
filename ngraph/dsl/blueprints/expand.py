"""Network topology blueprints and generation."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from itertools import product, zip_longest
from typing import Any, Dict, List, Set

from ngraph.model.network import Link, Network, Node


@dataclass
class Blueprint:
    """Represents a reusable blueprint for hierarchical sub-topologies.

    A blueprint may contain multiple groups of nodes (each can have a node_count
    and a name_template), plus adjacency rules describing how those groups connect.

    Attributes:
        name (str): Unique identifier of this blueprint.
        groups (Dict[str, Any]): A mapping of group_name -> group definition.
            Allowed top-level keys in each group definition here are the same
            as in normal group definitions (e.g. node_count, name_template,
            attrs, disabled, risk_groups, or nested use_blueprint references, etc.).
        adjacency (List[Dict[str, Any]]): A list of adjacency definitions
            describing how these groups are linked, using the DSL fields
            (source, target, pattern, link_params, etc.).
    """

    name: str
    groups: Dict[str, Any]
    adjacency: List[Dict[str, Any]]


@dataclass
class DSLExpansionContext:
    """Carries the blueprint definitions and the final Network instance
    to be populated during DSL expansion.

    Attributes:
        blueprints (Dict[str, Blueprint]): Dictionary of blueprint-name -> Blueprint.
        network (Network): The Network into which expanded nodes/links are inserted.
    """

    blueprints: Dict[str, Blueprint]
    network: Network


def expand_network_dsl(data: Dict[str, Any]) -> Network:
    """Expands a combined blueprint + network DSL into a complete Network object.

    Overall flow:
      1) Parse "blueprints" into Blueprint objects.
      2) Build a new Network from "network" metadata (e.g. name, version).
      3) Expand 'network["groups"]'.
         - If a group references a blueprint, incorporate that blueprint's subgroups
           while merging parent's attrs + disabled + risk_groups into subgroups.
         - Otherwise, directly create nodes (a "direct node group").
      4) Process any direct node definitions (network["nodes"]).
      5) Expand adjacency definitions in 'network["adjacency"]'.
      6) Process any direct link definitions (network["links"]).
      7) Process link overrides (in order if multiple overrides match).
      8) Process node overrides (in order if multiple overrides match).

    Under the new rules:
      - Only certain top-level fields are permitted in each structure. Any extra
        keys raise a ValueError. "attrs" is where arbitrary user fields go.
      - For link_params, recognized fields are "capacity", "cost", "disabled",
        "risk_groups", "attrs". Everything else must go inside link_params["attrs"].
      - For node/group definitions, recognized fields include "node_count",
        "name_template", "attrs", "disabled", "risk_groups" or "use_blueprint"
        for blueprint-based groups.

    Args:
        data (Dict[str, Any]): The YAML-parsed dictionary containing
            optional "blueprints" + "network".

    Returns:
        Network: The expanded Network object with all nodes and links.
    """
    # 1) Parse blueprint definitions
    blueprint_map: Dict[str, Blueprint] = {}
    if "blueprints" in data:
        if not isinstance(data["blueprints"], dict):
            raise ValueError("'blueprints' must be a dictionary.")
        for bp_name, bp_data in data["blueprints"].items():
            if not isinstance(bp_data, dict):
                raise ValueError(
                    f"Blueprint definition for '{bp_name}' must be a dict."
                )
            _check_no_extra_keys(
                bp_data,
                allowed={"groups", "adjacency"},
                context=f"blueprint '{bp_name}'",
            )
            blueprint_map[bp_name] = Blueprint(
                name=bp_name,
                groups=bp_data.get("groups", {}),
                adjacency=bp_data.get("adjacency", []),
            )

    # 2) Initialize the Network from "network" metadata
    network_data = data.get("network", {})
    if not isinstance(network_data, dict):
        raise ValueError("'network' must be a dictionary if present.")

    net = Network()
    # Pull recognized top-level fields from network_data
    for key in network_data.keys():
        if key not in (
            "name",
            "version",
            "groups",
            "nodes",
            "adjacency",
            "links",
            "link_overrides",
            "node_overrides",
        ):
            raise ValueError(f"Unrecognized top-level key in 'network': {key}")

    if "name" in network_data:
        net.attrs["name"] = network_data["name"]
    if "version" in network_data:
        net.attrs["version"] = network_data["version"]

    # Create a context
    ctx = DSLExpansionContext(blueprints=blueprint_map, network=net)

    # 3) Expand top-level groups
    for group_name, group_def in network_data.get("groups", {}).items():
        if not isinstance(group_def, dict):
            raise ValueError(f"Group definition for '{group_name}' must be a dict.")
        _expand_group(ctx, parent_path="", group_name=group_name, group_def=group_def)

    # 4) Process direct node definitions
    _process_direct_nodes(ctx.network, network_data)

    # 5) Expand adjacency definitions
    for adj_def in network_data.get("adjacency", []):
        if not isinstance(adj_def, dict):
            raise ValueError("Each adjacency entry must be a dictionary.")
        _expand_adjacency(ctx, adj_def)

    # 6) Process direct link definitions
    _process_direct_links(ctx.network, network_data)

    # 7) Process link overrides (in order)
    _process_link_overrides(ctx.network, network_data)

    # 8) Process node overrides (in order)
    _process_node_overrides(ctx.network, network_data)

    return net


def _expand_group(
    ctx: DSLExpansionContext,
    parent_path: str,
    group_name: str,
    group_def: Dict[str, Any],
    inherited_risk_groups: Set[str] | None = None,
) -> None:
    """Expands a single group definition into either:
      - Another blueprint's subgroups, or
      - A direct node group (with node_count, etc.),
      - Possibly replicating itself if group_name has bracket expansions.

    If 'use_blueprint' is present, we expand that blueprint. Otherwise, we
    create nodes directly.

    For blueprint usage:
      Allowed keys: {"use_blueprint", "parameters", "attrs", "disabled", "risk_groups"}.
      We merge 'attrs', 'disabled', and 'risk_groups' from this parent group
      into each blueprint subgroup's definition.

    For direct node groups (no 'use_blueprint'):
      Allowed keys: {"node_count", "name_template", "attrs", "disabled", "risk_groups"}.

    If group_name includes bracket expansions like "fa[1-2]", it replicates the
    same group_def for each expanded name.

    Args:
        ctx (DSLExpansionContext): The context containing blueprint info and the Network.
        parent_path (str): The parent path in the hierarchy.
        group_name (str): The current group's name (may have bracket expansions).
        group_def (Dict[str, Any]): The group definition (node_count, name_template, etc.).
        inherited_risk_groups (Set[str]): Risk groups inherited from a higher-level group.
    """
    if inherited_risk_groups is None:
        inherited_risk_groups = set()
    expanded_names = _expand_name_patterns(group_name)
    # If bracket expansions exist, replicate for each expansion
    if len(expanded_names) > 1 or expanded_names[0] != group_name:
        for expanded_name in expanded_names:
            _expand_group(
                ctx, parent_path, expanded_name, group_def, inherited_risk_groups
            )
        return

    # Compute the full path for this group
    if parent_path:
        effective_path = f"{parent_path}/{group_name}"
    else:
        effective_path = group_name

    if "use_blueprint" in group_def:
        # Blueprint usage => recognized keys
        _check_no_extra_keys(
            group_def,
            allowed={"use_blueprint", "parameters", "attrs", "disabled", "risk_groups"},
            context=f"group '{group_name}' using blueprint",
        )
        blueprint_name: str = group_def["use_blueprint"]
        bp = ctx.blueprints.get(blueprint_name)
        if not bp:
            raise ValueError(
                f"Group '{group_name}' references unknown blueprint '{blueprint_name}'."
            )

        parent_attrs = copy.deepcopy(group_def.get("attrs", {}))
        if not isinstance(parent_attrs, dict):
            raise ValueError(f"'attrs' must be a dict in group '{group_name}'.")
        parent_disabled = bool(group_def.get("disabled", False))

        # Merge parent's risk_groups
        parent_risk_groups = set(inherited_risk_groups)
        if "risk_groups" in group_def:
            rg_val = group_def["risk_groups"]
            if not isinstance(rg_val, (list, set)):
                raise ValueError(
                    f"'risk_groups' must be list or set in group '{group_name}'."
                )
            parent_risk_groups |= set(rg_val)

        param_overrides: Dict[str, Any] = group_def.get("parameters", {})
        if not isinstance(param_overrides, dict):
            raise ValueError(f"'parameters' must be a dict in group '{group_name}'.")

        # For each subgroup in the blueprint, apply param overrides and
        # merge parent's attrs/disabled/risk_groups
        for bp_sub_name, bp_sub_def in bp.groups.items():
            merged_def = _apply_parameters(bp_sub_name, bp_sub_def, param_overrides)
            merged_def = dict(merged_def)  # ensure we can mutate

            # Force disabled if parent is disabled
            if parent_disabled:
                merged_def["disabled"] = True

            # Merge parent's attrs
            child_attrs = merged_def.get("attrs", {})
            if not isinstance(child_attrs, dict):
                raise ValueError(
                    f"Subgroup '{bp_sub_name}' has non-dict 'attrs' inside blueprint '{blueprint_name}'."
                )
            merged_def["attrs"] = {**parent_attrs, **child_attrs}

            # Merge parent's risk_groups with child's
            child_rgs = set(merged_def.get("risk_groups", []))
            merged_def["risk_groups"] = parent_risk_groups | child_rgs

            # Recursively expand
            _expand_group(
                ctx,
                parent_path=effective_path,
                group_name=bp_sub_name,
                group_def=merged_def,
                inherited_risk_groups=merged_def["risk_groups"],
            )

        # Expand blueprint adjacency under this parent's path
        for adj_def in bp.adjacency:
            _expand_blueprint_adjacency(ctx, adj_def, effective_path)

    else:
        # Direct node group => recognized keys
        _check_no_extra_keys(
            group_def,
            allowed={"node_count", "name_template", "attrs", "disabled", "risk_groups"},
            context=f"group '{group_name}'",
        )
        node_count = group_def.get("node_count", 1)
        name_template = group_def.get("name_template", f"{group_name}-{{node_num}}")
        if not isinstance(node_count, int) or node_count < 1:
            raise ValueError(
                f"group '{group_name}' has invalid node_count: {node_count}"
            )

        combined_attrs = copy.deepcopy(group_def.get("attrs", {}))
        if not isinstance(combined_attrs, dict):
            raise ValueError(f"attrs must be a dict in group '{group_name}'.")
        group_disabled = bool(group_def.get("disabled", False))

        # Merge parent's risk groups
        parent_risk_groups = set(inherited_risk_groups)
        child_rgs = set(group_def.get("risk_groups", []))
        final_risk_groups = parent_risk_groups | child_rgs

        for i in range(1, node_count + 1):
            label = name_template.format(node_num=i)
            node_name = f"{effective_path}/{label}" if effective_path else label

            node = Node(
                name=node_name,
                disabled=group_disabled,
                attrs=copy.deepcopy(combined_attrs),
            )
            node.attrs.setdefault("type", "node")
            node.risk_groups = final_risk_groups.copy()
            ctx.network.add_node(node)


def _expand_blueprint_adjacency(
    ctx: DSLExpansionContext,
    adj_def: Dict[str, Any],
    parent_path: str,
) -> None:
    """Expands adjacency definitions from within a blueprint, using parent_path
    as the local root. This also handles optional expand_vars for repeated adjacency.

    Recognized adjacency keys:
      {"source", "target", "pattern", "link_count", "link_params",
       "expand_vars", "expansion_mode"}.

    Args:
        ctx (DSLExpansionContext): The context object with blueprint info and the network.
        adj_def (Dict[str, Any]): The adjacency definition inside the blueprint.
        parent_path (str): The path serving as the base for the blueprint's node paths.
    """
    _check_adjacency_keys(adj_def, context="blueprint adjacency")
    expand_vars = adj_def.get("expand_vars", {})
    if expand_vars:
        _expand_adjacency_with_variables(ctx, adj_def, parent_path)
        return

    source_rel = adj_def["source"]
    target_rel = adj_def["target"]
    pattern = adj_def.get("pattern", "mesh")
    link_params = adj_def.get("link_params", {})
    _check_link_params(link_params, context="blueprint adjacency")
    link_count = adj_def.get("link_count", 1)

    src_path = _join_paths(parent_path, source_rel)
    tgt_path = _join_paths(parent_path, target_rel)

    _expand_adjacency_pattern(ctx, src_path, tgt_path, pattern, link_params, link_count)


def _expand_adjacency(ctx: DSLExpansionContext, adj_def: Dict[str, Any]) -> None:
    """Expands a top-level adjacency definition from 'network.adjacency'. If 'expand_vars'
    is provided, we expand the source/target as templates repeatedly.

    Recognized adjacency keys:
      {"source", "target", "pattern", "link_count", "link_params",
       "expand_vars", "expansion_mode"}.

    Args:
        ctx (DSLExpansionContext): The context containing the target network.
        adj_def (Dict[str, Any]): The adjacency definition dict.
    """
    _check_adjacency_keys(adj_def, context="top-level adjacency")
    expand_vars = adj_def.get("expand_vars", {})
    if expand_vars:
        _expand_adjacency_with_variables(ctx, adj_def, parent_path="")
        return

    source_path_raw = adj_def["source"]
    target_path_raw = adj_def["target"]
    pattern = adj_def.get("pattern", "mesh")
    link_count = adj_def.get("link_count", 1)
    link_params = adj_def.get("link_params", {})
    _check_link_params(link_params, context="top-level adjacency")

    source_path = _join_paths("", source_path_raw)
    target_path = _join_paths("", target_path_raw)

    _expand_adjacency_pattern(
        ctx, source_path, target_path, pattern, link_params, link_count
    )


def _expand_adjacency_with_variables(
    ctx: DSLExpansionContext, adj_def: Dict[str, Any], parent_path: str
) -> None:
    """Handles adjacency expansions when 'expand_vars' is provided.
    We substitute variables into the 'source' and 'target' templates to produce
    multiple adjacency expansions. Then each expansion is passed to _expand_adjacency_pattern.

    Args:
        ctx (DSLExpansionContext): The DSL expansion context.
        adj_def (Dict[str, Any]): The adjacency definition including expand_vars, source, target, etc.
        parent_path (str): Prepended to source/target if they do not start with '/'.
    """
    source_template = adj_def["source"]
    target_template = adj_def["target"]
    pattern = adj_def.get("pattern", "mesh")
    link_params = adj_def.get("link_params", {})
    _check_link_params(link_params, context="adjacency with expand_vars")
    link_count = adj_def.get("link_count", 1)
    expand_vars = adj_def["expand_vars"]
    expansion_mode = adj_def.get("expansion_mode", "cartesian")

    var_names = sorted(expand_vars.keys())
    lists_of_values = [expand_vars[var] for var in var_names]

    if expansion_mode == "zip":
        lengths = [len(lst) for lst in lists_of_values]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"zip expansion requires all lists be the same length; got {lengths}"
            )

        for combo_tuple in zip_longest(*lists_of_values, fillvalue=None):
            combo_dict = dict(zip(var_names, combo_tuple, strict=False))
            expanded_src = _join_paths(
                parent_path, source_template.format(**combo_dict)
            )
            expanded_tgt = _join_paths(
                parent_path, target_template.format(**combo_dict)
            )
            _expand_adjacency_pattern(
                ctx, expanded_src, expanded_tgt, pattern, link_params, link_count
            )
    else:
        # "cartesian" default
        for combo_tuple in product(*lists_of_values):
            combo_dict = dict(zip(var_names, combo_tuple, strict=False))
            expanded_src = _join_paths(
                parent_path, source_template.format(**combo_dict)
            )
            expanded_tgt = _join_paths(
                parent_path, target_template.format(**combo_dict)
            )
            _expand_adjacency_pattern(
                ctx, expanded_src, expanded_tgt, pattern, link_params, link_count
            )


def _expand_adjacency_pattern(
    ctx: DSLExpansionContext,
    source_path: str,
    target_path: str,
    pattern: str,
    link_params: Dict[str, Any],
    link_count: int = 1,
) -> None:
    """Generates Link objects for the chosen adjacency pattern among matched nodes.

    Supported Patterns:
      * "mesh": Connect every source node to every target node
                (no self-loops, deduplicate reversed pairs).
      * "one_to_one": Pair each source node with exactly one target node
                      using wrap-around. The larger set size must be
                      a multiple of the smaller set size.

    link_params must only contain recognized keys: capacity, cost, disabled,
    risk_groups, attrs.

    Args:
        ctx (DSLExpansionContext): The context with the target network.
        source_path (str): Path pattern identifying source node group(s).
        target_path (str): Path pattern identifying target node group(s).
        pattern (str): "mesh" or "one_to_one".
        link_params (Dict[str, Any]): Additional link parameters (capacity, cost, disabled, risk_groups, attrs).
        link_count (int): Number of parallel links to create for each adjacency.
    """
    source_node_groups = ctx.network.select_node_groups_by_path(source_path)
    target_node_groups = ctx.network.select_node_groups_by_path(target_path)

    source_nodes = [node for _, nodes in source_node_groups.items() for node in nodes]
    target_nodes = [node for _, nodes in target_node_groups.items() for node in nodes]

    if not source_nodes or not target_nodes:
        return

    dedup_pairs = set()

    if pattern == "mesh":
        for sn in source_nodes:
            for tn in target_nodes:
                if sn.name == tn.name:
                    continue
                pair = tuple(sorted((sn.name, tn.name)))
                if pair not in dedup_pairs:
                    dedup_pairs.add(pair)
                    _create_link(ctx.network, sn.name, tn.name, link_params, link_count)

    elif pattern == "one_to_one":
        s_count = len(source_nodes)
        t_count = len(target_nodes)
        bigger_count = max(s_count, t_count)
        smaller_count = min(s_count, t_count)

        if bigger_count % smaller_count != 0:
            raise ValueError(
                "one_to_one pattern requires sizes with a multiple factor; "
                f"source={s_count}, target={t_count} do not align."
            )

        for i in range(bigger_count):
            if s_count >= t_count:
                sn = source_nodes[i].name
                tn = target_nodes[i % t_count].name
            else:
                sn = source_nodes[i % s_count].name
                tn = target_nodes[i].name

            if sn == tn:
                continue

            pair = tuple(sorted((sn, tn)))
            if pair not in dedup_pairs:
                dedup_pairs.add(pair)
                _create_link(ctx.network, sn, tn, link_params, link_count)
    else:
        raise ValueError(f"Unknown adjacency pattern: {pattern}")


def _create_link(
    net: Network,
    source: str,
    target: str,
    link_params: Dict[str, Any],
    link_count: int = 1,
) -> None:
    """Creates and adds one or more Links to the network, applying capacity, cost,
    disabled, risk_groups, and attrs from link_params if present.

    Args:
        net (Network): The network to which the new link(s) will be added.
        source (str): Source node name for the link.
        target (str): Target node name for the link.
        link_params (Dict[str, Any]): Dict possibly containing
            'capacity', 'cost', 'disabled', 'risk_groups', 'attrs'.
        link_count (int): Number of parallel links to create between source and target.
    """
    _check_link_params(link_params, context=f"creating link {source}->{target}")

    for _ in range(link_count):
        capacity = link_params.get("capacity", 1.0)
        cost = link_params.get("cost", 1.0)
        attrs = copy.deepcopy(link_params.get("attrs", {}))
        disabled_flag = bool(link_params.get("disabled", False))
        # If link_params has risk_groups, we set them (replace).
        link_rgs = set(link_params.get("risk_groups", []))

        link = Link(
            source=source,
            target=target,
            capacity=capacity,
            cost=cost,
            attrs=attrs,
            disabled=disabled_flag,
        )
        link.risk_groups = link_rgs
        net.add_link(link)


def _process_direct_nodes(net: Network, network_data: Dict[str, Any]) -> None:
    """Processes direct node definitions (network_data["nodes"]) and adds them to the network
    if they do not already exist. If the node name already exists, we do nothing.

    Allowed top-level keys for each node: {"disabled", "attrs", "risk_groups"}.
    Everything else must be placed inside "attrs" or it triggers an error.

    Args:
        net (Network): The network to which nodes are added.
        network_data (Dict[str, Any]): DSL data possibly containing a "nodes" dict.
    """
    nodes_dict = network_data.get("nodes", {})
    if not isinstance(nodes_dict, dict):
        return

    for node_name, raw_def in nodes_dict.items():
        if not isinstance(raw_def, dict):
            raise ValueError(f"Node definition for '{node_name}' must be a dict.")
        _check_no_extra_keys(
            raw_def,
            allowed={"disabled", "attrs", "risk_groups"},
            context=f"node '{node_name}'",
        )

        if node_name not in net.nodes:
            disabled_flag = bool(raw_def.get("disabled", False))
            attrs_dict = raw_def.get("attrs", {})
            if not isinstance(attrs_dict, dict):
                raise ValueError(f"'attrs' must be a dict in node '{node_name}'.")
            # risk_groups => set them if provided
            rgs = set(raw_def.get("risk_groups", []))

            new_node = Node(
                name=node_name,
                disabled=disabled_flag,
                attrs=copy.deepcopy(attrs_dict),
            )
            new_node.attrs.setdefault("type", "node")
            new_node.risk_groups = rgs
            net.add_node(new_node)


def _process_direct_links(net: Network, network_data: Dict[str, Any]) -> None:
    """Processes direct link definitions (network_data["links"]) and adds them to the network.

    Each link dict must contain {"source", "target"} plus optionally
    {"link_params", "link_count"}. No other top-level keys allowed.
    link_params must obey the recognized link_params format (including optional risk_groups).

    Args:
        net (Network): The network to which links are added.
        network_data (Dict[str, Any]): DSL data possibly containing a "links" list.
    """
    links_list = network_data.get("links", [])
    if not isinstance(links_list, list):
        return

    for link_info in links_list:
        if not isinstance(link_info, dict):
            raise ValueError("Each link definition must be a dictionary.")
        _check_no_extra_keys(
            link_info,
            allowed={"source", "target", "link_params", "link_count"},
            context="direct link",
        )

        source = link_info["source"]
        target = link_info["target"]
        if source not in net.nodes or target not in net.nodes:
            raise ValueError(f"Link references unknown node(s): {source}, {target}.")
        if source == target:
            raise ValueError(f"Link cannot have the same source and target: {source}")

        link_params = link_info.get("link_params", {})
        if not isinstance(link_params, dict):
            raise ValueError(f"link_params must be a dict for link {source}->{target}.")
        link_count = link_info.get("link_count", 1)
        if not isinstance(link_count, int) or link_count < 1:
            raise ValueError(
                f"Invalid link_count={link_count} for link {source}->{target}."
            )

        _create_link(net, source, target, link_params, link_count)


def _process_link_overrides(net: Network, network_data: Dict[str, Any]) -> None:
    """Processes the 'link_overrides' section of the network DSL, updating
    existing links with new parameters. Overrides are applied in order if
    multiple items match the same link.

    Each override must contain {"source", "target", "link_params"} plus
    optionally {"any_direction"}. link_params must obey recognized fields
    (capacity, cost, disabled, risk_groups, attrs).

    If link_params["risk_groups"] is given, it *replaces* the link's existing risk_groups.

    Args:
        net (Network): The Network whose links will be updated.
        network_data (Dict[str, Any]): DSL data possibly containing 'link_overrides'.
    """
    link_overrides = network_data.get("link_overrides", [])
    if not isinstance(link_overrides, list):
        return

    for link_override in link_overrides:
        if not isinstance(link_override, dict):
            raise ValueError("Each link_override must be a dict.")
        _check_no_extra_keys(
            link_override,
            allowed={"source", "target", "link_params", "any_direction"},
            context="link override",
        )
        source = link_override["source"]
        target = link_override["target"]
        link_params = link_override["link_params"]
        if not isinstance(link_params, dict):
            raise ValueError("link_params must be dict in link override.")
        any_direction = link_override.get("any_direction", True)

        _update_links(net, source, target, link_params, any_direction)


def _process_node_overrides(net: Network, network_data: Dict[str, Any]) -> None:
    """Processes the 'node_overrides' section of the network DSL, updating
    existing nodes with new attributes in bulk. Overrides are applied in order
    if multiple items match the same node.

    Each override must have {"path"} plus optionally {"attrs", "disabled", "risk_groups"}.

    - If "disabled" is present at top level, we set node.disabled.
    - If "risk_groups" is present, we *replace* the node's risk_groups.
    - Everything else merges into node.attrs.

    Args:
        net (Network): The Network whose nodes will be updated.
        network_data (Dict[str, Any]): DSL data possibly containing 'node_overrides'.
    """
    node_overrides = network_data.get("node_overrides", [])
    if not isinstance(node_overrides, list):
        return

    for override in node_overrides:
        if not isinstance(override, dict):
            raise ValueError("Each node_override must be a dict.")
        _check_no_extra_keys(
            override,
            allowed={"path", "attrs", "disabled", "risk_groups"},
            context="node override",
        )

        path = override["path"]
        top_level_disabled = override.get("disabled", None)
        override_risk_groups = override.get("risk_groups", None)

        if "attrs" in override and not isinstance(override["attrs"], dict):
            raise ValueError("attrs must be a dict in node override.")

        attrs_to_set = copy.deepcopy(override.get("attrs", {}))

        # We'll process disabled as a separate boolean
        # We'll process risk_groups as a direct replacement if given
        _update_nodes(net, path, attrs_to_set, top_level_disabled, override_risk_groups)


def _update_links(
    net: Network,
    source: str,
    target: str,
    link_params: Dict[str, Any],
    any_direction: bool = True,
) -> None:
    """Updates all Link objects between nodes matching 'source' and 'target' paths
    with new parameters (capacity, cost, disabled, risk_groups, attrs).

    If any_direction=True, both (source->target) and (target->source) links
    are updated if present.

    If link_params["risk_groups"] is given, it *replaces* the link's existing risk_groups.

    Args:
        net (Network): The network whose links should be updated.
        source (str): A path pattern identifying source node group(s).
        target (str): A path pattern identifying target node group(s).
        link_params (Dict[str, Any]): New parameter values
            (capacity, cost, disabled, risk_groups, attrs).
        any_direction (bool): If True, also update reversed direction links.
    """
    _check_link_params(link_params, context="link override processing")

    source_node_groups = net.select_node_groups_by_path(source)
    target_node_groups = net.select_node_groups_by_path(target)

    source_nodes = {
        node.name for _, nodes in source_node_groups.items() for node in nodes
    }
    target_nodes = {
        node.name for _, nodes in target_node_groups.items() for node in nodes
    }

    new_disabled_val = link_params.get("disabled", None)
    new_capacity = link_params.get("capacity", None)
    new_cost = link_params.get("cost", None)
    new_risk_groups = link_params.get("risk_groups", None)
    new_attrs = link_params.get("attrs", {})

    for link in net.links.values():
        forward_match = link.source in source_nodes and link.target in target_nodes
        reverse_match = (
            any_direction
            and link.source in target_nodes
            and link.target in source_nodes
        )
        if forward_match or reverse_match:
            if new_capacity is not None:
                link.capacity = new_capacity
            if new_cost is not None:
                link.cost = new_cost
            if new_disabled_val is not None:
                link.disabled = bool(new_disabled_val)
            if new_risk_groups is not None:
                link.risk_groups = set(new_risk_groups)
            if new_attrs:
                link.attrs.update(new_attrs)


def _update_nodes(
    net: Network,
    path: str,
    attrs: Dict[str, Any],
    disabled_val: Any = None,
    risk_groups_val: Any = None,
) -> None:
    """Updates attributes on all nodes matching a given path pattern.

    - If 'disabled_val' is not None, sets node.disabled to that boolean value.
    - If 'risk_groups_val' is not None, *replaces* the node's risk_groups with that new set.
    - Everything else in 'attrs' is merged into node.attrs.

    Args:
        net (Network): The network containing the nodes.
        path (str): A path pattern identifying which node group(s) to modify.
        attrs (Dict[str, Any]): A dictionary of new attributes to set/merge.
        disabled_val (Any): Boolean or None for disabling or enabling nodes.
        risk_groups_val (Any): List or set or None for replacing node.risk_groups.
    """
    node_groups = net.select_node_groups_by_path(path)
    for _, nodes in node_groups.items():
        for node in nodes:
            if disabled_val is not None:
                node.disabled = bool(disabled_val)
            if risk_groups_val is not None:
                if not isinstance(risk_groups_val, (list, set)):
                    raise ValueError(
                        f"risk_groups override must be list or set, got {type(risk_groups_val)}."
                    )
                node.risk_groups = set(risk_groups_val)
            node.attrs.update(attrs)


def _apply_parameters(
    subgroup_name: str, subgroup_def: Dict[str, Any], params_overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Applies user-provided parameter overrides to a blueprint subgroup.

    Example:
        If 'spine.node_count' = 6 is in params_overrides,
        it sets 'node_count' = 6 for the 'spine' subgroup.
        If 'spine.attrs.hw_type' = 'Dell', it sets subgroup_def['attrs']['hw_type'] = 'Dell'.

    Args:
        subgroup_name (str): Name of the subgroup in the blueprint.
        subgroup_def (Dict[str, Any]): The default definition of that subgroup.
        params_overrides (Dict[str, Any]): Overrides in the form of
            {'spine.node_count': 6, 'spine.attrs.hw_type': 'Dell'}.

    Returns:
        Dict[str, Any]: A copy of subgroup_def with parameter overrides applied.
    """
    out = copy.deepcopy(subgroup_def)

    for key, val in params_overrides.items():
        parts = key.split(".")
        if parts[0] == subgroup_name and len(parts) > 1:
            subpath = parts[1:]
            _apply_nested_path(out, subpath, val)

    return out


def _apply_nested_path(
    node_def: Dict[str, Any], path_parts: List[str], value: Any
) -> None:
    """Recursively applies a path like ["attrs", "role"] to set node_def["attrs"]["role"] = value.
    Creates intermediate dicts as needed.

    Args:
        node_def (Dict[str, Any]): The dictionary to update.
        path_parts (List[str]): List of keys specifying nested fields.
        value (Any): The value to set in the nested field.
    """
    if not path_parts:
        return
    key = path_parts[0]
    if len(path_parts) == 1:
        node_def[key] = value
        return
    if key not in node_def or not isinstance(node_def[key], dict):
        node_def[key] = {}
    _apply_nested_path(node_def[key], path_parts[1:], value)


_RANGE_REGEX = re.compile(r"\[([^\]]+)\]")


def _expand_name_patterns(name: str) -> List[str]:
    """Parses and expands bracketed expressions in a group name. For example:

        "fa[1-3]" -> ["fa1", "fa2", "fa3"]
        "dc[1,3,5-6]" -> ["dc1", "dc3", "dc5", "dc6"]
        "fa[1-2]_plane[5-6]" ->
          ["fa1_plane5", "fa1_plane6", "fa2_plane5", "fa2_plane6"]

    If no bracket expressions are present, returns [name] unchanged.

    Args:
        name (str): A group name that may contain bracket expansions.

    Returns:
        List[str]: All expanded names. If no expansion was needed, returns
            a single-element list with 'name' itself.
    """
    matches = list(_RANGE_REGEX.finditer(name))
    if not matches:
        return [name]  # no expansions

    expansions_list = []
    for match in matches:
        range_expr = match.group(1)
        expansions_list.append(_parse_range_expr(range_expr))

    expanded_names = []
    for combo in product(*expansions_list):
        result_str = ""
        last_end = 0
        for m_idx, match in enumerate(matches):
            start, end = match.span()
            result_str += name[last_end:start]
            result_str += combo[m_idx]
            last_end = end
        result_str += name[last_end:]
        expanded_names.append(result_str)

    return expanded_names


def _parse_range_expr(expr: str) -> List[str]:
    """Parses a bracket expression that might have commas, single values, and dash ranges.
    For example: "1-3,5,7-9" -> ["1", "2", "3", "5", "7", "8", "9"].

    Args:
        expr (str): The raw expression from inside brackets, e.g. "1-3,5,7-9".

    Returns:
        List[str]: A sorted list of all expansions.
    """
    values = []
    parts = [x.strip() for x in expr.split(",")]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            for val in range(start, end + 1):
                values.append(str(val))
        else:
            values.append(part)
    return values


def _join_paths(parent_path: str, rel_path: str) -> str:
    """Joins two path segments according to NetGraph's DSL conventions:

    - If rel_path starts with '/', we strip the leading slash and treat it as
      appended to parent_path if parent_path is not empty.
    - Otherwise, simply append rel_path to parent_path if parent_path is non-empty.

    Args:
        parent_path (str): The existing path prefix.
        rel_path (str): A relative path that may start with '/'.

    Returns:
        str: The combined path as a single string.
    """
    if rel_path.startswith("/"):
        rel_path = rel_path[1:]
        if parent_path:
            return f"{parent_path}/{rel_path}"
        return rel_path

    if parent_path:
        return f"{parent_path}/{rel_path}"
    return rel_path


def _check_no_extra_keys(
    data_dict: Dict[str, Any], allowed: set[str], context: str
) -> None:
    """Checks that data_dict only has keys in 'allowed'. Raises ValueError if not.

    Args:
        data_dict (Dict[str, Any]): The dict to check.
        allowed (set[str]): The set of recognized keys.
        context (str): A short description of what we are validating.
    """
    extra_keys = set(data_dict.keys()) - allowed
    if extra_keys:
        raise ValueError(
            f"Unrecognized key(s) in {context}: {', '.join(sorted(extra_keys))}. "
            f"Allowed keys are: {sorted(allowed)}"
        )


def _check_adjacency_keys(adj_def: Dict[str, Any], context: str) -> None:
    """Ensures adjacency definitions only contain recognized keys.

    Recognized adjacency keys are:
      {"source", "target", "pattern", "link_count", "link_params",
       "expand_vars", "expansion_mode"}.
    """
    _check_no_extra_keys(
        adj_def,
        allowed={
            "source",
            "target",
            "pattern",
            "link_count",
            "link_params",
            "expand_vars",
            "expansion_mode",
        },
        context=context,
    )
    if "source" not in adj_def or "target" not in adj_def:
        raise ValueError(f"Adjacency in {context} must have 'source' and 'target'.")


def _check_link_params(link_params: Dict[str, Any], context: str) -> None:
    """Checks that link_params only has recognized keys:
    {"capacity", "cost", "disabled", "risk_groups", "attrs"}.
    """
    recognized = {"capacity", "cost", "disabled", "risk_groups", "attrs"}
    extra = set(link_params.keys()) - recognized
    if extra:
        raise ValueError(
            f"Unrecognized link_params key(s) in {context}: {', '.join(sorted(extra))}. "
            f"Allowed: {sorted(recognized)}"
        )
