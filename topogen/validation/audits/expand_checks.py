"""Check expansion coverage without changing the network's selection rules."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Callable


def check_groups_adjacency_blueprints(
    dsl: dict[str, Any],
    ng_expand: Callable[[dict[str, Any]], Any],
    logger_obj,
) -> list[str]:
    """Check all network rules in one expansion and each blueprint in isolation.

    Copy the complete DSL, including node/link rules and variables. Dropping
    these changes selector meaning, especially for striped inter-site links.
    Blueprint checks remain independent so unused definitions are checked too.
    """
    tagged = deepcopy(dsl)
    network = tagged.setdefault("network", {})
    groups = network.get("nodes", {})
    adjacencies = network.get("links", [])
    for idx, definition in enumerate(groups.values()):
        definition.setdefault("attrs", {})["_tg_group_id"] = idx
    for idx, rule in enumerate(adjacencies):
        rule.setdefault("attrs", {})["_tg_adj_tag"] = f"adj_{idx}"
    expanded = ng_expand(tagged)
    group_counts = Counter(
        node.attrs.get("_tg_group_id") for node in expanded.nodes.values()
    )
    adjacency_counts = Counter(
        link.attrs.get("_tg_adj_tag") for link in expanded.links.values()
    )
    issues = [
        f"group '{path}' expands to 0 nodes"
        for idx, path in enumerate(groups)
        if not group_counts[idx]
    ]
    for idx, rule in enumerate(adjacencies):
        if not adjacency_counts[f"adj_{idx}"]:
            issues.append(
                f"adjacency[{idx}] expands to 0 links "
                f"(source={rule.get('source')}, target={rule.get('target')}, "
                f"pattern={rule.get('pattern')})"
            )

    for name, definition in (dsl.get("blueprints") or {}).items():
        blueprint = deepcopy(definition)
        rules = blueprint.get("links", [])
        if not rules:
            continue
        for idx, rule in enumerate(rules):
            rule.setdefault("attrs", {})["_tg_bp_adj_tag"] = f"{name}#{idx}"
        probe = {
            **dsl,
            "blueprints": {**dsl.get("blueprints", {}), name: blueprint},
            "network": {"nodes": {"probe": {"blueprint": name}}, "links": []},
        }
        expanded = ng_expand(probe)
        seen = {link.attrs.get("_tg_bp_adj_tag") for link in expanded.links.values()}
        issues.extend(
            f"blueprint '{name}' adjacency[{idx}] expands to 0 links"
            for idx in range(len(rules))
            if f"{name}#{idx}" not in seen
        )
    for issue in issues:
        logger_obj.error(issue)
    return issues
