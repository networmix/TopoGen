"""Strict DSL expansion checks: groups & adjacency yield something, blueprint edges exist."""

from __future__ import annotations

from copy import deepcopy as _deepcopy
from typing import Any, Callable


def check_groups_adjacency_blueprints(
    dsl: dict[str, Any],
    ng_expand: Callable[[dict[str, Any]], Any],
    logger_obj,
) -> list[str]:
    """Ensure groups expand to nodes and adjacency rules expand to links.

    Also validates each blueprint's adjacency rules expand to links.
    """
    issues: list[str] = []

    # --- Groups -> nodes ---
    groups_orig = (
        _deepcopy((dsl.get("network") or {}).get("groups", {}))
        if isinstance((dsl.get("network") or {}).get("groups", {}), dict)
        else {}
    )
    gid_to_path: dict[int, str] = {}
    if isinstance(groups_orig, dict) and groups_orig:
        for _i, (gpath, gdef) in enumerate(groups_orig.items()):
            if not isinstance(gdef, dict):
                continue
            attrs = gdef.get("attrs")
            if not isinstance(attrs, dict):
                attrs = {}
                gdef["attrs"] = attrs
            attrs["_tg_group_id"] = _i
            gid_to_path[_i] = str(gpath)
        dsl_groups = {
            "blueprints": dsl.get("blueprints") or {},
            "network": {
                "groups": groups_orig,
                "adjacency": (dsl.get("network") or {}).get("adjacency", []),
            },
        }
        net_groups = ng_expand(dsl_groups)
        counts: dict[int, int] = {k: 0 for k in gid_to_path}
        for node in net_groups.nodes.values():
            try:
                gid = int(getattr(node, "attrs", {}).get("_tg_group_id", -1))
            except Exception:
                gid = -1
            if gid in counts:
                counts[gid] += 1
        for gid, cnt in counts.items():
            if cnt <= 0:
                gpath = gid_to_path[gid]
                try:
                    logger_obj.error(
                        "validation: group expands to zero nodes: %s", gpath
                    )
                except Exception:
                    pass
                issues.append(f"group '{gpath}' expands to 0 nodes")

    # --- Scenario adjacency -> links ---
    adj_list = (dsl.get("network") or {}).get("adjacency", [])
    if isinstance(adj_list, list) and adj_list:
        tagged_adj: list[dict[str, Any]] = []
        tag_attr = "_tg_adj_tag"
        for idx, rule in enumerate(adj_list):
            if not isinstance(rule, dict):
                continue
            r = _deepcopy(rule)
            lp = r.get("link_params")
            if not isinstance(lp, dict):
                lp = {}
                r["link_params"] = lp
            attrs = lp.get("attrs")
            if not isinstance(attrs, dict):
                attrs = {}
                lp["attrs"] = attrs
            attrs[tag_attr] = f"adj_{idx}"
            tagged_adj.append(r)
        dsl_adj = {
            "blueprints": dsl.get("blueprints") or {},
            "network": {
                "groups": (
                    _deepcopy((dsl.get("network") or {}).get("groups", {}))
                    if isinstance((dsl.get("network") or {}).get("groups", {}), dict)
                    else {}
                ),
                "adjacency": tagged_adj,
            },
        }
        net_adj = ng_expand(dsl_adj)
        adj_counts_by_tag: dict[str, int] = {}
        for link in net_adj.links.values():
            tag = getattr(link, "attrs", {}).get(tag_attr)
            if isinstance(tag, str):
                adj_counts_by_tag[tag] = adj_counts_by_tag.get(tag, 0) + 1
        for idx, _rule in enumerate(adj_list):
            tag = f"adj_{idx}"
            if adj_counts_by_tag.get(tag, 0) <= 0:
                # Include a brief rule summary
                try:
                    src = _rule.get("source") if isinstance(_rule, dict) else None
                    dst = _rule.get("target") if isinstance(_rule, dict) else None
                    patt = _rule.get("pattern") if isinstance(_rule, dict) else None
                except Exception:
                    src = dst = patt = None
                try:
                    logger_obj.error(
                        (
                            "validation: adjacency[%d] expands to zero "
                            "links: source=%r target=%r pattern=%r"
                        ),
                        idx,
                        src,
                        dst,
                        patt,
                    )
                except Exception:
                    pass
                issues.append(
                    f"adjacency[{idx}] expands to 0 links (source={src}, target={dst}, pattern={patt})"
                )

    # --- Blueprint adjacency -> links ---
    blueprints = dsl.get("blueprints") or {}
    if isinstance(blueprints, dict) and blueprints:
        for bp_name, bp_def in blueprints.items():
            if not isinstance(bp_def, dict):
                continue
            bp_adj = bp_def.get("adjacency", [])
            if not isinstance(bp_adj, list) or not bp_adj:
                continue
            bp_copy = _deepcopy(bp_def)
            tag_attr = "_tg_bp_adj_tag"
            expected_tags: list[str] = []
            for i, rule in enumerate(bp_copy.get("adjacency", [])):
                if not isinstance(rule, dict):
                    continue
                lp = rule.get("link_params")
                if not isinstance(lp, dict):
                    lp = {}
                    rule["link_params"] = lp
                attrs = lp.get("attrs")
                if not isinstance(attrs, dict):
                    attrs = {}
                    lp["attrs"] = attrs
                tag_val = f"{bp_name}#{i}"
                expected_tags.append(tag_val)
                attrs[tag_attr] = tag_val
            dsl_bp = {
                "blueprints": {str(bp_name): bp_copy},
                "network": {"groups": {"__check__": {"use_blueprint": str(bp_name)}}},
            }
            net_bp = ng_expand(dsl_bp)
            seen: set[str] = set()
            for link in net_bp.links.values():
                tag_val = getattr(link, "attrs", {}).get(tag_attr)
                if isinstance(tag_val, str):
                    seen.add(tag_val)
            for i, tag_val in enumerate(expected_tags):
                if tag_val not in seen:
                    try:
                        logger_obj.error(
                            (
                                "validation: blueprint '%s' "
                                "adjacency[%d] expands to zero links"
                            ),
                            bp_name,
                            i,
                        )
                    except Exception:
                        pass
                    issues.append(
                        f"blueprint '{bp_name}' adjacency[{i}] expands to 0 links"
                    )

    return issues
