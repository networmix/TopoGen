"""Capacity allocation for inter-metro links.

Emits explicit per-POP-pair inter-metro adjacencies and computes per-link
capacities. When platform constraints are present via component capacities,
configured capacities are treated as minimums and remaining capacity is
distributed using a round-robin strategy. Without hardware constraints,
capacities remain at the configured base values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import networkx as nx

from topogen.log_config import get_logger

from .utils import _count_nodes_with_role

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def _apply_capacity_allocation(
    scenario: dict[str, Any],
    metros: list[dict[str, Any]],
    metro_settings: dict[str, dict[str, Any]],
    graph: nx.Graph,
    config: "TopologyConfig",
) -> None:
    """Allocate inter-metro capacities and emit per-POP-pair adjacencies.

    Keeps configured capacities as minimums and, when platform capacity is
    finite for any endpoint, distributes remaining capacity to inter-metro
    POP-to-POP links using a round-robin strategy.
    """
    components: dict[str, dict[str, Any]] = scenario.get("components", {})
    blueprints: dict[str, dict[str, Any]] = scenario.get("blueprints", {})
    metro_idx_map = {metro["name"]: idx for idx, metro in enumerate(metros, 1)}

    pop_capacity: dict[tuple[int, int], float] = {}
    dc_capacity: dict[tuple[int, int], float] = {}
    pop_constrained: dict[tuple[int, int], bool] = {}
    dc_constrained: dict[tuple[int, int], bool] = {}

    for idx, metro in enumerate(metros, 1):
        settings = metro_settings[metro["name"]]
        sites_count = int(settings["pop_per_metro"])
        dc_count = int(settings["dc_regions_per_metro"])
        pop_bp_name = settings["site_blueprint"]
        pop_bp = blueprints.get(pop_bp_name, {})
        pop_groups = pop_bp.get("groups", {})
        core_group = pop_groups.get("core", {})
        core_attrs = core_group.get("attrs", {})
        core_hw = core_attrs.get("hw_component")
        comp_entry = components.get(core_hw, {}) if core_hw else {}
        core_cap_val = comp_entry.get("capacity")
        core_cap = float(core_cap_val) if core_cap_val is not None else float("inf")
        for p in range(1, sites_count + 1):
            key = (idx, p)
            pop_capacity[key] = core_cap
            pop_constrained[key] = core_cap != float("inf")
        if dc_count > 0:
            dc_bp_name = settings["dc_region_blueprint"]
            dc_bp = blueprints.get(dc_bp_name, {})
            dc_groups = dc_bp.get("groups", {})
            dc_group = dc_groups.get("dc", {})
            dc_attrs = dc_group.get("attrs", {})
            dc_hw = dc_attrs.get("hw_component")
            comp_entry2 = components.get(dc_hw, {}) if dc_hw else {}
            dc_cap_val = comp_entry2.get("capacity")
            dc_cap = float(dc_cap_val) if dc_cap_val is not None else float("inf")
            for d in range(1, dc_count + 1):
                key = (idx, d)
                dc_capacity[key] = dc_cap
                dc_constrained[key] = dc_cap != float("inf")

    pop_budget: dict[tuple[int, int], float] = {k: v for k, v in pop_capacity.items()}
    dc_budget: dict[tuple[int, int], float] = {k: v for k, v in dc_capacity.items()}

    for idx, metro in enumerate(metros, 1):
        settings = metro_settings[metro["name"]]
        s = int(settings["pop_per_metro"])
        if s <= 1:
            continue
        c_intra = float(settings["intra_metro_link"]["capacity"])
        reserve = max(0, s - 1) * c_intra
        for p in range(1, s + 1):
            key = (idx, p)
            if key in pop_budget:
                pop_budget[key] = pop_budget.get(key, float("inf")) - reserve

    for idx, metro in enumerate(metros, 1):
        settings = metro_settings[metro["name"]]
        s = int(settings["pop_per_metro"])
        d = int(settings["dc_regions_per_metro"])
        if s <= 0 or d <= 0:
            continue
        c_dp = float(settings["dc_to_pop_link"]["capacity"])
        for p in range(1, s + 1):
            key = (idx, p)
            if key in pop_budget:
                pop_budget[key] = pop_budget.get(key, float("inf")) - d * c_dp
        for dc in range(1, d + 1):
            key = (idx, dc)
            if key in dc_budget:
                dc_budget[key] = dc_budget.get(key, float("inf")) - s * c_dp

    from .adjacency import _extract_corridor_edges

    corridor_edges = _extract_corridor_edges(graph)
    sites_per_metro: dict[int, int] = {
        metro_idx_map[m["name"]]: int(metro_settings[m["name"]]["pop_per_metro"])  # type: ignore[index]
        for m in metros
    }

    for edge in corridor_edges:
        source = edge["source"]
        target = edge["target"]
        src_metro = next(m for m in metros if m["node_key"] == source)
        tgt_metro = next(m for m in metros if m["node_key"] == target)
        s_idx = metro_idx_map[src_metro["name"]]
        t_idx = metro_idx_map[tgt_metro["name"]]
        s_sites = sites_per_metro[s_idx]
        t_sites = sites_per_metro[t_idx]
        src_settings = metro_settings[src_metro["name"]]
        default_c = float(src_settings["inter_metro_link"]["capacity"])
        base_c = float(edge.get("capacity", default_c))
        for p in range(1, s_sites + 1):
            key = (s_idx, p)
            if key in pop_budget:
                pop_budget[key] = pop_budget.get(key, float("inf")) - t_sites * base_c
        for q in range(1, t_sites + 1):
            key = (t_idx, q)
            if key in pop_budget:
                pop_budget[key] = pop_budget.get(key, float("inf")) - s_sites * base_c

    for (idx, p), remaining in pop_budget.items():
        if remaining < -1e-9:
            metro_name = next(
                m["name"] for m in metros if metro_idx_map[m["name"]] == idx
            )
            raise ValueError(
                f"Base capacities exceed platform at metro {metro_name} pop{p}: remaining {remaining:.0f} Gbps < 0"
            )
    for (idx, d), remaining in dc_budget.items():
        if remaining < -1e-9:
            metro_name = next(
                m["name"] for m in metros if metro_idx_map[m["name"]] == idx
            )
            raise ValueError(
                f"Base capacities exceed platform at metro {metro_name} dc{d}: remaining {remaining:.0f} Gbps < 0"
            )

    candidates: list[tuple[int, int, int, int, float]] = []
    for edge in corridor_edges:
        src_metro = next(m for m in metros if m["node_key"] == edge["source"])
        tgt_metro = next(m for m in metros if m["node_key"] == edge["target"])
        s_idx = metro_idx_map[src_metro["name"]]
        t_idx = metro_idx_map[tgt_metro["name"]]
        s_sites = sites_per_metro[s_idx]
        t_sites = sites_per_metro[t_idx]
        src_settings = metro_settings[src_metro["name"]]
        default_c = float(src_settings["inter_metro_link"]["capacity"])
        base_c = float(edge.get("capacity", default_c))
        step = base_c
        for p in range(1, s_sites + 1):
            for q in range(1, t_sites + 1):
                if pop_constrained.get((s_idx, p), False) or pop_constrained.get(
                    (t_idx, q), False
                ):
                    candidates.append((s_idx, p, t_idx, q, step))

    increments: dict[tuple[int, int, int, int], int] = {}
    if candidates:
        progressed = True
        while progressed:
            progressed = False
            for s_idx, p, t_idx, q, step in candidates:
                key_s = (s_idx, p)
                key_t = (t_idx, q)
                need_s = pop_constrained.get(key_s, False)
                need_t = pop_constrained.get(key_t, False)
                b_s = pop_budget.get(key_s, float("inf"))
                b_t = pop_budget.get(key_t, float("inf"))
                ok_s = (not need_s) or (b_s >= step)
                ok_t = (not need_t) or (b_t >= step)
                if ok_s and ok_t:
                    if need_s:
                        pop_budget[key_s] = b_s - step
                    if need_t:
                        pop_budget[key_t] = b_t - step
                    pair_key = (s_idx, p, t_idx, q)
                    increments[pair_key] = increments.get(pair_key, 0) + 1
                    progressed = True

    final_capacity: dict[tuple[int, int, int, int], int] = {}
    for edge in corridor_edges:
        src_metro = next(m for m in metros if m["node_key"] == edge["source"])
        tgt_metro = next(m for m in metros if m["node_key"] == edge["target"])
        s_idx = metro_idx_map[src_metro["name"]]
        t_idx = metro_idx_map[tgt_metro["name"]]
        s_sites = sites_per_metro[s_idx]
        t_sites = sites_per_metro[t_idx]
        src_settings = metro_settings[src_metro["name"]]
        default_c = float(src_settings["inter_metro_link"]["capacity"])
        base_c = float(edge.get("capacity", default_c))
        for p in range(1, s_sites + 1):
            for q in range(1, t_sites + 1):
                k = increments.get((s_idx, p, t_idx, q), 0)
                final_capacity[(s_idx, p, t_idx, q)] = int(base_c + k * base_c)

    network = scenario.setdefault("network", {})
    old_adj: list[dict[str, Any]] = network.get("adjacency", [])
    new_adj: list[dict[str, Any]] = []
    for adj in old_adj:
        try:
            attrs = adj.get("link_params", {}).get("attrs", {})
            if attrs.get("link_type") == "inter_metro_corridor":
                continue
        except Exception:
            pass
        new_adj.append(adj)

    bp_defs: dict[str, Any] = scenario.get("blueprints", {})
    for edge in corridor_edges:
        src_metro = next(m for m in metros if m["node_key"] == edge["source"])
        tgt_metro = next(m for m in metros if m["node_key"] == edge["target"])
        s_idx = metro_idx_map[src_metro["name"]]
        t_idx = metro_idx_map[tgt_metro["name"]]
        s_sites = sites_per_metro[s_idx]
        t_sites = sites_per_metro[t_idx]
        src_settings = metro_settings[src_metro["name"]]
        inter_attrs = dict(src_settings["inter_metro_link"]["attrs"])  # copy
        base_cost = math.ceil(
            edge.get("length_km", src_settings["inter_metro_link"]["cost"])
        )
        edge_risk_groups = edge.get("risk_groups", [])
        for p in range(1, s_sites + 1):
            for q in range(1, t_sites + 1):
                cap_total = final_capacity[(s_idx, p, t_idx, q)]
                src_bp = src_settings["site_blueprint"]
                tgt_bp = metro_settings[tgt_metro["name"]]["site_blueprint"]
                src_core = max(1, _count_nodes_with_role(bp_defs, src_bp, "core"))
                tgt_core = max(1, _count_nodes_with_role(bp_defs, tgt_bp, "core"))
                inter_divisor = max(1, min(src_core, tgt_core))
                cap_each = int(max(1, int(cap_total) // int(inter_divisor)))
                src_bp_name = src_settings["site_blueprint"]
                tgt_bp_name = metro_settings[tgt_metro["name"]]["site_blueprint"]
                _src_bp = bp_defs.get(src_bp_name, {})
                _tgt_bp = bp_defs.get(tgt_bp_name, {})
                _src_groups = _src_bp.get("groups", {})
                _tgt_groups = _tgt_bp.get("groups", {})
                src_total = int(
                    sum(int(g.get("node_count", 0)) for g in _src_groups.values())
                )
                tgt_total = int(
                    sum(int(g.get("node_count", 0)) for g in _tgt_groups.values())
                )
                src_core_count = max(
                    1, _count_nodes_with_role(bp_defs, src_bp_name, "core")
                )
                tgt_core_count = max(
                    1, _count_nodes_with_role(bp_defs, tgt_bp_name, "core")
                )
                needs_match = (src_core_count < src_total) or (
                    tgt_core_count < tgt_total
                )
                if needs_match:
                    entry = {
                        "source": {
                            "path": f"metro{s_idx}/pop{p}",
                            "match": {
                                "conditions": [
                                    {"attr": "role", "operator": "==", "value": "core"}
                                ]
                            },
                        },
                        "target": {
                            "path": f"metro{t_idx}/pop{q}",
                            "match": {
                                "conditions": [
                                    {"attr": "role", "operator": "==", "value": "core"}
                                ]
                            },
                        },
                        "pattern": "one_to_one",
                        "link_params": {
                            "capacity": cap_each,
                            "cost": base_cost,
                            "attrs": {
                                **inter_attrs,
                                "distance_km": base_cost,
                                "source_metro": src_metro["name"],
                                "source_metro_orig": src_metro.get(
                                    "name_orig", src_metro["name"]
                                ),
                                "target_metro": tgt_metro["name"],
                                "target_metro_orig": tgt_metro.get(
                                    "name_orig", tgt_metro["name"]
                                ),
                            },
                        },
                    }
                else:
                    entry = {
                        "source": f"metro{s_idx}/pop{p}",
                        "target": f"metro{t_idx}/pop{q}",
                        "pattern": "one_to_one",
                        "link_params": {
                            "capacity": cap_each,
                            "cost": base_cost,
                            "attrs": {
                                **inter_attrs,
                                "distance_km": base_cost,
                                "source_metro": src_metro["name"],
                                "source_metro_orig": src_metro.get(
                                    "name_orig", src_metro["name"]
                                ),
                                "target_metro": tgt_metro["name"],
                                "target_metro_orig": tgt_metro.get(
                                    "name_orig", tgt_metro["name"]
                                ),
                            },
                        },
                    }
                if edge_risk_groups:
                    entry["link_params"]["risk_groups"] = edge_risk_groups
                new_adj.append(entry)
    network["adjacency"] = new_adj
