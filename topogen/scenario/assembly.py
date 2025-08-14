"""Scenario assembly orchestrator and YAML post-processing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from topogen.blueprints_lib import get_builtin_blueprints
from topogen.log_config import get_logger

from .config import _determine_metro_settings
from .graph_pipeline import (
    assign_per_link_capacity,
    build_site_graph,
    tm_based_size_capacities,
    to_network_sections,
)
from .libraries import _build_blueprints_section, _build_components_section
from .network import _extract_metros_from_graph
from .policies import _build_failure_policy_set_section, _build_workflow_section
from .risk import _build_risk_groups_section
from .traffic import _build_traffic_matrix_section

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    import networkx as nx

    from topogen.config import TopologyConfig

logger = get_logger(__name__)


def _emit_yaml(scenario: dict[str, Any], *, yaml_anchors: bool = True) -> str:
    emit_anchors = bool(yaml_anchors)
    if emit_anchors:
        yaml_output = yaml.safe_dump(
            scenario, sort_keys=False, default_flow_style=False
        )
    else:

        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):  # type: ignore[override]
                return True

        yaml_output = yaml.dump(
            scenario,
            Dumper=NoAliasDumper,
            sort_keys=False,
            default_flow_style=False,
        )
    yaml_output = _add_adjacency_comments(yaml_output)
    logger.info("Generated NetGraph scenario YAML")
    return yaml_output


def build_scenario(graph: "nx.Graph", config: "TopologyConfig") -> str:
    """Build a NetGraph scenario YAML from the integrated graph.

    Transforms each metro node into a site-level MultiGraph (PoPs and DC regions),
    assigns base and per-link capacities, serializes `blueprints`, `components`,
    and `network` sections, and optionally augments class-level adjacency with
    per-end hardware derived from optics mapping.

    Args:
        graph: Integrated metro-highway graph loaded from JSON.
        config: Topology configuration object.

    Returns:
        YAML string representing the NetGraph scenario.

    Notes:
        - Validation is performed by the caller (CLI) after YAML emission using
          `topogen.validation.validate_scenario_yaml`.
        - Side artifacts may be written if enabled in `config` (network graph JSON,
          optional visualization exports).

    Raises:
        ValueError: If unknown blueprints are referenced; if ring-based adjacency
            requires a positive metro radius; or if capacity assignment detects an
            invalid or zero-size expansion.
    """
    logger.info("Building NetGraph scenario from integrated graph")

    metros = _extract_metros_from_graph(graph)
    logger.info(f"Found {len(metros)} metro nodes")

    metro_settings = _determine_metro_settings(metros, config)
    max_sites = max((s["pop_per_metro"] for s in metro_settings.values()), default=1)
    max_dc_regions = max(
        (s["dc_regions_per_metro"] for s in metro_settings.values()), default=0
    )
    logger.info(f"Maximum sites per metro: {max_sites}")
    logger.info(f"Maximum DC regions per metro: {max_dc_regions}")

    scenario: dict[str, Any] = {}

    try:
        scenario_seed = int(getattr(config.output, "scenario_seed", 42))
    except Exception:
        scenario_seed = 42
    scenario["seed"] = scenario_seed

    # New graph-based pipeline builds the authoritative site graph first
    logger.info("Building site-level MultiGraph")
    G = build_site_graph(metros, metro_settings, graph, config)
    # Optional TM-based sizing before per-link capacity split
    try:
        tm_based_size_capacities(G, metros, metro_settings, config)
    except Exception:
        # Fail fast with clear error; no silent fallback
        raise
    # Log clearly which capacity path is in effect
    try:
        tm_enabled = bool(
            getattr(getattr(config.build, "tm_sizing", object()), "enabled", False)
        )
    except Exception:
        tm_enabled = False
    if tm_enabled:
        logger.info("Assigning per-link capacities after TM-based sizing")
    else:
        logger.info("Assigning per-link capacities from configured base capacities")
    assign_per_link_capacity(G, config)

    # Persist the site-level network graph JSON artefact in configured output dir
    try:
        from .graph_pipeline import save_site_graph_json

        # Prefer configured output directory; fallback to CWD
        cfg_out = getattr(config, "_output_dir", None)
        output_dir = Path(cfg_out) if isinstance(cfg_out, (str, Path)) else Path.cwd()
        src_path = getattr(config, "_source_path", None)
        stem = Path(src_path).stem if isinstance(src_path, (str, Path)) else "scenario"
        network_graph_path = output_dir / f"{stem}_network_graph.json"
        logger.info(
            "Saving site-level network graph to JSON: %s",
            str(network_graph_path),
        )
        fmt = getattr(getattr(config, "output", None), "formatting", None)
        json_indent = int(getattr(fmt, "json_indent", 2)) if fmt is not None else 2
        save_site_graph_json(G, network_graph_path, json_indent=json_indent)
    except Exception as e:  # pragma: no cover - best-effort artefact save
        logger.warning("Failed to save site-level network graph: %s", e)

    # Optional: export a JPEG visualization of the site-level graph (match integrated graph DPI)
    try:
        if bool(getattr(config, "_export_site_graph", False)):
            from topogen.visualization import export_site_graph_map

            # Prefer configured output directory; fallback to CWD
            cfg_out = getattr(config, "_output_dir", None)
            output_dir = (
                Path(cfg_out) if isinstance(cfg_out, (str, Path)) else Path.cwd()
            )
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            src_path = getattr(config, "_source_path", None)
            stem = (
                Path(src_path).stem if isinstance(src_path, (str, Path)) else "scenario"
            )
            site_vis_path = output_dir / f"{stem}_site_graph.jpg"
            logger.info(
                "Exporting site-level graph visualization to: %s", str(site_vis_path)
            )
            dpi = int(getattr(config, "_visualization_dpi", 300))
            target_crs = str(
                getattr(getattr(config, "projection", object()), "target_crs", "")
            )
            export_site_graph_map(G, site_vis_path, dpi=dpi, target_crs=target_crs)
    except Exception as e:  # pragma: no cover - best-effort
        logger.warning("Failed to export site-level graph visualization: %s", e)

    # Determine used blueprints directly from the site graph (source of truth)
    used_blueprints = {
        str(data.get("site_blueprint", "")) for _n, data in G.nodes(data=True)
    }
    used_blueprints = {bp for bp in used_blueprints if bp}
    builtin_blueprints = get_builtin_blueprints()
    for bp_name in used_blueprints:
        if bp_name not in builtin_blueprints:
            available = ", ".join(sorted(builtin_blueprints.keys()))
            raise ValueError(f"Unknown blueprint '{bp_name}'. Available: {available}")

    # Emit libraries first to preserve expected YAML ordering
    scenario["blueprints"] = _build_blueprints_section(used_blueprints, config)
    scenario["components"] = _build_components_section(config, used_blueprints)

    logger.info("Serializing network sections from MultiGraph")
    groups, adjacency = to_network_sections(G, metros, metro_settings, config)
    scenario["network"] = {"groups": groups, "adjacency": adjacency}

    # Build risk groups and other sections prior to late HW so early exits still include them
    risk_groups = _build_risk_groups_section(graph, config)
    if risk_groups:
        scenario["risk_groups"] = risk_groups
    scenario["failure_policy_set"] = _build_failure_policy_set_section(config)
    traffic_section = _build_traffic_matrix_section(metros, metro_settings, config)
    if traffic_section:
        scenario["traffic_matrix_set"] = traffic_section
    scenario["workflow"] = _build_workflow_section(config)
    # nothing to embed; keep scenario dict YAML-serializable only

    # --- Late HW resolution using expanded DSL ---
    from collections.abc import Mapping as _Mapping

    comp_obj = getattr(config, "components", None)
    raw_optics = getattr(comp_obj, "optics", {}) if comp_obj is not None else {}
    optics_enabled = isinstance(raw_optics, _Mapping) and len(raw_optics) > 0

    try:
        from ngraph.dsl.blueprints.expand import (  # type: ignore[import-not-found]
            expand_network_dsl as _ng_expand,
        )

        from topogen.components_lib import (
            get_builtin_components as _get_components_lib,
        )
    except Exception as exc:
        # If optics are configured, expansion is required
        if optics_enabled:
            raise RuntimeError(
                "Late hardware resolution requires DSL expansion when optics mapping is configured"
            ) from exc
        _fmt = getattr(getattr(config, "output", None), "formatting", None)
        _anchors = (
            bool(getattr(_fmt, "yaml_anchors", True)) if _fmt is not None else True
        )
        return _emit_yaml(scenario, yaml_anchors=_anchors)

    def _count_for_optic(name: str, capacity: float) -> float:
        comps = _get_components_lib() or {}
        spec = comps.get(name)
        if spec is None:
            raise ValueError(f"Unknown component '{name}' in optics mapping")
        total = float(spec.get("capacity", 0.0))
        if total <= 0.0:
            raise ValueError(f"Optic '{name}' must have positive capacity")
        import math as _m

        modules = float(_m.ceil(capacity / total))
        return modules

    # Build probe network from current scenario to inspect concrete roles per adjacency
    try:
        probe = {
            "blueprints": scenario["blueprints"],
            "network": scenario["network"],
        }
        net = _ng_expand(probe)
    except Exception as exc:
        if optics_enabled:
            raise RuntimeError(
                "Late hardware resolution failed to expand network DSL with optics configured"
            ) from exc
        _fmt = getattr(getattr(config, "output", None), "formatting", None)
        _anchors = (
            bool(getattr(_fmt, "yaml_anchors", True)) if _fmt is not None else True
        )
        return _emit_yaml(scenario, yaml_anchors=_anchors)

    # Build endpoint role lookup
    node_role: dict[str, str] = {}
    for node in net.nodes.values():
        r = str(node.attrs.get("role", "")).strip()
        if r:
            node_role[str(node.name)] = r

    # Collect roles/capacities per adjacency id
    from collections import defaultdict

    per_adj: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for link in net.links.values():
        src = str(link.source)
        dst = str(link.target)
        rs = node_role.get(src, "")
        rd = node_role.get(dst, "")
        if not (rs and rd):
            continue
        aid = str(link.attrs.get("adjacency_id", ""))
        if not aid:
            # Fallback to link_type when adjacency id is absent
            aid = str(link.attrs.get("link_type", ""))
        per_adj[aid].append((rs, rd, float(link.capacity)))

    # Build optics lookup (unordered A|B and directional A-B)
    optics_lookup: dict[tuple[str, str], str] = {}
    if isinstance(raw_optics, _Mapping) and len(raw_optics) > 0:
        for k, v in raw_optics.items():
            key = str(k)
            if "|" in key:
                a, b = [p.strip() for p in key.split("|", 1)]
                if a and b:
                    optics_lookup[(a, b)] = str(v)
                    optics_lookup[(b, a)] = str(v)
            elif "-" in key:
                a, b = [p.strip() for p in key.split("-", 1)]
                optics_lookup[(a, b)] = str(v)
        try:
            logger.debug("Optics lookup built: %d entries", len(optics_lookup))
        except Exception:
            pass
    # Augment class-level link_params when a single role pair is present post-expansion
    if optics_lookup:
        augmented_total = 0
        augmented_by_pair: dict[tuple[str, str], int] = {}
        for adj in scenario["network"].get("adjacency", []):
            lp = adj.get("link_params", {})
            attrs = lp.get("attrs", {})
            aid = str(attrs.get("adjacency_id") or attrs.get("link_type") or "")
            if not aid or aid not in per_adj:
                continue
            entries = per_adj[aid]
            roles = {r for s, t, _ in entries for r in (s, t)}
            if len(roles) > 2:
                raise ValueError(
                    f"Adjacency '{aid}' expands to more than two roles: {sorted(roles)}"
                )
            pairs = {(s, t) for s, t, _ in entries}
            if len(pairs) != 1:
                # Mixed role-pairs in one class: leave HW unset
                try:
                    logger.debug(
                        "HW: skip adj='%s' due to multiple role pairs: %s",
                        aid,
                        sorted(list(pairs)),
                    )
                except Exception:
                    pass
                continue
            (sr, tr) = next(iter(pairs))
            optic = optics_lookup.get((sr, tr))
            if not optic:
                try:
                    logger.debug(
                        "HW: no optic configured for adj='%s' pair=(%s,%s); skipping",
                        aid,
                        sr,
                        tr,
                    )
                except Exception:
                    pass
                continue
            cap = float(entries[0][2])
            count = _count_for_optic(optic, cap)
            attrs.setdefault("hardware", {})
            attrs["hardware"]["source"] = {"component": optic, "count": float(count)}
            attrs["hardware"]["target"] = {"component": optic, "count": float(count)}
            lp["attrs"] = attrs
            adj["link_params"] = lp
            augmented_total += 1
            augmented_by_pair[(sr, tr)] = augmented_by_pair.get((sr, tr), 0) + 1
            try:
                logger.debug(
                    "HW: adj='%s' pair=(%s,%s) optic=%s capacity=%s count=%.3f",
                    aid,
                    sr,
                    tr,
                    optic,
                    f"{cap:.0f}",
                    float(count),
                )
            except Exception:
                pass

        # Aggregate summary at INFO level
        try:
            if augmented_total > 0:
                top_pairs = sorted(
                    augmented_by_pair.items(), key=lambda kv: kv[1], reverse=True
                )[:3]
                if top_pairs:
                    examples = ", ".join(
                        f"({a},{b})={cnt}" for ((a, b), cnt) in top_pairs
                    )
                    logger.info(
                        "HW: applied optics to %d adjacencies (top role-pairs: %s)",
                        augmented_total,
                        examples,
                    )
                else:
                    logger.info("HW: applied optics to %d adjacencies", augmented_total)
        except Exception:
            pass

    _fmt = getattr(getattr(config, "output", None), "formatting", None)
    _anchors = bool(getattr(_fmt, "yaml_anchors", True)) if _fmt is not None else True
    # Optional: export per-blueprint diagrams (abstract + concrete)
    try:
        if bool(getattr(config, "_export_blueprint_diagrams", False)):
            from topogen.visualization import export_blueprint_diagram

            # Determine used blueprints again (same as before)
            used_blueprints = set(scenario.get("blueprints", {}).keys())
            # Select a representative site with maximum attached capacity per blueprint
            # Build lookup: site node name -> total incident link capacity from expanded net
            attached: dict[str, float] = {}
            for link in net.links.values():
                cap = float(getattr(link, "capacity", 0.0) or 0.0)
                s = str(getattr(link, "source", ""))
                t = str(getattr(link, "target", ""))
                if s:
                    attached[s] = attached.get(s, 0.0) + cap
                if t:
                    attached[t] = attached.get(t, 0.0) + cap

            # Build mapping: site path -> blueprint used (from groups section attrs)
            # We use the authoritative groups section emitted earlier
            site_to_bp: dict[str, str] = {}
            for gpath, gdef in scenario.get("network", {}).get("groups", {}).items():
                bp = str(gdef.get("use_blueprint", ""))
                if not bp:
                    continue
                # Expand ranges like pop[1-4] into concrete prefixes
                import re as _re

                m = _re.match(r"^(?P<prefix>.+?)\[(?P<a>\d+)-(?:\d+)\]$", gpath)
                if m:
                    prefix = m.group("prefix")
                    # Find any nodes whose path starts with this prefix in expanded net
                    for node in net.nodes.values():
                        nname = str(node.name)
                        if nname.startswith(prefix.rstrip("/")):
                            head = nname.split("/", 2)[0:2]
                            site_path = "/".join(head)
                            site_to_bp[site_path] = bp
                else:
                    # Non-ranged path; take as-is
                    # Normalize to the first two components (site scope)
                    head = gpath.split("/", 2)[0:2]
                    site_path = "/".join(head)
                    site_to_bp[site_path] = bp

            # For each used blueprint, pick site with max attached capacity
            bp_to_best_site: dict[str, str] = {}
            for site_path, bp in site_to_bp.items():
                # Aggregate attached over concrete nodes under the site
                prefix = f"{site_path}/"
                tot = 0.0
                for node_name, val in attached.items():
                    if str(node_name).startswith(prefix):
                        tot += float(val)
                prev_site = bp_to_best_site.get(bp)
                if prev_site is None:
                    bp_to_best_site[bp] = site_path
                else:
                    # Compare totals
                    prev_tot = 0.0
                    for node_name, val in attached.items():
                        if str(node_name).startswith(f"{prev_site}/"):
                            prev_tot += float(val)
                    if tot > prev_tot:
                        bp_to_best_site[bp] = site_path

            # Export one diagram per blueprint actually used
            cfg_out = getattr(config, "_output_dir", None)
            output_dir = (
                Path(cfg_out) if isinstance(cfg_out, (str, Path)) else Path.cwd()
            )
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            src_path = getattr(config, "_source_path", None)
            stem = (
                Path(src_path).stem if isinstance(src_path, (str, Path)) else "scenario"
            )
            dpi = int(getattr(config, "_visualization_dpi", 300))
            blueprints_defs = scenario.get("blueprints", {})
            for bp in sorted(used_blueprints):
                if bp not in blueprints_defs:
                    continue
                site = bp_to_best_site.get(bp)
                if not site:
                    continue
                out_path = output_dir / f"{stem}_blueprint_{bp}.jpg"
                export_blueprint_diagram(
                    bp,
                    blueprints_defs[bp],
                    net,
                    site,
                    out_path,
                    dpi=dpi,
                )
    except Exception as e:  # pragma: no cover - non-fatal visualization optional
        logger.warning("Failed to export blueprint diagrams: %s", e)

    return _emit_yaml(scenario, yaml_anchors=_anchors)


def _add_adjacency_comments(yaml_content: str) -> str:
    """Add section comments to the adjacency section of the YAML."""
    lines = yaml_content.split("\n")
    result_lines: list[str] = []
    in_adjacency = False
    intra_metro_added = False
    inter_metro_added = False
    for i, line in enumerate(lines):
        if line.strip() == "adjacency:" and not in_adjacency:
            in_adjacency = True
            result_lines.append(line)
            continue
        if (
            in_adjacency
            and line
            and not line.startswith(" ")
            and not line.startswith("-")
        ):
            in_adjacency = False
        if in_adjacency and line.strip().startswith("- source:"):
            link_type = None
            for j in range(i, min(i + 15, len(lines))):
                if "link_type: intra_metro" in lines[j]:
                    link_type = "intra_metro"
                    break
                elif "link_type: inter_metro_corridor" in lines[j]:
                    link_type = "inter_metro"
                    break
            if link_type == "intra_metro" and not intra_metro_added:
                result_lines.append(
                    "  # Intra-metro adjacency (connectivity within each metro's sites)"
                )
                intra_metro_added = True
            elif link_type == "inter_metro" and not inter_metro_added:
                result_lines.append(
                    "  # Inter-metro corridor connectivity (backbone links between metros)"
                )
                inter_metro_added = True
        result_lines.append(line)
    return "\n".join(result_lines)
