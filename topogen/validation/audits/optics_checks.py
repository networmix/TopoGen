"""Link optics mapping and blueprint-provided hardware checks."""

from __future__ import annotations

from collections import defaultdict as _dd
from typing import Any, Dict


def _role_of(net: Any, node_name: str) -> str:
    try:
        return str(getattr(net.nodes[node_name], "attrs", {}).get("role", ""))
    except Exception:
        return ""


def _build_optics_map(
    optics_cfg: Dict[str, Any],
) -> tuple[
    Dict[tuple[str, str], str],
    Dict[tuple[str, str], set[str]],
    Dict[tuple[str, str], set[str]],
]:
    """Return (optics_map, optics_values, mapping_conflicts)."""
    optics_map: dict[tuple[str, str], str] = {}
    optics_values: dict[tuple[str, str], set[str]] = {}
    mapping_conflicts: dict[tuple[str, str], set[str]] = {}
    if isinstance(optics_cfg, dict) and optics_cfg:
        unordered_by_pair: dict[frozenset[str], list[tuple[str, str, str]]] = {}
        for key, val in optics_cfg.items():
            if not isinstance(val, str):
                continue
            k = str(key)
            if "-" in k and "|" not in k:
                a, b = [p.strip() for p in k.split("-", 1)]
                if a and b:
                    key2 = (a, b)
                    ov = str(val)
                    if key2 not in optics_map:
                        optics_map[key2] = ov
                    if key2 not in optics_values:
                        optics_values[key2] = set()
                    optics_values[key2].add(ov)
            elif "|" in k:
                a, b = [p.strip() for p in k.split("|", 1)]
                if a and b:
                    sig = frozenset((a, b))
                    unordered_by_pair.setdefault(sig, []).append((a, b, str(val)))
        # Resolve unordered declarations
        for _sig, entries in unordered_by_pair.items():
            if len(entries) == 1:
                a, b, ov = entries[0]
                for key2 in ((a, b), (b, a)):
                    if key2 not in optics_map:
                        optics_map[key2] = ov
                    if key2 not in optics_values:
                        optics_values[key2] = set()
                    optics_values[key2].add(ov)
            else:
                for a, b, ov in entries:
                    key2 = (a, b)
                    if key2 not in optics_map:
                        optics_map[key2] = ov
                    if key2 not in optics_values:
                        optics_values[key2] = set()
                    optics_values[key2].add(ov)

        # Conflicts
        for key2, values in optics_values.items():
            if len(values) > 1:
                mapping_conflicts[key2] = values

    return optics_map, optics_values, mapping_conflicts


def check_link_optics(
    net: Any, d: Dict[str, Any], comp_lib: Dict[str, Any]
) -> list[str]:
    """Aggregate optics mapping, blueprint hardware presence/capacity checks."""
    issues: list[str] = []

    optics_cfg = (d.get("components") or {}).get("optics", {})
    optics_map, _optics_values, mapping_conflicts = _build_optics_map(optics_cfg)

    if not optics_map and not mapping_conflicts:
        return issues

    miss_src: dict[tuple[str, str], int] = _dd(int)
    miss_tgt: dict[tuple[str, str], int] = _dd(int)
    miss_src_samples: dict[tuple[str, str], list[str]] = _dd(list)
    miss_tgt_samples: dict[tuple[str, str], list[str]] = _dd(list)
    # Missing mapping when hardware absent
    miss_map_src: dict[tuple[str, str], int] = _dd(int)
    miss_map_tgt: dict[tuple[str, str], int] = _dd(int)
    miss_map_src_samples: dict[tuple[str, str], list[str]] = _dd(list)
    miss_map_tgt_samples: dict[tuple[str, str], list[str]] = _dd(list)
    # Aggregation for blueprint-provided hardware checks
    bhw_missing_src: dict[tuple[str, str], int] = _dd(int)
    bhw_missing_tgt: dict[tuple[str, str], int] = _dd(int)
    bhw_cap_short: dict[tuple[str, str], int] = _dd(int)
    bhw_unknown_src: dict[tuple[str, str], int] = _dd(int)
    bhw_unknown_tgt: dict[tuple[str, str], int] = _dd(int)
    bhw_calc_err: dict[tuple[str, str], int] = _dd(int)
    bhw_samples: dict[tuple[str, str], list[str]] = _dd(list)

    for link in net.links.values():
        src = str(getattr(link, "source", ""))
        dst = str(getattr(link, "target", ""))
        if not src or not dst:
            continue
        rs = _role_of(net, src)
        rd = _role_of(net, dst)
        if not rs or not rd:
            continue
        attrs_link = getattr(link, "attrs", {}) or {}
        aid = str(attrs_link.get("adjacency_id", "")).strip()
        if not aid:
            aid = str(attrs_link.get("link_type", "")).strip()
        hw = attrs_link.get("hardware")
        # If hardware present on the link, validate presence and capacity
        if isinstance(hw, dict):
            src_hw = hw.get("source") if isinstance(hw.get("source"), dict) else None
            tgt_hw = hw.get("target") if isinstance(hw.get("target"), dict) else None
            if not src_hw:
                key = (rs, rd)
                bhw_missing_src[key] += 1
                if aid and len(bhw_samples[key]) < 3 and aid not in bhw_samples[key]:
                    bhw_samples[key].append(aid)
                continue
            if not tgt_hw:
                key = (rd, rs)
                bhw_missing_tgt[key] += 1
                if aid and len(bhw_samples[key]) < 3 and aid not in bhw_samples[key]:
                    bhw_samples[key].append(aid)
                continue
            try:
                comp_src = str(src_hw.get("component", "")).strip()
                comp_tgt = str(tgt_hw.get("component", "")).strip()
                c_src = comp_lib.get(comp_src) if comp_src else None
                c_tgt = comp_lib.get(comp_tgt) if comp_tgt else None
                if comp_src and not c_src:
                    key = (rs, rd)
                    bhw_unknown_src[key] += 1
                    if (
                        aid
                        and len(bhw_samples[key]) < 3
                        and aid not in bhw_samples[key]
                    ):
                        bhw_samples[key].append(aid)
                if comp_tgt and not c_tgt:
                    key = (rd, rs)
                    bhw_unknown_tgt[key] += 1
                    if (
                        aid
                        and len(bhw_samples[key]) < 3
                        and aid not in bhw_samples[key]
                    ):
                        bhw_samples[key].append(aid)
                cnt_src = float(src_hw.get("count", 1.0))
                cnt_tgt = float(tgt_hw.get("count", 1.0))
                av_src = (float(c_src.get("capacity", 0.0)) * cnt_src) if c_src else 0.0
                av_tgt = (float(c_tgt.get("capacity", 0.0)) * cnt_tgt) if c_tgt else 0.0
                need = float(getattr(link, "capacity", 0.0) or 0.0)
                if av_src + 1e-9 < need or av_tgt + 1e-9 < need:
                    key = (rs, rd)
                    bhw_cap_short[key] += 1
                    if (
                        aid
                        and len(bhw_samples[key]) < 3
                        and aid not in bhw_samples[key]
                    ):
                        bhw_samples[key].append(aid)
            except Exception:
                key = (rs, rd)
                bhw_calc_err[key] += 1
                if aid and len(bhw_samples[key]) < 3 and aid not in bhw_samples[key]:
                    bhw_samples[key].append(aid)
            continue

        # Enforce mapping only when hardware is not specified on the link
        req_src = optics_map.get((rs, rd))
        req_tgt = optics_map.get((rd, rs))
        if isinstance(req_src, str):
            key = (rs, rd)
            miss_src[key] += 1
            if (
                aid
                and len(miss_src_samples[key]) < 3
                and aid not in miss_src_samples[key]
            ):
                miss_src_samples[key].append(aid)
        else:
            key = (rs, rd)
            miss_map_src[key] += 1
            if (
                aid
                and len(miss_map_src_samples[key]) < 3
                and aid not in miss_map_src_samples[key]
            ):
                miss_map_src_samples[key].append(aid)
        if isinstance(req_tgt, str):
            key = (rd, rs)
            miss_tgt[key] += 1
            if (
                aid
                and len(miss_tgt_samples[key]) < 3
                and aid not in miss_tgt_samples[key]
            ):
                miss_tgt_samples[key].append(aid)
        else:
            key = (rd, rs)
            miss_map_tgt[key] += 1
            if (
                aid
                and len(miss_map_tgt_samples[key]) < 3
                and aid not in miss_map_tgt_samples[key]
            ):
                miss_map_tgt_samples[key].append(aid)

    # Emit aggregated issues (message text preserved)
    for (a, b), vals in sorted(mapping_conflicts.items()):
        vstr = ", ".join(sorted(vals))
        issues.append(
            f"optics mapping: conflicting values for roles ({a},{b}) -> {{{vstr}}}"
        )
    for (a, b), cnt in sorted(bhw_missing_src.items()):
        samples = bhw_samples.get((a, b), [])
        prefix = f"optics (blueprint): missing hardware on source end for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join("'" + s + "'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)
    for (a, b), cnt in sorted(bhw_missing_tgt.items()):
        samples = bhw_samples.get((a, b), [])
        prefix = f"optics (blueprint): missing hardware on target end for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join("'" + s + "'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)
    for (a, b), cnt in sorted(bhw_unknown_src.items()):
        samples = bhw_samples.get((a, b), [])
        prefix = f"optics (blueprint): unknown component on source end for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join("'" + s + "'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)
    for (a, b), cnt in sorted(bhw_unknown_tgt.items()):
        samples = bhw_samples.get((a, b), [])
        prefix = f"optics (blueprint): unknown component on target end for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join("'" + s + "'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)
    for (a, b), cnt in sorted(bhw_cap_short.items()):
        samples = bhw_samples.get((a, b), [])
        prefix = f"optics (blueprint): hardware capacity shortfall for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join("'" + s + "'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)
    for (a, b), cnt in sorted(bhw_calc_err.items()):
        samples = bhw_samples.get((a, b), [])
        prefix = f"optics (blueprint): capacity calculation error for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join("'" + s + "'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)
    for (a, b), cnt in sorted(miss_map_src.items()):
        samples = miss_map_src_samples.get((a, b), [])
        prefix = f"optics mapping: missing source-end mapping for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join("'" + s + "'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)
    for (a, b), cnt in sorted(miss_map_tgt.items()):
        samples = miss_map_tgt_samples.get((a, b), [])
        prefix = f"optics mapping: missing target-end mapping for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join("'" + s + "'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)
    for (a, b), cnt in sorted(miss_src.items()):
        samples = miss_src_samples.get((a, b), [])
        prefix = f"optics: missing hardware required by mapping on source end for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join(f"'{s}'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)
    for (a, b), cnt in sorted(miss_tgt.items()):
        samples = miss_tgt_samples.get((a, b), [])
        prefix = f"optics: missing hardware required by mapping on target end for roles ({a},{b}) - {cnt} links"
        if samples:
            sample_str = ", ".join(f"'{s}'" for s in samples)
            issues.append(f"{prefix} (e.g., adj={sample_str})")
        else:
            issues.append(prefix)

    return issues
