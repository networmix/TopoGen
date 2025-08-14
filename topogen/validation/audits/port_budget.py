"""Dedicated port budget audit used by the pipeline."""

from __future__ import annotations

from typing import Any

from topogen.log_config import get_logger

from ..helpers import _node_hw_from_attrs

logger = get_logger(__name__)


def audit_port_budget(
    net: Any, scenario_dict: dict[str, Any], comp_lib: dict[str, Any]
) -> list[str]:
    """Audit node platform port budgets against per-link optics.

    For each link end, compute modules = ceil(link_capacity / optic.capacity)
    and multiply by optic.ports to get required platform ports. Sum per node,
    compare to available ports on node platform (component.ports × count).
    """
    issues: list[str] = []
    try:
        from math import ceil as _ceil

        def _role_of(node_name: str) -> str:
            try:
                return str(getattr(net.nodes[node_name], "attrs", {}).get("role", ""))
            except Exception:
                return ""

        # Build optics mapping from components.optics config
        optics_cfg = (scenario_dict.get("components") or {}).get("optics", {})
        optics_map: dict[tuple[str, str], str] = {}
        if isinstance(optics_cfg, dict) and optics_cfg:
            for key, val in optics_cfg.items():
                if not isinstance(val, str):
                    continue
                k = str(key)
                if "|" in k:
                    a, b = [p.strip() for p in k.split("|", 1)]
                    if a and b:
                        optics_map[(a, b)] = str(val)
                        optics_map[(b, a)] = str(val)
                elif "-" in k:
                    a, b = [p.strip() for p in k.split("-", 1)]
                    if a and b:
                        optics_map[(a, b)] = str(val)

        # Available platform ports per node
        available_ports: dict[str, int] = {}
        for node in net.nodes.values():
            node_name = str(getattr(node, "name", ""))
            if not node_name:
                continue
            comp_name, count = _node_hw_from_attrs(getattr(node, "attrs", {}) or {})
            if not comp_name or count <= 0.0:
                continue
            comp = comp_lib.get(comp_name)
            if comp is None:
                continue
            try:
                ports_per_platform = int(comp.get("ports", 0))
            except Exception:
                ports_per_platform = 0
            if ports_per_platform <= 0:
                continue
            total_ports = int(ports_per_platform * float(count))
            if total_ports > 0:
                available_ports[node_name] = total_ports

        if not available_ports:
            return issues

        # Sum required ports per node across incident links
        required_ports: dict[str, int] = {}
        for link in net.links.values():
            cap = float(getattr(link, "capacity", 0.0) or 0.0)
            if cap <= 0.0:
                continue
            src = str(getattr(link, "source", ""))
            dst = str(getattr(link, "target", ""))
            if not src or not dst:
                continue

            src_optic: str | None = None
            tgt_optic: str | None = None
            hw = getattr(link, "attrs", {}).get("hardware")
            if isinstance(hw, dict):
                s_hw = hw.get("source") if isinstance(hw.get("source"), dict) else None
                t_hw = hw.get("target") if isinstance(hw.get("target"), dict) else None
                if s_hw:
                    src_optic = str(s_hw.get("component", "")).strip() or None
                if t_hw:
                    tgt_optic = str(t_hw.get("component", "")).strip() or None

            if src_optic is None or tgt_optic is None:
                rs = _role_of(src)
                rd = _role_of(dst)
                if not src_optic:
                    src_optic = optics_map.get((rs, rd))
                if not tgt_optic:
                    tgt_optic = optics_map.get((rd, rs))

            def _accumulate(
                node_name: str, optic_name: str | None, link_cap: float
            ) -> None:
                if not optic_name:
                    return
                comp = comp_lib.get(optic_name)
                if comp is None:
                    return
                try:
                    module_capacity = float(comp.get("capacity", 0.0))
                    ports_per_optic = int(comp.get("ports", 1))
                except Exception:
                    module_capacity = 0.0
                    ports_per_optic = 1
                if module_capacity <= 0.0:
                    return
                modules_needed = int(_ceil(link_cap / module_capacity))
                ports_needed = modules_needed * max(ports_per_optic, 1)
                if ports_needed <= 0:
                    return
                required_ports[node_name] = (
                    required_ports.get(node_name, 0) + ports_needed
                )

            _accumulate(src, src_optic, cap)
            _accumulate(dst, tgt_optic, cap)

        for node_name, need_ports in sorted(required_ports.items()):
            avail_ports = int(available_ports.get(node_name, 0))
            if avail_ports <= 0:
                continue
            if need_ports > avail_ports:
                attrs = getattr(net.nodes[node_name], "attrs", {}) or {}
                comp_name, count = _node_hw_from_attrs(attrs)
                issues.append(
                    (
                        "hardware ports: node '\n"
                        f"{node_name}\n' requires {need_ports} ports for link optics but only "
                        f"{avail_ports} ports are available on '{comp_name}' (count={count:g})."
                    ).replace("\n", "")
                )
    except Exception as e:  # pragma: no cover
        issues.append(f"port budget audit failed: {e}")
    return issues
