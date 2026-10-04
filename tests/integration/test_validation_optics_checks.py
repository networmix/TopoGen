from __future__ import annotations

from types import SimpleNamespace

from topogen.validation.audits.optics_checks import check_link_optics


def _net_two_roles():
    # Nodes with roles
    nA = SimpleNamespace(name="A", attrs={"role": "core"})
    nB = SimpleNamespace(name="B", attrs={"role": "agg"})

    # Helper to create link with adjacency id and optional hardware dict
    def link(a: str, b: str, cap: float, aid: str, hw: dict | None = None):
        return SimpleNamespace(
            source=a,
            target=b,
            capacity=cap,
            attrs={"adjacency_id": aid, **({"hardware": hw} if hw is not None else {})},
        )

    net = SimpleNamespace(nodes={"A": nA, "B": nB}, links={})
    return net, link


def test_legacy_optics_mapping_is_rejected():
    net, _ = _net_two_roles()
    issues = check_link_optics(net, {"core|agg": "X"}, {})
    assert any("role->role" in issue for issue in issues)


def test_missing_hardware_and_unknown_and_capacity_shortfall():
    net, make_link = _net_two_roles()
    # Three links: missing source hw, unknown component, and capacity shortfall
    l_missing_src = make_link(
        "A", "B", 100.0, "adj1", hw={"target": {"component": "O", "count": 1}}
    )
    l_unknown_src = make_link(
        "A",
        "B",
        50.0,
        "adj2",
        hw={
            "source": {"component": "UNKNOWN", "count": 1},
            "target": {"component": "O", "count": 1},
        },
    )
    l_cap_short = make_link(
        "A",
        "B",
        200.0,
        "adj3",
        hw={
            "source": {"component": "O", "count": 1},
            "target": {"component": "O", "count": 1},
        },
    )
    net.links = {"l1": l_missing_src, "l2": l_unknown_src, "l3": l_cap_short}
    comp_lib = {"O": {"capacity": 100.0}}
    # Provide a mapping so the function does not early-return
    issues = check_link_optics(net, {"core->agg": "O", "agg->core": "O"}, comp_lib)
    assert any("missing hardware on source end" in s for s in issues)
    assert any("unknown component on source end" in s for s in issues)
    assert any("hardware capacity shortfall" in s for s in issues)


def test_missing_mapping_paths_when_no_hardware_present():
    net, make_link = _net_two_roles()
    # One link without hardware
    l_plain = make_link("A", "B", 10.0, "plain")
    net.links = {"e": l_plain}
    # Mapping exists only for (core,agg); this will require source-end hardware
    d = {"core->agg": "O", "agg->core": "O"}
    issues = check_link_optics(net, d, {"O": {"capacity": 100}})
    assert any(
        "missing source-end mapping" in s
        or "missing hardware required by mapping on source" in s
        for s in issues
    )
