from __future__ import annotations

from types import SimpleNamespace

from topogen.validation.audits.port_budget import audit_port_budget


def _net_for_ports():
    # Two nodes with platform hardware 'P' providing 8 ports each
    nA = SimpleNamespace(
        name="A", attrs={"role": "core", "hardware": {"component": "P", "count": 1}}
    )
    nB = SimpleNamespace(
        name="B", attrs={"role": "core", "hardware": {"component": "P", "count": 1}}
    )
    # One link requiring optics O on both ends; 4x capacity → ports needed = ceil(400/100)*1 = 4 per end
    l1 = SimpleNamespace(
        source="A",
        target="B",
        capacity=400.0,
        attrs={
            "hardware": {"source": {"component": "O"}, "target": {"component": "O"}}
        },
    )
    net = SimpleNamespace(nodes={"A": nA, "B": nB}, links={"l1": l1})
    return net


def test_audit_port_budget_no_issue_when_ports_sufficient():
    net = _net_for_ports()
    scenario = {"components": {"optics": {"core|core": "O"}}}
    comp_lib = {"P": {"ports": 8}, "O": {"capacity": 100, "ports": 1}}
    issues = audit_port_budget(net, scenario, comp_lib)
    assert issues == []


def test_audit_port_budget_detects_shortage():
    net = _net_for_ports()
    scenario = {"components": {"optics": {"core|core": "O"}}}
    # Only 2 ports available per node → shortage (need 4)
    comp_lib = {"P": {"ports": 2}, "O": {"capacity": 100, "ports": 1}}
    issues = audit_port_budget(net, scenario, comp_lib)
    assert issues and "requires 4 ports" in issues[0]
