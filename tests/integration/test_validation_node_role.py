from __future__ import annotations

from types import SimpleNamespace

from topogen.validation.audits.node_role import check_node_roles


def test_check_node_roles_reports_missing() -> None:
    n1 = SimpleNamespace(name="A", attrs={"role": "core"})
    n2 = SimpleNamespace(name="B", attrs={})
    n3 = SimpleNamespace(name="C", attrs={"role": ""})
    net = SimpleNamespace(nodes={"A": n1, "B": n2, "C": n3})
    issues = check_node_roles(net)
    assert issues and "missing role" in issues[0]
