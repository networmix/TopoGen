"""Malformed custom hardware and roles must not bypass scenario audits."""

import pytest
from ngraph import Network, Node

from topogen.validation.audits.hw_capacity import check_node_hw_capacity
from topogen.validation.audits.node_role import check_node_roles


@pytest.mark.parametrize("role", [None, False, 12, ""])
def test_role_must_be_a_nonempty_string(role):
    network = Network()
    network.add_node(Node("r", attrs={"role": role}))
    assert check_node_roles(network)


@pytest.mark.parametrize("capacity", [float("nan"), float("inf"), -1])
def test_invalid_chassis_capacity_is_reported(capacity):
    network = Network()
    network.add_node(Node("r", attrs={"hardware": {"component": "C", "count": 1}}))
    assert check_node_hw_capacity(network, {"C": {"capacity": capacity}})


@pytest.mark.parametrize(
    "cap,limit,ports,count,disabled,expected",
    [
        (10, 20, 10, 2, False, (0, 0)),
        (10, 5, 10, 2, False, (2, 0)),
        (10, 5, 10, 2, True, (2, 0)),
        (10, 20, 1, 2, True, (0, 2)),
        (10, 0, 10, 2, False, (2, 0)),
        (10, 20, 0, 2, False, (0, 2)),
        (10.0000000005, 10, 10, 2, False, (0, 0)),
        (10, 20, 2, 2.0000000005, False, (0, 2)),
        (44000.0, 43999.9999999995, 10, 2, False, (0, 0)),
    ],
)
def test_installed_hardware_rules_keep_baseline_boundaries(
    cap, limit, ports, count, disabled, expected
):
    from ngraph import Link

    from topogen.validation.audits.port_budget import audit_port_budget

    net = Network()
    for name in ("A", "B"):
        net.add_node(Node(name, attrs={"hardware": {"component": "P", "count": 1}}))
    net.add_link(
        Link(
            "A",
            "B",
            capacity=cap,
            cost=1,
            disabled=disabled,
            attrs={
                "hardware": {
                    end: {"component": "O", "count": count}
                    for end in ("source", "target")
                }
            },
        )
    )
    components = {
        "P": {"capacity": limit, "ports": ports},
        "O": {"capacity": 100000, "ports": 1},
    }
    assert (
        len(check_node_hw_capacity(net, components)),
        len(audit_port_budget(net, components)),
    ) == expected


def test_zero_platform_capacity_accepts_zero_attached_load():
    from ngraph import Link

    net = Network()
    for name in ("A", "B"):
        net.add_node(Node(name, attrs={"hardware": {"component": "P", "count": 1}}))
    components = {"P": {"capacity": 0}}
    assert check_node_hw_capacity(net, components) == []
    net.add_link(Link("A", "B", capacity=1, cost=1))
    assert len(check_node_hw_capacity(net, components)) == 2
