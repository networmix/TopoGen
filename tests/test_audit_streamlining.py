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
