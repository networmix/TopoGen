"""Require a nonempty string role on every device."""

from ngraph import Network


def check_node_roles(net: Network) -> list[str]:
    missing = [
        node.name
        for node in net.nodes.values()
        if not isinstance(node.attrs.get("role"), str) or not node.attrs["role"].strip()
    ]
    if not missing:
        return []
    examples = ", ".join(repr(name) for name in missing[:5])
    return [f"node roles: {len(missing)} node(s) missing role (e.g., {examples})"]
