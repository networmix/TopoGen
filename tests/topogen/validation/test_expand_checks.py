from __future__ import annotations

from types import SimpleNamespace

from topogen.validation.audits.expand_checks import check_groups_adjacency_blueprints


class _Node:
    def __init__(self, gid: int | None = None) -> None:
        self.attrs = {} if gid is None else {"_tg_group_id": gid}


class _Link:
    def __init__(self, attrs: dict | None = None) -> None:
        self.attrs = attrs or {}


def test_expand_checks_flags_empty_groups_and_adjacency_and_bp():
    # DSL with one group and one adjacency and one blueprint
    dsl = {
        "blueprints": {
            "BP": {
                "groups": {"core": {"node_count": 1, "attrs": {}}},
                "adjacency": [
                    {"source": "/core", "target": "/core", "link_params": {}}
                ],
            }
        },
        "network": {
            "groups": {"g1": {"use_blueprint": "BP", "attrs": {}}},
            "adjacency": [
                {
                    "source": "g1",
                    "target": "g1",
                    "pattern": "one_to_one",
                    "link_params": {},
                }
            ],
        },
    }

    # ng_expand stub: for tagged group expansion, produce zero nodes to trigger issue
    def _expand(stub):  # type: ignore[no-untyped-def]
        class Net:
            def __init__(self):
                self.nodes = {}
                self.links = {}

        n = Net()
        net = stub.get("network", {})
        bps = stub.get("blueprints", {})
        # Groups expansion: read _tg_group_id for counting
        for gpath, gdef in (net.get("groups", {}) or {}).items():
            # Simulate zero-node expansion for g1 only
            gid = gdef.get("attrs", {}).get("_tg_group_id")
            if gpath == "g1":
                continue
            n.nodes[gpath] = _Node(gid)
        # Adjacency expansion: only create links when special tag present
        for rule in net.get("adjacency", []) or []:
            lp = rule.get("link_params", {}) or {}
            attrs = lp.get("attrs", {}) or {}
            if attrs.get("_tg_adj_tag") == "adj_0":
                # omit to trigger adjacency issue
                continue
            n.links[f"L{len(n.links)}"] = _Link(attrs)
        # Blueprint expansion: only if bp is not BP to trigger bp adjacency issue
        if bps:
            bp_name = next(iter(bps.keys()))
            # Do not emit links for BP to trigger bp check issue
            if bp_name != "BP":
                n.links["BP0"] = _Link({"_tg_bp_adj_tag": f"{bp_name}#0"})
        return n

    # Logger stub with .error method
    logger = SimpleNamespace(error=lambda *args, **kwargs: None)

    issues = check_groups_adjacency_blueprints(dsl, _expand, logger)
    # Expect group, adjacency, and blueprint issues
    assert any("group 'g1' expands to 0 nodes" in s for s in issues)
    assert any("adjacency[0] expands to 0 links" in s for s in issues)
    assert any("blueprint 'BP' adjacency[0] expands to 0 links" in s for s in issues)
