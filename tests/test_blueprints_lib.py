"""Every builtin blueprint has devices through the actual public parser."""

import pytest
import yaml
from ngraph import Scenario

from topogen.library import load_library, thaw


@pytest.mark.parametrize(
    "name,nodes,links",
    [
        ("DCRegion", 1, 0),
        ("SingleRouter", 1, 0),
        ("FullMesh4", 4, 6),
        ("Clos_L16_S4", 20, 64),
        ("DragonFly_CustomG4", 16, 40),
        ("Dragonfly_A3H2G7", 21, 42),
    ],
)
def test_blueprint_inventory(name, nodes, links):
    document = {
        "blueprints": thaw(load_library(None).blueprints),
        "network": {"nodes": {"site": {"blueprint": name}}, "links": []},
    }
    network = Scenario.from_yaml(yaml.safe_dump(document)).network
    assert (len(network.nodes), len(network.links)) == (nodes, links)
