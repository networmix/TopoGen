"""Topology Generator for Network Analysis.

Generates continental-scale backbone network topologies from population
density and highway infrastructure data.
"""

# Core classes and utilities
from . import visualization
from .config import TopologyConfig
from .integrated_graph import build_integrated_graph, load_from_json, save_to_json

__all__ = [
    "TopologyConfig",
    "build_integrated_graph",
    "load_from_json",
    "save_to_json",
    "visualization",
]
