"""Project-used subset of Contextily's API, checked against version 1.7.1.

Only add_basemap and the arguments used by TopoGen are declared. Check upstream
contextily/plotting.py before extending this subset or updating its signature.
"""

from typing import Literal

from matplotlib.axes import Axes

def add_basemap(
    ax: Axes,
    *,
    crs: str | None = ...,
    attribution: str | Literal[False] | None = ...,
    alpha: float | None = ...,
) -> None: ...
