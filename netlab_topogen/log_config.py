"""Logging configuration for NetLab-TopoGen."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return the named logger without configuring it."""
    return logging.getLogger(name)


def set_global_log_level(level: int) -> None:
    """Configure root logging and set the level for NetLab-TopoGen loggers."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    netlab_topogen_logger = logging.getLogger("netlab_topogen")
    netlab_topogen_logger.setLevel(level)
