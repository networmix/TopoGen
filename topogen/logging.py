"""Logging configuration for TopologyGenerator."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Logger name, typically __name__ from calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def set_global_log_level(level: int) -> None:
    """Set the global logging level for all topogen loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    # Set level for topogen package loggers
    topogen_logger = logging.getLogger("topogen")
    topogen_logger.setLevel(level)
