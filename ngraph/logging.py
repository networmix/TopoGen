"""Centralized logging configuration for NetGraph."""

import logging
import sys
from typing import Optional

# Flag to track if we've already set up the root logger
_ROOT_LOGGER_CONFIGURED = False


def setup_root_logger(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> None:
    """Set up the root NetGraph logger with a single handler.

    This should only be called once to avoid duplicate handlers.

    Args:
        level: Logging level (default: INFO).
        format_string: Custom format string (optional).
        handler: Custom handler (optional, defaults to StreamHandler).
    """
    global _ROOT_LOGGER_CONFIGURED

    if _ROOT_LOGGER_CONFIGURED:
        return

    root_logger = logging.getLogger("ngraph")
    root_logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Default format with timestamps, level, logger name, and message
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Default to console output, but allow override for testing
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Let logs propagate to root logger so pytest can capture them
    root_logger.propagate = True

    _ROOT_LOGGER_CONFIGURED = True

    _ROOT_LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with NetGraph's standard configuration.

    This is the main function that should be used throughout the package.
    All loggers will inherit from the root 'ngraph' logger configuration.

    Args:
        name: Logger name (typically __name__ from calling module).

    Returns:
        Configured logger instance.
    """
    # Ensure root logger is set up
    setup_root_logger()

    # Get the logger - it will inherit from the root ngraph logger
    logger = logging.getLogger(name)

    # Don't add handlers to child loggers - they inherit from root
    # Just set the level
    logger.setLevel(logging.NOTSET)  # Inherit from parent

    return logger


def set_global_log_level(level: int) -> None:
    """Set the log level for all NetGraph loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    # Ensure root logger is set up
    setup_root_logger()

    # Set the root level for all ngraph loggers
    root_logger = logging.getLogger("ngraph")
    root_logger.setLevel(level)

    # Also update handlers to respect the new level
    for handler in root_logger.handlers:
        handler.setLevel(level)


def enable_debug_logging() -> None:
    """Enable debug logging for the entire package."""
    set_global_log_level(logging.DEBUG)


def disable_debug_logging() -> None:
    """Disable debug logging, set to INFO level."""
    set_global_log_level(logging.INFO)


def reset_logging() -> None:
    """Reset logging configuration (mainly for testing)."""
    global _ROOT_LOGGER_CONFIGURED
    _ROOT_LOGGER_CONFIGURED = False

    # Clear any existing handlers from ngraph logger
    root_logger = logging.getLogger("ngraph")
    root_logger.handlers.clear()
    root_logger.setLevel(logging.NOTSET)


# Initialize the root logger when the module is imported
setup_root_logger()
