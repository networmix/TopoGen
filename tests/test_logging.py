"""Test the centralized logging functionality."""

import logging
from io import StringIO

from topogen.log_config import get_logger, set_global_log_level


def test_set_global_log_level():
    """Test that set_global_log_level configures logging properly."""
    # Test setting WARNING level
    set_global_log_level(logging.WARNING)
    topogen_logger = logging.getLogger("topogen")
    assert topogen_logger.level == logging.WARNING

    # Test setting DEBUG level
    set_global_log_level(logging.DEBUG)
    topogen_logger = logging.getLogger("topogen")
    assert topogen_logger.level == logging.DEBUG

    # Test setting INFO level
    set_global_log_level(logging.INFO)
    topogen_logger = logging.getLogger("topogen")
    assert topogen_logger.level == logging.INFO


def test_logger_hierarchy():
    """Test that child loggers inherit from parent."""
    # Set global level
    set_global_log_level(logging.WARNING)

    # Create child logger
    child_logger = get_logger("topogen.test.child")

    # Child should inherit effective level from parent
    assert child_logger.getEffectiveLevel() == logging.WARNING


def test_logging_output():
    """Test that logging outputs at correct levels."""
    logger = get_logger("topogen.test.output")

    # Capture log output
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)

    # Clear any existing handlers and add our test handler
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    log_output = log_capture.getvalue()
    assert "Debug message" in log_output
    assert "Info message" in log_output
    assert "Warning message" in log_output
    assert "Error message" in log_output
