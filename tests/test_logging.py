"""Test the centralized logging functionality."""

import logging
from io import StringIO

from topogen.log_config import get_logger, set_global_log_level


def test_set_global_log_level():
    set_global_log_level(logging.WARNING)
    topogen_logger = logging.getLogger("topogen")
    assert topogen_logger.level == logging.WARNING

    set_global_log_level(logging.DEBUG)
    topogen_logger = logging.getLogger("topogen")
    assert topogen_logger.level == logging.DEBUG

    set_global_log_level(logging.INFO)
    topogen_logger = logging.getLogger("topogen")
    assert topogen_logger.level == logging.INFO


def test_logger_hierarchy():
    set_global_log_level(logging.WARNING)

    child_logger = get_logger("topogen.test.child")

    assert child_logger.getEffectiveLevel() == logging.WARNING


def test_logging_output():
    logger = get_logger("topogen.test.output")

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)

    # Clear any existing handlers and add our test handler
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    log_output = log_capture.getvalue()
    assert "Debug message" in log_output
    assert "Info message" in log_output
    assert "Warning message" in log_output
    assert "Error message" in log_output
