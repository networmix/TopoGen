"""Test package structure and imports."""

import sys
from pathlib import Path


def test_package_import():
    """Test that the topogen package can be imported."""
    import topogen

    assert hasattr(topogen, "__version__")
    assert topogen.__version__ == "0.1.0"


def test_main_module_import():
    """Test that topogen.__main__ can be imported."""
    import topogen.__main__

    # Should not raise any errors
    assert topogen.__main__


def test_cli_module_import():
    """Test that topogen.cli can be imported."""
    import topogen.cli

    assert hasattr(topogen.cli, "main")
    assert callable(topogen.cli.main)


def test_logging_module_import():
    """Test that topogen.logging can be imported."""
    import topogen.logging

    assert hasattr(topogen.logging, "get_logger")
    assert hasattr(topogen.logging, "set_global_log_level")
    assert callable(topogen.logging.get_logger)
    assert callable(topogen.logging.set_global_log_level)


def test_main_module_calls_cli():
    """Test that __main__ module calls cli.main()."""
    # Import and check the __main__ module source content
    import topogen.__main__

    main_file = Path(topogen.__main__.__file__)
    content = main_file.read_text()

    assert "from topogen.cli import main" in content
    assert "main()" in content


def test_no_missing_dependencies():
    """Test that all imports work without missing dependencies."""
    # This tests that all required dependencies are available
    try:
        # Test specific imports used in the modules
        import argparse  # noqa: F401
        import logging  # noqa: F401
        import pathlib  # noqa: F401

        import yaml  # noqa: F401

        import topogen.cli  # noqa: F401
        import topogen.logging  # noqa: F401

    except ImportError as e:
        import pytest

        pytest.fail(f"Missing required dependency: {e}")


def test_python_version_compatibility():
    """Test that package works with supported Python versions."""
    # Check that we're running on a supported version
    assert sys.version_info >= (3, 11), "Package requires Python 3.11+"
