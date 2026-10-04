"""CLI tests focused on argument parsing and dispatch behavior.

These tests avoid heavy I/O by stubbing subcommand functions and log setup.
"""

from __future__ import annotations

import importlib
import logging
from argparse import Namespace
from contextlib import ExitStack
from functools import partial
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _invoke_main(argv: list[str], *, stub_subcommand: bool = False):
    """Run the CLI and capture exit code, stdout, selected command, and log level.

    With ``stub_subcommand``, replace subcommands with a recorder that prints once.
    """
    import topogen.cli as cli

    importlib.reload(cli)

    called: dict[str, bool] = {"build": False, "generate": False, "info": False}
    level_holder: dict[str, int | None] = {"level": None}

    def record(command, _args):
        called[command] = True
        print("stub command output")

    out = SimpleNamespace(code=0, stdout="", called=None, level=None)
    with (
        ExitStack() as patches,
        patch("sys.stdout", new_callable=StringIO) as buf,
        patch("sys.argv", ["topogen"] + argv),
    ):
        patches.enter_context(
            patch(
                "topogen.log_config.set_global_log_level",
                side_effect=lambda lvl: level_holder.__setitem__("level", lvl),
            )
        )
        if stub_subcommand:
            for command in called:
                patches.enter_context(
                    patch.object(
                        cli, f"{command}_command", side_effect=partial(record, command)
                    )
                )
        try:
            cli.main()
        except SystemExit as exc:
            out.code = exc.code
        out.stdout = buf.getvalue()
        out.level = level_holder["level"]
        out.called = next(
            (name for name, was_called in called.items() if was_called), None
        )

    return out


def test_no_args_shows_help_and_exits_nonzero():
    res = _invoke_main([])
    assert res.code == 1
    assert "Available commands" in res.stdout


def test_verbose_flag_sets_debug_level_and_dispatches_info():
    res = _invoke_main(["-v", "info", "config.yml"], stub_subcommand=True)
    assert res.called == "info"
    assert res.level == logging.DEBUG


def test_default_log_level_is_info():
    res = _invoke_main(["info", "config.yml"], stub_subcommand=True)
    assert res.level == logging.INFO


def test_quiet_suppresses_print_output():
    res = _invoke_main(["--quiet", "info", "config.yml"], stub_subcommand=True)
    assert res.code == 0
    assert res.called == "info"
    assert res.stdout == ""
    visible = _invoke_main(["info", "config.yml"], stub_subcommand=True)
    assert visible.code == 0
    assert "stub command output" in visible.stdout


def test_subcommand_dispatch_build_generate_info():
    for cmd in ("build", "generate", "info"):
        res = _invoke_main([cmd], stub_subcommand=True)
        assert res.called == cmd


def test_timer_context_manager_success_and_error():
    from topogen.cli import Timer

    with patch("sys.stdout", new_callable=StringIO) as buf:
        with Timer("Unit test op"):
            pass
        s = buf.getvalue()
        assert "Unit test op" in s

    with (
        patch("sys.stdout", new_callable=StringIO) as buf,
        pytest.raises(RuntimeError, match="boom"),
        Timer("Failing op"),
    ):
        raise RuntimeError("boom")
    assert "failed after" in buf.getvalue()


def test__load_config_file_not_found_exits_with_code_2(tmp_path):
    from topogen.cli import _load_config

    missing = tmp_path / "does_not_exist.yml"
    with pytest.raises(SystemExit) as exc:
        _load_config(missing)
    assert exc.value.code == 2


def test__load_config_generic_error_exits_with_code_2():
    import topogen.cli as cli

    with (
        patch.object(cli.TopologyConfig, "from_yaml", side_effect=ValueError("bad")),
        pytest.raises(SystemExit) as exc,
    ):
        cli._load_config(Path("config.yml"))
    assert exc.value.code == 2


def test_build_command_success_print_and_non_print():
    import topogen.cli as cli

    importlib.reload(cli)

    with (
        patch.object(cli, "_load_config", return_value=Namespace()),
        patch.object(cli, "_run_pipeline", return_value="YAML"),
        patch("sys.stdout", new_callable=StringIO) as buf,
    ):
        args = Namespace(
            config="config.yml",
            output="config_scenario.yml",
            print=True,
            debug_dir=None,
        )
        cli.build_command(args)
        out = buf.getvalue()
        assert "GENERATED SCENARIO YAML" in out
        assert "YAML" in out

    with (
        patch.object(cli, "_load_config", return_value=Namespace()),
        patch.object(cli, "_run_pipeline", return_value="YAML"),
        patch("sys.stdout", new_callable=StringIO) as buf,
    ):
        args = Namespace(
            config="config.yml",
            output="config_scenario.yml",
            print=False,
            debug_dir=None,
        )
        cli.build_command(args)
        out = buf.getvalue()
        assert "SUCCESS! Generated topology" in out


def test_build_command_failure_exit_codes():
    import topogen.cli as cli

    importlib.reload(cli)

    with patch.object(cli, "_load_config", return_value=Namespace()):
        for exc, expected in [
            (FileNotFoundError("x"), 3),
            (ValueError("x"), 3),
            (RuntimeError("x"), 1),
        ]:
            with (
                patch.object(cli, "_run_pipeline", side_effect=exc),
                pytest.raises(SystemExit) as error,
            ):
                cli.build_command(
                    Namespace(
                        config="c.yml", output="o.yaml", print=False, debug_dir=None
                    )
                )
            assert error.value.code == expected


def test_run_pipeline_missing_integrated_graph(tmp_path):
    from topogen import RunContext, TopologyConfig
    from topogen.cli import _run_pipeline

    with pytest.raises(FileNotFoundError, match="Run topogen generate first"):
        _run_pipeline(
            TopologyConfig(),
            tmp_path / "scenario.yml",
            context=RunContext(tmp_path, "config"),
        )


def test_generate_command_success_and_failure():
    import topogen.cli as cli

    importlib.reload(cli)

    with (
        patch.object(cli, "_load_config", return_value=Namespace()),
        patch.object(cli, "_run_generation", return_value=None),
    ):
        cli.generate_command(Namespace(config="config.yml", output=None))

    with (
        patch.object(cli, "_load_config", return_value=Namespace()),
        patch.object(cli, "_run_generation", side_effect=RuntimeError("boom")),
        pytest.raises(SystemExit) as exc,
    ):
        cli.generate_command(Namespace(config="config.yml", output=None))
    assert exc.value.code == 1


def test_info_command_prints_status(tmp_path):
    import topogen.cli as cli

    importlib.reload(cli)

    uac = tmp_path / "uac.zip"
    tiger = tmp_path / "tiger.zip"
    uac.write_text("x")
    # tiger intentionally missing

    fake_cfg = Namespace(
        data_sources=Namespace(
            uac_polygons=str(uac),
            tiger_roads=str(tiger),
            conus_boundary=str(tmp_path / "boundary.zip"),
        ),
        projection=Namespace(target_crs="EPSG:5070"),
        clustering=Namespace(metro_clusters=1),
    )

    with (
        patch.object(cli, "_load_config", return_value=fake_cfg),
        patch("sys.stdout", new_callable=StringIO) as buf,
    ):
        cli.info_command(Namespace(config="config.yml", output=None))
        out = buf.getvalue()
        assert "TopoGen Configuration" in out
        assert "UAC polygons:" in out
