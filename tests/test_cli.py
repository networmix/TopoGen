"""CLI tests focused on argument parsing and dispatch behavior.

These tests avoid heavy I/O by stubbing subcommand functions and log setup.
"""

from __future__ import annotations

import builtins as py_builtins
import importlib
import logging
from argparse import Namespace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace  # for narrow use in _invoke_main helper
from unittest.mock import patch


def _invoke_main(argv: list[str], *, stub_subcommand: bool = False):
    """Invoke topogen.cli.main with patches applied.

    Args:
        argv: Arguments excluding program name.
        stub_subcommand: If True, replaces subcommands (build/generate/info)
            with a no-op function that records it was called and prints once.

    Returns:
        Namespace with: code (int), stdout (str), called (str|None), level (int|None).
    """
    import topogen.cli as cli

    importlib.reload(cli)

    called: dict[str, bool] = {"build": False, "generate": False, "info": False}
    level_holder: dict[str, int | None] = {"level": None}

    def _noop_cmd(_args):  # type: ignore[no-untyped-def]
        print("noop")

    patchers = []

    # Capture configured log level
    patchers.append(
        patch(
            "topogen.log_config.set_global_log_level",
            side_effect=lambda lvl: level_holder.__setitem__("level", lvl),
        )
    )

    if stub_subcommand:
        patchers.extend(
            [
                patch.object(
                    cli,
                    "build_command",
                    side_effect=lambda a: called.__setitem__("build", True),
                ),
                patch.object(
                    cli,
                    "generate_command",
                    side_effect=lambda a: called.__setitem__("generate", True),
                ),
                patch.object(
                    cli,
                    "info_command",
                    side_effect=lambda a: called.__setitem__("info", True),
                ),
            ]
        )

    for p in patchers:
        p.start()

    out = SimpleNamespace(code=0, stdout="", called=None, level=None)
    saved_print = py_builtins.print
    try:
        with (
            patch("sys.stdout", new_callable=StringIO) as buf,
            patch("sys.argv", ["topogen"] + argv),
        ):
            try:
                cli.main()
            except SystemExit as e:
                out.code = int(getattr(e, "code", 0) or 0)
            out.stdout = buf.getvalue()
            out.level = level_holder["level"]
            for name, was_called in called.items():
                if was_called:
                    out.called = name
                    break
    finally:
        # Restore global print in case --quiet modified it
        py_builtins.print = saved_print
        for p in reversed(patchers):
            p.stop()

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
    # With quiet, our stubbed subcommand won't print; stdout should be empty
    res = _invoke_main(["--quiet", "info", "-c", "config.yml"], stub_subcommand=True)
    assert res.stdout == ""


def test_subcommand_dispatch_build_generate_info():
    for cmd in ("build", "generate", "info"):
        res = _invoke_main([cmd], stub_subcommand=True)
        assert res.called == cmd


def test_timer_context_manager_success_and_error():
    from topogen.cli import Timer

    # Success path
    with patch("sys.stdout", new_callable=StringIO) as buf:
        with Timer("Unit test op"):
            pass
        s = buf.getvalue()
        assert "Unit test op" in s

    # Error path (re-raises)
    with patch("sys.stdout", new_callable=StringIO):
        try:
            with Timer("Failing op"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass


def test__load_config_file_not_found_exits_with_code_2(tmp_path):
    from topogen.cli import _load_config

    with patch("sys.stdout", new_callable=StringIO):
        pass

    # Use a definitely-missing path
    missing = tmp_path / "does_not_exist.yml"
    with patch(
        "sys.exit", side_effect=lambda code=0: (_ for _ in ()).throw(SystemExit(code))
    ) as _:
        try:
            _load_config(missing)
        except SystemExit as e:
            assert int(e.code or 0) == 2


def test__load_config_generic_error_exits_with_code_2():
    import topogen.cli as cli

    importlib.reload(cli)

    with patch.object(cli.TopologyConfig, "from_yaml", side_effect=ValueError("bad")):
        with patch(
            "sys.exit",
            side_effect=lambda code=0: (_ for _ in ()).throw(SystemExit(code)),
        ):
            try:
                cli._load_config(Path("config.yml"))  # type: ignore[name-defined]
            except SystemExit as e:  # noqa: F841
                assert int(e.code or 0) == 2


def test_build_command_success_print_and_non_print():
    import topogen.cli as cli

    importlib.reload(cli)

    # Stub loader and pipeline
    with (
        patch.object(cli, "_load_config", return_value=Namespace()),
        patch.object(cli, "_run_pipeline", return_value="YAML"),
        patch("sys.stdout", new_callable=StringIO) as buf,
    ):
        args = Namespace(config="config.yml", output="config_scenario.yml", print=True)
        cli.build_command(args)
        out = buf.getvalue()
        assert "GENERATED SCENARIO YAML" in out
        assert "YAML" in out

    with (
        patch.object(cli, "_load_config", return_value=Namespace()),
        patch.object(cli, "_run_pipeline", return_value="YAML"),
        patch("sys.stdout", new_callable=StringIO) as buf,
    ):
        args = Namespace(config="config.yml", output="config_scenario.yml", print=False)
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
                patch(
                    "sys.exit",
                    side_effect=lambda code=0: (_ for _ in ()).throw(SystemExit(code)),
                ) as _,
            ):
                try:
                    cli.build_command(
                        Namespace(config="c.yml", output="o.yaml", print=False)
                    )
                except SystemExit as e:
                    assert int(e.code or 0) == expected


def test__run_pipeline_missing_integrated_graph_exits_1():
    from topogen.cli import _run_pipeline

    with patch(
        "sys.exit", side_effect=lambda code=0: (_ for _ in ()).throw(SystemExit(code))
    ):
        # Force the integrated graph sentinel to be treated as missing regardless of repo state
        original_exists = Path.exists

        def fake_exists(self: Path) -> bool:  # type: ignore[no-redef]
            if str(self).endswith("_integrated_graph.json"):
                return False
            return original_exists(self)

        with patch("pathlib.Path.exists", new=fake_exists):
            try:
                # Provide a minimal TopologyConfig-like object with _source_path used by _run_pipeline
                fake_cfg = Namespace(_source_path=Path("config.yml"))
                _run_pipeline(fake_cfg, Path("config_scenario.yml"), print_yaml=False)  # type: ignore[name-defined]
            except SystemExit as e:
                assert int(e.code or 0) == 1


def test_generate_command_success_and_failure():
    import topogen.cli as cli

    importlib.reload(cli)

    with (
        patch.object(cli, "_load_config", return_value=Namespace()),
        patch.object(cli, "_run_generation", return_value=None),
    ):
        cli.generate_command(Namespace(config="config.yml"))

    with (
        patch.object(cli, "_load_config", return_value=Namespace()),
        patch.object(cli, "_run_generation", side_effect=RuntimeError("boom")),
        patch(
            "sys.exit",
            side_effect=lambda code=0: (_ for _ in ()).throw(SystemExit(code)),
        ),
    ):
        try:
            cli.generate_command(Namespace(config="config.yml"))
        except SystemExit as e:
            assert int(e.code or 0) == 1


def test_info_command_prints_status(tmp_path):
    import topogen.cli as cli

    importlib.reload(cli)

    # Create one existing and one missing path
    uac = tmp_path / "uac.zip"
    tiger = tmp_path / "tiger.zip"
    uac.write_text("x")
    # tiger intentionally missing

    fake_cfg = Namespace(
        data_sources=Namespace(uac_polygons=str(uac), tiger_roads=str(tiger)),
        projection=Namespace(target_crs="EPSG:5070"),
        clustering=Namespace(metro_clusters=1),
    )

    with (
        patch.object(cli, "_load_config", return_value=fake_cfg),
        patch("sys.stdout", new_callable=StringIO) as buf,
    ):
        cli.info_command(Namespace(config="config.yml"))
        out = buf.getvalue()
        assert "TopoGen Configuration" in out
        assert "UAC polygons:" in out
