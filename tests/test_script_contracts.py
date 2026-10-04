"""Batch commands must propagate failures to their caller."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "stage,code",
    [("generate", 1), ("generate", 2), ("build", 1), ("build", 3), ("none", 0)],
)
def test_batch_exit_status(tmp_path, stage, code):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "one.yml").write_text("{}")
    runner = tmp_path / "topogen"
    runner.write_text(
        f'#!/bin/sh\nif [ "$1" = "{stage}" ]; then exit {code}; fi\nexit 0\n'
    )
    runner.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(Path("build.sh").resolve()),
            str(configs),
            str(tmp_path / "output"),
        ],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (code == 0), result.stdout + result.stderr
