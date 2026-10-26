"""Atomic publication of one completely written artifact."""

import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4


def write_atomic(path: Path, writer: Callable[[Path], object]) -> None:
    """Retain the destination on writer failure; remove the private temporary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        candidate = path.with_name(f".{path.stem}.{uuid4().hex}{path.suffix}")
        descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        temporary = candidate
        os.close(descriptor)
        writer(temporary)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
