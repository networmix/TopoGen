"""Explicit, per-invocation artifact destinations; never part of topology settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    output_dir: Path
    stem: str = "scenario"
    debug_dir: Path | None = None

    def path(self, suffix: str) -> Path:
        return self.output_dir / f"{self.stem}_{suffix}"
