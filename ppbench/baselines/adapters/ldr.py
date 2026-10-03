"""LegoACE and the reference arm: LDraw straight through."""
from __future__ import annotations

from ppbench.core.canon import resolve
from ppbench.core.ir import Design
from ppbench.core.partlib import PartLib


def from_ldr_file(path, source="ldr", lib: PartLib | None = None) -> Design:
    return resolve(Design.load(path, source=source), lib or PartLib())
