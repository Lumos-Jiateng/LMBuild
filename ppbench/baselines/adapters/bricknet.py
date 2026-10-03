"""BrickNet: a connector graph, realised to absolute poses by its own decoder.

`graph_to_ldr` is BrickNet's function and is the only correct way to realise the
poses, because the connector algebra (`M = f_a . R . f_b^-1`) and the per-family
residual degrees of freedom live inside it. Reimplementing it would be a second
source of truth for the same thing.
"""
from __future__ import annotations

import os
import sys

from ppbench import config as C
from ppbench.core.canon import resolve
from ppbench.core.ir import Design
from ppbench.core.partlib import PartLib


def _bn():
    os.environ.setdefault("BRICKNET_DATA", str(C.BRICKNET_DATA))
    for p in (str(C.BRICKNET_SRC), str(C.ROOT / ".deps")):
        if p not in sys.path:
            sys.path.append(p)
    import bricknet
    return bricknet


def n_graphs(npz_path) -> int:
    return len(_bn().load_graphs(npz_path))


def from_bricknet_npz(npz_path, index: int, source="bricknet",
                      lib: PartLib | None = None) -> Design:
    bn = _bn()
    graphs = bn.load_graphs(npz_path)
    g = graphs[index]
    text = bn.graph_to_ldr(g)
    d = Design.from_ldr(text, source=source,
                        design_id=f"{os.path.basename(str(npz_path))}#{index}")
    return resolve(d, lib or PartLib())
