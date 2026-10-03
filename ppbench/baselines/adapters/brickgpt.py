"""BrickGPT: a text list of 1-unit-tall rectangular bricks on an integer grid.

Format is `HxW (x,y,z)` per line. The conversion to LDraw is BrickGPT's own,
from `brickgpt.data.brick_structure.Brick.to_ldr`, reimplemented here so a
BrickGPT output can be read without importing BrickGPT and pulling in its
Llama-3.2 dependency:

    x_ldu = (x + h/2) * 20     z_ldu = (y + w/2) * 20     y_ldu = -(z) * 24

with the rotation matrix selecting the brick's orientation. Note BrickGPT's
`z` is the vertical index while LDraw's vertical axis is `y`, pointing down --
the axis relabelling is the whole conversion, and getting it wrong yields a
model lying on its side.

The brick-id to part-id table comes from BrickGPT's own `brick_library.json`.
"""
from __future__ import annotations

import json
import re

import numpy as np

from ppbench import config as C
from ppbench.core.canon import resolve
from ppbench.core.ir import Design
from ppbench.core.partlib import PartLib

_LINE = re.compile(r"^\s*(\d+)x(\d+)\s*\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)\s*$")

_ORI0 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
_ORI1 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float)


def _library():
    p = C.BRICKGPT_SRC / "brickgpt/data/brick_library.json"
    lib = json.load(open(p))
    dims = {}
    for v in lib.values():
        dims[(int(v["height"]), int(v["width"]))] = v["partID"].lower().removesuffix(".dat")
    return dims


def from_brickgpt_txt(text: str, source="brickgpt", design_id="",
                      lib: PartLib | None = None) -> Design:
    dims = _library()
    stems, poses, steps, bad = [], [], [], []
    for i, ln in enumerate(text.splitlines()):
        if not ln.strip():
            continue
        m = _LINE.match(ln)
        if m is None:
            bad.append(ln.strip()[:40])
            continue
        h, w, x, y, z = (int(g) for g in m.groups())
        stem = dims.get((h, w)) or dims.get((w, h))
        if stem is None:
            bad.append(f"{h}x{w} not in library")
            continue
        ori = 1 if h > w else 0
        T = np.eye(4)
        T[:3, :3] = _ORI1 if ori == 1 else _ORI0
        T[:3, 3] = [(x + h * 0.5) * 20.0, -(z) * 24.0, (y + w * 0.5) * 20.0]
        stems.append(stem)
        poses.append(T)
        steps.append(len(steps))

    d = Design(stems=stems,
               colors=np.full(len(stems), 15, np.int32),
               poses=np.array(poses, float).reshape(-1, 4, 4),
               steps=np.array(steps, np.int32),
               source=source, design_id=design_id,
               ingest={"n_unparsed_lines": len(bad), "unparsed_examples": bad[:5],
                       "parse_rate": len(stems) / max(len(stems) + len(bad), 1)})
    return resolve(d, lib or PartLib())


def from_brickgpt_json(obj, **kw) -> Design:
    """BrickGPT's json form: {"1": {"brick_id", "x", "y", "z", "ori"}, ...}."""
    inv = {}
    p = C.BRICKGPT_SRC / "brickgpt/data/brick_library.json"
    for k, v in json.load(open(p)).items():
        inv[int(k)] = (int(v["height"]), int(v["width"]))
    lines = []
    for k, v in obj.items():
        if not str(k).isdigit():
            continue
        h, w = inv[int(v["brick_id"])]
        if int(v.get("ori", 0)) == 1:
            h, w = w, h
        lines.append(f"{h}x{w} ({v['x']},{v['y']},{v['z']})")
    return from_brickgpt_txt("\n".join(lines), **kw)
