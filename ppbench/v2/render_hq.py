"""Page renders: larger, cleaner, and coloured by canonical role so the same colour
means the same kind of part in every system's picture (a seat is always the same
blue). Parts without a recognised role share a neutral grey; external generators
without roles get a per-part palette instead, flagged in the caption.

Reuses the Blender worker of `render.py` (Cycles, CPU) without modifying it.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np

from ppbench.v2 import render
from ppbench.v2.lexicon import Lexicon

ROLE_COLORS = {   # sRGB-ish, tuned to read on the light ground plane
    "seat": (0.20, 0.45, 0.78), "backrest": (0.36, 0.62, 0.86), "armrest": (0.55, 0.40, 0.78),
    "gas lift cylinder": (0.80, 0.52, 0.16), "swivel mechanism": (0.86, 0.70, 0.22), "base": (0.30, 0.33, 0.37),
    "caster": (0.15, 0.58, 0.52), "caster wheel": (0.12, 0.40, 0.36), "lever": (0.82, 0.30, 0.30),
}
UNKNOWN = (0.62, 0.64, 0.66)
VIEWS = ("cond", "iso", "front", "side")


def _role_color(role: str):
    """A stable colour for any role string, so the same role reads the same in every system and every task.

    ROLE_COLORS is hand-tuned for the swivel-office-chair demo and covers nothing else; until 2026-09-16 every
    other task fell through to UNKNOWN, so a page render of, say, a dining table came out as seven identical
    grey parts. Deriving the hue from the role name keeps the module's promise (same colour = same kind of part)
    without a per-task lexicon.
    """
    h = 0
    for ch in role:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return colorsys.hsv_to_rgb((h % 997) / 997.0, 0.48, 0.80)


def colors_for(design, task_id):
    lex = Lexicon(task_id)
    roles = [lex.canon(p.role) or (p.role or "") for p in design.parts]
    if any(p.role for p in design.parts):
        cols = [ROLE_COLORS[r] if r in ROLE_COLORS else (_role_color(r) if r else UNKNOWN) for r in roles]
        return cols, "role"
    return [colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.45, 0.82) for i in range(len(design.parts))], "part"


def render_design(design, task_id, out_prefix, size=768, samples=32):
    import os
    import trimesh
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cols, mode = colors_for(design, task_id)
    scene = trimesh.Scene()
    colors = {}
    for i, (p, c) in enumerate(zip(design.parts, cols)):
        name = f"n{i:03d}"
        scene.add_geometry(trimesh.Trimesh(np.asarray(p.vertices) @ render.ZUP_TO_YUP.T, p.faces, process=False),
                           node_name=name, geom_name=name)
        colors[name] = [float(x) ** 2.2 for x in c]     # the palette is sRGB; Blender wants linear
    glb = str(out_prefix) + "_scene.glb"
    scene.export(glb)
    render._run({"mode": "design", "glb": glb, "out": str(out_prefix), "size": size, "samples": samples,
                 "views": {v: render.VIEWS[v] for v in VIEWS}, "colors": colors, "ground": True,
                 "view_transform": "Standard", "world_strength": 0.35, "light_scale": 0.55,
                 "ground_rgb": [0.30, 0.31, 0.31]}, timeout=1800)
    os.unlink(glb)
    from PIL import Image
    outs = {}
    for v in VIEWS:
        png = Path(f"{out_prefix}_{v}.png")
        jpg = png.with_suffix(".jpg")
        Image.open(png).convert("RGB").save(jpg, "JPEG", quality=86, optimize=True)
        png.unlink()
        outs[v] = str(jpg)
    return outs, mode
