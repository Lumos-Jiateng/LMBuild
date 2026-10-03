"""Renders of a design, and the pool contact sheet.

Views are (azimuth, elevation) in degrees in the world frame (+Z up, front -Y),
camera at azimuth a sits along (cos a, sin a). `cond` matches the camera of
the task's condition image, which was generated from the reference render at
azimuth 300, elevation 22.
"""
from __future__ import annotations

import colorsys
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# A Python with the `bpy` module (pip install bpy==5.2.1, Python 3.13), or Blender itself run as `blender -b --python`.
BPY = os.environ.get("PPBENCH_BPY", str(Path(__file__).resolve().parents[2] / ".venv_render" / "bin" / "python"))
WORKER = Path(__file__).with_name("render_worker.py")
VIEWS = {"cond": (300, 22), "iso": (-55, 26), "front": (-90, 8), "side": (0, 8), "back_iso": (125, 26), "top": (-90, 80)}
ZUP_TO_YUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def palette(n):
    return [list(colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.55, 0.85)) for i in range(n)]


def _run(job, timeout=900):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=os.environ.get("TMPDIR")) as fh:
        json.dump(job, fh)
    env = dict(os.environ, RENDER_THREADS=os.environ.get("RENDER_THREADS", "16"))
    r = subprocess.run([BPY, str(WORKER), "--", fh.name], capture_output=True, text=True, timeout=timeout, env=env)
    os.unlink(fh.name)
    if r.returncode != 0:
        raise RuntimeError("blender failed: " + (r.stderr or r.stdout)[-1500:])


def render_parts(parts, out_prefix, views=("iso", "front", "side", "cond"), size=512, samples=24,
                 color_key=None, ground=True, mono=None):
    """parts: objects with id, vertices (world), faces, and optionally role. Returns {view: png path}.

    `mono` is an explicit RGB for every part, for neutral clay: `color_key` groups parts and then
    picks hues off the palette, which has no entry that is actually grey."""
    import trimesh
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    keys = [(color_key(p) if color_key else p.id) for p in parts]
    uniq = list(dict.fromkeys(keys))
    pal = dict(zip(uniq, palette(len(uniq))))
    colors = {}
    for i, (p, k) in enumerate(zip(parts, keys)):
        name = f"n{i:03d}"
        m = trimesh.Trimesh(np.asarray(p.vertices) @ ZUP_TO_YUP.T, p.faces, process=False)
        scene.add_geometry(m, node_name=name, geom_name=name)
        colors[name] = list(mono) if mono else pal[k]
    glb_path = str(out_prefix) + "_scene.glb"
    scene.export(glb_path)
    job = {"mode": "design", "glb": glb_path, "out": str(out_prefix), "size": size, "samples": samples,
           "views": {v: VIEWS[v] for v in views}, "colors": colors, "ground": ground}
    try:
        _run(job)
    finally:
        # the scratch GLB is the caller's to clean up, but a parallel run on a shared filesystem can
        # lose the directory entry under it; that must not mask whatever really went wrong
        try:
            os.unlink(glb_path)
        except FileNotFoundError:
            pass
    out = {v: f"{out_prefix}_{v}.png" for v in views}
    missing = [v for v, f in out.items() if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"blender wrote no image for {missing} of {out_prefix}")
    return out


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def grid(paths, out, labels=None, cols=4, cell=256, font_size=17, label_h=26):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(FONT, font_size)
    except OSError:
        font = ImageFont.load_default()
    ims = [Image.open(p).convert("RGB").resize((cell, cell)) for p in paths]
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(ims):
        x, y = (i % cols) * cell, (i // cols) * (cell + label_h)
        sheet.paste(im, (x, y + label_h))
        if labels:
            d.text((x + 3, y + 3), labels[i][:36], fill=(0, 0, 0), font=font)
    sheet.save(out)
    return str(out)


def pool_sheet(task, out_dir, per_sheet=30, cols=6):
    """One thumbnail per pool part plus labelled sheets. Cached on disk."""
    out_dir = Path(out_dir)
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    items = [{"glb": p["mesh"], "out": str(thumbs / f"{p['pool_part_id']}.png")} for p in task.pool
             if not (thumbs / f"{p['pool_part_id']}.png").exists()]
    if items:
        k = max(1, len(items) // 8 + 1)
        chunks = [items[i::k] for i in range(k)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(chunks)) as ex:
            list(ex.map(lambda ch: _run({"mode": "thumbs", "items": ch, "size": 256, "samples": 16}, timeout=3600), chunks))
    sheets = []
    pool = task.pool
    for s in range(0, len(pool), per_sheet):
        chunk = pool[s:s + per_sheet]
        labels = [f"{p['pool_part_id']} {p['name']} {'x'.join(f'{v:.2f}' for v in p['size_m'])}m" for p in chunk]
        out = out_dir / f"pool_sheet_{s // per_sheet + 1}.png"
        grid([str(thumbs / f"{p['pool_part_id']}.png") for p in chunk], out, labels, cols=cols)
        sheets.append(str(out))
    # one image with every part, for transports that limit images per request
    labels = [f"{p['pool_part_id']} {p['name'][:14]} {'x'.join(f'{v:.2f}' for v in p['size_m'])}" for p in pool]
    grid([str(thumbs / f"{p['pool_part_id']}.png") for p in pool], out_dir / "pool_sheet_all.png", labels,
         cols=8, cell=220, font_size=14, label_h=22)
    return sheets


def sheet_all(task_id):
    from ppbench.v2.task import task_dir
    p = task_dir(task_id) / "pool_sheet" / "pool_sheet_all.png"
    return str(p) if p.exists() else None
