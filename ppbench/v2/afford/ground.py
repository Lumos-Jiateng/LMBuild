"""Which segment of a design is which component: a blind visual judge.

Each segment is shown as one 3-panel sheet: two views of the whole object in grey with the segment in
orange (the part of it hidden behind other geometry drawn semi-transparent, so internal components are
still visible), and the segment alone, enlarged. The judge gets the object's name and the task's
component list as lettered options, plus "none of these", and answers with one letter; the answer
distribution is read from the first token's log-probabilities. The design's declared names are never
shown: a part is a wheel because it looks and sits like one.

Two non-contestant judges (Qwen2.5-VL-32B, gemma-3-27b) answer every sheet; A.x use their mean.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import string
import time
from pathlib import Path

import numpy as np

from ppbench.v2.afford import raster
from ppbench.v2.afford.core import CACHE

SHEETS = Path(os.environ.get("PPB_AFFORD_SHEETS", "/tmp/ppbench_afford/sheets"))
PANEL = 256
VIEW_ORDER = list(raster.VIEWS)
ORANGE = np.array([1.0, 0.45, 0.05])
LETTERS = string.ascii_uppercase
JUDGES = {"qwen2.5-vl-32b": "http://127.0.0.1:8411/v1", "gemma-3-27b": "http://127.0.0.1:8412/v1"}
PROMPT_VERSION = "g1"


def _meshes(d):
    return [(p.vertices, p.faces) for p in d.parts]


def render_design(d, segs, out_dir: Path):
    """Write one sheet per segment; returns [(segment index, png path, visible_frac)]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes = _meshes(d)
    V = np.vstack([p.vertices for p in d.parts])
    lo, hi = V.min(0), V.max(0)
    views = {}
    for v in VIEW_ORDER:
        fr = raster.Frame(lo, hi, v, PANEL)
        ib, dp, sh = raster.rasterize(meshes, fr)
        views[v] = (fr, ib, sh)
    part_seg = np.full(len(d.parts), -1)
    for si, s in enumerate(segs):
        part_seg[s] = si
    res = []
    for si, s in enumerate(segs):
        sub = [meshes[i] for i in s]
        best = []
        for v in VIEW_ORDER:
            fr, ib, sh = views[v]
            solo, _, _ = raster.rasterize(sub, fr)
            vis = np.isin(ib, s)
            m = solo >= 0
            best.append((vis.sum() + 0.3 * (m & ~vis).sum(), v, vis, m))
        best.sort(key=lambda t: -t[0])
        pick = [best[0]] + [b for b in best[1:] if b[1] != best[0][1]][:1]
        panels = []
        for _, v, vis, m in pick:
            fr, ib, sh = views[v]
            img = np.ones((PANEL, PANEL, 3))
            bg = ib >= 0
            g = (0.42 + 0.5 * sh)[..., None] * np.ones(3)
            img[bg] = g[bg]
            occl = m & ~vis
            img[occl] = 0.45 * img[occl] + 0.55 * ORANGE
            img[vis] = (0.35 + 0.65 * sh[vis])[:, None] * ORANGE
            panels.append(img)
        # the segment alone, enlarged, in the view that showed it best
        Vs = np.vstack([meshes[i][0] for i in s])
        fr = raster.Frame(Vs.min(0), Vs.max(0), pick[0][1], PANEL, margin=0.1)
        ib, _, sh = raster.rasterize(sub, fr)
        img = np.ones((PANEL, PANEL, 3))
        m = ib >= 0
        img[m] = (0.35 + 0.65 * sh[m])[:, None] * ORANGE
        panels.append(img)
        sheet = np.concatenate([np.pad(p, ((0, 0), (0, 4), (0, 0)), constant_values=1.0) if k < 2 else p
                                for k, p in enumerate(panels)], axis=1)
        from PIL import Image
        path = out_dir / f"seg{si:03d}.png"
        Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(path, optimize=True)
        total_px = sum(int(b[3].sum()) for b in best[:1])
        vis_px = int(pick[0][2].sum())
        res.append((si, str(path), vis_px / max(total_px, 1)))
    return res


def options(comps):
    """Lettered options: the components (at most 25), then 'none of these'."""
    opts = [c for c in comps][:25]
    return opts


def prompt(obj_name, comps):
    lines = []
    for k, c in enumerate(options(comps)):
        why = f" ({c['why']})" if c.get("why") else ""
        lines.append(f"{LETTERS[k]}. {c['name']}{why}")
    none = LETTERS[len(options(comps))]
    lines.append(f"{none}. none of these (another part of the {obj_name}, or not recognisable)")
    art = "an" if obj_name[:1].lower() in "aeiou" else "a"
    return (f"The picture shows a 3D model of {art} {obj_name}. The two left panels show the whole model in grey "
            f"with ONE component highlighted in orange (where other geometry hides it, it is drawn "
            f"semi-transparent). The right panel shows the highlighted component on its own, enlarged.\n\n"
            f"Which component of the {obj_name} is the orange component?\n" + "\n".join(lines) +
            f"\n\nAnswer with the single letter of the best option.")


def ask(client, model, png_path, text, retries=4):
    b64 = base64.b64encode(Path(png_path).read_bytes()).decode()
    msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                        {"type": "text", "text": text}]}]
    for a in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=msg, max_tokens=1, temperature=0.0,
                                               logprobs=True, top_logprobs=20)
            ch = r.choices[0]
            top = ch.logprobs.content[0].top_logprobs if ch.logprobs and ch.logprobs.content else []
            return {"text": ch.message.content, "top": [(t.token, t.logprob) for t in top]}
        except Exception as e:   # the server restarts or is saturated: back off
            err = str(e)
            time.sleep(3 * (a + 1))
    return {"error": err}


def distribution(ans, n_opts):
    """Probabilities over the n_opts+1 letters from the first-token top log-probabilities."""
    p = np.zeros(n_opts + 1)
    for tok, lp in ans.get("top") or []:
        t = tok.strip().strip(".").upper()
        if len(t) == 1 and t in LETTERS[:n_opts + 1]:
            p[LETTERS.index(t)] += math.exp(lp)
    if p.sum() <= 0:
        t = (ans.get("text") or "").strip().strip(".").upper()[:1]
        if t and t in LETTERS[:n_opts + 1]:
            p[LETTERS.index(t)] = 1.0
    return (p / p.sum()).tolist() if p.sum() > 0 else None
