"""Render sheets the Level 3 judge reads: neutral clay, and a decomposition sheet.

Written for the 2026-09-20 Level 3 review (docs/evaluation/metrics_feedback.md, P8-P13). The judge is
never shown a part list or coordinates, so every question it is asked has to be answerable from a
picture:

    <key>_claygrid.png   4 views (cond, iso, front, side) in one grey clay, no part boundaries visible
                         -> 3.2 aesthetics (shown alone) and 3.3 alignment (shown with the photo)
    <key>_decomp.png     2 part-coloured views + 2 exploded views, one colour per part
                         -> 3.1 decomposition

Clay is deliberately colourless: a design that declared nice materials must not win an aesthetics
point for its palette. The decomposition sheet is the opposite -- one hue per part is the whole point.

    .venv_eval/bin/python -m ppbench.v2.judge_views build [--tasks all] [--workers 40] [--force]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ppbench.v2.task import RESULTS, Task

# Blender writes eight panels and a scratch GLB per design. Doing that straight onto the shared
# results filesystem, from tens of processes at once, loses files: panels come back truncated
# (UnidentifiedImageError) and the scratch GLB vanishes under its own writer. Everything is rendered
# on local scratch and only the two finished sheets are moved across.
WORK = Path(os.environ.get("PPBENCH_VIEW_TMP", "/tmp/ppbench_views"))

SIZE = 384
SAMPLES = 16
EXPLODE_FRAC = 0.25      # each part slides this fraction of the bbox diagonal away from the centre
EXPLODE_MAX_PARTS = 40   # past this the exploded view is a cloud; two more part-coloured views say more
CLAY = (0.72, 0.72, 0.72)   # neutral grey; the palette has no grey entry
CLAY_VIEWS = ("cond", "back_iso", "front", "side")
PART_VIEWS = ("cond", "back_iso")
EXPLODE_VIEWS = ("iso", "front")
MANY_VIEWS = ("front", "top")


def _explode(parts, frac=EXPLODE_FRAC):
    """Push every part away from the assembly centre along the direction of its own centroid."""
    V = np.vstack([p.vertices for p in parts])
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2
    d = float(np.linalg.norm(hi - lo))
    out = []
    for p in parts:
        pc = (np.asarray(p.vertices).min(0) + np.asarray(p.vertices).max(0)) / 2
        v = pc - c
        n = float(np.linalg.norm(v))
        step = (v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])) * frac * d
        out.append(dataclasses.replace(p, vertices=np.asarray(p.vertices) + step))
    return out


def _retry(fn, n=3, wait=3.0):
    """Renders go through Blender on a shared filesystem; a lost write is worth one more try before
    the design is dropped from the judge set."""
    last = None
    for k in range(n):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(wait * (k + 1))
    raise last


def sheets(design, out_prefix, force=False):
    """Both sheets for one design. Returns {"clay": path, "decomp": path}."""
    from ppbench.v2 import render
    out_prefix = Path(out_prefix)
    clay, decomp = out_prefix.with_name(out_prefix.name + "_claygrid.png"), out_prefix.with_name(out_prefix.name + "_decomp.png")
    if not force and clay.exists() and decomp.exists():
        return {"clay": str(clay), "decomp": str(decomp), "status": "cached"}
    parts = design.parts
    if not parts:
        return {"clay": None, "decomp": None, "status": "empty"}

    _retry(lambda: render.render_parts(parts, out_prefix.with_name(out_prefix.name + "_clay"),
                                       views=CLAY_VIEWS, size=SIZE, samples=SAMPLES,
                                       color_key=lambda p: "all", mono=CLAY))
    render.grid([f"{out_prefix}_clay_{v}.png" for v in CLAY_VIEWS], clay, cols=2, cell=SIZE,
                labels=["front-right (condition camera)", "rear-left", "front", "side"])

    _retry(lambda: render.render_parts(parts, out_prefix.with_name(out_prefix.name + "_part"),
                                       views=PART_VIEWS, size=SIZE, samples=SAMPLES, color_key=lambda p: p.id))
    if len(parts) <= EXPLODE_MAX_PARTS:
        _retry(lambda: render.render_parts(_explode(parts), out_prefix.with_name(out_prefix.name + "_exp"),
                                           views=EXPLODE_VIEWS, size=SIZE, samples=SAMPLES,
                                           color_key=lambda p: p.id, ground=False))
        tail, tlab = [f"{out_prefix}_exp_{v}.png" for v in EXPLODE_VIEWS], ["exploded iso", "exploded front"]
    else:
        _retry(lambda: render.render_parts(parts, out_prefix.with_name(out_prefix.name + "_more"),
                                           views=MANY_VIEWS, size=SIZE, samples=SAMPLES, color_key=lambda p: p.id))
        tail, tlab = [f"{out_prefix}_more_{v}.png" for v in MANY_VIEWS], ["front", "top"]
    render.grid([f"{out_prefix}_part_{v}.png" for v in PART_VIEWS] + tail, decomp, cols=2, cell=SIZE,
                labels=["parts, front-right", "parts, rear-left"] + tlab)

    # drop exactly this design's panels. The old glob "<key>_*_*.png" also matched the panels of <key>__round<n>
    # (a round checkpoint of the same trajectory rendering at the same time in the same task folder) and deleted
    # them mid-render (2026-09-22, once round checkpoints joined the judge set).
    for kind, views in (("clay", CLAY_VIEWS), ("part", PART_VIEWS), ("exp", EXPLODE_VIEWS), ("more", MANY_VIEWS)):
        for v in views:
            out_prefix.with_name(f"{out_prefix.name}_{kind}_{v}.png").unlink(missing_ok=True)
    return {"clay": str(clay), "decomp": str(decomp), "status": "ok"}


# ---------------------------------------------------------------- the set of designs the judge scores
def main_setting(task_id, source="eval_v33"):
    """Main setting only: name_only+image, the 3-round trajectory's final design, seed 0, tiers A and B,
    plus every domain-specific baseline and the reference design as the calibration anchor."""
    d = RESULTS / task_id / source
    if not d.is_dir():
        return []
    # PPB_JUDGE_SCOPE=all (2026-09-21, the user asked for 3.1-3.3 on every setting): also Tier C, the prompt
    # ablations and the round-1/round-2 checkpoints of 3-round trajectories (round 3 is the final design, judged
    # under its own key). Same sheets, prompts and judges; the default stays the main setting.
    if os.environ.get("PPB_JUDGE_SCOPE", "main") == "all":
        return _all_settings(d)
    out = []
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".error.txt") or "__round" in f.name:
            continue
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        it = rec.get("item") or {}
        k = it.get("kind", "tool")
        if k == "external":
            out.append(it)
        elif k == "tool" and it.get("tier") in ("A", "B") and it.get("condition") == "name_only+image" \
                and it.get("rounds") == 3 and it.get("seed") == 0:
            out.append(it)
    return out


GRID = {("name_only+image", "C", 3), ("name_only+image", "B", 3), ("name_only+image", "A", 3),   # the six phases of
        ("attributes+image", "B", 1), ("attributes+image", "A", 1), ("functional+image", "B", 1)}  # settings.md


def _all_settings(d):
    skip = set(x for x in os.environ.get("PPB_JUDGE_EXCLUDE", "").split(",") if x)
    # PPB_JUDGE_GRID_EXTRA="cond:tier:rounds,..." (2026-09-25) adds cells outside the six phases, e.g. the GPT Tier A
    # one-round ablation runs "name_only+image:A:1"; unset = the six phases only
    grid = set(GRID) | {(c, t, int(r)) for c, t, r in (x.split(":") for x in
                        os.environ.get("PPB_JUDGE_GRID_EXTRA", "").split(",") if x)}
    out = []
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".error.txt"):
            continue
        try:
            it = json.loads(f.read_text()).get("item") or {}
        except (OSError, ValueError):
            continue
        if it.get("system") in skip:
            continue
        if it.get("kind") == "external":
            out.append(it)
        elif it.get("kind", "tool") == "tool" and it.get("seed") == 0 \
                and (it.get("condition"), it.get("tier"), it.get("trajectory_rounds", it.get("rounds"))) in grid:
            cp = it.get("checkpoint_round")
            if cp is None or cp < it.get("trajectory_rounds", 0):
                out.append(it)
    return out


def _one(args):
    task_id, item, force = args
    from ppbench.v2 import report as R
    import shutil
    key = item["key"]
    final = RESULTS / task_id / "judge_views"
    try:
        if not force and (final / f"{key}_claygrid.png").exists() and (final / f"{key}_decomp.png").exists():
            return key, "cached"
        task = Task(task_id, snapshot=RESULTS / task_id / "task_snapshot_core.json")
        design = R.load_design(task, item)
        work = WORK / task_id
        work.mkdir(parents=True, exist_ok=True)
        res = sheets(design, work / key, force=True)
        if res["status"] != "ok":
            return key, res["status"]
        final.mkdir(parents=True, exist_ok=True)
        for kind in ("claygrid", "decomp"):
            src = work / f"{key}_{kind}.png"
            shutil.move(str(src), str(final / f"{key}_{kind}.png"))
        return key, "ok"
    except Exception as e:
        if item.get("kind") == "reference":
            return key, "no_reference"     # the 10 tasks with no segmented reference asset
        import traceback
        final.mkdir(parents=True, exist_ok=True)
        (final / f"{key}.error.txt").write_text(traceback.format_exc())
        return key, f"error {type(e).__name__}: {e}"


def reference_item(task_id):
    return {"kind": "reference", "key": "reference", "system": "reference", "tier": "ref", "task": task_id}


def build(tasks=None, workers=40, force=False, limit=None):
    ids = tasks or [t["task_id"] for t in json.loads(
        (__import__("ppbench.v2.task", fromlist=["CORE"]).CORE).read_text())]
    jobs = []
    for t in ids:
        jobs += [(t, it, force) for it in main_setting(t)]
        jobs.append((t, reference_item(t), force))
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} sheets over {len(ids)} tasks", flush=True)
    os.environ.setdefault("RENDER_THREADS", "3")
    tot, t0 = defaultdict(int), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for n, (key, st) in enumerate(ex.map(_one, jobs, chunksize=4), 1):
            tot[st.split(":")[0] if isinstance(st, str) else st] += 1
            if st.startswith("error"):
                print(f"  {key:70s} {st}", flush=True)
            if n % 200 == 0:
                print(f"{n}/{len(jobs)}  {dict(tot)}  {time.time()-t0:.0f}s", flush=True)
    print(f"done {dict(tot)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    ns = ap.parse_args()
    ts = None if ns.tasks == "all" else [x.strip() for x in ns.tasks.split(",") if x.strip()]
    build(ts, ns.workers, ns.force, ns.limit)
