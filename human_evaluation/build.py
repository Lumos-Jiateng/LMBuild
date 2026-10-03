"""Build the human-evaluation set for Level 3 (3.1 decomposition, 3.2 aesthetics, 3.3 structure alignment).

v2 (2026-09-21, the owner's revision of v1):
  - six systems, 15 objects -> 90 designs
  - ONE question per dimension, a single 1-5 rating (one click per item)
  - every dimension is shown the same bright part-coloured sheet (white background, one colour per part:
    two assembled views over two exploded views), in the look of delivery/agent_trajectories --style white
  - 3.2 is still shown without the photograph; 3.1 and 3.3 with it, as for the VLM judge
  - every item names the object being built (e.g. "dining table"), so 3.2 is judged as that kind of object

Designs: Tier B, name_only+image, round-1 checkpoint of the 3-round trajectory (seed 0), for the tool
systems; the single final output for the two domain-specific generators.

Assignment: 90 designs x 3 dimensions = 270 items. Each of the 4 annotators gets 90 (30 per dimension);
the 360 - 270 = 90 spare slots double-annotate 30 items per dimension (5 per system) for inter-annotator
agreement.

    .venv_eval/bin/python human_evaluation/build.py [--workers 24] [--force]
"""
from __future__ import annotations

import argparse
import colorsys
import dataclasses
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ppbench.v2.task import RESULTS, Task     # noqa: E402

SYSTEMS = [   # (system id, role, eval key suffix)
    ("gpt-6-astra", "closed API (OpenAI)", "__B__name_only+image__r3__s0__round1"),
    ("claude-sonnet-5", "closed API (Anthropic)", "__B__name_only+image__r3__s0__round1"),
    ("qwen3.5-27b", "open, strong (Qwen)", "__B__name_only+image__r3__s0__round1"),
    ("gemma-4-31b-it", "open, weaker (Gemma)", "__B__name_only+image__r3__s0__round1"),
    ("partpacker", "domain-specific: image -> part-level 3D", "__ext__image__s0"),
    ("brickgpt", "domain-specific: LEGO (text -> bricks)", "__ext__name_only__s0"),
]
# 15 of the v1 20, one or two per kind of object; gemma-4-31b-it had 0-1 parts on farm_tractor,
# helicopter, dining_chair and casement_window, which is why those are not in the set.
TASKS = ["offroad_jeep", "passenger_car", "wheeled_excavator", "scissor_car_jack", "bench_drill_press",
         "oscillating_steam_engine", "dining_table", "bed_frame", "chest_of_drawers", "swivel_office_chair",
         "stepladder", "refrigerator", "desk_lamp", "oscillating_fan", "wheelchair"]
ANNOTATORS = ["A1", "A2", "A3", "A4"]
PER_DIM = 30                       # items per annotator per dimension
DIM_ORDER = ["3.2", "3.3", "3.1"]  # aesthetics first, before the annotator has seen any photograph
MIN_PARTS = 2
SEED = 20260921

QUESTIONS = {
    "3.2": {"name": "aesthetics", "photo": False,
            "question": "Is the design itself elegant — does it look like a well-designed object that agrees with human aesthetic sense?",
            "anchors": {"1": "ugly or incoherent: a heap of unrelated shapes, grossly distorted proportions",
                        "3": "acceptable but plain or awkward: crude shapes, parts that do not quite belong together",
                        "5": "elegant: balanced proportions, parts that belong together, a finished-looking form"}},
    "3.3": {"name": "structure_alignment", "photo": True,
            "question": "How faithfully does the design reproduce the structure of the object in the photograph — its parts, their placement, their relative sizes and the fine details (e.g. how many legs, wheels or shelves)?",
            "anchors": {"1": "a different object, or the structure is unrelated",
                        "3": "the right kind of object, but clear structural differences (missing / invented / misplaced parts, wrong counts)",
                        "5": "structurally faithful in every visible respect"}},
    "3.1": {"name": "decomposition", "photo": True,
            "question": "Is the object broken into parts the way the real product is made and assembled? (One colour = one part.)",
            "anchors": {"1": "unusable: a pile of fragments / bricks, or the whole object is one lump",
                        "3": "partly right: some real components chopped into pieces, or moving components fused to what they move against",
                        "5": "every real component is one part, and components that move or are made separately are separate parts"}},
}

# ---------------------------------------------------------------- bright part-coloured sheet
WHITE_WORKER = ROOT / "delivery" / "agent_trajectories" / "white_worker.py"
CACHE = HERE / "renders"
SIZE, SAMPLES = 512, 24
EXPLODE_FRAC, EXPLODE_MAX_PARTS = 0.25, 40


def part_rgb(i):
    """One bright, well-separated hue per part (golden-ratio hue walk, the white-style saturation)."""
    return list(colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.62, 0.97))


def _single_sided(v, f):
    """Catalog meshes store every face twice (once per winding); the coincident twins shade black."""
    import trimesh
    m = trimesh.Trimesh(v, f, process=True)
    _, first = np.unique(np.sort(m.faces, 1), axis=0, return_index=True)
    if len(first) == len(m.faces):
        return trimesh.Trimesh(v, f, process=False)
    return trimesh.Trimesh(m.vertices, m.faces[np.sort(first)], process=False)


def white_render(parts, out_prefix, views, shadow=True):
    """parts: objects with vertices (world, z-up) and faces. Writes <out_prefix>_<view>.png on white."""
    import trimesh
    from PIL import Image
    from ppbench.v2 import render
    scene, colors = trimesh.Scene(), {}
    for i, p in enumerate(parts):
        name = f"n{i:03d}"
        scene.add_geometry(_single_sided(np.asarray(p.vertices) @ render.ZUP_TO_YUP.T, np.asarray(p.faces)),
                           node_name=name, geom_name=name)
        colors[name] = [float(x) ** 2.2 for x in part_rgb(i)]
    tmp = Path(tempfile.mkdtemp(prefix="he_", dir="/dev/shm"))
    try:
        scene.export(str(tmp / "scene.glb"))
        job = {"glb": str(tmp / "scene.glb"), "out": str(tmp / "o"), "size": SIZE, "samples": SAMPLES,
               "views": {v: render.VIEWS[v] for v in views}, "colors": colors, "shadow": shadow}
        (tmp / "job.json").write_text(json.dumps(job))
        r = subprocess.run([render.BPY, str(WHITE_WORKER), "--", str(tmp / "job.json")], capture_output=True,
                           text=True, timeout=1800, env=dict(os.environ, RENDER_THREADS=os.environ.get("RENDER_THREADS", "3")))
        if r.returncode != 0:
            raise RuntimeError("blender failed: " + (r.stderr or r.stdout)[-1500:])
        for v in views:
            a = np.asarray(Image.open(tmp / f"o_{v}.png").convert("RGBA")).astype(np.float32) / 255.0
            rgb, al = a[..., :3], a[..., 3]
            sh = (rgb.max(-1) < 0.1) & (al < 0.995)            # shadow-catcher pixels: keep the contact shadow faint
            al = np.where(sh, al * 0.4, al)
            img = np.clip(rgb * al[..., None] + (1.0 - al[..., None]), 0, 1) * 255
            lum = img.min(-1)
            t = np.clip((lum - 225.0) / 25.0, 0.0, 1.0)[..., None]
            t = t * t * (3 - 2 * t)
            Image.fromarray((img * (1 - t) + 255.0 * t).astype(np.uint8)).save(f"{out_prefix}_{v}.png")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _explode(parts):
    V = np.vstack([p.vertices for p in parts])
    lo, hi = V.min(0), V.max(0)
    c, d = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
    out = []
    for p in parts:
        pv = np.asarray(p.vertices)
        v = (pv.min(0) + pv.max(0)) / 2 - c
        n = float(np.linalg.norm(v))
        step = (v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])) * EXPLODE_FRAC * d
        out.append(dataclasses.replace(p, vertices=pv + step))
    return out


def sheet(task_id, item, out_png):
    from ppbench.v2 import report as R
    from ppbench.v2 import render
    task = Task(task_id, snapshot=RESULTS / task_id / "task_snapshot_core.json")
    parts = R.load_design(task, item).parts
    if len(parts) < MIN_PARTS:
        raise RuntimeError(f"{len(parts)} parts")
    work = Path(tempfile.mkdtemp(prefix="hes_", dir="/dev/shm"))
    try:
        white_render(parts, work / "a", ("cond", "back_iso"))
        if len(parts) <= EXPLODE_MAX_PARTS:
            white_render(_explode(parts), work / "e", ("iso", "front"), shadow=False)
            tail, lab = [work / "e_iso.png", work / "e_front.png"], ["exploded, front-left", "exploded, front"]
        else:   # past ~40 parts an exploded view is a cloud; two more assembled views say more
            white_render(parts, work / "e", ("front", "top"))
            tail, lab = [work / "e_front.png", work / "e_top.png"], ["front", "top"]
        render.grid([work / "a_cond.png", work / "a_back_iso.png", *tail], out_png, cols=2, cell=SIZE,
                    labels=["assembled, front-right", "assembled, rear-left", *lab])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return len(parts)


def _job(args):
    task_id, item, out = args
    try:
        return task_id, item["key"], f"ok {sheet(task_id, item, out)} parts"
    except Exception as e:
        return task_id, item["key"], f"error {type(e).__name__}: {e}"


# ---------------------------------------------------------------- the set
def _record(task, key):
    return json.loads((RESULTS / task / "eval_v33" / f"{key}.json").read_text())


def _n_parts(rec):
    for d in ("3.1", "1.1"):
        n = ((rec["dims"].get(d) or {}).get("metrics") or {}).get("n_parts")
        if n is not None:
            return int(n)
    return 0


def assign(designs):
    """Latin square per dimension: design (task t, system j) goes to annotator (t + j + dim offset) mod 4, so each
    annotator gets every system 3-4 times per dimension, and the dimension offset sends the three ratings of one
    design to three different annotators (no carry-over between dimensions). The spare slots double-annotate 5
    designs per system per dimension, never to the design's own rater in that dimension, preferably to
    an annotator who has not seen the design at all."""
    ti = {t: n for n, t in enumerate(TASKS)}
    sj = {s[0]: n for n, s in enumerate(SYSTEMS)}
    mine = {dim: {a: [] for a in ANNOTATORS} for dim in DIM_ORDER}
    seen = {a: set() for a in ANNOTATORS}
    for k, dim in enumerate(DIM_ORDER):
        for d in designs:
            a = ANNOTATORS[(ti[d["task"]] + sj[d["system"]] + k) % len(ANNOTATORS)]
            mine[dim][a].append(d)
            seen[a].add(d["id"])
    for k, dim in enumerate(DIM_ORDER):
        rng = random.Random(f"{SEED}-{dim}")
        per_sys = (PER_DIM * len(ANNOTATORS) - len(designs)) // len(SYSTEMS)
        dups = []
        for s in SYSTEMS:
            grp = [d for d in designs if d["system"] == s[0]]
            dups += rng.sample(grp, per_sys)
        for _ in range(1000):
            trial = {a: list(v) for a, v in mine[dim].items()}
            sn = {a: set(v) for a, v in seen.items()}
            rng.shuffle(dups)
            ok = True
            for d in dups:
                need = [a for a in ANNOTATORS if d not in trial[a] and len(trial[a]) < PER_DIM]
                if not need:
                    ok = False
                    break
                # prefer an annotator who has not seen this design in another dimension
                a = min(need, key=lambda x: (d["id"] in sn[x], len(trial[x]), rng.random()))
                trial[a].append(d)
                sn[a].add(d["id"])
            if ok:
                mine[dim], seen = trial, sn
                break
        else:
            raise SystemExit(f"could not place the duplicates for {dim}")
    out = {a: [] for a in ANNOTATORS}
    for dim in DIM_ORDER:
        for a in ANNOTATORS:
            assert len(mine[dim][a]) == PER_DIM, (dim, a, len(mine[dim][a]))
            block = [{"item_id": f"{d['id']}_{dim}", "design": d["id"], "task": d["task"], "dim": dim}
                     for d in mine[dim][a]]
            random.Random(f"{SEED}-{a}-{dim}").shuffle(block)
            out[a] += block
    return out


def build(workers=24, force=False):
    img = HERE / "static" / "img"
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    CACHE.mkdir(exist_ok=True)

    designs, jobs = [], []
    for t in TASKS:
        for sysid, role, suf in SYSTEMS:
            key = sysid + suf
            rec = _record(t, key)
            n = _n_parts(rec)
            if n < MIN_PARTS:
                raise SystemExit(f"{t} {key}: {n} parts -- the set must not contain empty designs")
            designs.append({"task": t, "system": sysid, "role": role, "key": key, "n_parts": n,
                            "design_path": rec["item"].get("path")})
            out = CACHE / t / f"{key}.png"
            if force or not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((t, rec["item"], out))
    print(f"{len(designs)} designs, {len(jobs)} sheets to render", flush=True)
    if jobs:
        os.environ.setdefault("RENDER_THREADS", "3")
        with ProcessPoolExecutor(workers) as ex:
            for t, k, st in ex.map(_job, jobs):
                print(f"  {t:26s} {k:55s} {st}", flush=True)

    rng = random.Random(SEED)
    for d, i in zip(designs, rng.sample(range(1000, 10000), len(designs))):
        d["id"] = f"d{i}"
        src = CACHE / d["task"] / f"{d['key']}.png"
        if not src.exists():
            raise SystemExit(f"missing sheet {src}")
        from PIL import Image
        Image.open(src).convert("RGB").save(img / f"{d['id']}.jpg", quality=92)   # opaque name: no system leak
    photos = {}
    from PIL import Image
    for t in TASKS:
        task = Task(t, snapshot=RESULTS / t / "task_snapshot_core.json")
        im = Image.open(task.condition("name_only+image")["image"]["path"]).convert("RGB")
        im.thumbnail((768, 768))          # the size the VLM judge is sent
        im.save(img / f"photo_{t}.jpg", quality=92)
        photos[t] = f"img/photo_{t}.jpg"

    asg = assign(designs)
    public = {"version": 2, "tasks": {t: {"photo": photos[t], "name": t.replace("_", " ")} for t in TASKS},
              "designs": {d["id"]: {"task": d["task"], "n_parts": d["n_parts"]} for d in designs},
              "questions": QUESTIONS, "dim_order": DIM_ORDER, "assignments": asg}
    (HERE / "static" / "set.json").write_text(json.dumps(public, indent=1))
    (HERE / "key.json").write_text(json.dumps({"version": 2, "systems": SYSTEMS, "tasks": TASKS, "designs": designs},
                                              indent=1))
    print("wrote " + ", ".join(f"{a} {len(v)}" for a, v in asg.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()
    build(ns.workers, ns.force)
