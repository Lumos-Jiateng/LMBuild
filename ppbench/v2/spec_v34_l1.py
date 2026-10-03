"""L1 v3.4: Level 1 (1.1 connectivity, 1.2 collision, 1.3 stability) at the scale of the design itself.

v3.3 measured every design on a fixed 4 mm voxel grid with absolute floors (1 cm^3 collision, 0.1 cm^3
interlock, 1 cm^2 fastening area, 2 cm joint-origin tolerance, 5 mm ground band). Those values were chosen
for metre-scale furniture. A toy-scale design (a LEGO model 6-17 cm tall built from 7.8 mm bricks) is
then smaller than the instrument: a 1-stud brick is two cells wide, its eroded core is empty, and a whole
2x2 brick (2.5 cm^3) is below the 1 cm^3 floor, so two bricks occupying the same place read as no
collision (audit 2026-10-02: 0-2 of 10 injected duplicate bricks detected).

v3.4 applies one rule to every design, with no reference to who made it:

    D      = diagonal of the design's own bounding box (all parts, metres, after the oracle scale and yaw
             that every external design already receives)
    t      = median over parts of the part's smallest bounding-box extent (its thickness)
    pitch  = clip(min(D / 250, t / 4), 0.4 mm, 4 mm)   # 250 cells across the object and 4 cells through a
                                                      # typical part; 4 mm for metre-scale designs, as in v3.3
             (coarsened by 1.25x steps, never beyond 4 mm, if the parts' boxes would exceed 3e8 cells)
    s      = pitch / 4 mm                         # 1 for metre-scale designs
    every length constant x s, every area x s^2, every volume x s^3:
        voxel pitch            4 mm      -> 4 mm * s
        collision floor        1 cm^3    -> 1 cm^3 * s^3      (= 15.6 cells, unchanged in cells)
        interlock floor        0.1 cm^3  -> 0.1 cm^3 * s^3
        fastening area         1 cm^2    -> 1 cm^2 * s^2
        joint-origin tolerance 2 cm      -> 2 cm * s
        ground band floor      5 mm      -> 5 mm * s
    dimensionless rules (fraction of the smaller part, 10 % / 25 %, tilt and push targets) are unchanged.

The pitch is never coarser than v3.3's 4 mm: a large design keeps v3.3's numbers exactly, and a small one
is resolved as finely as a metre-scale one is (in cells per object). The 0.4 mm floor is one LDraw unit, the
smallest feature any part system in the benchmark uses. The LDraw stud-stacking tolerance (1.5 mm) is a
property of the brick system, not of the design, and stays as it is. Reference anchors (stand / push
targets) are measured with the same rule on the reference.

    .venv_eval/bin/python -m ppbench.v2.spec_v34_l1 build [--tasks all] [--workers 48] [--force]
Records: results/v2/<task>/eval_v34l1/<key>.json (Level 1 only is recomputed; Levels 2-4 are carried from
the v3.2 record exactly as spec_v33 does). Never writes to any other eval directory.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ppbench.v2 import analysis, voxel
from ppbench.v2 import spec_v33 as V
from ppbench.v2.evaluate import _dim
from ppbench.v2.task import RESULTS

SPEC_VERSION = "v3.4-L1"
OUT_DIR = "eval_v34l1"

BASE_PITCH_M = 0.004            # v3.3's grid
CELLS_ACROSS = 250              # pitch = D / 250 -> 4 mm at D = 1 m
MIN_PITCH_M = 0.0004            # one LDraw unit
# the v3.3 constants this spec scales (read from v3.3 so the two never drift apart)
BASE = {"INTERLOCK_MIN_M3": V.INTERLOCK_MIN_M3, "FASTEN_MIN_AREA_M2": V.FASTEN_MIN_AREA_M2,
        "JOINT_ORIGIN_TOL_M": V.JOINT_ORIGIN_TOL_M, "COLLIDE_ABS_M3": V.COLLIDE_ABS_M3}
BASE_BAND_M = 0.005


CELLS_PER_THICKNESS = 4         # pitch <= median part thickness / 4: a typical part keeps an eroded core
MAX_DESIGN_CELLS = 300_000_000  # memory guard: coarsen (never beyond 4 mm) until the bounding boxes fit


def _cells(parts, pitch):
    return float(sum(np.prod((p.vertices.max(0) - p.vertices.min(0)) / pitch + 5) for p in parts))


def scale_for(parts):
    """D, pitch, s for one design. Two lengths of the design itself set the pitch: its diagonal (250 cells
    across the object) and the median thickness of its parts (4 cells through a typical part), whichever
    is finer; clipped to [0.4 mm, 4 mm] and coarsened only if the voxel budget would be exceeded."""
    V_ = np.vstack([p.vertices for p in parts])
    d = float(np.linalg.norm(V_.max(0) - V_.min(0)))
    thick = float(np.median([float(np.min(p.vertices.max(0) - p.vertices.min(0))) for p in parts]))
    pitch = float(np.clip(min(d / CELLS_ACROSS, thick / CELLS_PER_THICKNESS), MIN_PITCH_M, BASE_PITCH_M))
    while pitch < BASE_PITCH_M and _cells(parts, pitch) > MAX_DESIGN_CELLS:
        pitch = min(BASE_PITCH_M, pitch * 1.25)
    return d, pitch, pitch / BASE_PITCH_M


def _apply(s):
    """Set v3.3's module constants for one design (each worker handles one design at a time)."""
    V.INTERLOCK_MIN_M3 = BASE["INTERLOCK_MIN_M3"] * s ** 3
    V.FASTEN_MIN_AREA_M2 = BASE["FASTEN_MIN_AREA_M2"] * s ** 2
    V.JOINT_ORIGIN_TOL_M = BASE["JOINT_ORIGIN_TOL_M"] * s
    V.COLLIDE_ABS_M3 = BASE["COLLIDE_ABS_M3"] * s ** 3


def _restore():
    for k, v in BASE.items():
        setattr(V, k, v)


def stability(parts, occs, main, weights, corrected, anchors, n_parts, s):
    """v3.3's 1.3 with the ground-band floor scaled. Same targets and formula."""
    band = max(BASE_BAND_M * s, 0.01 * float(np.ptp(np.vstack([p.vertices for p in parts])[:, 2])))
    real = analysis.stability
    analysis.stability = lambda occs_, masses=None, extra_masses=(), band=0.005, _b=band: real(occs_, masses, extra_masses, _b)
    try:
        out = V.stability(parts, occs, main, weights, corrected, anchors, n_parts)
    finally:
        analysis.stability = real
    out.setdefault("metrics", {})["ground_band_m"] = band
    return out


def level1(design, task, anchors):
    parts, joints = design.parts, design.joints
    if not parts:
        z = _dim("computed", "fail", 0.0, {"n_parts": 0}, ["empty design"])
        return {"1.1": z, "1.2": dict(z), "1.3": dict(z)}
    over = V._oversize(parts, anchors)
    if over:
        return over
    d, pitch, s = scale_for(parts)
    scale_meta = {"design_diagonal_m": d, "voxel_res_m": pitch, "scale_factor": s,
                  "collision_floor_m3": BASE["COLLIDE_ABS_M3"] * s ** 3,
                  "joint_origin_tol_m": BASE["JOINT_ORIGIN_TOL_M"] * s}
    _apply(s)
    try:
        try:
            occs = analysis.occupancies(parts, res=pitch)
        except voxel.VoxelTooLarge as e:
            spans = [float(np.linalg.norm(p.vertices.max(0) - p.vertices.min(0))) for p in parts]
            met = {"n_parts": len(parts), "largest_part_span_m": max(spans), **scale_meta}
            return {k: _dim("computed", "fail", 0.0, dict(met), [f"a part is too large to voxelise: {e}"])
                    for k in ("1.1", "1.2", "1.3")}
        rows = V._pairs(parts, occs, joints)
        conn, main = V.connectivity(parts, occs, rows, joints)
        coll = V.collision(parts, occs, rows)
        weights, corrected = V._masses(parts, occs)
        stab = stability(parts, occs, main, weights, corrected, anchors or {}, len(parts), s)
    finally:
        _restore()
    note = (f"L1 v3.4: design-scaled instrument, D = {d:.3f} m, pitch = {pitch * 1000:.2f} mm (s = {s:.3f}); "
            "collision/interlock floors x s^3, fastening area x s^2, joint tolerance and ground band x s")
    for dim in (conn, coll, stab):
        dim.setdefault("metrics", {}).update({"l1_scale": scale_meta})
        dim.setdefault("notes", []).append(note)
    return {"1.1": conn, "1.2": coll, "1.3": stab}


# ---------------------------------------------------------------- reference anchors (same rule)
def anchors_v34(task):
    from ppbench.v2.evaluate import reference_design
    ref = reference_design(task)
    parts = ref.parts
    d, pitch, s = scale_for(parts)
    occs = analysis.occupancies(parts, res=pitch)
    _apply(s)
    try:
        rows = V._pairs(parts, occs, ref.joints)
    finally:
        _restore()
    vols = np.array([o.volume for o in occs])
    comps = sorted(analysis.components(len(parts), [(r["i"], r["k"]) for r in rows if r["connected"]]),
                   key=lambda c: -vols[c].sum())
    weights, corrected = V._masses(parts, occs)
    band = max(BASE_BAND_M * s, 0.01 * float(np.ptp(np.vstack([p.vertices for p in parts])[:, 2])))
    st = analysis.stability(occs, masses=list(weights), band=band)
    z_top = float(max(p.vertices[:, 2].max() for p in parts))
    tilt = float(st["critical_tilt_deg"])
    return {"spec": SPEC_VERSION, "task": task.id, "reference_tilt_deg": tilt,
            "reference_push_ratio": V._push(st, z_top), "free_standing": tilt > 0,
            "reference_n_parts": len(parts), "reference_n_components": len(comps),
            "reference_shell_corrected": len(corrected), "reference_diagonal_m": d,
            "reference_voxel_res_m": pitch}


def get_anchors(task_id, task):
    out = RESULTS / task_id / "anchors_v34l1.json"
    if out.exists():
        try:
            a = json.loads(out.read_text())
            if a.get("spec") == SPEC_VERSION:
                return a
        except (OSError, ValueError):
            pass
    try:
        a = anchors_v34(task)
    except Exception as e:
        # the v3.3 anchors (4 mm) are the fallback, e.g. an LDraw-only reference
        a = dict(V.get_anchors(task_id, task))
        a.update({"spec": SPEC_VERSION, "fallback": f"v3.3 anchors: {type(e).__name__}: {e}"})
    out.write_text(json.dumps(a, indent=1, default=float))
    return a


# ---------------------------------------------------------------- records
def upgrade(rec32, dims1, note):
    out = V.upgrade(rec32, dims1, note)
    out["spec"] = SPEC_VERSION
    out["stamp"] = f"{rec32.get('stamp')}|{SPEC_VERSION}"
    return out


def _one(args):
    task_id, name, force = args
    from ppbench.v2 import report as R
    base = RESULTS / task_id
    src, dst = base / "eval_v32" / name, base / OUT_DIR / name
    try:
        rec32 = json.loads(src.read_text())
    except (OSError, ValueError):
        return "bad"
    want = f"{rec32.get('stamp')}|{SPEC_VERSION}"
    if not force and dst.exists():
        try:
            if json.loads(dst.read_text()).get("stamp") == want:
                return "cached"
        except (OSError, ValueError):
            pass
    item = rec32.get("item")
    if not item or item.get("kind") == "reference":
        return "skip"
    path = Path(item.get("path", ""))
    if item.get("kind", "tool") in ("tool", "annotated") and not (path / "design.json").exists():
        return "gone"
    try:
        task = V._task(task_id)
        design = R.load_design(task, item)
        note = f"Level 1 recomputed under spec {SPEC_VERSION} (design-scaled instrument)"
        if item.get("kind") == "external":
            prep = V.prepare_external(design, task)
            note += " (external design: oracle scale/yaw applied first)" + (f": {'; '.join(prep)}" if prep else "")
        dims1 = level1(design, task, get_anchors(task_id, task))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(upgrade(rec32, dims1, note), indent=1, default=float))
        return "ok"
    except Exception:
        dst.parent.mkdir(parents=True, exist_ok=True)
        import traceback
        dst.with_suffix(".error.txt").write_text(traceback.format_exc())
        return "error"


# main-setting record names (scripts/make_tables.py) plus the published-50 r3 final
AGENT_SUFFIXES = ("__B__name_only+image__r1__s0", "__B__name_only+image__r1__s0__round1",
                  "__B__name_only+image__r3__s0", "__B__name_only+image__r3__s0__round1")
EXT_SUFFIXES = ("__ext__image__s0", "__ext__name_only__s0")


def _systems():
    from ppbench.v2.systems import AGENTS, EXT
    return list(AGENTS), list(EXT)


_KC = {}


def _KEYS(path):
    if path not in _KC:
        _KC[path] = [tuple(l.rstrip("\n").split("\t")) for l in open(path) if l.strip()]
    return _KC[path]


def jobs_for(ids, force, scope="main"):
    agents, ext = _systems()
    jobs = []
    for t in ids:
        d = RESULTS / t / "eval_v32"
        if not d.is_dir():
            continue
        if scope == "all":
            names = [f.name for f in sorted(d.glob("*.json"))]
        elif scope.startswith("keys:"):          # keys:<tsv of task\tkey> (2026-10-02, release export scope)
            names = [f"{k}.json" for tt, k in _KEYS(scope[5:]) if tt == t]
        else:
            names = [f"{s}{x}.json" for s in agents for x in AGENT_SUFFIXES] + [f"{s}{x}.json" for s in ext for x in EXT_SUFFIXES]
        jobs += [(t, n, force) for n in names if (d / n).exists()]
    return jobs


def build(tasks=None, workers=48, force=False, limit=None, scope="main"):
    ids = tasks or [t["task_id"] for t in json.loads(
        (__import__("ppbench.v2.task", fromlist=["CORE"]).CORE).read_text())]
    jobs = jobs_for(ids, force, scope)
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} records over {len(ids)} tasks", flush=True)
    for t in ids:
        try:
            get_anchors(t, V._task(t))
        except Exception as e:
            print(f"{t:28s} anchors failed: {e}", flush=True)
    # big designs first so the pool does not finish on a straggler
    jobs.sort(key=lambda j: 0 if j[1].startswith(("cubepart", "physx", "particulate")) else 1)
    tot = defaultdict(int)
    t0 = time.time()
    with ProcessPoolExecutor(workers, max_tasks_per_child=200) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=4), 1):
            tot[r] += 1
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}  {dict(tot)}  {time.time() - t0:.0f}s", flush=True)
    print(f"TOTAL {dict(tot)}  {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scope", default="main", help="main | all | keys:<tsv>")
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args(argv)
    ids = None if ns.tasks == "all" else ns.tasks.split(",")
    build(ids, ns.workers, ns.force, ns.limit, ns.scope)


if __name__ == "__main__":
    main()
