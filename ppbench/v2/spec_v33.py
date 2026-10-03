"""v3.3 Level 1: connectivity, collision and stability recomputed from geometry.

Written against docs/evaluation/metrics_feedback.md (2026-09-20 review). v3.2 re-partitioned the
stored v3.1 numbers; v3.3 is the first spec that changes what Level 1 measures, so it reloads every
design and voxelises it again. Levels 2-4 are carried through from the v3.2 record untouched.

    1.1 connectivity   a connection needs physical evidence: interlock, a fastened declared joint, or
                       a lattice stud. Proximity is not evidence. Scored on breaks, not on the
                       largest piece, with the whole-body term at one third instead of one half.
    1.2 collision      every interfacing pair counts, declared joints included, at a wider tolerance.
    1.3 stability      does it stand, and how hard must you push at its top edge to tip it over.

    .venv_eval/bin/python -m ppbench.v2.spec_v33 build [--tasks all] [--workers 24] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ppbench.v2 import analysis, voxel
from ppbench.v2.evaluate import _dim, _mean
from ppbench.v2.spec_v32 import CRITICAL, NAMES, STRUCTURE
from ppbench.v2.task import RESULTS, ROOT, Task

SPEC_VERSION = "v3.3"

# ---------------------------------------------------------------- constants (none is task-specific)
INTERLOCK_MIN_M3 = 1e-7          # 0.1 cm^3 of core-core overlap: material of one part inside the other
FASTEN_MIN_AREA_M2 = 1e-4        # 1 cm^2 of shared interface for a declared joint to be fastenable
JOINT_ORIGIN_TOL_M = analysis.JOINT_CONNECT_TOL_M        # 2 cm, the protocol's own rule
LATTICE_STACK_TOL_M = 0.0015     # a brick sits on the brick below: vertical extents adjacent within 1.5 mm
LATTICE_KINDS = {"ldraw"}        # snap-fit part systems whose lattice is itself the fastener
COLLIDE_ABS_M3 = analysis.COLLIDE_ABS_M3                 # 1 cm^3
COLLIDE_FRAC_FREE = analysis.COLLIDE_FRAC                # 0.10 for a pair with no declared joint
COLLIDE_FRAC_JOINT = 0.25        # a fastened or articulated interface may legitimately share material
PEN_FRAC_FAIL = 0.10
ANCHORED_VOLUME_FAIL = 0.95
BREAK_SCORE_FAIL = 0.90
SHELL_FRAC = 0.05                # filled_frac below this is an open mesh whose fill failed
TILT_FULL_DEG = 30.0
TILT_TARGET_MIN_DEG = 5.0
TILT_TARGET_MAX_DEG = 30.0
PUSH_FULL = 0.20                 # force target when the task has no reference object
PUSH_TARGET_MIN = 0.05
PUSH_TARGET_MAX = 0.35


# ---------------------------------------------------------------- geometry pass
def _masses(parts, occs):
    """Part masses under uniform density. An open mesh (the fill failed) would weigh almost nothing
    and drag the centre of mass with it, so its volume is the convex hull of its own surface samples,
    capped at its bounding box."""
    w, corrected = [], []
    for p, o in zip(parts, occs):
        v = o.volume
        if o.filled_frac < SHELL_FRAC:
            ext = p.vertices.max(0) - p.vertices.min(0)
            bbox = float(np.prod(np.maximum(ext, o.res)))
            hull = 0.0
            try:
                from scipy.spatial import ConvexHull
                pts = o.points if len(o.points) <= 20000 else o.points[::max(1, len(o.points) // 20000)]
                hull = float(ConvexHull(pts).volume)
            except Exception:
                hull = 0.0
            better = min(max(hull, v), bbox)
            if better > v * 1.05:
                corrected.append({"part": p.id, "voxel_m3": v, "hull_m3": better})
                v = better
        w.append(max(v, 1e-12))
    return np.array(w), corrected


def _pairs(parts, occs, joints):
    """Every interfacing pair with the evidence that decides whether it is a connection."""
    lattice = [bool((getattr(p, "source", None) or {}).get("kind") in LATTICE_KINDS) for p in parts]
    zlo = [float(p.vertices[:, 2].min()) for p in parts]
    zhi = [float(p.vertices[:, 2].max()) for p in parts]
    idx = {p.id: i for i, p in enumerate(parts)}
    jmap = defaultdict(list)
    for j in joints:
        if j.parent in idx and j.child in idx and j.parent != j.child:
            jmap[frozenset((idx[j.parent], idx[j.child]))].append(j)
    lo = np.array([o.lo for o in occs])
    hi = np.array([o.hi for o in occs])
    rows = []
    for i in range(len(parts)):
        near = np.where(np.all(lo[i] - 1 <= hi[i + 1:], 1) & np.all(lo[i + 1:] <= hi[i] + 1, 1))[0] + i + 1
        for k in near:
            k = int(k)
            a, b = occs[i], occs[k]
            if not voxel.touching(a, b):
                continue
            deep = voxel.overlap_cells(a, b, deep=True) * a.res ** 3
            cells = voxel.contact_cells(a, b)
            area = cells * a.res ** 2
            small = max(min(a.volume, b.volume), 1e-12)
            js = jmap.get(frozenset((i, k)), [])
            origin_ok = None
            if js:
                origin_ok = True
                for j in js:
                    if j.origin is None:
                        continue
                    d = max(analysis.surface_distance(j.origin, a), analysis.surface_distance(j.origin, b))
                    origin_ok = bool(d <= JOINT_ORIGIN_TOL_M)
                    if origin_ok:
                        break
            interlock = deep >= INTERLOCK_MIN_M3
            fastened = bool(js) and origin_ok is not False and (interlock or area >= FASTEN_MIN_AREA_M2)
            stud = False
            if lattice[i] and lattice[k]:
                stud = (abs(zhi[i] - zlo[k]) <= LATTICE_STACK_TOL_M
                        or abs(zhi[k] - zlo[i]) <= LATTICE_STACK_TOL_M)
            rows.append({"i": i, "k": k, "a": parts[i].id, "b": parts[k].id, "deep_m3": deep,
                         "contact_area_m2": area, "frac_of_smaller": deep / small,
                         "declared_joint": bool(js), "joint_origin_ok": origin_ok,
                         "interlock": interlock, "fastened_joint": fastened, "lattice_stud": stud,
                         "connected": bool(interlock or fastened or stud)})
    return rows


# ---------------------------------------------------------------- 1.1 connectivity
def joint_edges(parts, occs, joints):
    """Pairs a declared joint joins at a place where both parts are: the protocol's own rule
    (origin within 2 cm of each surface), including pairs whose surfaces do not touch."""
    idx = {p.id: i for i, p in enumerate(parts)}
    out = set()
    for j in joints:
        if j.parent not in idx or j.child not in idx or j.parent == j.child:
            continue
        i, k = idx[j.parent], idx[j.child]
        if j.origin is None:
            continue
        if max(analysis.surface_distance(j.origin, occs[i]),
               analysis.surface_distance(j.origin, occs[k])) <= JOINT_ORIGIN_TOL_M:
            out.add((min(i, k), max(i, k)))
    return out


def connectivity(parts, occs, rows, joints):
    """Does it hold together, and does the design say how.

    The first two terms use the rule the protocol promised the agents (surfaces touching, or a joint
    joining the parts where both parts are), so no design is marked down for a rule it was never
    given. The third asks the fastening question as a *credit* rather than a penalty: a body of n
    parts needs n-1 connections to span, and this is the share of that minimum the design actually
    backs with physical evidence -- an interlock, a declared joint on a real interface, or a lattice
    stud. Declaring the joints raises it; leaving parts to rest against each other does not."""
    n = len(parts)
    vols = np.array([o.volume for o in occs])
    if n == 1:
        return _dim("computed", "not_applicable", None,
                    {"n_parts": 1, "n_interfacing_pairs": len(rows)},
                    ["a single-part design has no connection to evaluate: not scored, not zeroed"]), [0]
    jedge = joint_edges(parts, occs, joints)
    contact = sorted({(min(r["i"], r["k"]), max(r["i"], r["k"])) for r in rows} | jedge)
    comps = sorted(analysis.components(n, contact), key=lambda c: -vols[c].sum())
    anchored = float(vols[comps[0]].sum() / max(vols.sum(), 1e-12))
    break_score = float(max(0.0, 1.0 - (len(comps) - 1) / max(n - 1, 1)))
    fastened = [r for r in rows if r["connected"]]
    spec = float(min(1.0, len(fastened) / max(n - 1, 1)))
    from scipy import ndimage
    whole = []
    for o in occs:
        lab, k = ndimage.label(o.grid, structure=np.ones((3, 3, 3)))
        if k <= 1:
            whole.append(1.0)
            continue
        sizes = np.bincount(lab.ravel())[1:]
        whole.append(1.0 if (sizes >= 0.05 * sizes.sum()).sum() == 1 else 0.0)
    whole_rate = float(np.mean(whole))
    score = float(np.mean([anchored, break_score, spec]))
    # the same three terms with the fastening rule applied to the graph as well: what 1.1 would be if
    # a resting contact held nothing. Reported, not scored -- it grades the agents against a protocol
    # rule they were never given (see docs/evaluation/metrics_feedback.md).
    fcomps = sorted(analysis.components(n, [(r["i"], r["k"]) for r in fastened]), key=lambda c: -vols[c].sum())
    strict = float(np.mean([float(vols[fcomps[0]].sum() / max(vols.sum(), 1e-12)),
                            max(0.0, 1.0 - (len(fcomps) - 1) / max(n - 1, 1)), whole_rate]))
    shells = [p.id for p, o in zip(parts, occs) if o.filled_frac < SHELL_FRAC]
    resting = [r for r in rows if not r["connected"]]
    st = ("fail" if (anchored < ANCHORED_VOLUME_FAIL or break_score < BREAK_SCORE_FAIL)
          else ("degraded" if (shells or resting) else "pass"))
    ev = {"interlock": sum(1 for r in rows if r["interlock"]),
          "fastened_joint": sum(1 for r in rows if r["fastened_joint"] and not r["interlock"]),
          "lattice_stud": sum(1 for r in rows if r["lattice_stud"] and not (r["interlock"] or r["fastened_joint"]))}
    return _dim("computed", st, score, {
        "n_parts": n, "n_components": len(comps), "n_breaks": len(comps) - 1,
        "anchored_volume_frac": anchored, "break_score": break_score,
        "fastening_spec_rate": spec, "n_fastened_connections": len(fastened),
        "connections_needed_to_span": n - 1,
        "whole_body_rate": whole_rate, "fastening_strict_score": strict,
        "n_components_fastened_only": len(fcomps),
        "n_interfacing_pairs": len(rows), "connection_evidence": ev,
        "n_resting_contacts": len(resting),
        "resting_contacts": [[r["a"], r["b"]] for r in resting][:20],
        "n_declared_joint_pairs": sum(1 for r in rows if r["declared_joint"]),
        "n_declared_joints_rejected": sum(1 for r in rows if r["declared_joint"] and not r["fastened_joint"]),
        "unattached_parts": [parts[i].id for c in comps[1:] for i in c][:30],
        "shell_only_parts": shells[:20]}, [
        "score = mean(anchored_volume_frac, break_score, fastening_spec_rate), one third each",
        "break_score = 1 - (n_components - 1) / (n_parts - 1); the graph is the protocol's rule: "
        "surfaces touching, or a declared joint whose origin is within 2 cm of both surfaces",
        "fastening_spec_rate = min(1, n_fastened_connections / (n_parts - 1)): of the n-1 connections a "
        "body of n parts needs to span, the share the design backs with an interlock, a declared joint "
        "on a real interface (>= 1 cm2) or a lattice stud. Resting contacts do not count toward it",
        "whole_body_rate and fastening_strict_score are reported, not scored"]
    ), comps[0]


# ---------------------------------------------------------------- 1.2 collision
def collision(parts, occs, rows):
    vols = np.array([o.volume for o in occs])
    if not rows:
        return _dim("computed", "not_applicable", None,
                    {"n_parts": len(parts), "n_interfacing_pairs": 0},
                    ["no pair of parts interfaces at all: there is no interface to check for collision"])
    coll = [r for r in rows if r["deep_m3"] > COLLIDE_ABS_M3 and r["frac_of_smaller"] >
            (COLLIDE_FRAC_JOINT if r["declared_joint"] else COLLIDE_FRAC_FREE)]
    rate = 1 - len(coll) / len(rows)
    pen_total = sum(r["deep_m3"] for r in rows)
    pen_frac = pen_total / max(vols.sum(), 1e-12)
    st = "fail" if pen_frac > PEN_FRAC_FAIL else ("degraded" if coll else "pass")
    return _dim("computed", st, float(rate), {
        "collision_free_pair_rate": float(rate), "n_interfacing_pairs": len(rows),
        "n_colliding_pairs": len(coll), "n_declared_joint_pairs": sum(1 for r in rows if r["declared_joint"]),
        "n_colliding_joint_pairs": sum(1 for r in coll if r["declared_joint"]),
        "pen_volume_frac": float(pen_frac), "pen_volume_cm3": float(pen_total * 1e6),
        "worst_pair_frac_of_smaller": float(max(r["frac_of_smaller"] for r in rows)),
        "graded_penetration_score": float(max(0.0, 1 - pen_frac / PEN_FRAC_FAIL)),
        "voxel_res_m": occs[0].res if occs else None,
        "colliding_pairs": [[r["a"], r["b"], round(r["frac_of_smaller"], 3)] for r in coll][:20]}, [
        "score = 1 - n_colliding / n_interfacing_pairs over EVERY interfacing pair, declared joints included",
        f"a pair collides when the eroded-core overlap exceeds 1 cm3 and {COLLIDE_FRAC_FREE:.0%} of the "
        f"smaller part ({COLLIDE_FRAC_JOINT:.0%} when the design declares a joint there)"])


# ---------------------------------------------------------------- 1.3 stability
def _push(st, z_top):
    """Horizontal force at the object's highest point, as a multiple of its own weight, that tips it."""
    h = max(float(z_top) - float(st["ground_z"]), 1e-6)
    return float(st["support_margin_m"]) / h


def stability(parts, occs, main, weights, corrected, anchors, n_parts):
    """The assembly as placed, on the ground plane the design itself rests on.

    Measured on every part, not only on the largest connected group: what is put on the floor is the
    whole arrangement, and a group picked by the connectivity rule would make 1.3 a second reading of
    1.1 -- and, when that group floats in mid-air, would take its own lowest rim for the ground."""
    band = max(0.005, 0.01 * float(np.ptp(np.vstack([p.vertices for p in parts])[:, 2])))
    st = analysis.stability(occs, masses=list(weights), band=band)
    tilt = float(st["critical_tilt_deg"])
    z_top = float(max(p.vertices[:, 2].max() for p in parts))
    f = _push(st, z_top)
    notes = ["statics only; no settling simulation",
             "score = mean(stand_score, push_score)",
             "stand_score = clip(critical_tilt, 0, target) / target",
             "push_score = clip(f / f_target, 0, 1), f = the horizontal force at the top of the object, "
             "in multiples of its own weight, that tips it about its weakest support edge"]
    if corrected:
        notes.append(f"{len(corrected)} open-mesh part(s) re-weighted by their convex hull")
    if len(main) < n_parts:
        notes.append(f"the design is in pieces ({len(main)} of {n_parts} parts in the largest connected group): "
                     "unattached parts still carry their mass, and support only where they reach the ground")
    met = {**{k: v for k, v in st.items() if k != "com"}, "com": st["com"], "ground_band_m": band,
           "z_top_m": z_top, "push_force_ratio": f, "shell_corrected_parts": corrected[:10],
           "n_parts_in_main_group": len(main), "n_parts": n_parts}
    if anchors.get("free_standing") is False:
        met["reference_tilt_deg"] = anchors.get("reference_tilt_deg")
        return _dim("computed", "not_applicable", None, met,
                    notes + ["the reference object is not free-standing on a ground plane: not measurable here"])
    rt, rf = anchors.get("reference_tilt_deg"), anchors.get("reference_push_ratio")
    target = float(min(max(rt, TILT_TARGET_MIN_DEG), TILT_TARGET_MAX_DEG)) if isinstance(rt, (int, float)) and rt > 0 else TILT_FULL_DEG
    ftar = float(min(max(rf, PUSH_TARGET_MIN), PUSH_TARGET_MAX)) if isinstance(rf, (int, float)) and rf > 0 else PUSH_FULL
    stand = float(min(max(tilt, 0.0), target) / target)
    push = float(min(max(f, 0.0), ftar) / ftar)
    score = float((stand + push) / 2)
    few = st["n_support_vertices"] <= 2
    thin = isinstance(rt, (int, float)) and 0 < rt < TILT_TARGET_MIN_DEG
    met.update({"stand_score": stand, "push_score": push, "target_tilt_deg": target,
                "target_push_ratio": ftar, "reference_tilt_deg": rt, "reference_push_ratio": rf,
                "pose_sensitive": bool(few)})
    return _dim("computed", "fail" if tilt <= 0 else ("degraded" if (few or thin or len(main) < n_parts) else "pass"),
                score, met, notes)


# ---------------------------------------------------------------- one design
OVERSIZE_FACTOR = 5.0             # a single part this many times the reference's diagonal is a scale failure
OVERSIZE_ABS_M = 10.0             # ... or this many metres when the task has no reference object


def _oversize(parts, anchors):
    """A part several times the size of the whole reference object is a design that got the units
    wrong. v3.1 scored that as the structural failure it is instead of dropping the design; so does
    this, and it also keeps a 40 m part out of the voxeliser."""
    spans = [float(np.linalg.norm(p.vertices.max(0) - p.vertices.min(0))) for p in parts]
    rd = (anchors or {}).get("reference_diagonal_m")
    limit = OVERSIZE_FACTOR * rd if isinstance(rd, (int, float)) and rd > 0 else OVERSIZE_ABS_M
    big = [p.id for p, sp in zip(parts, spans) if sp > limit]
    if not big:
        return None
    met = {"n_parts": len(parts), "largest_part_span_m": max(spans), "oversize_limit_m": limit,
           "oversized_parts": big[:20], "n_oversized_parts": len(big)}
    note = (f"{len(big)} part(s) span more than {limit:.1f} m, {OVERSIZE_FACTOR:.0f}x the reference "
            "object: the design's scale is wrong")
    return {k: _dim("computed", "fail", 0.0, dict(met), [note]) for k in ("1.1", "1.2", "1.3")}


def _ldraw_reference(task):
    """The 5 bc_* tasks have an LDraw model instead of a reference GLB (task.reference() raises there). Its parts,
    in metres and Z-up, in the shape evaluate.prepare expects: enough for the oracle scale and yaw."""
    from ppbench.v2 import external
    cad = [c for c in (task.raw.get("reference") or {}).get("cad_files") or [] if c.get("format") == "LDR"]
    if not cad:
        return None
    d = external.ldr_design(ROOT / cad[0]["path"], "reference")
    if not d.parts:
        return None
    V = np.vstack([p.vertices for p in d.parts])
    return {"bounds": [V.min(0).tolist(), V.max(0).tolist()],
            "parts": [{"vertices": p.vertices, "faces": p.faces} for p in d.parts]}


def prepare_external(design, task):
    """The oracle scale and yaw evaluate.prepare gives an external design (unitless output scaled to the reference
    diagonal, a front-less output turned to its best-matching yaw). Until 2026-09-21 v3.3 reloaded external designs
    and measured Level 1 on the generator's raw units, so the 4 mm contact, 1 cm3 collision and 5x-oversize rules
    were applied at the wrong scale. Tool-run designs are metric with a declared front and never pass through here."""
    from ppbench.v2.evaluate import prepare
    try:
        ref = task.reference()
    except Exception:
        ref = _ldraw_reference(task)
    return prepare(design, ref) if ref else []


def level1(design, task, anchors):
    parts, joints = design.parts, design.joints
    if not parts:
        z = _dim("computed", "fail", 0.0, {"n_parts": 0}, ["empty design"])
        return {"1.1": z, "1.2": dict(z), "1.3": dict(z)}
    over = _oversize(parts, anchors)
    if over:
        return over
    try:
        occs = analysis.occupancies(parts)
    except voxel.VoxelTooLarge as e:
        spans = [float(np.linalg.norm(p.vertices.max(0) - p.vertices.min(0))) for p in parts]
        met = {"n_parts": len(parts), "largest_part_span_m": max(spans),
               "oversized_parts": [p.id for p, sp in zip(parts, spans) if sp > 10.0][:20]}
        return {k: _dim("computed", "fail", 0.0, dict(met), [f"a part is too large to voxelise: {e}"])
                for k in ("1.1", "1.2", "1.3")}
    rows = _pairs(parts, occs, joints)
    conn, main = connectivity(parts, occs, rows, joints)
    coll = collision(parts, occs, rows)
    weights, corrected = _masses(parts, occs)
    stab = stability(parts, occs, main, weights, corrected, anchors or {}, len(parts))
    return {"1.1": conn, "1.2": coll, "1.3": stab}


# ---------------------------------------------------------------- reference anchors
def anchors_v33(task: Task):
    from ppbench.v2.evaluate import reference_design
    d = reference_design(task)
    parts = d.parts
    occs = analysis.occupancies(parts)
    rows = _pairs(parts, occs, d.joints)
    vols = np.array([o.volume for o in occs])
    comps = sorted(analysis.components(len(parts), [(r["i"], r["k"]) for r in rows if r["connected"]]),
                   key=lambda c: -vols[c].sum())
    # the reference's own stand and push are measured on its whole body: a dataset asset is one object,
    # and its own part graph must not decide the target the designs are held to.
    weights, corrected = _masses(parts, occs)
    band = max(0.005, 0.01 * float(np.ptp(np.vstack([p.vertices for p in parts])[:, 2])))
    st = analysis.stability(occs, masses=list(weights), band=band)
    z_top = float(max(p.vertices[:, 2].max() for p in parts))
    tilt = float(st["critical_tilt_deg"])
    V = np.vstack([p.vertices for p in parts])
    return {"spec": SPEC_VERSION, "task": task.id, "reference_tilt_deg": tilt,
            "reference_push_ratio": _push(st, z_top), "free_standing": tilt > 0,
            "reference_n_parts": len(parts), "reference_n_components": len(comps),
            "reference_shell_corrected": len(corrected),
            "reference_diagonal_m": float(np.linalg.norm(V.max(0) - V.min(0)))}


def get_anchors(task_id, task):
    out = RESULTS / task_id / "anchors_v33.json"
    if out.exists():
        try:
            a = json.loads(out.read_text())
            if a.get("spec") == SPEC_VERSION:
                return a
        except (OSError, ValueError):
            pass
    try:
        a = anchors_v33(task)
    except Exception as e:
        a = {"spec": SPEC_VERSION, "task": task_id, "error": f"{type(e).__name__}: {e}"}
    out.write_text(json.dumps(a, indent=1, default=float))
    return a


# ---------------------------------------------------------------- record
def upgrade(rec32, dims1, note):
    D = dict(rec32.get("dims") or {})
    for k, v in dims1.items():
        D[k] = v
    crit = [k for k in CRITICAL if (D.get(k) or {}).get("status") == "fail"]
    scored = [v["score"] for v in D.values() if v.get("score") is not None]
    out = {k: rec32[k] for k in ("system", "tier", "meta", "item") if k in rec32}
    out["notes"] = (rec32.get("notes") or []) + [note]
    out.update({"spec": SPEC_VERSION, "derived_from": rec32.get("spec"), "dims": D,
                "headline": {"critical_fail": crit,
                             "buildable": not any(k in crit for k in STRUCTURE),
                             "overall": (sum(scored) / len(scored)) if scored else None,
                             "n_scored_dims": len(scored), "gate_pending": True},
                "stamp": f"{rec32.get('stamp')}|{SPEC_VERSION}"})
    return out


def _one(args):
    task_id, name, force = args
    from ppbench.v2 import report as R
    base = RESULTS / task_id
    src, dst = base / "eval_v32" / name, base / "eval_v33" / name
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
        return "gone"      # a v3.2 record whose round checkpoint is no longer on disk
    try:
        task = _task(task_id)
        design = R.load_design(task, item)
        note = "Level 1 recomputed under spec v3.3"
        if item.get("kind") == "external":
            prep = prepare_external(design, task)
            note += " (external design: oracle scale/yaw applied first, fix of 2026-09-21)" + (f": {'; '.join(prep)}" if prep else "")
        dims1 = level1(design, task, get_anchors(task_id, task))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(upgrade(rec32, dims1, note),
                                  indent=1, default=float))
        return "ok"
    except Exception as e:
        dst.parent.mkdir(parents=True, exist_ok=True)
        import traceback
        (dst.with_suffix(".error.txt")).write_text(traceback.format_exc())
        return "error"


_TASKS = {}


def _task(task_id):
    if task_id not in _TASKS:
        _TASKS[task_id] = Task(task_id, snapshot=RESULTS / task_id / "task_snapshot_core.json")
    return _TASKS[task_id]


def build(tasks=None, workers=24, force=False, limit=None):
    ids = tasks or [t["task_id"] for t in json.loads(
        (__import__("ppbench.v2.task", fromlist=["CORE"]).CORE).read_text())]
    jobs = []
    for t in ids:
        d = RESULTS / t / "eval_v32"
        if d.is_dir():
            jobs += [(t, f.name, force) for f in sorted(d.glob("*.json"))]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} records over {len(ids)} tasks", flush=True)
    # anchors first, one process, so the pool does not race on the cache file
    for t in ids:
        try:
            get_anchors(t, _task(t))
        except Exception as e:
            print(f"{t:28s} anchors failed: {e}", flush=True)
    tot = defaultdict(int)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=8), 1):
            tot[r] += 1
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}  {dict(tot)}  {time.time() - t0:.0f}s", flush=True)
    print(f"TOTAL {dict(tot)}  {time.time() - t0:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "anchors"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args(argv)
    ids = None if ns.tasks == "all" else ns.tasks.split(",")
    if ns.cmd == "anchors":
        for t in (ids or [x["task_id"] for x in json.loads(
                (__import__("ppbench.v2.task", fromlist=["CORE"]).CORE).read_text())]):
            a = get_anchors(t, _task(t))
            print(f"{t:28s} tilt {a.get('reference_tilt_deg')} push {a.get('reference_push_ratio')} "
                  f"free {a.get('free_standing')} shell_fixed {a.get('reference_shell_corrected')}")
        return
    build(ids, ns.workers, ns.force, ns.limit)


if __name__ == "__main__":
    main()
