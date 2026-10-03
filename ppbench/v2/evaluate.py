"""The v2 evaluator: every dimension of docs/evaluation, graded, with depth labels.

    ev = evaluate(design, task, anchors)          # one record, never an average

Every dimension returns {depth, status, score, metrics, notes}. `score` is in
[0, 1] where a graded summary is meaningful and None where the taxonomy says
not to merge (4.3 keeps presence, placement and connection apart). Raw numbers
always travel next to the score, and every constant that turns a raw number
into a score is a named module constant below.

What the design is compared against, per dimension:
  claims       required parts P, attributes A, subsystems F, kinematics K (cited, in the task)
  reference    the task's reference object: geometry, roles, Artiverse materials, annotated joints
  population   117 other Artiverse swivel chairs: extents, role counts, material by role (population.json)

Status follows v1: pass, degraded (nothing critical fired but part of the
measurement used a weaker instrument), fail, skipped (the producer does not
declare what this dimension scores), not_built.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

import numpy as np

from ppbench.v2 import analysis, materials, voxel
from ppbench.v2.design import Design, Joint, Part, axis_angle
from ppbench.v2.lexicon import Lexicon
from ppbench.v2.task import RESULTS, Task

# ---------------------------------------------------------------- constants
CONTACT_JOINT_TOL_M = analysis.JOINT_CONNECT_TOL_M   # a joint origin within this of both parts sits on their boundary
PEN_FRAC_FAIL = 0.10              # interpenetrating volume / total volume that fails 1.1 (reference chair: 0.039)
LARGEST_COMPONENT_FAIL = 0.95     # volume fraction of the largest connected group below which 1.1 fails
CONNECTED_PARTS_FAIL = 0.90       # fraction of parts in the largest connected group below which 1.1 fails
SEAT_DECAY_M = 0.15               # seat height this far outside the reference range scores 0
OCCUPANT_KG = 75.0                # added load for the loaded tilt in 4.1 (an assumption, stated)
FSCORE_TAU_M = (0.02, 0.05)
N_SHAPE_POINTS = 20000
SWEEP_STEPS_ROT = 12
SWEEP_STEPS_LIN = 8
SWEEP_BLOCK_ABS_M3 = 1e-6
SWEEP_BLOCK_FRAC = 0.05
ORIGIN_SCALE_FRAC = 0.25          # origin error normalised by this fraction of the reference diagonal
AXIS_TOL_DEG = 45.0
INSERT_EXTRA_M = 0.05
TYPE_COMPAT = {("prismatic", "cylindrical"): 0.75, ("cylindrical", "prismatic"): 0.75,
               ("revolute", "continuous"): 0.75, ("continuous", "revolute"): 0.75,
               ("cylindrical", "revolute"): 0.5, ("cylindrical", "continuous"): 0.5,
               ("revolute", "cylindrical"): 0.5, ("continuous", "cylindrical"): 0.5}


def _dim(depth, status, score, metrics, notes=None):
    return {"depth": depth, "status": status, "score": None if score is None else float(np.clip(score, 0, 1)),
            "metrics": metrics, "notes": notes or []}


def _mean(xs):
    """Mean of the terms that could be measured. A term the task has no data for (no reference object, no
    annotated reference joints, no population) is None and is dropped; it is never counted as a zero, which
    would score the object down for a gap in our data rather than in the design."""
    v = [float(x) for x in xs if x is not None]
    return float(np.mean(v)) if v else None


def _declares(design, key):
    return bool((design.meta.get("declares") or {}).get(key, False))


# ================================================================ preparation

def prepare(design: Design, ref: dict):
    """Unitless producers are scaled to the reference diagonal (flagged), and
    every design is grounded (lowest point to z = 0) and centred in XY on the
    reference's XY centre for shape comparison only (a copy)."""
    notes = []
    unitless = design.meta.get("scale_mode") == "unitless"
    if unitless and design.parts:
        V = np.vstack([p.vertices for p in design.parts])
        diag = np.linalg.norm(V.max(0) - V.min(0))
        rdiag = np.linalg.norm(np.subtract(ref["bounds"][1], ref["bounds"][0]))
        s = rdiag / max(diag, 1e-9)
        c = np.r_[(V.max(0)[:2] + V.min(0)[:2]) / 2, V.min(0)[2]]
        for p in design.parts:
            p.vertices = (p.vertices - c) * s
        for j in design.joints:
            if j.origin is not None:
                j.origin = ((np.asarray(j.origin) - c) * s).tolist()
            if j.type == "prismatic" and j.limits is not None:   # linear limits are lengths and scale with the parts
                j.limits = [float(x) * s for x in j.limits]
        design.meta["oracle_scale"] = float(s)
        notes.append(f"unitless output scaled by {s:.3f} to the reference diagonal (oracle scale); size is not scored")
    if design.parts and (unitless or design.meta.get("front_declared") is False):
        # no declared front: pick the yaw (0/90/180/270) that best matches the reference, and say so
        rparts = [Part("r", p["vertices"], p["faces"]) for p in ref["parts"]]
        rpts = _center_ground(sample_points(rparts, 6000, seed=1), ref["bounds"])
        best = None
        for yaw in (0, 90, 180, 270):
            R = axis_angle([0, 0, 1], math.radians(yaw))
            dp = _center_ground(sample_points(design.parts, 6000) @ R.T, ref["bounds"])
            f = fscore(dp, rpts)["fscore@0.05"]
            if best is None or f > best[1]:
                best = (yaw, f)
        if best[0]:
            R = axis_angle([0, 0, 1], math.radians(best[0]))
            for p in design.parts:
                p.vertices = p.vertices @ R.T
            for j in design.joints:   # joints turn with the parts (no scored design so far had joints and a non-zero yaw)
                if j.origin is not None:
                    j.origin = (np.asarray(j.origin, float) @ R.T).tolist()
                if j.axis is not None:
                    j.axis = (np.asarray(j.axis, float) @ R.T).tolist()
        design.meta["oracle_yaw_deg"] = best[0]
        notes.append(f"no declared front: rotated {best[0]} deg about Z to best match the reference (oracle yaw)")
    return notes


# ================================================================ helpers

def rigid_groups(parts, rows, joints):
    """Connected groups of the contact graph with moving-joint pairs cut."""
    idx = {p.id: i for i, p in enumerate(parts)}
    moving = {frozenset((j.parent, j.child)) for j in joints if j.type != "fixed"}
    edges = [(r["i"], r["k"]) for r in rows if frozenset((r["a"], r["b"])) not in moving]
    for j in joints:
        if j.type == "fixed" and j.parent in idx and j.child in idx:
            edges.append((idx[j.parent], idx[j.child]))
    comps = analysis.components(len(parts), edges)
    g = {}
    for c in comps:
        for i in c:
            g[parts[i].id] = [parts[k].id for k in c]
    return g


def moving_set(parts, rows, joints, jt: Joint):
    """Parts that move with the child of `jt`: reachable from the child without
    passing through the parent. `welded` when the parent is reachable too."""
    ids = [p.id for p in parts]
    adj = defaultdict(set)
    for r in rows:
        if {r["a"], r["b"]} == {jt.parent, jt.child}:
            continue
        adj[r["a"]].add(r["b"])
        adj[r["b"]].add(r["a"])
    for j in joints:
        if j.id == jt.id or j.parent not in ids or j.child not in ids:
            continue
        adj[j.parent].add(j.child)
        adj[j.child].add(j.parent)
    seen, stack, welded = {jt.child}, [jt.child], False
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v == jt.parent:
                welded = True
                continue
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen, welded


def sample_points(parts, n=N_SHAPE_POINTS, seed=0):
    import trimesh
    rng = np.random.default_rng(seed)
    areas = np.array([trimesh.Trimesh(p.vertices, p.faces, process=False).area for p in parts])
    if areas.sum() <= 0:
        return np.zeros((0, 3))
    counts = rng.multinomial(n, areas / areas.sum())
    out = []
    for p, k in zip(parts, counts):
        if k:
            pts, _ = trimesh.sample.sample_surface(trimesh.Trimesh(p.vertices, p.faces, process=False), int(k), seed=int(rng.integers(1 << 30)))
            out.append(pts)
    return np.vstack(out)


def fscore(a, b, taus=FSCORE_TAU_M):
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return {f"fscore@{t}": 0.0 for t in taus} | {"chamfer_m": float("inf")}
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    out = {"chamfer_m": float((da.mean() + db.mean()) / 2)}
    for t in taus:
        p, r = (da < t).mean(), (db < t).mean()
        out[f"fscore@{t}"] = float(2 * p * r / (p + r)) if p + r > 0 else 0.0
    return out


def _center_ground(pts, bounds_ref):
    lo, hi = pts.min(0), pts.max(0)
    rc = (np.asarray(bounds_ref[0])[:2] + np.asarray(bounds_ref[1])[:2]) / 2
    return pts - np.r_[(lo[:2] + hi[:2]) / 2 - rc, lo[2]]


def extents(parts):
    V = np.vstack([p.vertices for p in parts])
    e = V.max(0) - V.min(0)
    return {"width": float(max(e[0], e[1])), "depth": float(min(e[0], e[1])), "height": float(e[2]),
            "x": float(e[0]), "y": float(e[1])}


def role_centres(parts, canon, bounds):
    lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    ext = np.maximum(hi - lo, 1e-6)
    out = defaultdict(list)
    for p in parts:
        c = canon.get(p.id)
        if c:
            out[c].append(((p.vertices.min(0) + p.vertices.max(0)) / 2 - lo) / ext)
    return {k: np.mean(v, 0) for k, v in out.items()}


def symmetry_iou(occs, res=0.01):
    """Best mirror IoU over vertical planes through the XY centre of mass, on a 1 cm grid."""
    pts = []
    for o in occs:
        idx = np.argwhere(o.grid)
        if len(idx):
            pts.append((idx + o.lo + 0.5) * o.res)
    if not pts:
        return 0.0, 0.0
    P = np.vstack(pts)
    c = P.mean(0)
    Q = P - c
    base = set(map(tuple, np.floor(Q / res).astype(np.int64)))
    best, best_ang = 0.0, 0.0
    for ang in range(0, 180, 5):
        n = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang)), 0.0])
        M = Q - 2 * np.outer(Q @ n, n)
        mir = set(map(tuple, np.floor(M / res).astype(np.int64)))
        iou = len(base & mir) / max(len(base | mir), 1)
        if iou > best:
            best, best_ang = iou, ang
    return float(best), float(best_ang)


def roundness(part, n=4000):
    """Surface of revolution test: 1 - CV/0.2 of the outer radius over 36 angular bins.

    Tried about each principal axis of dense surface samples, best kept: a caster
    wheel is nearly as wide as it is tall, so "the thinnest axis" is not reliably its axle."""
    pts = sample_points([part], n)
    if len(pts) < 50:
        return 0.0, None
    P = pts - (pts.min(0) + pts.max(0)) / 2
    _, V = np.linalg.eigh(np.cov(P.T))
    best = (0.0, V[:, 0])
    for k in range(3):
        axis, u, v = V[:, k], V[:, (k + 1) % 3], V[:, (k + 2) % 3]
        x, y = P @ u, P @ v
        ang = np.arctan2(y, x)
        r = np.hypot(x, y)
        bins = np.floor((ang + np.pi) / (2 * np.pi) * 36).astype(int).clip(0, 35)
        mx = np.array([r[bins == b].max() if np.any(bins == b) else 0.0 for b in range(36)])
        if mx.mean() <= 0:
            continue
        s = float(max(0.0, 1 - (mx.std() / mx.mean()) / 0.2))
        if s > best[0]:
            best = (s, axis)
    return best


def ground_clusters(parts, cell=0.03, band=0.01):
    V = np.vstack([p.vertices for p in parts])
    z0 = V[:, 2].min()
    g = V[V[:, 2] <= z0 + band][:, :2]
    if not len(g):
        return 0
    cells = set(map(tuple, np.floor(g / cell).astype(int)))
    comps, seen = 0, set()
    for c in cells:
        if c in seen:
            continue
        comps += 1
        stack = [c]
        seen.add(c)
        while stack:
            a = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    b = (a[0] + dx, a[1] + dy)
                    if b in cells and b not in seen:
                        seen.add(b)
                        stack.append(b)
    return comps


def axis_requirement(text):
    t = (text or "").lower()
    if "vertical" in t:
        return "vertical"
    if "horizontal" in t or "axle" in t:
        return "horizontal"
    return None


def axis_score(axis, req):
    if axis is None:
        return 0.0
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    if req == "vertical":
        err = math.degrees(math.acos(min(1.0, abs(a[2]))))
    elif req == "horizontal":
        err = math.degrees(math.asin(min(1.0, abs(a[2]))))
    else:
        return 1.0
    return max(0.0, 1 - err / AXIS_TOL_DEG)


def type_score(declared, required):
    if declared == required:
        return 1.0
    return TYPE_COMPAT.get((declared, required), 0.0)


def range_score(jt: Joint, claim_range: str | None, joint_type: str):
    t = (claim_range or "").lower()
    full = "continuous" in t or "360" in t or "freely" in t
    if jt.type == "continuous":
        return 1.0 if full or joint_type in ("revolute", "continuous") else 0.5
    if jt.limits is None:
        return 0.5 if not full else 0.25
    span = jt.limits[1] - jt.limits[0]
    if full:
        return float(min(1.0, span / (2 * math.pi)))
    return 1.0 if span > 0 else 0.0


# ================================================================ evaluate

def evaluate(design: Design, task: Task, anchors: dict | None = None, population: dict | None = None,
             res=voxel.RES) -> dict:
    lex = Lexicon(task.id, task)
    try:
        ref = task.reference()
    except Exception:      # the 5 LDraw tasks have no reference object
        ref = None
    notes = prepare(design, ref) if ref else ["no reference object for this task: reference terms are dropped, not scored 0"]
    if getattr(lex, "auto", False):
        notes.append("roles matched with an automatic lexicon built from the task's own part names, not a curated table")
    parts, joints = design.parts, design.joints
    anchors = anchors or {}
    pop = (population or {}).get("aggregate", {})
    rec = {"system": design.system, "tier": design.tier, "meta": design.meta, "notes": notes, "dims": {}}
    if not parts:
        rec["dims"] = {k: _dim("computed", "fail", 0.0, {}, ["empty design"]) for k in
                       ("1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1", "4.2")}
        rec["headline"] = {"critical_fail": ["1.1"], "buildable": False}
        return rec

    canon = {p.id: lex.canon(p.role) for p in parts}
    occs = analysis.occupancies(parts, res)
    jp = frozenset(frozenset((j.parent, j.child)) for j in joints)
    rows = analysis.pair_table(parts, occs, jp)
    idx = {p.id: i for i, p in enumerate(parts)}
    vols = np.array([o.volume for o in occs])
    D = rec["dims"]

    # ------------------------------------------------------------ 1.1 integrity
    nonjoint = [r for r in rows if not r["joint_pair"]]
    pen_total = sum(r["deep_m3"] for r in nonjoint)
    pen_frac = pen_total / max(vols.sum(), 1e-12)
    coll = [r for r in nonjoint if r["collides"]]
    sweep = {str(f): sum(1 for r in nonjoint if r["deep_m3"] > analysis.COLLIDE_ABS_M3 and r["frac_of_smaller"] > f)
             for f in analysis.FRAC_SWEEP}
    edges = [(r["i"], r["k"]) for r in rows]
    for j in joints:
        if j.parent in idx and j.child in idx and j.origin is not None:
            if max(analysis.surface_distance(j.origin, occs[idx[j.parent]]),
                   analysis.surface_distance(j.origin, occs[idx[j.child]])) <= CONTACT_JOINT_TOL_M:
                edges.append((idx[j.parent], idx[j.child]))
    comps = analysis.components(len(parts), edges)
    comps = sorted(comps, key=lambda c: -vols[c].sum())
    largest_frac = vols[comps[0]].sum() / max(vols.sum(), 1e-12)
    # by count as well: five loose casters are 1% of the volume and a broken chair
    connected_part_frac = len(comps[0]) / len(parts)
    whole = []
    from scipy import ndimage
    for o in occs:
        lab, n = ndimage.label(o.grid, structure=np.ones((3, 3, 3)))
        if n <= 1:
            whole.append(1.0)
            continue
        sizes = np.bincount(lab.ravel())[1:]
        whole.append(1.0 if (sizes >= 0.05 * sizes.sum()).sum() == 1 else 0.0)
    zmin = float(min(p.vertices[:, 2].min() for p in parts))
    shells = [p.id for p, o in zip(parts, occs) if o.filled_frac < 0.05]
    touching_pairs = len(nonjoint)
    cfree = 1 - len(coll) / max(touching_pairs, 1)
    s11 = np.mean([cfree, (largest_frac + connected_part_frac) / 2, np.mean(whole)])
    st11 = "fail" if (largest_frac < LARGEST_COMPONENT_FAIL or connected_part_frac < CONNECTED_PARTS_FAIL
                      or pen_frac > PEN_FRAC_FAIL) else ("degraded" if shells else "pass")
    D["1.1"] = _dim("computed", st11, s11, {
        "n_parts": len(parts), "voxel_res_m": res,
        "collision_free_pair_rate": cfree, "n_touching_pairs": touching_pairs, "n_colliding_pairs": len(coll),
        "colliding_pairs": [{"a": r["a"], "b": r["b"], "cm3": round(r["deep_m3"] * 1e6, 2), "frac": round(r["frac_of_smaller"], 3)}
                            for r in sorted(coll, key=lambda r: -r["deep_m3"])[:10]],
        "pen_volume_frac": pen_frac, "pen_volume_cm3": pen_total * 1e6, "collision_frac_sweep": sweep,
        "joint_pair_overlap_cm3": sum(r["deep_m3"] for r in rows if r["joint_pair"]) * 1e6,
        "n_components": len(comps), "largest_component_volume_frac": float(largest_frac),
        "connected_part_frac": float(connected_part_frac),
        "floating_parts": [parts[i].id for c in comps[1:] for i in c][:30],
        "whole_body_rate": float(np.mean(whole)), "lowest_z_m": zmin,
        "below_ground_m": max(0.0, -zmin), "lifted_m": max(0.0, zmin),
        "shell_only_parts": shells[:20]},
        ["volumes of shell-only parts (open meshes) are under-counted"] if shells else [])

    # ------------------------------------------------------------ 1.2 stability (uniform density)
    # The object is the largest connected group; loose parts lying on the floor support nothing.
    main = comps[0]
    # ground-contact band: 1% of the object's height, at least 5 mm (one rule for every design; a
    # generated mesh never has a perfectly planar underside, and a real object settles that far)
    band = max(0.005, 0.01 * float(np.ptp(np.vstack([p.vertices for p in parts])[:, 2])))
    st = analysis.stability([occs[i] for i in main], band=band)
    tilt_ref = anchors.get("tilt_deg")
    s12 = st["critical_tilt_deg"] / tilt_ref if tilt_ref else (1.0 if st["critical_tilt_deg"] > 0 else 0.0)
    few = st["n_support_vertices"] <= 2
    lifted_main = float(min(parts[i].vertices[:, 2].min() for i in main))
    n12 = ["settling simulation not built; statics only"]
    if len(comps) > 1:
        n12.append(f"computed on the largest connected group ({len(main)} of {len(parts)} parts)")
    D["1.2"] = _dim("computed", "fail" if st["critical_tilt_deg"] <= 0 else ("degraded" if few or len(comps) > 1 else "pass"),
                    max(0.0, s12), {**{k: v for k, v in st.items() if k != "com"}, "com": st["com"],
                                    "reference_tilt_deg": tilt_ref, "pose_sensitive": few,
                                    "n_parts_in_main_group": len(main), "main_group_lowest_z_m": lifted_main,
                                    "ground_band_m": band},
                    n12)

    # ------------------------------------------------------------ 2.1 geometry
    ext = extents(parts)
    rext = {"width": max(ref["bounds"][1][0] - ref["bounds"][0][0], ref["bounds"][1][1] - ref["bounds"][0][1]),
            "depth": min(ref["bounds"][1][0] - ref["bounds"][0][0], ref["bounds"][1][1] - ref["bounds"][0][1]),
            "height": ref["bounds"][1][2] - ref["bounds"][0][2]} if ref else None
    m21 = {"extent_m": ext}
    sub = []
    pred = {}
    f0 = None
    rparts = []
    ref_canon = {}
    seat_parts = [p for p in parts if canon[p.id] == "seat"]
    if rext and design.meta.get("scale_mode") != "unitless":
        lr = {k: math.log2(max(ext[k], 1e-6) / rext[k]) for k in ("width", "depth", "height")}
        size_score = 2 ** (-np.mean([abs(v) for v in lr.values()]))
        pe = pop.get("extent_m", {})
        inpop = [1.0 if pe.get(k) and pe[k]["p5"] <= ext[k] <= pe[k]["p95"] else 0.0 for k in ("width", "depth", "height")] if pe else []
        m21.update({"size_log2_ratio_vs_reference": lr, "size_score": size_score,
                    "within_population_p5_p95_frac": float(np.mean(inpop)) if inpop else None})
        sub.append(size_score)
    if ref:
        dpts = _center_ground(sample_points(parts), ref["bounds"])
        rparts = [Part(f"r{p['node']}", p["vertices"], p["faces"]) for p in ref["parts"]]
        rpts = _center_ground(sample_points(rparts, seed=1), ref["bounds"])
        best = None
        for yaw in (0, 90, 180, 270):
            R = axis_angle([0, 0, 1], math.radians(yaw))
            f = fscore(dpts @ R.T, rpts)
            if best is None or f["fscore@0.05"] > best[1]["fscore@0.05"]:
                best = (yaw, f)
        m21.update({"shape_vs_reference": best[1], "best_yaw_deg": best[0], "front_consistent": best[0] == 0})
        f0 = fscore(dpts, rpts)
        m21["shape_vs_reference_as_oriented"] = f0
        sub.append(f0["fscore@0.05"])
        ref_canon = {f"r{p['node']}": lex.canon(p["role_raw"]) for p in ref["parts"]}
    # role predicates
    ref_seat = [p for p in rparts if ref_canon.get(p.id) == "seat"]
    if seat_parts and ref_seat:
        top = max(float(p.vertices[:, 2].max()) for p in seat_parts)
        rtop = max(float(p.vertices[:, 2].max()) for p in ref_seat)
        rng_hi = rtop + max((j.get("prismatic_range") or [0, 0])[1] for j in ref["joints"]) if ref["joints"] else rtop
        out = 0.0 if rtop <= top <= rng_hi else min(abs(top - rtop), abs(top - rng_hi))
        pred["seat_height"] = {"seat_top_m": top, "reference_range_m": [rtop, rng_hi], "score": max(0.0, 1 - out / SEAT_DECAY_M)}
    back = [p for p in parts if canon[p.id] == "backrest"]
    if back and seat_parts:
        s_lo = np.min([p.vertices.min(0) for p in seat_parts], 0)
        s_hi = np.max([p.vertices.max(0) for p in seat_parts], 0)
        b_lo = np.min([p.vertices.min(0) for p in back], 0)
        b_hi = np.max([p.vertices.max(0) for p in back], 0)
        checks = [float((b_lo[2] + b_hi[2]) / 2 > s_hi[2] - 0.02), float((b_lo[1] + b_hi[1]) / 2 > (s_lo[1] + s_hi[1]) / 2),
                  float(b_hi[2] - b_lo[2] >= b_hi[1] - b_lo[1])]
        pred["backrest_above_behind_upright"] = {"checks": checks, "score": float(np.mean(checks))}
    wheels = [p for p in parts if canon[p.id] == "caster wheel"]
    if wheels:
        rs = [roundness(p) for p in wheels]
        hz = [1.0 if a is not None and abs(a[2]) < 0.5 else 0.0 for _, a in rs]
        pred["wheel_round_horizontal_axle"] = {"roundness": float(np.mean([r for r, _ in rs])), "axle_horizontal": float(np.mean(hz)),
                                               "score": float(np.mean([r for r, _ in rs]) * 0.5 + np.mean(hz) * 0.5)}
    cols = [p for p in parts if canon[p.id] == "gas lift cylinder"]
    if cols:
        vs = []
        for p in cols:
            P = p.vertices - p.vertices.mean(0)
            w, V = np.linalg.eigh(np.cov(P.T))
            vs.append(abs(V[:, 2][2]))
        pred["column_vertical"] = {"score": float(np.mean(vs))}
    nclu = ground_clusters(parts)
    rclu = anchors.get("ground_clusters")   # no reference evaluation for this task: report the count, score nothing
    pred["ground_contact_points"] = {"n": nclu, "reference_n": rclu,
                                     **({"score": float(min(nclu, rclu) / max(rclu, 1))} if rclu else {})}
    arms = [p for p in parts if canon[p.id] == "armrest"]
    if arms and seat_parts:
        sc = np.mean([(p.vertices.min(0) + p.vertices.max(0)) / 2 for p in seat_parts], 0)
        sides = {np.sign(((p.vertices.min(0) + p.vertices.max(0)) / 2 - sc)[0]) for p in arms}
        pred["armrests_both_sides"] = {"score": float(len(sides - {0.0}) / 2)}
    m21["role_predicates"] = pred
    pred_mean = _mean([v.get("score") for v in pred.values()])
    if pred_mean is not None:
        sub.append(pred_mean)
    s21 = _mean(sub)
    D["2.1"] = _dim("computed", "pass" if s21 is not None else "skipped", s21, m21,
                    ["role predicates are computed only for roles the design declares; presence is scored in 3.1"]
                    + ([] if ref else ["no reference object: shape and size are not measurable for this task"]))

    # ------------------------------------------------------------ 2.2 kinematics
    moving = [j for j in joints if j.type != "fixed" and j.parent in idx and j.child in idx]
    groups = rigid_groups(parts, rows, joints)
    if not _declares(design, "joints") and not joints:
        D["2.2"] = _dim("computed", "skipped", None, {"declared": False}, ["the producer does not declare joints"])
    else:
        comp = []
        for j in moving:
            ok_axis = j.type == "ball" or (j.axis is not None and np.linalg.norm(j.axis) > 0)
            ok_lim = j.type in ("continuous", "ball") or j.limits is not None
            comp.append(float(ok_axis and j.origin is not None and ok_lim))
        placement = []
        for j in moving:
            if j.origin is None:
                placement.append(0.0)
                continue
            d = max(analysis.surface_distance(j.origin, occs[idx[j.parent]]), analysis.surface_distance(j.origin, occs[idx[j.child]]))
            placement.append(max(0.0, 1 - max(0.0, d - CONTACT_JOINT_TOL_M) / 0.1))
        # claims K
        claim_rows = []
        for k in task.kinematics:
            creq = axis_requirement(k.get("axis"))
            kids = [p.id for p in parts if lex.satisfies(k["moving_part"], canon[p.id], broad=False)]
            per_child = []
            for cid in kids:
                bestq, bestj = 0.0, None
                for j in moving:
                    cg = set(groups.get(j.child, [j.child]))
                    pg = set(groups.get(j.parent, [j.parent]))
                    if cid not in cg:
                        continue
                    par_ok = any(lex.satisfies(k["relative_to"], canon[x], broad=False) for x in pg)
                    rs = 1.0 if par_ok else 0.5
                    q = rs * (0.4 * type_score(j.type, k["joint_type"]) + 0.4 * axis_score(j.axis, creq)
                              + 0.2 * range_score(j, k.get("range"), k["joint_type"]))
                    if q > bestq:
                        bestq, bestj = q, j.id
                per_child.append((cid, bestq, bestj))
            score = float(np.mean([q for _, q, _ in per_child])) if per_child else 0.0
            claim_rows.append({"id": k["id"], "moving_part": k["moving_part"], "relative_to": k["relative_to"],
                               "joint_type": k["joint_type"], "axis": creq, "n_candidate_parts": len(kids),
                               "score": score, "best": [{"part": c, "q": round(q, 3), "joint": jj} for c, q, jj in per_child][:8]})
        recall_k = float(np.mean([c["score"] for c in claim_rows])) if claim_rows else None
        # reference joints, Hungarian
        rdiag = float(np.linalg.norm(np.subtract(ref["bounds"][1], ref["bounds"][0]))) if ref else 1.0
        rnode_canon = {p["node"]: lex.canon(p["role_raw"]) for p in (ref["parts"] if ref else [])}
        rj = []
        for j in (ref["joints"] if ref else []):
            child_roles = {rnode_canon[n] for n in j["child_nodes"]} - {None}
            parent_roles = {rnode_canon[n] for n in j["parent_nodes"]} - {None}
            rtype = "prismatic" if j["type"] == "cylindrical" and j.get("prismatic_range") else j["type"]
            rj.append({"id": j["id"], "child_roles": child_roles, "parent_roles": parent_roles, "type": j["type"],
                       "axis": np.asarray(j["axis"]), "origin": np.asarray(j["origin"]), "alt_type": rtype})
        Q = np.zeros((len(rj), len(moving)))
        for a, r in enumerate(rj):
            for b, j in enumerate(moving):
                cg = {canon[x] for x in groups.get(j.child, [j.child])}
                pg = {canon[x] for x in groups.get(j.parent, [j.parent])}
                role = (0.5 if r["child_roles"] & cg else 0.0) + (0.5 if (r["parent_roles"] & pg or not r["parent_roles"]) else 0.0)
                if not r["child_roles"] & cg:
                    role = 0.0
                ts = max(type_score(j.type, r["type"]), type_score(j.type, r["alt_type"]))
                if j.axis is not None:
                    cosv = abs(float(np.dot(np.asarray(j.axis) / max(np.linalg.norm(j.axis), 1e-12), r["axis"])))
                    ang = math.degrees(math.acos(min(1.0, cosv)))
                    ax = max(0.0, 1 - ang / AXIS_TOL_DEG)
                else:
                    ax = 0.0
                org = max(0.0, 1 - np.linalg.norm(np.asarray(j.origin) - r["origin"]) / (ORIGIN_SCALE_FRAC * rdiag)) if j.origin is not None else 0.0
                Q[a, b] = role * (0.35 * ts + 0.35 * ax + 0.3 * org)
        matches = []
        if len(rj) and len(moving):
            from scipy.optimize import linear_sum_assignment
            ra, cb = linear_sum_assignment(-Q)
            matches = [{"reference": rj[a]["id"], "design": moving[b].id, "q": float(Q[a, b])} for a, b in zip(ra, cb) if Q[a, b] > 0]
        tp = sum(m["q"] for m in matches)
        rec_ref = tp / len(rj) if rj else None
        prec_ref = tp / len(moving) if moving else 0.0
        # a task whose reference has no annotated joints cannot support this term: drop it instead of scoring 0
        f1_ref = None if not rj else (2 * prec_ref * rec_ref / (prec_ref + rec_ref) if rec_ref and prec_ref else 0.0)
        # achievable range
        sweeps = []
        rest_overlap = {}
        for j in moving[:24]:
            if j.type == "ball" or j.axis is None or j.origin is None:
                sweeps.append({"joint": j.id, "ratio": 0.0, "note": "no axis/origin"})
                continue
            mset, welded = moving_set(parts, rows, joints, j)
            static = [i for i, p in enumerate(parts) if p.id not in mset]
            mids = [idx[x] for x in mset]
            mvol = sum(occs[i].volume for i in mids)
            if j.type in ("continuous",) or (j.type in ("revolute",) and j.limits is None):
                samples = [("rot", a) for a in np.linspace(0, 2 * math.pi, SWEEP_STEPS_ROT, endpoint=False)[1:]]
            elif j.type == "revolute":
                samples = [("rot", a) for a in np.linspace(j.limits[0], j.limits[1], SWEEP_STEPS_LIN)]
            else:
                lo_, hi_ = (j.limits if j.limits is not None else (0.0, 0.0))
                samples = [("lin", s) for s in np.linspace(lo_, hi_, SWEEP_STEPS_LIN)]
                if j.type == "cylindrical":
                    samples += [("rot", a) for a in np.linspace(0, 2 * math.pi, SWEEP_STEPS_ROT, endpoint=False)[1:]]
            base = {}
            for i in mids:
                for s_ in static:
                    base[(i, s_)] = voxel.overlap_cells(occs[i], occs[s_])
            free, blocked_at = 0, None
            ax = np.asarray(j.axis) / np.linalg.norm(j.axis)
            org = np.asarray(j.origin)
            for kind, val in samples:
                inc = 0
                for i in mids:
                    v = parts[i].vertices
                    if kind == "rot":
                        v2 = (v - org) @ axis_angle(ax, val).T + org
                    else:
                        v2 = v + ax * val
                    o2 = voxel.voxelize(v2, parts[i].faces, res)
                    for s_ in static:
                        inc += max(0, voxel.overlap_cells(o2, occs[s_]) - base[(i, s_)])
                if inc * res ** 3 > max(SWEEP_BLOCK_ABS_M3, SWEEP_BLOCK_FRAC * mvol):
                    blocked_at = blocked_at if blocked_at is not None else (kind, round(float(val), 3))
                else:
                    free += 1
            sweeps.append({"joint": j.id, "type": j.type, "ratio": free / max(len(samples), 1), "n_samples": len(samples),
                           "moving_parts": sorted(mset)[:12], "welded_by_contact": welded, "first_block": blocked_at})
        range_ratio = float(np.mean([s["ratio"] for s in sweeps])) if sweeps else None
        parts_sub = [recall_k, f1_ref, range_ratio, float(np.mean(comp)) if comp else None]
        s22 = _mean(parts_sub)
        # nothing to score: the task claims no kinematics and its reference has no annotated joints (a static object)
        st22 = "skipped" if s22 is None else ("fail" if not moving else "pass")
        D["2.2"] = _dim("computed", st22, s22, {
            "n_joints": len(joints), "n_moving": len(moving), "reference_n_moving": len(ref["joints"]) if ref else None,
            "declaration_completeness": float(np.mean(comp)) if comp else 0.0,
            "joint_on_boundary": float(np.mean(placement)) if placement else 0.0,
            "claims_recall": recall_k, "claims": claim_rows,
            "reference_recall": rec_ref, "reference_precision": prec_ref, "reference_f1": f1_ref, "reference_matches": matches,
            "achievable_range_ratio": range_ratio, "sweeps": sweeps},
            ["joints swept independently; coupled mechanisms are not sampled jointly"]
            + ([] if s22 is not None else ["this task claims no kinematics and its reference has no annotated joints"]))

    D["2.3"] = _dim("computed", "not_built", None, {}, ["operability probes (grip, reach, pass) are not built"])

    # ------------------------------------------------------------ 3.1 decomposition
    declares_roles = _declares(design, "roles")
    ref_counts = Counter(lex.canon(p["role_raw"]) for p in (ref["parts"] if ref else []))
    ref_counts.pop(None, None)
    des_counts = Counter(c for c in canon.values() if c)
    cov_rows = []
    for pc in task.required_parts:
        c = lex.canon(pc["part"])
        acc = lex.accept.get(c, {c}) if c else set()
        n = sum(v for k, v in des_counts.items() if k in acc)
        need = max(1, max((ref_counts.get(a, 0) for a in acc), default=1))
        cov_rows.append({"id": pc["id"], "part": pc["part"], "canonical": c, "n_design": n, "n_reference": need,
                         "score": float(min(n, need) / need) if c else 0.0})
    coverage = float(np.mean([r["score"] for r in cov_rows])) if cov_rows else None
    keys = set(ref_counts) | set(des_counts)
    l1 = sum(abs(des_counts.get(k, 0) - ref_counts.get(k, 0)) for k in keys)
    hist = max(0.0, 1 - l1 / max(sum(ref_counts.values()), 1)) if ref_counts else None
    gran = 2 ** (-abs(math.log2(len(parts) / max(len(ref["parts"]), 1)))) if ref else None
    unrec = sum(1 for p in parts if p.role and canon[p.id] is None) / len(parts)
    consol = 0
    for r in nonjoint:
        a, b = parts[r["i"]], parts[r["k"]]
        if canon[a.id] and canon[a.id] == canon[b.id] and a.material and a.material == b.material:
            consol += 1
    # part correspondence to the reference (role-free)
    shift = np.r_[(np.asarray(ref["bounds"][0])[:2] + np.asarray(ref["bounds"][1])[:2]) / 2, 0.0] if ref else np.zeros(3)
    V = np.vstack([p.vertices for p in parts])
    dshift = np.r_[(V.max(0)[:2] + V.min(0)[:2]) / 2, V.min(0)[2]] - shift
    ious = []
    coarse = 0.01
    dsets = []
    for p in (parts if rparts else []):
        pts = p.vertices - dshift
        o = voxel.voxelize(pts, p.faces, coarse)
        dsets.append(set(map(tuple, np.argwhere(o.grid) + o.lo)))
    for rp in rparts:
        o = voxel.voxelize(rp.vertices, rp.faces, coarse)
        rs = set(map(tuple, np.argwhere(o.grid) + o.lo))
        best_iou = max((len(rs & ds) / max(len(rs | ds), 1) for ds in dsets), default=0.0)
        ious.append((best_iou, len(rs)))
    miou = float(sum(i * n for i, n in ious) / max(sum(n for _, n in ious), 1)) if ious else None
    if declares_roles or any(p.role for p in parts):
        s31, st31 = _mean([coverage, hist, gran, 1 - unrec]), "pass"
    else:
        s31, st31 = gran, "degraded"   # without roles only granularity is measurable: a weaker instrument, not a skip
    D["3.1"] = _dim("computed", st31, s31, {
        "required_part_coverage": coverage, "required_parts": cov_rows, "role_counts": dict(des_counts),
        "reference_role_counts": dict(ref_counts), "role_histogram_score": hist, "n_parts": len(parts),
        "reference_n_parts": len(ref["parts"]) if ref else None, "granularity_score": gran, "unrecognised_role_frac": unrec,
        "unrecognised_roles": sorted({p.role for p in parts if p.role and canon[p.id] is None})[:20],
        "consolidation_violations": consol, "part_iou_vs_reference_volume_weighted": miou},
        [] if st31 != "skipped" else ["no roles declared: only granularity and part IoU are computed"])

    # ------------------------------------------------------------ 3.2 aesthetics (computed half)
    sym, sym_ang = symmetry_iou(occs)
    sym_ref = anchors.get("symmetry_iou")
    prop = 2 ** (-np.mean([abs(math.log2(max(ext["height"], 1e-6) / max(ext["width"], 1e-6) / (rext["height"] / rext["width"]))),
                            abs(math.log2(max(ext["depth"], 1e-6) / max(ext["width"], 1e-6) / (rext["depth"] / rext["width"])))])) if rext else None
    D["3.2"] = _dim("computed+adjudicated", "pass", _mean([min(1.0, sym / sym_ref) if sym_ref else sym, prop]), {
        "symmetry_iou": sym, "symmetry_plane_normal_deg": sym_ang, "reference_symmetry_iou": sym_ref,
        "proportion_score": prop}, ["image similarity and the VLM judge are attached by the report step"])

    # ------------------------------------------------------------ 3.3 creativity
    used = Counter((p.source or {}).get("id") for p in parts if (p.source or {}).get("kind") == "pool")
    origin = task.raw["evaluator_only"]["pool"]["pool_part_origin"]
    kinds = Counter(origin.get(k, {}).get("kind", "?") for k in used.elements())
    created = sum(1 for p in parts if (p.source or {}).get("kind") == "created")
    D["3.3"] = _dim("computed", "pass", None, {
        "novelty_1_minus_fscore@0.05": (1 - f0["fscore@0.05"]) if f0 else None, "distinct_pool_parts": len(used),
        "pool_instances_by_origin": dict(kinds), "created_instances": created,
        "created_frac": created / len(parts)}, ["diversity across seeds is attached by the report step"])

    # ------------------------------------------------------------ 4.1 material
    if not _declares(design, "materials") and not any(p.material for p in parts):
        D["4.1"] = _dim("computed", "skipped", None, {"declared": False}, ["the producer does not declare materials"])
    else:
        res_m = [materials.resolve(p.material) for p in parts]
        cov = float(sum(v for v, r in zip(vols, res_m) if r) / max(vols.sum(), 1e-12))
        groups_m = []
        for a in task.attributes:
            if a.get("kind") == "material":
                groups_m += [g for g in materials.claim_classes(a["statement"]) if g not in groups_m]
        claimed = set().union(*groups_m) if groups_m else set()
        dclasses = {r[1] for r in res_m if r}
        claim_cov = sum(1 for g in groups_m if g & dclasses) / len(groups_m) if groups_m else None
        in_claim = float(sum(v for v, r in zip(vols, res_m) if r and r[1] in claimed) / max(vols.sum(), 1e-12)) if claimed else None
        ref_cls = defaultdict(Counter)
        for p in (ref["parts"] if ref else []):
            c = lex.canon(p["role_raw"])
            if c and p["material_class"]:
                ref_cls[c][p["material_class"]] += 1
        agree_w, agree_n = 0.0, 0.0
        like_w, like_n = 0.0, 0.0
        pm = pop.get("material_class_by_role", {})
        for p, v, r in zip(parts, vols, res_m):
            c = canon[p.id]
            if not c:
                continue
            if c in ref_cls:
                agree_n += v
                agree_w += v * float(r is not None and r[1] == ref_cls[c].most_common(1)[0][0])
            if c in pm:
                cnt = pm[c]
                tot = sum(cnt.values()) + len(materials.CLASSES)
                probs = {k: (cnt.get(k, 0) + 1) / tot for k in materials.CLASSES}
                like_n += v
                like_w += v * ((probs.get(r[1], 0.0) / max(probs.values())) if r else 0.0)
        agree = agree_w / agree_n if agree_n else None
        like = like_w / like_n if like_n else None
        masses = [(r[2] if r else 0.0) * v for r, v in zip(res_m, vols)]
        loaded = None
        if seat_parts and sum(masses) > 0:
            sp = np.vstack([p.vertices for p in seat_parts])
            top_pt = [float((sp[:, 0].min() + sp[:, 0].max()) / 2), float((sp[:, 1].min() + sp[:, 1].max()) / 2), float(sp[:, 2].max())]
            loaded = analysis.stability([occs[i] for i in main], [masses[i] for i in main], [(top_pt, OCCUPANT_KG)],
                                        band=band)["critical_tilt_deg"]
        lref = anchors.get("loaded_tilt_deg")
        lscore = None if loaded is None else (max(0.0, loaded) / lref if lref else float(loaded > 0))
        # reference-material agreement and population likelihood exist only for tasks with that data: drop, never zero
        subs = [cov, like, agree, None if lscore is None else min(1.0, lscore)]
        D["4.1"] = _dim("declared+computed", "pass" if cov > 0.99 else ("fail" if cov == 0 else "degraded"), _mean(subs), {
            "coverage_volume_frac": cov, "claimed_class_groups": [sorted(g) for g in groups_m], "design_classes": sorted(dclasses),
            "claim_class_coverage": claim_cov, "volume_frac_in_claimed_classes": in_claim,
            "reference_role_material_agreement": agree, "population_material_likelihood": like,
            "total_mass_kg": float(sum(masses)), "reference_mass_kg": anchors.get("mass_kg"),
            "loaded_tilt_deg": loaded, "reference_loaded_tilt_deg": lref, "occupant_kg": OCCUPANT_KG,
            "per_part": [{"part": p.id, "role": canon[p.id], "material": p.material, "class": r[1] if r else None}
                         for p, r in zip(parts, res_m)][:40]},
            ["densities are Artiverse medians; volumes of open meshes are under-counted"])

    # ------------------------------------------------------------ 4.2 assembly sequence
    seq = design.sequence
    if not seq:
        D["4.2"] = _dim("computed", "fail" if _declares(design, "sequence") else "skipped", 0.0 if _declares(design, "sequence") else None,
                        {"declared": False}, ["no assembly sequence declared"])
    else:
        order = [s["part"] for s in seq if s["part"] in idx]
        coverage42 = len(set(order)) / len(parts)
        dirs = [np.array(v, float) for v in ([0, 0, -1], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])]
        feas, feas_decl, supported, stable = [], [], [], []
        placed = []
        grounded = lambda i: parts[i].vertices[:, 2].min() <= zmin + 0.01
        pair_set = {(r["i"], r["k"]) for r in rows} | {(r["k"], r["i"]) for r in rows}
        for step, pid in enumerate(order):
            i = idx[pid]
            declared = next((s.get("direction") for s in seq if s["part"] == pid), None)
            cands = ([np.asarray(declared, float)] if declared is not None else []) + dirs
            ok_any, ok_decl = False, None
            if placed:
                P = voxel.union([occs[k] for k in placed])
                rest = voxel.overlap_cells(occs[i], P, deep=True)
                for ci, dvec in enumerate(cands):
                    ext_m = float(np.linalg.norm(parts[i].vertices.max(0) - parts[i].vertices.min(0))) + INSERT_EXTRA_M
                    nsteps = int(ext_m / res) + 1
                    blocked = False
                    for t in range(1, nsteps + 1, 2):
                        off = np.round(-dvec * t).astype(np.int64)
                        shifted = voxel.Occ(occs[i].lo + off, occs[i].grid, occs[i].points[:0], res, 0)
                        shifted._core = occs[i].core
                        if voxel.overlap_cells(shifted, P, deep=True) > rest + 2:
                            blocked = True
                            break
                    if declared is not None and ci == 0:
                        ok_decl = not blocked
                    if not blocked:
                        ok_any = True
                        if declared is None or ci > 0 or ok_decl is not None:
                            break
            else:
                ok_any, ok_decl = True, True if declared is not None else None
            feas.append(float(ok_any))
            if ok_decl is not None:
                feas_decl.append(float(ok_decl))
            placed.append(i)
            sub_edges = [(a, b) for (a, b) in pair_set if a in placed and b in placed]
            comps_p = analysis.components(len(parts), sub_edges)
            comps_p = [c for c in comps_p if c[0] in placed or any(x in placed for x in c)]
            comps_p = [[x for x in c if x in placed] for c in comps_p]
            comps_p = [c for c in comps_p if c]
            unsupported = [c for c in comps_p if not any(grounded(x) for x in c)]
            supported.append(float(not unsupported))
            gcomp = [x for c in comps_p if any(grounded(x) for x in c) for x in c]
            if gcomp:
                stp = analysis.stability([occs[x] for x in gcomp])
                stable.append(float(stp["critical_tilt_deg"] > 0))
            else:
                stable.append(0.0)
        s42 = np.mean([coverage42, np.mean(feas), np.mean(supported), np.mean(stable)])
        D["4.2"] = _dim("computed", "pass" if coverage42 == 1 else "degraded", s42, {
            "coverage": coverage42, "insertion_feasible_frac": float(np.mean(feas)),
            "declared_direction_feasible_frac": float(np.mean(feas_decl)) if feas_decl else None,
            "first_infeasible_step": next((k for k, v in enumerate(feas) if not v), None),
            "supported_prefix_frac": float(np.mean(supported)), "stable_prefix_frac": float(np.mean(stable)),
            "n_steps": len(order)},
            ["insertion paths are straight translations along the declared direction or the six axes"])

    # ------------------------------------------------------------ 4.3 functional completeness
    rref_centres = role_centres(rparts, ref_canon, ref["bounds"]) if ref else {}
    V = np.vstack([p.vertices for p in parts])
    dcent = role_centres(parts, canon, [V.min(0), V.max(0)])
    pairs_adj = defaultdict(set)
    moving_pairs = {frozenset((j.parent, j.child)) for j in moving}
    for r in rows:
        pairs_adj[r["a"]].add(r["b"])
        pairs_adj[r["b"]].add(r["a"])
    for j in joints:
        if j.parent in idx and j.child in idx:
            pairs_adj[j.parent].add(j.child)
            pairs_adj[j.child].add(j.parent)

    def path(A, B, need_motion, max_hops=4):
        """1.0: a path of at most max_hops edges from an A part to a different B part
        (through a moving joint when the connection carries motion); 0.5: a path
        exists but crosses no moving joint although motion is required; 0.0: none."""
        from collections import deque
        best = 0.0
        for a in A:
            q = deque([(a, 0, False)])
            seen = {(a, False)}
            while q:
                u, h, mot = q.popleft()
                if h > 0 and u in B and u != a:
                    best = max(best, 1.0 if (mot or not need_motion) else 0.5)
                    if best == 1.0:
                        return 1.0
                if h >= max_hops:
                    continue
                for v in pairs_adj[u]:
                    m2 = mot or frozenset((u, v)) in moving_pairs
                    if (v, m2) not in seen:
                        seen.add((v, m2))
                        q.append((v, h + 1, m2))
        return best

    subs_rows = []
    for F in task.subsystems:
        pres = [any(lex.satisfies(n, canon[p.id]) for p in parts) for n in F["parts"]]
        plac = []
        for n in F["parts"]:
            c = lex.canon(n)
            acc = lex.accept.get(c, {c}) if c else set()
            dc = [dcent[a] for a in acc if a in dcent]
            rc = [rref_centres[a] for a in acc if a in rref_centres]
            if dc and rc:
                plac.append(max(0.0, 1 - float(np.linalg.norm(np.mean(dc, 0) - np.mean(rc, 0))) / 0.5))
        conns = []
        for cn in F.get("connections", []):
            A = {p.id for p in parts if lex.satisfies(cn["from"], canon[p.id], broad=False)}
            B = {p.id for p in parts if lex.satisfies(cn["to"], canon[p.id], broad=False)}
            conns.append(path(A, B, cn.get("carries") == "motion") if A and B else 0.0)
        subs_rows.append({"id": F["id"], "subsystem": F["subsystem"], "presence": float(np.mean(pres)) if pres else None,
                          "placement": float(np.mean(plac)) if plac else None,
                          "connection": float(np.mean(conns)) if conns else None})
    if not declares_roles and not any(p.role for p in parts):
        D["4.3"] = _dim("computed", "skipped", None, {"declared": False, "subsystems": subs_rows}, ["no roles: subsystems cannot be located"])
    else:
        D["4.3"] = _dim("computed", "pass", None, {
            "presence": _mean([s["presence"] for s in subs_rows]),
            "placement": _mean([s["placement"] for s in subs_rows]),
            "connection": _mean([s["connection"] for s in subs_rows]),
            "subsystems": subs_rows}, ["presence, placement and connection are reported separately, never merged"])

    # ------------------------------------------------------------ attributes A (computed proxies)
    attr = []
    for a in task.attributes:
        if a.get("kind") == "material":
            m = D.get("4.1", {}).get("metrics", {})
            attr.append({"id": a["id"], "proxy": "declared material classes cover the claimed classes",
                         "score": m.get("claim_class_coverage")})
        elif "desk" in a["statement"].lower():
            attr.append({"id": a["id"], "proxy": "seat height within the reference height range",
                         "score": pred.get("seat_height", {}).get("score", 0.0)})
        elif "adjust" in a["statement"].lower() and "lever" not in a["statement"].lower():
            adj = [j for j in moving if j.type in ("prismatic", "cylindrical") and
                   any(canon[x] in ("seat", "armrest", "backrest", "gas lift cylinder") for x in groups.get(j.child, [j.child]))]
            attr.append({"id": a["id"], "proxy": "an adjustable (prismatic/cylindrical) joint moves the seat, back or arms",
                         "score": float(bool(adj))})
        elif "lever" in a["statement"].lower():
            levers = [p for p in parts if canon[p.id] == "lever"]
            near = 0.0
            for lv in levers:
                for p in parts:
                    if canon[p.id] in ("gas lift cylinder", "swivel mechanism", "seat"):
                        if analysis.surface_distance(occs[idx[lv.id]].centroid(), occs[idx[p.id]]) < 0.08:
                            near = 1.0
            attr.append({"id": a["id"], "proxy": "a lever exists within 8 cm of the column, swivel mechanism or seat",
                         "score": (0.5 + 0.5 * near) if levers else 0.0})
        else:
            attr.append({"id": a["id"], "proxy": None, "score": None})
    rec["attributes"] = attr

    # ------------------------------------------------------------ headline
    crit = [k for k in ("1.1", "1.2", "2.1", "2.2", "4.1", "4.2") if D.get(k, {}).get("status") == "fail"]
    rec["headline"] = {"critical_fail": crit, "buildable": not any(k in crit for k in ("1.1", "1.2")),
                       "gate_pending": True}
    return rec


# ================================================================ reference arm

def reference_design(task: Task) -> Design:
    ref = task.reference()
    parts = []
    for p in ref["parts"]:
        parts.append(Part(f"{p['role_raw'].replace(' ', '_')}_{p['node']}", p["vertices"], p["faces"], p["role_raw"],
                          p["material"], {"kind": "reference", "node": p["node"]}))
    joints = []
    node_id = {p["node"]: parts[i].id for i, p in enumerate(ref["parts"])}
    for j in ref["joints"]:
        if not j["parent_nodes"] or not j["child_nodes"]:
            continue
        child = j["child_nodes"][0]
        if len(j["child_nodes"]) > 1:   # the member nearest the pivot is the one that sits on the parent
            child = min(j["child_nodes"], key=lambda n: np.min(np.linalg.norm(ref["parts"][n]["vertices"] - j["origin"], axis=1)))
        parent = min(j["parent_nodes"], key=lambda n: np.min(np.linalg.norm(ref["parts"][n]["vertices"] - j["origin"], axis=1)))
        t = j["type"]
        lim = None
        if t == "revolute" and j["range"] and None not in j["range"]:
            lim = [float(j["range"][0]), float(j["range"][1])]
        if t == "cylindrical" and j.get("prismatic_range"):
            lim = j["prismatic_range"]
        joints.append(Joint(j["id"], t, node_id[parent], node_id[child], j["axis"], j["origin"], lim))
    return Design("reference", "ref", parts, joints, None,
                  {"scale_mode": "metric", "declares": {"roles": True, "materials": True, "joints": True, "sequence": False},
                   "note": "the task's reference object with its Artiverse materials and annotated joints"})


def anchors_from(rec_ref: dict) -> dict:
    D = rec_ref["dims"]
    return {"tilt_deg": D["1.2"]["metrics"]["critical_tilt_deg"],
            "symmetry_iou": D["3.2"]["metrics"]["symmetry_iou"],
            "ground_clusters": D["2.1"]["metrics"]["role_predicates"]["ground_contact_points"]["n"],
            "loaded_tilt_deg": D.get("4.1", {}).get("metrics", {}).get("loaded_tilt_deg"),
            "mass_kg": D.get("4.1", {}).get("metrics", {}).get("total_mass_kg")}


def load_population(task_id):
    p = RESULTS / task_id / "population.json"
    return json.loads(p.read_text()) if p.exists() else None
