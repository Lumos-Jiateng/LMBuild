"""v3 metrics: task-independent, every score in [0, 1]. See docs/evaluation/metrics_v3.md.

    ev = evaluate_v3(design, task, anchors)      # one record, never an average

No metric here contains a fact about a specific object. Everything comes from the
design, the task file's own claims, or the reference object. The v2 evaluator in
evaluate.py is left alone so its records stay reproducible; v3 writes to eval_v3/.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

import numpy as np

from ppbench.v2 import analysis, materials, voxel
from ppbench.v2.design import Design, Joint, Part, axis_angle
from ppbench.v2.evaluate import (_declares, _dim, _mean, axis_requirement, axis_score, fscore,
                                 moving_set, prepare, range_score, reference_design, rigid_groups,
                                 sample_points, symmetry_iou, type_score)
from ppbench.v2.lexicon import Lexicon
from ppbench.v2.task import RESULTS, Task

SPEC_VERSION = "v3.1"

# ---------------------------------------------------------------- constants
# Every constant below turns a raw number into a [0,1] score. None of them is task-specific.
PEN_FRAC_FAIL = 0.10              # interpenetrating volume / total volume that fails 1.1
LARGEST_COMPONENT_FAIL = 0.95     # volume fraction of the largest connected group below which 1.1 fails
CONNECTED_PARTS_FAIL = 0.90       # fraction of parts in the largest connected group below which 1.1 fails
TILT_FULL_DEG = 30.0              # tilt target when the task has no reference (reference median 22.3, p90 50.2)
TILT_TARGET_MIN_DEG = 5.0         # ... and the target is clamped into this band around the reference's own tilt,
TILT_TARGET_MAX_DEG = 30.0        #     so a door is not asked to stand like a table, nor a table let off at 0.5 deg
FSCORE_TAU_REL = (0.01, 0.025)    # shape tolerances as fractions of the bbox diagonal, after rescaling
SHAPE_TAU_SCORED = 0.025          # the one that becomes the 2.1 score
N_SHAPE_POINTS = 20000
JOINT_PLACEMENT_DECAY_M = 0.10    # joint origin this far off the shared boundary scores 0
CONTACT_JOINT_TOL_M = analysis.JOINT_CONNECT_TOL_M
SWEEP_STEPS_ROT = 12
SWEEP_STEPS_LIN = 8
SWEEP_BLOCK_ABS_M3 = 1e-6
SWEEP_BLOCK_FRAC = 0.05
MOVING_FRAC_MAX = 0.995           # a joint that moves (all but) the whole object is not a joint
MOVING_VOL_MIN_M3 = 1e-9          # ... and one whose moving side has no volume is not moving anything.
# There is deliberately no lower bound in *fraction*. v3.0 had one at 0.5% and it never fired, because the
# contact flood made every moving set huge; with the declared moving set of v3.1 it fires on exactly the
# joints it should not -- a skateboard wheel is 0.3% of the board, a knob less. "Moves a proper
# subassembly" is a question about the partition, not about size: something moves, something stays.
MAX_SWEPT_JOINTS = 24
AXIS_TOL_DEG = 45.0
ORIGIN_SCALE_FRAC = 0.25
INSERT_EXTRA_M = 0.05
PART_IOU_RES_M = 0.01
OVERSIZE_FACTOR = 5.0             # a single part this many times the reference's diagonal is a scale failure
OVERSIZE_ABS_M = 10.0             # ... or this many metres when the task has no reference object


# ---------------------------------------------------------------- shape, scale-normalised

def _normalise(pts):
    """Centre on the XY footprint centre, drop to z = 0, divide by the bbox diagonal.
    After this the reference is the design's size and only shape is left."""
    lo, hi = pts.min(0), pts.max(0)
    c = np.r_[(lo[:2] + hi[:2]) / 2, lo[2]]
    diag = float(np.linalg.norm(hi - lo))
    return (pts - c) / max(diag, 1e-9), diag


def shape_similarity(design_parts, ref_parts, seed_d=0, seed_r=1):
    """Scale-normalised, yaw-searched shape agreement. Returns metrics and the score."""
    dp = sample_points(design_parts, n=N_SHAPE_POINTS, seed=seed_d)
    rp = sample_points(ref_parts, n=N_SHAPE_POINTS, seed=seed_r)
    if len(dp) == 0 or len(rp) == 0:
        return {"note": "no surface to sample"}, None
    dn, ddiag = _normalise(dp)
    rn, rdiag = _normalise(rp)
    best_yaw, best = None, None
    for yaw in (0, 90, 180, 270):
        R = axis_angle([0, 0, 1], math.radians(yaw))
        f = fscore(dn @ R.T, rn, taus=FSCORE_TAU_REL)
        if best is None or f[f"fscore@{SHAPE_TAU_SCORED}"] > best[f"fscore@{SHAPE_TAU_SCORED}"]:
            best_yaw, best = yaw, f
    as_oriented = fscore(dn, rn, taus=FSCORE_TAU_REL)
    m = {"shape_normalised": best, "best_yaw_deg": best_yaw, "front_consistent": best_yaw == 0,
         "shape_normalised_as_oriented": as_oriented,
         "design_diagonal_m": ddiag, "reference_diagonal_m": rdiag,
         "scale_ratio": ddiag / max(rdiag, 1e-9),
         "size_log2_ratio_vs_reference": math.log2(max(ddiag, 1e-9) / max(rdiag, 1e-9)),
         "tau_rel": list(FSCORE_TAU_REL)}
    return m, float(best[f"fscore@{SHAPE_TAU_SCORED}"])


# ---------------------------------------------------------------- what a joint moves

def kinematic_moving_set(parts, rows, joints, jt: Joint):
    """The parts that move with the child of `jt`, and whether the declaration loops back.

    The declaration is the source of truth: walk the declared joint graph out of the child
    and never through the parent. Contact only decides where the parts that no joint mentions
    end up, and it decides that conservatively -- whatever the static side can reach by
    touching stays put -- so a fender that brushes a wheel does not ride with the wheel.

    Contact alone cannot answer this question. `voxel.touching` is a one-cell test, so a wheel
    sitting in its arch "touches" the body with zero shared volume, and a flood fill over
    contacts hands the axle joint 95% of the car. That is what the corpus showed: 47% of all
    scored joints moved more than half the object's volume, and they swept freely, because the
    block threshold scales with the moving volume. See docs/LIMITATIONS.md.
    """
    ids = {p.id for p in parts}
    adj = defaultdict(set)
    for j in joints:
        if j.id == jt.id or j.parent not in ids or j.child not in ids:
            continue
        adj[j.parent].add(j.child)
        adj[j.child].add(j.parent)
    moving, stack, looped = {jt.child}, [jt.child], False
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v == jt.parent:
                looped = True
                continue
            if v not in moving:
                moving.add(v)
                stack.append(v)
    declared = {x for j in joints for x in (j.parent, j.child)} & ids
    loose = ids - declared
    if loose:
        cadj = defaultdict(set)
        for r in rows:
            cadj[r["a"]].add(r["b"])
            cadj[r["b"]].add(r["a"])

        def reach(seed):
            seen, st = set(), [x for x in seed]
            while st:
                u = st.pop()
                for v in cadj[u]:
                    if v in loose and v not in seen:
                        seen.add(v)
                        st.append(v)
            return seen
        moving |= reach(moving) - reach(declared - moving)
    return moving & ids, looped


# ---------------------------------------------------------------- per-joint quality

def joint_quality(j: Joint, parts, occs, rows, joints, idx, res, vol_total):
    """One joint, one [0,1] score, from four terms that need no knowledge of the object."""
    q = {"joint": j.id, "type": j.type}
    ok_axis = j.type == "ball" or (j.axis is not None and np.linalg.norm(j.axis) > 0)
    ok_lim = j.type in ("continuous", "ball") or j.limits is not None
    completeness = float(np.mean([float(ok_axis), float(j.origin is not None), float(ok_lim)]))
    if j.origin is not None:
        d = max(analysis.surface_distance(j.origin, occs[idx[j.parent]]),
                analysis.surface_distance(j.origin, occs[idx[j.child]]))
        placement = max(0.0, 1 - max(0.0, d - CONTACT_JOINT_TOL_M) / JOINT_PLACEMENT_DECAY_M)
    else:
        d, placement = None, 0.0
    mset, looped = kinematic_moving_set(parts, rows, joints, j)
    welded = moving_set(parts, rows, joints, j)[1]      # the old contact-reachability flag, reported only
    mids = [idx[x] for x in mset if x in idx]
    mvol = sum(occs[i].volume for i in mids)
    mfrac = mvol / max(vol_total, 1e-12)
    # a joint is effective when it moves a proper subassembly: not nothing, not the whole object.
    # Neither weld flag gates the score -- a closed drawer touches its case and a hinged lid touches
    # its box, so contact-reachability marks almost every real joint welded, and a declared loop is
    # often just a redundant `fixed` joint. Whether motion is actually possible is what the sweep
    # below measures, and it now sweeps the declared subassembly instead of the whole object.
    effectiveness = float(0 < len(mset) < len(parts) and mvol > MOVING_VOL_MIN_M3 and mfrac < MOVING_FRAC_MAX)
    mobility, first_block, n_samples = 0.0, None, 0
    if j.axis is not None and j.origin is not None and j.type != "ball" and mids:
        ax = np.asarray(j.axis, float)
        ax = ax / max(np.linalg.norm(ax), 1e-12)
        org = np.asarray(j.origin, float)
        static = [i for i, p in enumerate(parts) if p.id not in mset]
        if j.type == "continuous" or (j.type == "revolute" and j.limits is None):
            samples = [("rot", a) for a in np.linspace(0, 2 * math.pi, SWEEP_STEPS_ROT, endpoint=False)[1:]]
        elif j.type == "revolute":
            samples = [("rot", a) for a in np.linspace(j.limits[0], j.limits[1], SWEEP_STEPS_LIN)]
        else:
            lo_, hi_ = (j.limits if j.limits is not None else (0.0, 0.0))
            samples = [("lin", s) for s in np.linspace(lo_, hi_, SWEEP_STEPS_LIN)]
            if j.type == "cylindrical":
                samples += [("rot", a) for a in np.linspace(0, 2 * math.pi, SWEEP_STEPS_ROT, endpoint=False)[1:]]
        base = {(i, s_): voxel.overlap_cells(occs[i], occs[s_]) for i in mids for s_ in static}
        free = 0
        for kind, val in samples:
            inc = 0
            for i in mids:
                v = parts[i].vertices
                v2 = (v - org) @ axis_angle(ax, val).T + org if kind == "rot" else v + ax * val
                o2 = voxel.voxelize(v2, parts[i].faces, res)
                for s_ in static:
                    inc += max(0, voxel.overlap_cells(o2, occs[s_]) - base[(i, s_)])
            if inc * res ** 3 > max(SWEEP_BLOCK_ABS_M3, SWEEP_BLOCK_FRAC * max(mvol, 1e-12)):
                first_block = first_block if first_block is not None else (kind, round(float(val), 3))
            else:
                free += 1
        n_samples = len(samples)
        mobility = free / max(n_samples, 1)
    q.update({"completeness": completeness, "placement": placement, "mobility": mobility,
              "effectiveness": effectiveness, "origin_offset_m": d, "moving_volume_frac": float(mfrac),
              "welded_by_contact": welded, "declared_loop": looped,
              "n_moving_parts": len(mset), "n_sweep_samples": n_samples,
              "first_block": first_block})
    q["score"] = float(np.mean([completeness, placement, mobility, effectiveness]))
    return q


# ---------------------------------------------------------------- anchors (reference-derived, per task)

def anchors_v3(task: Task, res=voxel.RES) -> dict:
    """What the reference object says about its own task. Derived, never hand-written."""
    ref_design = reference_design(task)
    parts = ref_design.parts
    occs = analysis.occupancies(parts, res)
    jp = frozenset(frozenset((j.parent, j.child)) for j in ref_design.joints)
    rows = analysis.pair_table(parts, occs, jp)
    vols = np.array([o.volume for o in occs])
    edges = [(r["i"], r["k"]) for r in rows]
    comps = sorted(analysis.components(len(parts), edges), key=lambda c: -vols[c].sum())
    band = max(0.005, 0.01 * float(np.ptp(np.vstack([p.vertices for p in parts])[:, 2])))
    st = analysis.stability([occs[i] for i in comps[0]], band=band)
    tilt = float(st["critical_tilt_deg"])
    V = np.vstack([p.vertices for p in parts])
    return {"spec": SPEC_VERSION, "task": task.id,
            "reference_tilt_deg": tilt,
            "free_standing": tilt > 0,
            "reference_n_parts": len(parts),
            "reference_n_moving_joints": sum(1 for j in ref_design.joints if j.type != "fixed"),
            "reference_diagonal_m": float(np.linalg.norm(V.max(0) - V.min(0)))}


# ---------------------------------------------------------------- the evaluator

def evaluate_v3(design: Design, task: Task, anchors: dict | None = None, res=voxel.RES,
                is_reference: bool = False) -> dict:
    lex = Lexicon(task.id, task)
    try:
        ref = task.reference()
    except Exception:
        ref = None
    notes = prepare(design, ref) if ref else ["no reference object for this task: reference terms are dropped, not scored 0"]
    anchors = anchors or {}
    parts, joints = design.parts, design.joints
    rec = {"spec": SPEC_VERSION, "system": design.system, "tier": design.tier, "meta": design.meta,
           "notes": notes, "dims": {}}
    if not parts:
        rec["dims"] = {k: _dim("computed", "fail", 0.0, {}, ["empty design"]) for k in
                       ("1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1", "4.2")}
        rec["headline"] = {"critical_fail": ["1.1"], "buildable": False, "overall": 0.0, "gate_pending": True}
        return rec

    canon = {p.id: lex.canon(p.role) for p in parts}
    # Catch a blown scale before voxelising: a part several times the size of the whole reference
    # object is a design that got the units wrong, and at 4 mm it also costs minutes of grid to
    # confirm what its bounding box already says.
    spans = [float(np.linalg.norm(p.vertices.max(0) - p.vertices.min(0))) for p in parts]
    rdiag = anchors.get("reference_diagonal_m")
    limit = OVERSIZE_FACTOR * rdiag if isinstance(rdiag, (int, float)) and rdiag > 0 else OVERSIZE_ABS_M
    if max(spans) > limit:
        big = [p.id for p, sp in zip(parts, spans) if sp > limit]
        rec["dims"] = {"1.1": _dim("computed", "fail", 0.0,
                                   {"n_parts": len(parts), "largest_part_span_m": max(spans),
                                    "oversize_limit_m": limit, "oversized_parts": big[:20],
                                    "n_oversized_parts": len(big)},
                                   [f"{len(big)} part(s) span more than {limit:.1f} m, "
                                    f"{OVERSIZE_FACTOR:.0f}x the reference object: the design's scale is wrong"])}
        rec["headline"] = {"critical_fail": ["1.1"], "buildable": False, "overall": 0.0,
                           "n_scored_dims": 1, "gate_pending": True}
        return rec
    try:
        occs = analysis.occupancies(parts, res)
    except voxel.VoxelTooLarge as e:
        # a part tens of metres across is a design that got the scale wrong, not an evaluator
        # failure: score it as the structural failure it is instead of dropping the design.
        spans = [float(np.linalg.norm(p.vertices.max(0) - p.vertices.min(0))) for p in parts]
        rec["dims"] = {"1.1": _dim("computed", "fail", 0.0,
                                   {"n_parts": len(parts), "largest_part_span_m": max(spans),
                                    "oversized_parts": [p.id for p, sp in zip(parts, spans) if sp > 10.0][:20]},
                                   [f"a part is too large to voxelise: {e}"])}
        rec["headline"] = {"critical_fail": ["1.1"], "buildable": False, "overall": 0.0,
                           "n_scored_dims": 1, "gate_pending": True}
        return rec
    jp = frozenset(frozenset((j.parent, j.child)) for j in joints)
    rows = analysis.pair_table(parts, occs, jp)
    idx = {p.id: i for i, p in enumerate(parts)}
    vols = np.array([o.volume for o in occs])
    D = rec["dims"]

    # ------------------------------------------------------------ 1.1 integrity (unchanged from v2)
    nonjoint = [r for r in rows if not r["joint_pair"]]
    pen_total = sum(r["deep_m3"] for r in nonjoint)
    pen_frac = pen_total / max(vols.sum(), 1e-12)
    coll = [r for r in nonjoint if r["collides"]]
    edges = [(r["i"], r["k"]) for r in rows]
    for j in joints:
        if j.parent in idx and j.child in idx and j.origin is not None:
            if max(analysis.surface_distance(j.origin, occs[idx[j.parent]]),
                   analysis.surface_distance(j.origin, occs[idx[j.child]])) <= CONTACT_JOINT_TOL_M:
                edges.append((idx[j.parent], idx[j.child]))
    comps = sorted(analysis.components(len(parts), edges), key=lambda c: -vols[c].sum())
    largest_frac = vols[comps[0]].sum() / max(vols.sum(), 1e-12)
    connected_part_frac = len(comps[0]) / len(parts)
    from scipy import ndimage
    whole = []
    for o in occs:
        lab, n = ndimage.label(o.grid, structure=np.ones((3, 3, 3)))
        if n <= 1:
            whole.append(1.0)
            continue
        sizes = np.bincount(lab.ravel())[1:]
        whole.append(1.0 if (sizes >= 0.05 * sizes.sum()).sum() == 1 else 0.0)
    shells = [p.id for p, o in zip(parts, occs) if o.filled_frac < 0.05]
    cfree = 1 - len(coll) / max(len(nonjoint), 1)
    s11 = float(np.mean([cfree, (largest_frac + connected_part_frac) / 2, np.mean(whole)]))
    st11 = "fail" if (largest_frac < LARGEST_COMPONENT_FAIL or connected_part_frac < CONNECTED_PARTS_FAIL
                      or pen_frac > PEN_FRAC_FAIL) else ("degraded" if shells else "pass")
    D["1.1"] = _dim("computed", st11, s11, {
        "n_parts": len(parts), "voxel_res_m": res, "collision_free_pair_rate": cfree,
        "n_touching_pairs": len(nonjoint), "n_colliding_pairs": len(coll),
        "pen_volume_frac": pen_frac, "pen_volume_cm3": pen_total * 1e6,
        "n_components": len(comps), "largest_component_volume_frac": float(largest_frac),
        "connected_part_frac": float(connected_part_frac), "whole_body_rate": float(np.mean(whole)),
        "floating_parts": [parts[i].id for c in comps[1:] for i in c][:30],
        "shell_only_parts": shells[:20]},
        ["volumes of shell-only parts (open meshes) are under-counted"] if shells else [])

    # ------------------------------------------------------------ 1.2 stability (no reference term)
    main = comps[0]
    band = max(0.005, 0.01 * float(np.ptp(np.vstack([p.vertices for p in parts])[:, 2])))
    st = analysis.stability([occs[i] for i in main], band=band)
    tilt = float(st["critical_tilt_deg"])
    free_standing = anchors.get("free_standing")
    n12 = ["settling simulation not built; statics only",
           "score = clip(tilt, 0, target) / target, target = the reference's own tilt clamped to [5, 30] degrees"]
    if len(comps) > 1:
        n12.append(f"computed on the largest connected group ({len(main)} of {len(parts)} parts)")
    if free_standing is False:
        # the reference for this task is not free-standing either (it hangs, clamps or leans):
        # free-ground statics does not apply, so this is not measurable rather than a zero
        D["1.2"] = _dim("computed", "not_applicable", None,
                        {**{k: v for k, v in st.items() if k != "com"}, "com": st["com"],
                         "reference_tilt_deg": anchors.get("reference_tilt_deg"), "ground_band_m": band},
                        n12 + ["the reference object is not free-standing on a ground plane: 1.2 is not measurable for this task"])
    else:
        # the target is the reference object's own critical tilt, clamped into [5, 30] degrees: 1.0 means
        # "stands at least as well as the real object". A flat door leans at 2 deg and a dining table at 63,
        # and neither should be scored against the other's number.
        rt = anchors.get("reference_tilt_deg")
        target = float(min(max(rt, TILT_TARGET_MIN_DEG), TILT_TARGET_MAX_DEG)) if isinstance(rt, (int, float)) and rt > 0 else TILT_FULL_DEG
        s12 = float(min(max(tilt, 0.0), target) / target)
        few = st["n_support_vertices"] <= 2
        # the reference itself tips below the floor (a door panel, a window sash): the target is one the
        # real object does not reach either, so the number ranks designs within the task and must not be
        # read across tasks
        thin = isinstance(rt, (int, float)) and 0 < rt < TILT_TARGET_MIN_DEG
        if thin:
            n12.append(f"the reference object itself tips at {rt:.1f} deg, below the {TILT_TARGET_MIN_DEG:.0f} deg floor: "
                       "this score ranks designs within the task, it is not comparable across tasks")
        D["1.2"] = _dim("computed", "fail" if tilt <= 0 else ("degraded" if few or thin or len(comps) > 1 else "pass"), s12,
                        {**{k: v for k, v in st.items() if k != "com"}, "com": st["com"],
                         "target_tilt_deg": target, "reference_tilt_deg": rt,
                         "pose_sensitive": few, "n_parts_in_main_group": len(main), "ground_band_m": band}, n12)

    # ------------------------------------------------------------ 2.1 geometry (shape only, scale-normalised)
    rparts = [Part(f"r{p['node']}", p["vertices"], p["faces"]) for p in ref["parts"]] if ref else []
    if rparts:
        m21, s21 = shape_similarity(parts, rparts)
        D["2.1"] = _dim("computed", "pass" if s21 is not None else "skipped", s21, m21,
                        ["the reference is rescaled to the design's bounding diagonal: 2.1 is shape only",
                         "absolute size is reported as scale_ratio and size_log2_ratio_vs_reference, and is not scored"])
    else:
        D["2.1"] = _dim("computed", "skipped", None, {},
                        ["no reference object: shape is not measurable for this task"])

    # ------------------------------------------------------------ 2.2 kinematics + operability (per joint)
    moving = [j for j in joints if j.type != "fixed" and j.parent in idx and j.child in idx]
    groups = rigid_groups(parts, rows, joints)
    per_joint = []
    claims_k = task.kinematics
    ref_moving = [j for j in (ref["joints"] if ref else []) if j.get("type") != "fixed"]
    if not moving:
        if is_reference:
            # the reference arm: no annotated joints in the source data is a gap in our data,
            # not a design that failed to declare one
            D["2.2"] = _dim("computed", "skipped", None, {"n_joints": len(joints), "n_moving": 0},
                            ["the reference object has no annotated joints: kinematics is not measurable for this task"])
        elif claims_k:
            D["2.2"] = _dim("computed", "fail", 0.0, {"n_joints": len(joints), "n_moving": 0,
                                                      "n_claimed_mechanisms": len(claims_k)},
                            ["the task claims kinematics and the design declares no moving joint"])
        else:
            D["2.2"] = _dim("computed", "skipped", None, {"n_joints": len(joints), "n_moving": 0},
                            ["no moving joint declared, the task claims none, so there is nothing to score"])
    else:
        vol_total = float(vols.sum())
        per_joint[:] = [joint_quality(j, parts, occs, rows, joints, idx, res, vol_total)
                        for j in moving[:MAX_SWEPT_JOINTS]]
        s22 = float(np.mean([q["score"] for q in per_joint]))
        # ---- coverage terms: reported, never folded into the score
        claim_rows = []
        for k in claims_k:
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
                    rs = 1.0 if any(lex.satisfies(k["relative_to"], canon[x], broad=False) for x in pg) else 0.5
                    q = rs * (0.4 * type_score(j.type, k["joint_type"]) + 0.4 * axis_score(j.axis, creq)
                              + 0.2 * range_score(j, k.get("range"), k["joint_type"]))
                    if q > bestq:
                        bestq, bestj = q, j.id
                per_child.append((cid, bestq, bestj))
            claim_rows.append({"id": k["id"], "moving_part": k["moving_part"], "joint_type": k["joint_type"],
                               "n_candidate_parts": len(kids),
                               "score": float(np.mean([q for _, q, _ in per_child])) if per_child else 0.0})
        recall_k = float(np.mean([c["score"] for c in claim_rows])) if claim_rows else None
        ratio = len(moving) / len(ref_moving) if ref_moving else None
        D["2.2"] = _dim("computed", "pass", s22, {
            "n_joints": len(joints), "n_moving": len(moving), "n_scored_joints": len(per_joint),
            "reference_n_moving": len(ref_moving) if ref else None,
            "joint_count_ratio": ratio, "claims_recall": recall_k, "claims": claim_rows,
            "mean_completeness": float(np.mean([q["completeness"] for q in per_joint])),
            "mean_placement": float(np.mean([q["placement"] for q in per_joint])),
            "mean_mobility": float(np.mean([q["mobility"] for q in per_joint])),
            "mean_effectiveness": float(np.mean([q["effectiveness"] for q in per_joint])),
            "per_joint": per_joint},
            ["2.2 is the mean per-joint quality; claim recall and joint count are reported, not scored",
             "joints are swept independently; coupled mechanisms are not sampled jointly"]
            + ([f"only the first {MAX_SWEPT_JOINTS} joints are swept"] if len(moving) > MAX_SWEPT_JOINTS else []))

    # ------------------------------------------------------------ 2.3 operability (semantic capabilities)
    # 2.2 asks whether each joint is a good joint; 2.3 asks whether the object can be used --
    # sat on, driven, stopped -- against the task's own capability sheet.
    from ppbench.v2 import operability as OP
    D["2.3"] = OP.evaluate_operability(design, task, lex, parts, occs, rows, joints, per_joint, res,
                                       is_reference=is_reference)

    # ------------------------------------------------------------ 3.1 decomposition (computed + judge)
    gran = 2 ** (-abs(math.log2(len(parts) / max(len(ref["parts"]), 1)))) if ref else None
    miou = None
    if rparts:
        dpts_all = np.vstack([p.vertices for p in parts])
        dlo, dhi = dpts_all.min(0), dpts_all.max(0)
        dc = np.r_[(dlo[:2] + dhi[:2]) / 2, dlo[2]]
        ddiag = max(float(np.linalg.norm(dhi - dlo)), 1e-9)
        rall = np.vstack([p.vertices for p in rparts])
        rlo, rhi = rall.min(0), rall.max(0)
        rc = np.r_[(rlo[:2] + rhi[:2]) / 2, rlo[2]]
        rdiag = max(float(np.linalg.norm(rhi - rlo)), 1e-9)
        dsets = []
        for p in parts:
            o = voxel.voxelize((p.vertices - dc) / ddiag, p.faces, PART_IOU_RES_M)
            dsets.append(set(map(tuple, np.argwhere(o.grid) + o.lo)))
        ious = []
        for rp in rparts:
            o = voxel.voxelize((rp.vertices - rc) / rdiag, rp.faces, PART_IOU_RES_M)
            rs = set(map(tuple, np.argwhere(o.grid) + o.lo))
            best = max((len(rs & ds) / max(len(rs | ds), 1) for ds in dsets), default=0.0)
            ious.append((best, len(rs)))
        miou = float(sum(i * n for i, n in ious) / max(sum(n for _, n in ious), 1)) if ious else None
    consol = 0
    for r in nonjoint:
        a, b = parts[r["i"]], parts[r["k"]]
        if canon[a.id] and canon[a.id] == canon[b.id] and a.material and a.material == b.material:
            consol += 1
    s31 = _mean([gran, miou])
    D["3.1"] = _dim("computed+adjudicated", "pass" if s31 is not None else "skipped", s31, {
        "granularity_score": gran, "n_parts": len(parts),
        "reference_n_parts": len(ref["parts"]) if ref else None,
        "part_correspondence_iou": miou, "part_iou_res_m": PART_IOU_RES_M,
        "consolidation_violations": consol},
        ["computed half only; the decomposition judge is attached by the report step",
         "part correspondence is measured on the scale-normalised alignment, like 2.1"])

    # ------------------------------------------------------------ 3.2 aesthetics (computed half)
    sym, sym_ang = symmetry_iou(occs)
    V = np.vstack([p.vertices for p in parts])
    e = V.max(0) - V.min(0)
    ext = {"width": float(max(e[0], e[1])), "depth": float(min(e[0], e[1])), "height": float(e[2])}
    prop = None
    if ref:
        rb = ref["bounds"]
        rext = {"width": max(rb[1][0] - rb[0][0], rb[1][1] - rb[0][1]),
                "depth": min(rb[1][0] - rb[0][0], rb[1][1] - rb[0][1]),
                "height": rb[1][2] - rb[0][2]}
        prop = 2 ** (-np.mean([abs(math.log2(max(ext["height"], 1e-6) / max(ext["width"], 1e-6) / (rext["height"] / rext["width"]))),
                               abs(math.log2(max(ext["depth"], 1e-6) / max(ext["width"], 1e-6) / (rext["depth"] / rext["width"])))]))
        prop = float(prop)
    D["3.2"] = _dim("computed+adjudicated", "pass", _mean([sym, prop]), {
        "symmetry_iou": sym, "symmetry_plane_normal_deg": sym_ang, "proportion_score": prop,
        "extent_m": ext}, ["image similarity and the VLM judge are attached by the report step"])

    # 3.3 creativity is not scored in v3

    # ------------------------------------------------------------ 4.1 material
    if not any(p.material for p in parts):
        # nothing carries a material: measure nothing rather than score a zero. How often this
        # happens is itself a result, reported per system as material coverage across the run.
        D["4.1"] = _dim("computed", "skipped", None,
                        {"declared": bool(_declares(design, "materials")), "n_parts_with_material": 0},
                        ["no part carries a material"])
    else:
        res_m = [materials.resolve(p.material) for p in parts]
        cov = float(sum(v for v, r in zip(vols, res_m) if r) / max(vols.sum(), 1e-12))
        groups_m = []
        for a in task.attributes:
            if a.get("kind") == "material":
                groups_m += [g for g in materials.claim_classes(a["statement"]) if g not in groups_m]
        dclasses = {r[1] for r in res_m if r}
        claim_cov = sum(1 for g in groups_m if g & dclasses) / len(groups_m) if groups_m else None
        masses = [(r[2] if r else 0.0) * v for r, v in zip(res_m, vols)]
        D["4.1"] = _dim("declared+computed", "pass" if cov > 0.99 else ("fail" if cov == 0 else "degraded"),
                        _mean([cov, claim_cov]), {
                            "coverage_volume_frac": cov, "claim_class_coverage": claim_cov,
                            "claimed_class_groups": [sorted(g) for g in groups_m],
                            "design_classes": sorted(dclasses), "total_mass_kg": float(sum(masses))},
                        ["densities are Artiverse medians; volumes of open meshes are under-counted",
                         "the population prior and the per-task load case of v2 are dropped as task-specific"])

    # ------------------------------------------------------------ 4.2 assembly sequence (unchanged, generic)
    seq = design.sequence
    zmin = float(min(p.vertices[:, 2].min() for p in parts))
    if not seq:
        D["4.2"] = _dim("computed", "fail" if _declares(design, "sequence") else "skipped",
                        0.0 if _declares(design, "sequence") else None, {"declared": False},
                        ["no assembly sequence declared"])
    else:
        order = [s["part"] for s in seq if s["part"] in idx]
        coverage42 = len(set(order)) / len(parts)
        dirs = [np.array(v, float) for v in ([0, 0, -1], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])]
        feas, feas_decl, supported, stable = [], [], [], []
        placed = []
        grounded = lambda i: parts[i].vertices[:, 2].min() <= zmin + 0.01
        pair_set = {(r["i"], r["k"]) for r in rows} | {(r["k"], r["i"]) for r in rows}
        for pid in order:
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
            comps_p = [[x for x in c if x in placed] for c in analysis.components(len(parts), sub_edges)]
            comps_p = [c for c in comps_p if c]
            supported.append(float(not [c for c in comps_p if not any(grounded(x) for x in c)]))
            gcomp = [x for c in comps_p if any(grounded(x) for x in c) for x in c]
            stable.append(float(analysis.stability([occs[x] for x in gcomp])["critical_tilt_deg"] > 0) if gcomp else 0.0)
        D["4.2"] = _dim("computed", "pass" if coverage42 == 1 else "degraded",
                        float(np.mean([coverage42, np.mean(feas), np.mean(supported), np.mean(stable)])), {
                            "coverage": coverage42, "insertion_feasible_frac": float(np.mean(feas)),
                            "declared_direction_feasible_frac": float(np.mean(feas_decl)) if feas_decl else None,
                            "first_infeasible_step": next((k for k, v in enumerate(feas) if not v), None),
                            "supported_prefix_frac": float(np.mean(supported)),
                            "stable_prefix_frac": float(np.mean(stable)), "n_steps": len(order)},
                        ["insertion paths are straight translations along the declared direction or the six axes"])

    # ------------------------------------------------------------ 4.3 functional completeness (unmerged)
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
        conns = []
        for cn in F.get("connections", []):
            A = {p.id for p in parts if lex.satisfies(cn["from"], canon[p.id], broad=False)}
            B = {p.id for p in parts if lex.satisfies(cn["to"], canon[p.id], broad=False)}
            conns.append(path(A, B, cn.get("carries") == "motion") if A and B else 0.0)
        subs_rows.append({"id": F["id"], "subsystem": F["subsystem"],
                          "presence": float(np.mean(pres)) if pres else None,
                          "connection": float(np.mean(conns)) if conns else None})
    if not _declares(design, "roles") and not any(p.role for p in parts):
        D["4.3"] = _dim("computed", "skipped", None, {"declared": False, "subsystems": subs_rows},
                        ["no roles: subsystems cannot be located"])
    else:
        D["4.3"] = _dim("computed", "pass", None, {
            "presence": _mean([s["presence"] for s in subs_rows]),
            "connection": _mean([s["connection"] for s in subs_rows]),
            "subsystems": subs_rows},
            ["presence and connection are reported separately, never merged"])

    # ------------------------------------------------------------ headline
    crit = [k for k in ("1.1", "1.2", "2.1", "2.2", "4.1", "4.2") if D.get(k, {}).get("status") == "fail"]
    scored = [v["score"] for v in D.values() if v.get("score") is not None]
    rec["headline"] = {"critical_fail": crit, "buildable": not any(k in crit for k in ("1.1", "1.2")),
                       "overall": float(np.mean(scored)) if scored else None,
                       "n_scored_dims": len(scored), "gate_pending": True}
    return rec
