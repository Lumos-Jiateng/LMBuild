"""2.3 Operability: can the object actually be used for what it is for?

A chair that is sittable, a car that can be stopped. The question is semantic, so the
checkpoints are per object -- but nothing here is hand-written per object twice:

  capability            what a user can do with it ("can be stopped")
    requirement         one line, each citing the claim it comes from
      predicate         one of the nine below, implemented once, returns [0, 1]

The sheets live in ppbench/v2/opsheets/<task>.json. Their mechanical part is generated
from the task's own cited claims (functional subsystems F, kinematics K, required parts
P) by `generate`; the human-interaction part is authored in opsheets/_human.json and
merged in, so the whole sheet stays reproducible and every requirement carries a claim id.

    capability = min(its requirements)      a brake that is not connected is not a brake
    2.3        = mean(capabilities)

What this verifies is that the design *affords* the function: the part is there, it is
placed where it has to be, the force or motion path reaches it, it can move, a person
fits and can reach it. Whether the friction is enough or the engine would fire stays
Declared and out of scope, exactly as the taxonomy says.

    python -m ppbench.v2.operability generate            # rebuild every sheet
    python -m ppbench.v2.operability validate            # sheets are well formed
    python -m ppbench.v2.operability ceiling             # the references score their own sheets
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from ppbench.v2 import analysis, voxel
from ppbench.v2.design import Design, axis_angle
from ppbench.v2.evaluate import _dim, _mean, axis_requirement, axis_score, type_score
from ppbench.v2.lexicon import Lexicon
from ppbench.v2.task import RESULTS, Task

SPEC_VERSION = "op-v1"
SHEETS = Path(__file__).parent / "opsheets"

# One P50 adult, in metres, shared by every task. Not per object: a person is a person.
HUMAN = {
    "seated_torso": (0.45, 0.35, 0.65),      # w, d, h above the seat surface
    "seated_legs": (0.45, 0.45, 0.40),       # in front of the seat, down to the floor
    "standing": (0.50, 0.35, 1.70),
    "hand": (0.10, 0.10, 0.10),
    "arm_reach_m": 0.70,
    "shoulder_above_seat_m": 0.60,
    "standing_shoulder_m": 1.40,
    "grip_min_m": 0.015,
    "grip_max_m": 0.070,
}
CLEAR_SAMPLES = 10                # per axis, so 1000 points per clearance box
COLOCATED_DECAY = 2.0             # gap of decay x the allowance scores 0
REACH_DECAY = 2.0
MIN_SWEEP_DEFAULT = 0.5
CEILING_MIN = 0.5                 # a requirement the task's own reference object scores below this is
                                  # not a checkpoint: it is disabled for every design and reported
CEILING_EXEMPT = {"part_present"}  # ... except presence, which is handled differently. A reference that does
                                  # not segment a magnetron is a gap in the reference's segmentation, not proof
                                  # the claim is wrong -- the claim is cited and a design that declares the part
                                  # does satisfy it. But such a part cannot GATE a capability: with min(), one
                                  # invisible internal would zero the capability for every design and for the
                                  # real object too. So it becomes Declared: reported as coverage, never as
                                  # correctness, exactly as the taxonomy's scoping rule says. Geometric and
                                  # relational checks the real object fails are simply mis-specified: disabled.
CALIB = SHEETS / "_calibration.json"


# ---------------------------------------------------------------- context

class OpCtx:
    """Everything the predicates need, built once per design."""

    def __init__(self, design: Design, task: Task, lex: Lexicon, parts, occs, rows, joints, per_joint, res):
        self.design, self.task, self.lex = design, task, lex
        self.parts, self.occs, self.rows, self.joints, self.res = parts, occs, rows, joints, res
        self.per_joint = {q["joint"]: q for q in (per_joint or [])}
        self.idx = {p.id: i for i, p in enumerate(parts)}
        self.canon = {p.id: lex.canon(p.role) for p in parts}
        self.moving = [j for j in joints if j.type != "fixed" and j.parent in self.idx and j.child in self.idx]
        from ppbench.v2.evaluate import rigid_groups
        self.groups = rigid_groups(parts, rows, joints)
        V = np.vstack([p.vertices for p in parts])
        self.lo, self.hi = V.min(0), V.max(0)
        self.z0 = float(self.lo[2])
        self.centre_xy = (self.lo[:2] + self.hi[:2]) / 2
        self.diag = float(np.linalg.norm(self.hi - self.lo))
        self._trees = {}
        adj = {p.id: set() for p in parts}
        for r in rows:
            adj[r["a"]].add(r["b"])
            adj[r["b"]].add(r["a"])
        for j in joints:
            if j.parent in self.idx and j.child in self.idx:
                adj[j.parent].add(j.child)
                adj[j.child].add(j.parent)
        self.adj = adj
        self.moving_pairs = {frozenset((j.parent, j.child)) for j in self.moving}

    # -- roles
    def roles(self, phrase, broad=True):
        """Indices of the parts that realise a claim phrase."""
        return [i for i, p in enumerate(self.parts)
                if self.lex.satisfies(phrase, self.canon[p.id], broad=broad)]

    def names(self, ids):
        return [self.parts[i].id for i in ids][:8]

    # -- geometry
    def tree(self, i):
        if i not in self._trees:
            from scipy.spatial import cKDTree
            pts = self.occs[i].points
            if len(pts) == 0:
                pts = self.parts[i].vertices
            self._trees[i] = cKDTree(pts)
        return self._trees[i]

    def gap(self, i, k):
        """Smallest surface-to-surface distance between two parts, 0 when they touch."""
        d, _ = self.tree(i).query(self.occs[k].points if len(self.occs[k].points) else self.parts[k].vertices)
        return float(d.min()) if len(d) else float("inf")

    def occupied(self, pts):
        """Boolean per point: inside any part."""
        out = np.zeros(len(pts), bool)
        cells = np.floor(np.asarray(pts) / self.res).astype(np.int64)
        for o in self.occs:
            rel = cells - o.lo
            ok = np.all((rel >= 0) & (rel < np.array(o.grid.shape)), axis=1) & ~out
            if not ok.any():
                continue
            r = rel[ok]
            out[np.where(ok)[0][o.grid[r[:, 0], r[:, 1], r[:, 2]]]] = True
        return out

    def free_fraction(self, lo, hi):
        """Fraction of a box that no part occupies, and the box's volume."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if np.any(hi <= lo):
            return 0.0, 0.0
        g = [np.linspace(lo[a], hi[a], CLEAR_SAMPLES) for a in range(3)]
        pts = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        free = float((~self.occupied(pts)).mean())
        return free, float(np.prod(hi - lo))

    def top_centre(self, ids):
        """The middle of the top face of a set of parts: where a person or a load sits."""
        V = np.vstack([self.parts[i].vertices for i in ids])
        return np.r_[(V.min(0)[:2] + V.max(0)[:2]) / 2, V.max(0)[2]]

    def outward(self, ids):
        """Unit XY direction from the object's centre towards these parts: the 'front' of a seat,
        the open side of a cabinet. Derived from the design, never declared."""
        V = np.vstack([self.parts[i].vertices for i in ids])
        c = (V.min(0)[:2] + V.max(0)[:2]) / 2 - self.centre_xy
        n = float(np.linalg.norm(c))
        return (c / n) if n > 1e-6 else np.array([0.0, -1.0])

    def path(self, A, B, need_motion, max_hops=4):
        """1.0 a path of at most max_hops edges (through a moving joint when motion is required),
        0.5 a path that carries no motion where motion is needed, 0.0 none."""
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
                for v in self.adj[u]:
                    m2 = mot or frozenset((u, v)) in self.moving_pairs
                    if (v, m2) not in seen:
                        seen.add((v, m2))
                        q.append((v, h + 1, m2))
        return best


# ---------------------------------------------------------------- predicates
# Each returns (score in [0,1], detail dict). None of them knows what object it is looking at.

def p_part_present(ctx, r):
    ids = ctx.roles(r["role"])
    need = int(r.get("min_count", 1))
    return min(len(ids), need) / need, {"n": len(ids), "need": need, "parts": ctx.names(ids)}


def p_ground_contact(ctx, r):
    ids = ctx.roles(r["role"])
    need = int(r.get("min_clusters", 1))
    band = float(r.get("band_m", 0.02))
    cell = float(r.get("cell_m", 0.05))
    cells = set()
    for i in ids:
        v = ctx.parts[i].vertices
        low = v[v[:, 2] <= ctx.z0 + band][:, :2]
        cells |= set(map(tuple, np.floor(low / cell).astype(int)))
    seen, n = set(), 0
    for c in cells:
        if c in seen:
            continue
        n += 1
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
    return min(n, need) / need, {"clusters": n, "need": need, "parts": ctx.names(ids)}


def p_joint_enables(ctx, r):
    """A joint exists that moves this role the way the claim says, and it can actually move."""
    ids = set(ctx.parts[i].id for i in ctx.roles(r["role"], broad=False)) or \
          set(ctx.parts[i].id for i in ctx.roles(r["role"]))
    if not ids:
        return 0.0, {"reason": "no part with this role"}
    req_axis = axis_requirement(r.get("axis"))
    want = r.get("type")
    best, detail = 0.0, {"reason": "no joint moves this role"}
    for j in ctx.moving:
        grp = set(ctx.groups.get(j.child, [j.child]))
        if not (ids & grp):
            continue
        q = ctx.per_joint.get(j.id, {})
        mob = q.get("mobility", 0.0)
        terms = [mob]
        if want:
            terms.append(type_score(j.type, want))
        if req_axis:
            terms.append(axis_score(j.axis, req_axis))
        s = float(np.mean(terms))
        if s > best:
            best, detail = s, {"joint": j.id, "type": j.type, "mobility": mob,
                               "type_score": type_score(j.type, want) if want else None,
                               "axis_score": axis_score(j.axis, req_axis) if req_axis else None}
    floor = float(r.get("min_sweep", MIN_SWEEP_DEFAULT))
    detail["min_sweep"] = floor
    return best, detail


def p_chain(ctx, r):
    A = {ctx.parts[i].id for i in ctx.roles(r["from"], broad=False)}
    B = {ctx.parts[i].id for i in ctx.roles(r["to"], broad=False)}
    if not A or not B:
        return 0.0, {"reason": "role missing", "from_n": len(A), "to_n": len(B)}
    s = ctx.path(A, B, r.get("carries") == "motion", int(r.get("max_hops", 4)))
    return s, {"from_n": len(A), "to_n": len(B), "carries": r.get("carries")}


def p_colocated(ctx, r):
    A, B = ctx.roles(r["a"]), ctx.roles(r["b"])
    if not A or not B:
        return 0.0, {"reason": "role missing", "a_n": len(A), "b_n": len(B)}
    allow = float(r.get("max_gap_m", 0.10))
    g = min(ctx.gap(i, k) for i in A for k in B)
    return float(max(0.0, 1 - max(0.0, g - allow) / (COLOCATED_DECAY * allow))), {"gap_m": g, "allow_m": allow}


def p_clearance(ctx, r):
    """A person or a load has to fit: how much of the required free volume is actually free."""
    ids = ctx.roles(r["anchor"])
    if not ids:
        return 0.0, {"reason": "no anchor part"}
    box = r.get("box")
    w, d, h = HUMAN[box] if box in HUMAN else (float(r["w"]), float(r["d"]), float(r["h"]))
    where = r.get("placement", "above")
    c = ctx.top_centre(ids)
    if where == "above":
        lo = np.r_[c[:2] - [w / 2, d / 2], c[2] + 0.01]
        hi = lo + [w, d, h]
    elif where == "below":
        # legroom under a work surface, standing room under a canopy
        V = np.vstack([ctx.parts[i].vertices for i in ids])
        zt = float(V.min(0)[2])
        lo = np.r_[c[:2] - [w / 2, d / 2], max(ctx.z0, zt - h)]
        hi = np.r_[lo[:2] + [w, d], zt - 0.01]
    elif where == "in_front":
        o = ctx.outward(ids)
        base = np.r_[c[:2] + o * (d / 2 + 0.02), max(ctx.z0, c[2] - h)]
        lo = np.r_[base[:2] - [w / 2, d / 2], base[2]]
        hi = lo + [w, d, h]
    else:  # inside: the anchor's own bounding box, minus a wall margin
        V = np.vstack([ctx.parts[i].vertices for i in ids])
        m = float(r.get("margin_m", 0.02))
        lo, hi = V.min(0) + m, V.max(0) - m
    free, vol = ctx.free_fraction(lo, hi)
    need = float(r.get("min_free", 0.9))
    return float(min(free / need, 1.0)), {"free_frac": free, "need_free": need, "box_m3": vol,
                                          "placement": where, "anchor": ctx.names(ids)}


def p_reachable(ctx, r):
    """The control is within a person's reach from where the person is."""
    tgt = ctx.roles(r["target"])
    if not tgt:
        return 0.0, {"reason": "no target part"}
    frm = ctx.roles(r["from"]) if r.get("from") else []
    if frm:
        c = ctx.top_centre(frm)
        eye = np.r_[c[:2], c[2] + HUMAN["shoulder_above_seat_m"]]
    else:
        eye = np.r_[ctx.centre_xy, ctx.z0 + HUMAN["standing_shoulder_m"]]
    reach = float(r.get("reach_m", HUMAN["arm_reach_m"]))
    d = min(float(ctx.tree(i).query(eye[None, :])[0][0]) for i in tgt)
    return float(max(0.0, 1 - max(0.0, d - reach) / (REACH_DECAY * reach))), \
        {"distance_m": d, "reach_m": reach, "from": ctx.names(frm) or "standing", "target": ctx.names(tgt)}


def p_grip(ctx, r):
    """A handle a hand can close around: the two smaller principal extents of the part."""
    ids = ctx.roles(r["role"])
    if not ids:
        return 0.0, {"reason": "no part with this role"}
    lo_m = float(r.get("min_m", HUMAN["grip_min_m"]))
    hi_m = float(r.get("max_m", HUMAN["grip_max_m"]))
    best, detail = 0.0, {}
    for i in ids:
        V = ctx.parts[i].vertices
        P = V - V.mean(0)
        _, vecs = np.linalg.eigh(np.cov(P.T))
        ext = sorted(float(np.ptp(P @ vecs[:, a])) for a in range(3))[:2]   # the two thin directions
        s = float(np.mean([1.0 if lo_m <= e <= hi_m else max(0.0, 1 - abs(e - np.clip(e, lo_m, hi_m)) / hi_m)
                           for e in ext]))
        if s > best:
            best, detail = s, {"part": ctx.parts[i].id, "cross_section_m": [round(e, 4) for e in ext],
                               "band_m": [lo_m, hi_m]}
    return best, detail


def p_surface(ctx, r):
    """A usable upward surface: its projected area, and optionally the height it sits at."""
    ids = ctx.roles(r["role"])
    if not ids:
        return 0.0, {"reason": "no part with this role"}
    cols = set()
    top = -1e9
    for i in ids:
        o = ctx.occs[i]
        occ = np.argwhere(o.grid.any(2))
        cols |= set(map(tuple, occ + o.lo[:2]))
        top = max(top, float(ctx.parts[i].vertices[:, 2].max()))
    area = len(cols) * ctx.res ** 2
    need = float(r.get("min_area_m2", 0.05))
    s = min(area / need, 1.0)
    detail = {"area_m2": area, "need_m2": need, "top_z_m": top}
    rng = r.get("height_range_m")
    if rng:
        lo_h, hi_h = float(rng[0]), float(rng[1])
        h = top - ctx.z0
        span = max(hi_h - lo_h, 1e-6)
        s = min(s, float(max(0.0, 1 - max(0.0, max(lo_h - h, h - hi_h)) / span)))
        detail.update({"height_m": h, "height_range_m": [lo_h, hi_h]})
    return float(s), detail


PREDICATES = {"part_present": p_part_present, "ground_contact": p_ground_contact,
              "joint_enables": p_joint_enables, "chain": p_chain, "colocated": p_colocated,
              "clearance": p_clearance, "reachable": p_reachable, "grip": p_grip, "surface": p_surface}


# ---------------------------------------------------------------- scoring

def req_key(cap_id, r):
    """Stable identity of a requirement, so calibration survives a sheet regeneration."""
    return "|".join([cap_id, r["pred"]] + [f"{k}={r[k]}" for k in ROLE_KEYS if r.get(k)])


def load_calibration():
    return json.loads(CALIB.read_text()) if CALIB.exists() else {}


def sheet_path(task_id):
    return SHEETS / f"{task_id}.json"


def load_sheet(task_id):
    p = sheet_path(task_id)
    return json.loads(p.read_text()) if p.exists() else None


def evaluate_operability(design, task, lex, parts, occs, rows, joints, per_joint, res=voxel.RES,
                         is_reference=False, calibration=None):
    """The 2.3 record. Returns a _dim() dict; `skipped` when the task has no sheet."""
    sheet = load_sheet(task.id)
    if not sheet:
        return _dim("computed", "skipped", None, {}, ["no capability sheet for this task"])
    ctx = OpCtx(design, task, lex, parts, occs, rows, joints, per_joint, res)
    calib = (calibration if calibration is not None else load_calibration()).get(task.id, {}) if not is_reference else {}
    caps, notes, n_disabled, n_declared = [], [], 0, 0
    declared_scores = []
    for cap in sheet["capabilities"]:
        reqs = []
        for r in cap["requirements"]:
            fn = PREDICATES.get(r["pred"])
            if fn is None:
                notes.append(f"unknown predicate {r['pred']}")
                continue
            try:
                s, detail = fn(ctx, r)
            except Exception as e:                    # one bad requirement must not lose the design
                s, detail = 0.0, {"error": f"{type(e).__name__}: {e}"}
            ref = calib.get(req_key(cap["id"], r))
            row = {**{k: v for k, v in r.items() if k != "pred"}, "pred": r["pred"],
                   "score": float(s), "detail": detail}
            if ref is not None:
                row["reference_score"] = ref
                if ref < CEILING_MIN and r["pred"] not in CEILING_EXEMPT:
                    row["disabled_by_ceiling"] = True
                    n_disabled += 1
                elif ref < CEILING_MIN:
                    row["declared_only"] = True      # the reference cannot show it either: coverage, not a gate
                    n_declared += 1
            reqs.append(row)
        declared_scores += [x["score"] for x in reqs if x.get("declared_only")]
        live = [x for x in reqs if not x.get("disabled_by_ceiling") and not x.get("declared_only")]
        if not live:
            continue
        vals = [x["score"] for x in live]
        caps.append({"id": cap["id"], "name": cap["name"], "claims": cap.get("claims", []),
                     "score": float(min(vals)), "mean": float(np.mean(vals)),
                     "n_requirements": len(live), "n_disabled": sum(1 for x in reqs if x.get("disabled_by_ceiling")),
                     "n_declared_only": sum(1 for x in reqs if x.get("declared_only")),
                     "weakest": min(live, key=lambda x: x["score"])["pred"], "requirements": reqs})
    if not caps:
        return _dim("computed", "skipped", None, {"sheet": sheet.get("spec")}, ["the sheet has no usable capability"])
    score = float(np.mean([c["score"] for c in caps]))
    n_zero = sum(1 for c in caps if c["score"] == 0.0)
    return _dim("computed", "fail" if n_zero == len(caps) else ("degraded" if n_zero else "pass"), score, {
        "sheet_spec": sheet.get("spec"), "operator": sheet.get("operator"),
        # 5 tasks have no reference object, so nothing disables their unverifiable checkpoints:
        # they are scored strictly and must not be pooled with the calibrated 35 without saying so
        "calibrated": bool(calib), "n_requirements_disabled_by_ceiling": n_disabled,
        "n_requirements_declared_only": n_declared,
        "declared_part_coverage": float(np.mean(declared_scores)) if declared_scores else None,
        "n_capabilities": len(caps), "n_capabilities_zero": n_zero,
        "capability_scores": {c["id"]: round(c["score"], 3) for c in caps},
        "mean_of_means": float(np.mean([c["mean"] for c in caps])),
        "capabilities": caps},
        ["capability = min(its requirements): one broken requirement disables the capability",
         "affordance only: geometry, placement, path and motion. The internal physics stays declared"]
        + ([f"{n_disabled} requirement(s) disabled: the task's own reference object does not pass them"] if n_disabled else [])
        + ([f"{n_declared} part(s) the reference cannot show either are reported as declared_part_coverage, not scored"] if n_declared else [])
        + notes)


# ---------------------------------------------------------------- sheet generation

ROLE_KEYS = ("role", "a", "b", "from", "to", "anchor", "target")


def generate_sheet(task_id, human_blocks=None):
    """The mechanical half of a sheet, straight from the task's cited claims."""
    task = Task(task_id, snapshot=RESULTS / task_id / "task_snapshot_core.json")
    raw = task.raw
    caps = []
    ks = raw.get("required_kinematics", [])
    used_k = set()
    for f in raw.get("functional_subsystems", []):
        reqs = [{"pred": "part_present", "role": n, "claim": f["id"]} for n in f.get("parts", [])]
        for c in f.get("connections", []):
            reqs.append({"pred": "chain", "from": c["from"], "to": c["to"],
                         "carries": c.get("carries"), "claim": f["id"]})
        fparts = {n.lower() for n in f.get("parts", [])}
        for k in ks:
            if (k.get("moving_part") or "").lower() in fparts:
                reqs.append({"pred": "joint_enables", "role": k["moving_part"], "type": k.get("joint_type"),
                             "axis": k.get("axis"), "claim": k["id"]})
                used_k.add(k["id"])
        caps.append({"id": f["id"], "name": f["subsystem"], "function": f.get("function", ""),
                     "claims": [f["id"]], "requirements": reqs})
    left = [k for k in ks if k["id"] not in used_k]
    if left:
        caps.append({"id": "K", "name": "declared mechanisms move", "claims": [k["id"] for k in left],
                     "requirements": [{"pred": "joint_enables", "role": k["moving_part"], "type": k.get("joint_type"),
                                       "axis": k.get("axis"), "claim": k["id"]} for k in left]})
    covered = {n.lower() for f in raw.get("functional_subsystems", []) for n in f.get("parts", [])}
    orphan = [p for p in raw.get("required_parts", []) if p["part"].lower() not in covered]
    if orphan:
        caps.append({"id": "P", "name": "required subsystems the claims name but no subsystem covers",
                     "claims": [p["id"] for p in orphan],
                     "requirements": [{"pred": "part_present", "role": p["part"], "claim": p["id"]} for p in orphan]})
    caps += list((human_blocks or {}).get(task_id, []))
    # A claim can name something the object's own vocabulary has no word for: a tractor's F3
    # connects to an "implement" that is not part of the tractor, and a laptop's "central
    # processing unit (CPU)" is an internal nobody can see. Those requirements are dropped
    # and listed, never silently scored 0 against the design.
    lex = Lexicon(task_id, task)
    dropped = []
    for c in caps:
        keep = []
        for r in c["requirements"]:
            miss = [f"{k}={r[k]!r}" for k in ROLE_KEYS if r.get(k) and lex.canon(r[k]) is None]
            if miss:
                dropped.append({"capability": c["id"], "pred": r["pred"], "unresolved": miss,
                                "claim": r.get("claim")})
            else:
                keep.append(r)
        c["requirements"] = keep
    caps = [c for c in caps if c["requirements"]]
    return {"task": task_id, "spec": SPEC_VERSION, "operator": "adult_p50", "dropped_requirements": dropped,
            "generated_from": "functional_subsystems F, required_kinematics K, required_parts P, plus the authored human layer",
            "capabilities": caps}


def generate(tasks=None):
    human = json.loads((SHEETS / "_human.json").read_text()) if (SHEETS / "_human.json").exists() else {}
    SHEETS.mkdir(parents=True, exist_ok=True)
    ids = tasks or sorted(p.name for p in RESULTS.iterdir()
                          if (p / "task_snapshot_core.json").exists())
    for t in ids:
        try:
            s = generate_sheet(t, human)
        except Exception as e:
            print(f"{t:28s} SKIP {type(e).__name__}: {e}")
            continue
        sheet_path(t).write_text(json.dumps(s, indent=1))
        n_req = sum(len(c["requirements"]) for c in s["capabilities"])
        print(f"{t:28s} {len(s['capabilities']):2d} capabilities, {n_req:3d} requirements"
              f"{'  (+human)' if t in human else ''}")


def validate(tasks=None):
    """Every predicate known, every role resolvable against the task's vocabulary, every requirement cited."""
    ids = tasks or sorted(p.stem for p in SHEETS.glob("*.json") if not p.stem.startswith("_"))
    bad = 0
    for t in ids:
        s = load_sheet(t)
        task = Task(t, snapshot=RESULTS / t / "task_snapshot_core.json")
        lex = Lexicon(t, task)
        msgs = []
        for c in s["capabilities"]:
            if not c.get("requirements"):
                msgs.append(f"{c['id']} has no requirement")
            for r in c["requirements"]:
                if r["pred"] not in PREDICATES:
                    msgs.append(f"{c['id']} unknown predicate {r['pred']}")
                if not r.get("claim"):
                    msgs.append(f"{c['id']} requirement {r['pred']} cites no claim")
                for key in ("role", "a", "b", "from", "to", "anchor", "target"):
                    if key in r and r[key] and lex.canon(r[key]) is None:
                        msgs.append(f"{c['id']} {r['pred']}.{key}={r[key]!r} is not in the task vocabulary")
        bad += len(msgs)
        print(f"{t:28s} {len(s['capabilities']):2d} caps  " + ("ok" if not msgs else f"{len(msgs)} issue(s)"))
        for m in msgs[:8]:
            print(f"      {m}")
    print(f"\n{bad} issue(s) over {len(ids)} sheets")
    return bad


def calibrate(tasks=None, workers=12):
    """Score every requirement against the task's own reference object and store the result.

    A requirement the real object fails is not a checkpoint -- the claim names a part the
    reference never segments, or a motion its annotation does not carry. Those are disabled
    for every design and reported, instead of quietly scoring every design 0."""
    from concurrent.futures import ProcessPoolExecutor
    ids = tasks or sorted(p.stem for p in SHEETS.glob("*.json") if not p.stem.startswith("_"))
    out, skipped = {}, []
    with ProcessPoolExecutor(workers) as ex:
        for t, scores, err in ex.map(_calibrate_one, ids):
            if err:
                skipped.append((t, err))
                continue
            out[t] = scores
            low = sum(1 for v in scores.values() if v < CEILING_MIN)
            print(f"{t:28s} {len(scores):3d} requirements, {low:2d} below the ceiling")
    CALIB.write_text(json.dumps(out, indent=1, sort_keys=True))
    for t, err in skipped:
        print(f"{t:28s} no calibration ({err})")
    print(f"\ncalibration written for {len(out)} tasks -> {CALIB}")
    return out


def _calibrate_one(task_id):
    from ppbench.v2 import metrics_v3 as M
    from ppbench.v2.evaluate import reference_design
    try:
        task = Task(task_id, snapshot=RESULTS / task_id / "task_snapshot_core.json")
        rec = M.evaluate_v3(reference_design(task), task, M.anchors_v3(task), is_reference=True)
    except Exception as e:
        return task_id, {}, f"{type(e).__name__}: {e}"
    m = (rec["dims"].get("2.3") or {}).get("metrics") or {}
    scores = {}
    for c in m.get("capabilities", []):
        for r in c["requirements"]:
            scores[req_key(c["id"], r)] = round(float(r["score"]), 4)
    return task_id, scores, ""


def ceiling(tasks=None, res=voxel.RES):
    """The reference object scored against its own sheet. Anything it fails is a bad checkpoint."""
    from ppbench.v2 import metrics_v3 as M
    from ppbench.v2.evaluate import reference_design
    ids = tasks or sorted(p.stem for p in SHEETS.glob("*.json") if not p.stem.startswith("_"))
    rows = []
    for t in ids:
        try:
            task = Task(t, snapshot=RESULTS / t / "task_snapshot_core.json")
            d = reference_design(task)
            rec = M.evaluate_v3(d, task, M.anchors_v3(task), is_reference=True)
        except Exception as e:
            print(f"{t:28s} SKIP {type(e).__name__}: {e}")
            continue
        dim = rec["dims"].get("2.3") or {}
        m = dim.get("metrics") or {}
        rows.append((t, dim.get("score"), m))
        weak = sorted((c["score"], c["id"], c["name"], c["weakest"]) for c in m.get("capabilities", []))[:3]
        print(f"{t:28s} 2.3={dim.get('score') if dim.get('score') is None else round(dim['score'], 3)}  "
              f"caps={m.get('n_capabilities')}  zero={m.get('n_capabilities_zero')}")
        for s, cid, name, w in weak:
            if s < 0.999:
                print(f"      {cid} {name[:40]:40s} {s:.2f}  weakest: {w}")
    ok = [r for r in rows if isinstance(r[1], (int, float))]
    if ok:
        print(f"\nreference mean 2.3 = {np.mean([r[1] for r in ok]):.3f} over {len(ok)} tasks")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "validate", "calibrate", "ceiling"])
    ap.add_argument("--tasks", help="comma-separated task ids (default: all)")
    ns = ap.parse_args(argv)
    tasks = [t.strip() for t in ns.tasks.split(",")] if ns.tasks else None
    {"generate": lambda: generate(tasks), "validate": lambda: validate(tasks),
     "calibrate": lambda: calibrate(tasks), "ceiling": lambda: ceiling(tasks)}[ns.cmd]()


if __name__ == "__main__":
    main()
