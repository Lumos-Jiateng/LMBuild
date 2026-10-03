"""A.3 kinematics: does the claimed motion work, on the part that is the claimed component?

For one claim (e.g. "wheels rotate about the axle centreline") and one instance of its moving component
in a design, every declared non-fixed joint that carries that instance is scored, and the best counts:

    q = type_ok * axis_ok * offset_ok * range_ok * free_ok * moveset_ok

A product, not a mean: a wheel joint about the wrong axis is not a working wheel, however complete its
declaration is. What each factor requires comes from the real instances that realise the same claim
(afford/calib.expectation), never from a per-object rule:

* type_ok     the joint type is one real instances use (1), or a compatible one (cylindrical for a
              revolute or prismatic claim, 0.75), else 0;
* axis_ok     the axis matches the relation the real joints have to their moving part -- the part's own
              symmetry axis (wheels, turntables, knobs), or a fixed principal axis of it (a door's long
              edge), and vertical/horizontal when every real joint is -- 1 within 10 deg, 0 past 30;
* offset_ok   the axis line runs through the part (wheels) or along its boundary (doors, lids), as the
              real joints do; an axis that misses the part altogether scores 0;
* range_ok    the declared limits allow the range real instances have (0.8 x their lower quartile, or a
              full turn when real instances turn freely);
* free_ok     the part actually moves through that range: the fraction of poses at which it enters the
              static rest of the object by no more than 2% of its own volume, at a voxel size of 1/150 of
              the object (4 mm cells were coarser than a LEGO wheel); a part already buried in the static
              body at rest (over 20% of its volume, not counting the shaft of a rotation's own parent)
              cannot move and scores 0;
* moveset_ok  the joint moves the part without dragging along what must stay (the claim's
              relative_to component, or more of the object than real instances move).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ppbench.v2 import voxel
from ppbench.v2.afford import core
from ppbench.v2.design import axis_angle

ANGLE_OK, ANGLE_ZERO = 10.0, 30.0
BLOCK_FRAC = 0.02
EMBED_FRAC = 0.20
ROT_SAMPLES_FULL = 12
SAMPLES = 8


def moving_set(d, j, contact=None):
    """Parts that travel with j's child: walk the declared joint graph out of the child, never through the
    parent; parts no joint mentions follow by contact, the static side claiming contested ones first."""
    ids = {p.id for p in d.parts}
    adj = defaultdict(set)
    for o in d.joints:
        if o.id == j.id or o.parent not in ids or o.child not in ids:
            continue
        adj[o.parent].add(o.child)
        adj[o.child].add(o.parent)
    ms, st = {j.child}, [j.child]
    while st:
        u = st.pop()
        for v in adj[u]:
            if v != j.parent and v not in ms:
                ms.add(v)
                st.append(v)
    if contact:
        declared = {x for o in d.joints for x in (o.parent, o.child)} & ids
        loose = ids - declared

        def reach(seed):
            seen, stk = set(), list(seed)
            while stk:
                u = stk.pop()
                for v in contact.get(u, ()):
                    if v in loose and v not in seen:
                        seen.add(v)
                        stk.append(v)
            return seen
        ms |= reach(ms) - reach((ids - ms) | {j.parent})
    return ms & ids


def axis_offset(P, c, a, o):
    """Distance from the part's centre to the joint axis line, as a fraction of the part's half-extent in that
    same direction: 0 = the axis runs through the middle (a wheel), 1 = along the part's boundary (a door's edge).
    Until 2026-09-22 this divided by half the part's *longest* extent, so a door hinged on its vertical edge
    (0.4 m from a 2 m-tall leaf's centre) read 0.4 -- mid-way -- instead of 1.0."""
    v = c - o
    perp = v - (v @ a) * a
    dist = float(np.linalg.norm(perp))
    if dist < 1e-9:
        return 0.0
    u = perp / dist
    proj = (P - c) @ u
    half = max(float(np.abs(proj).max()), 1e-9)
    return dist / half


def moveset_score(mf, mf_max):
    """1 up to the moving fraction real instances allow, then linearly to 0 at the whole object. A hard cliff at
    the limit zeroed a correct drawer whose solid box happened to be 61% of the voxels against a 60% cap."""
    if mf <= mf_max:
        return 1.0
    return float(np.clip(1.0 - (mf - mf_max) / max(1.0 - mf_max, 1e-9), 0.0, 1.0))


def _moving_extent(d, ms, a, P):
    """Extent along the slide axis of everything the joint moves: a drawer is its front, box and handle together.
    Measuring the one labelled part made a real drawer's 2 cm front read as travelling 20x its size."""
    byid = {p.id: p for p in d.parts}
    V = [byid[x].vertices for x in ms if x in byid]
    if not V:
        return float(np.ptp(P @ a))
    return float(np.ptp(np.vstack(V) @ a))


def _vol_proxy(p):
    lo, hi = p.vertices.min(0), p.vertices.max(0)
    return float(np.prod(np.maximum(hi - lo, 1e-6)))


def relation(d, j, inst, ms, scene=None):
    """How joint j relates to the moving instance: which of its axes, where the line sits, how far it goes."""
    byid = {p.id: p for p in d.parts}
    parts = [byid[x] for x in inst if x in byid]
    if not parts or j.axis is None:
        return None
    P = core.sample(parts, 2000)
    if len(P) < 20:
        return None
    axes, ext, c = core.pca_frame(P)
    rnd, rk = core.roundness(P, axes)
    a = np.asarray(j.axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    cos = np.abs(axes @ a)
    o = np.asarray(j.origin if j.origin is not None else c, float)
    off = axis_offset(P, c, a, o)
    span, travel = None, None
    if j.type in ("continuous",) or (j.type == "revolute" and j.limits is None):
        span = 2 * math.pi
    elif j.type == "revolute":
        span = abs(float(j.limits[1]) - float(j.limits[0]))
    if j.type in ("prismatic", "cylindrical") and j.limits is not None:
        along = _moving_extent(d, ms, a, P)
        travel = abs(float(j.limits[1]) - float(j.limits[0])) / max(along, 1e-9)
    if scene is not None:
        mf = scene.volume_frac(ms)
    else:
        tot = sum(_vol_proxy(p) for p in d.parts)
        mf = sum(_vol_proxy(byid[x]) for x in ms if x in byid) / max(tot, 1e-12)
    return {"type": j.type, "pca_index": int(np.argmax(cos)), "axis_z": float(abs(a[2])),
            "sym_aligned": (bool(abs(a @ axes[rk]) >= math.cos(math.radians(20)))) if rnd >= 0.6 else None,
            "roundness": rnd, "offset": off, "span": span, "travel_ratio": travel, "moving_frac": mf}


def _angle_score(err):
    return float(np.clip((ANGLE_ZERO - err) / (ANGLE_ZERO - ANGLE_OK), 0.0, 1.0))


def _err(a, b):
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))


class Scene:
    """Voxel occupancies of a design at a size relative to the object, built lazily and shared by all
    joints of the design."""

    def __init__(self, d):
        self.d = d
        V = np.vstack([p.vertices for p in d.parts])
        self.diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        self.res = float(np.clip(self.diag / 150, 0.0005, 0.004))
        self._occ = {}
        self.byid = {p.id: p for p in d.parts}

    def occ(self, pid):
        if pid not in self._occ:
            p = self.byid[pid]
            try:
                self._occ[pid] = voxel.voxelize(p.vertices, p.faces, self.res)
            except Exception:
                self._occ[pid] = None
        return self._occ[pid]

    def volume_frac(self, ids):
        vol = {pid: (self.occ(pid).count if self.occ(pid) is not None else 0) for pid in self.byid}
        return sum(vol[x] for x in ids if x in vol) / max(sum(vol.values()), 1)

    def contact(self):
        """Touching pairs (one-cell test), for parts no joint mentions."""
        if hasattr(self, "_contact"):
            return self._contact
        ids = list(self.byid)
        occs = [self.occ(i) for i in ids]
        out = defaultdict(set)
        lo = [o.lo if o is not None else None for o in occs]
        hi = [o.hi if o is not None else None for o in occs]
        for a in range(len(ids)):
            if occs[a] is None:
                continue
            for b in range(a + 1, len(ids)):
                if occs[b] is None:
                    continue
                if np.any(lo[a] - 1 > hi[b]) or np.any(lo[b] - 1 > hi[a]):
                    continue
                if voxel.touching(occs[a], occs[b]):
                    out[ids[a]].add(ids[b])
                    out[ids[b]].add(ids[a])
        self._contact = out
        return out


def _overlap(scene, moved_occs, static_ids):
    tot = 0
    for mo in moved_occs:
        if mo is None:
            continue
        for s in static_ids:
            so = scene.occ(s)
            if so is None or np.any(mo.lo > so.hi) or np.any(so.lo > mo.hi):
                continue
            tot += voxel.overlap_cells(mo, so, deep=True)
    return tot


def _overlap_off_axis(scene, moved_occs, static_ids, org, ax, r_ax):
    """Deep-overlap cells farther than r_ax from the rotation axis (the shaft region is exempt)."""
    tot = 0
    for mo in moved_occs:
        if mo is None:
            continue
        for s in static_ids:
            so = scene.occ(s)
            if so is None:
                continue
            sl = voxel._slices(mo, so)
            if sl is None:
                continue
            inter = mo.core[sl[0]] & so.core[sl[1]]
            if not inter.any():
                continue
            lo = np.maximum(mo.lo, so.lo)
            idx = np.argwhere(inter) + lo
            P = (idx + 0.5) * scene.res - org
            dist = np.linalg.norm(P - np.outer(P @ ax, ax), axis=1)
            tot += int((dist > r_ax).sum())
    return tot


def sweep(scene, j, ms, motion, amount, rot_full):
    """Fraction of poses over the required motion that stay collision-free, and whether the part is buried."""
    d = scene.d
    moving = [scene.byid[x] for x in ms]
    static = [p.id for p in d.parts if p.id not in ms]
    rest_occ = [scene.occ(p.id) for p in moving]
    mcells = sum(o.count for o in rest_occ if o is not None)
    if mcells == 0 or j.axis is None:
        return 0.0, True, {}
    ax = np.asarray(j.axis, float)
    ax = ax / max(np.linalg.norm(ax), 1e-12)
    org = np.asarray(j.origin if j.origin is not None else np.vstack([p.vertices for p in moving]).mean(0), float)
    rest_all = _overlap(scene, rest_occ, static)
    if motion == "rotate":   # a shaft through the hub is how a rotating part is held: exempt the region near the axis
        V = np.vstack([p.vertices for p in moving]) - org
        radial = np.linalg.norm(V - np.outer(V @ ax, ax), axis=1)
        rest_excl = _overlap_off_axis(scene, rest_occ, static, org, ax, 0.35 * float(radial.max()))
    else:
        rest_excl = rest_all
    buried = rest_excl > EMBED_FRAC * mcells
    lim = j.limits
    if motion == "rotate":
        if rot_full:
            poses = list(np.linspace(0, 2 * math.pi, ROT_SAMPLES_FULL, endpoint=False)[1:])
        else:
            poses = _interval(amount, lim, SAMPLES)
    else:
        poses = _interval(amount, lim, SAMPLES)
    free = 0
    first_block = None
    for val in poses:
        occs = []
        for p in moving:
            if motion == "rotate":
                v2 = (p.vertices - org) @ axis_angle(ax, val).T + org
            else:
                v2 = p.vertices + ax * val
            try:
                occs.append(voxel.voxelize(v2, p.faces, scene.res))
            except Exception:
                occs.append(None)
        new = _overlap(scene, occs, static) - rest_all
        if new <= max(2, BLOCK_FRAC * mcells):
            free += 1
        elif first_block is None:
            first_block = round(float(val), 3)
    frac = free / max(len(poses), 1)
    return frac, bool(buried), {"rest_overlap_frac": rest_excl / mcells, "first_block": first_block,
                                "n_poses": len(poses), "res_m": scene.res}


def _interval(amount, lim, n):
    """Poses covering `amount` of motion from rest, inside the declared limits when there are any."""
    if lim is None:
        a, b = 0.0, amount
    else:
        lo, hi = sorted((float(lim[0]), float(lim[1])))
        if hi >= amount:
            a, b = 0.0, amount
        elif -lo >= amount:
            a, b = -amount, 0.0
        else:
            a, b = lo, hi
    return [x for x in np.linspace(a, b, n + 1) if abs(x) > 1e-9][:n]


def joint_quality(scene, j, inst, expect, motion, relative_parts, contact=None):
    d = scene.d
    ms = moving_set(d, j, contact)
    if not (set(inst) & ms):
        return None
    types = set(expect.get("types") or [])
    compat = {"rotate": {"revolute", "continuous"}, "slide": {"prismatic"}, "ball": {"ball"}}[motion]
    if j.type in types or j.type in compat:
        type_ok = 1.0
    elif j.type == "cylindrical" and motion in ("rotate", "slide"):
        type_ok = 0.75
    else:
        type_ok = 0.0
    parts = [scene.byid[x] for x in inst]
    P = core.sample(parts, 2000)
    axes, ext, c = core.pca_frame(P)
    rnd, rk = core.roundness(P, axes)
    if j.axis is None:
        return {"joint": j.id, "q": 0.0, "why": "no axis"}
    a = np.asarray(j.axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    axis_ok = 1.0
    ax_rule = expect.get("axis")
    if ax_rule == "symmetry":
        axis_ok *= _angle_score(_err(a, axes[rk]))
    elif ax_rule and ax_rule.startswith("pca"):
        axis_ok *= _angle_score(_err(a, axes[int(ax_rule[3:])]))
    if expect.get("axis_dir") == "vertical":
        axis_ok *= _angle_score(math.degrees(math.acos(min(1.0, abs(a[2])))))
    elif expect.get("axis_dir") == "horizontal":
        axis_ok *= _angle_score(math.degrees(math.asin(min(1.0, abs(a[2])))))
    o = np.asarray(j.origin if j.origin is not None else c, float)
    off = axis_offset(P, c, a, o)
    if off > 1.5:
        offset_ok = 0.0
    elif expect.get("offset") == "centre":
        offset_ok = float(np.clip((0.7 - off) / 0.4, 0, 1))
    elif expect.get("offset") == "edge":
        offset_ok = float(np.clip((off - 0.2) / 0.3, 0, 1))
    else:
        offset_ok = 1.0
    # range the claim needs, in the joint's own units
    rot_full = False
    if motion == "rotate":
        need = float(expect.get("span") or math.radians(30))
        rot_full = need >= 2 * math.pi - 1e-3
        if j.type == "continuous" or (j.type in ("revolute", "cylindrical") and j.limits is None):
            have = 2 * math.pi
        elif j.limits is not None and j.type == "revolute":
            have = abs(float(j.limits[1]) - float(j.limits[0]))
        else:
            have = 0.0 if j.type == "prismatic" else 2 * math.pi
        range_ok = float(min(1.0, have / max(need, 1e-9)))
        amount = need
    elif motion == "slide":
        along = _moving_extent(d, ms, a, P)
        need = float(expect.get("travel_ratio") or 0.2) * along
        have = abs(float(j.limits[1]) - float(j.limits[0])) if (j.limits is not None and j.type in ("prismatic", "cylindrical")) else 0.0
        range_ok = float(min(1.0, have / max(need, 1e-9)))
        amount = need
    else:
        range_ok, amount = 1.0, 0.0
    mf = scene.volume_frac(ms)
    moveset_ok = 1.0
    why = []
    if relative_parts and (set(relative_parts) & ms):
        moveset_ok = 0.0
        why.append("moves the part it should move relative to")
    if len(ms) >= len(d.parts):
        moveset_ok = 0.0
        why.append("moves the whole object")
    else:
        ms_soft = moveset_score(mf, float(expect.get("moving_frac_max") or 0.6))
        if ms_soft < 1.0:
            why.append(f"moves {mf:.0%} of the object")
        moveset_ok = min(moveset_ok, ms_soft)
    q0 = type_ok * axis_ok * offset_ok * range_ok * moveset_ok
    free, buried, sw = (1.0, False, {})
    if q0 > 0 and motion in ("rotate", "slide"):
        free, buried, sw = sweep(scene, j, ms, motion, amount, rot_full)
    free_ok = 0.0 if buried else free
    q = q0 * free_ok
    return {"joint": j.id, "type": j.type, "q": q, "type_ok": type_ok, "axis_ok": axis_ok, "offset_ok": offset_ok,
            "offset": off, "range_ok": range_ok, "free_ok": free_ok, "buried": buried, "moveset_ok": moveset_ok,
            "moving_frac": mf, "n_moving_parts": len(ms), "sweep": sw, "why": why}
