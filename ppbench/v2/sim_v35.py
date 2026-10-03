"""Spec v3.5 P.3, simulated tier: does each claimed motion actually work in a physics engine (MuJoCo).

Generic by construction -- the wiki claims choose the tests, physics decides the outcome, nothing is per task:

  * bodies    parts connected by contact or a fixed joint are one solid (the same rule as P.1), except that
              contact never welds across a declared moving joint: what moves with the joint's child (its
              kinematic moving set) is its own body. Declared moving joints articulate; bodies not reachable
              from the base through joints are free and fall where gravity puts them. Masses are 1.3's: uniform
              density, an open mesh whose voxel fill failed weighed by its convex hull
  * settle    the object stands on a ground plane under gravity, every joint held (an object whose task is not
              free-standing is mounted: its base is fixed to the world); if the base tips past
              SETTLE_TILT_DEG or is thrown, nothing can be operated and every test scores 0
  * actuate   for each claimed motion K whose moving part is not a rolling element: the design joint that moves
              that role is driven through its range (declared limits, else a quarter turn / half the part's
              length, both directions tried) by a position servo, every other joint held. Score = fraction of
              the commanded travel achieved x the object stayed upright
  * roll      for each claimed motion K whose moving part is a rolling element (wheel, caster, roller, track):
              the wheel joints are released and a horizontal push of ROLL_PUSH x own weight acts on the base
              for ROLL_T s across the wheel axes, every hinge between a ground-touching body and the base
              released. Score = min(1, distance / (ROLL_FULL x the frictionless distance)) x upright

Collision shapes are each part's convex hull (a part turning on a floor-touching horizontal hinge: its cylinder of
revolution). A moving part collides with the body it hangs on (MuJoCo's parent filter is off), except a rolling
body with its carrier; body pairs that already interpenetrate in the design or in the engine at t = 0 never collide. A claimed motion is tested on the design joints between the
moving part and the claim's `relative_to` part (the base when that part is absent), of the claimed type. Mass is
uniform density: designs model doors and panels as solid slabs, so a declared density would make a steel door
outweigh the cabinet it hangs on (material realism is P.2's job; 1.3 uses the same convention).
"""
from __future__ import annotations

import math

import numpy as np

SERVO_W = 60.0                      # rad/s, natural frequency of every holding / driving servo
ROLL_FULL = 0.25                    # rolling this share of the frictionless distance is full credit
SETTLE_T = 0.6
SETTLE_TILT_DEG = 20.0
ACT_T = 1.5
HOLD_T = 0.4
UPRIGHT_TILT_DEG = 15.0
ROLL_T = 1.5
ROLL_PUSH = 0.1                     # x own weight: a gentle push, far below the unit friction coefficient, so sliding
                                    # cannot pass for rolling and a sound design does not tip
ROLL_HEADS = {"wheel", "caster", "castor", "roller", "tire", "tyre", "track", "tread"}
DEFAULT_DENSITY = 1000.0
MAX_HULL_PTS = 512


def _hull_pts(v, seed=0):
    """Collision points of one part: its exact convex hull's vertices. A very large hull is reduced to the
    extreme point along each of MAX_HULL_PTS directions, never by random sampling (which drops the corners a
    part stands on)."""
    v = np.asarray(v, float)
    ext = v.max(0) - v.min(0)
    if np.min(ext) < 1e-4:
        # flat or degenerate part: give it a 1 mm skin so the hull has a volume
        pad = np.zeros(3)
        pad[int(np.argmin(ext))] = 1e-3
        v = np.vstack([v, v + pad])
    try:
        from scipy.spatial import ConvexHull
        v = v[ConvexHull(v).vertices]
    except Exception:
        pass
    if len(v) > MAX_HULL_PTS:
        k = np.arange(MAX_HULL_PTS) + 0.5
        phi, th = np.arccos(1 - 2 * k / MAX_HULL_PTS), np.pi * (1 + 5 ** 0.5) * k
        dirs = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
        dirs = np.vstack([dirs, np.eye(3), -np.eye(3)])
        v = v[np.unique(np.argmax(v @ dirs.T, axis=0))]
    return v


OVERLAP_EXCLUDE_FRAC = 0.01          # bodies whose parts already interpenetrate this much in the design never collide
HULL_PEN_FRAC = 0.002                # ... nor bodies whose collision hulls already penetrate this deep (x diagonal) at t=0

def _revolve(v, axis, origin):
    """The solid of revolution of a part about a hinge: a cylinder on the axis, as long as the part along it and as
    wide as its farthest point from it."""
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    o = np.asarray(origin, float)
    t = (v - o) @ a
    r = float(np.linalg.norm((v - o) - np.outer(t, a), axis=1).max())
    t0, t1 = float(t.min()), float(t.max())
    if t1 - t0 < 1e-4:
        t0, t1 = t0 - 5e-4, t1 + 5e-4
    return o + a * t0, o + a * t1, max(r, 1e-4)


ROUND_OFFSET = 0.2                   # a rolling body's hinge passes within this share of its radius of its centre ...
ROUND_ASPECT = 0.75                  # ... and its two widths across the hinge agree to this ratio (a disc, not a door)


def _round_about(v, axis, origin):
    """Is this body a wheel on this hinge: centred on it, and as wide one way across it as the other?"""
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    o = np.asarray(origin, float)
    u = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    pu, pw = (v - o) @ u, (v - o) @ w
    r = max(float(np.ptp(pu)), float(np.ptp(pw))) / 2
    if r <= 0:
        return False
    off = float(np.hypot((pu.max() + pu.min()) / 2, (pw.max() + pw.min()) / 2))
    return off <= ROUND_OFFSET * r and min(np.ptp(pu), np.ptp(pw)) >= ROUND_ASPECT * max(np.ptp(pu), np.ptp(pw))


def compile_mjcf(parts, groups, joints, masses, root_gid, tree, floor_z=0.0, excludes=(), mounted=False, round_groups=()):
    """groups: list[list[part index]]; tree: {child_gid: (parent_gid, joint)}; bodies in world coordinates."""
    lines = ['<mujoco model="design">',
             '<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"><flag filterparent="disable"/></option>',
             '<default><geom friction="1.0 0.02 0.002" solref="0.004 1" condim="3"/>'
             '<joint damping="0.0" armature="0.0001"/></default>', "<asset>"]
    for gi, g in enumerate(groups):
        for i in g:
            v = _hull_pts(parts[i].vertices, i)
            lines.append(f'<mesh name="m{i}" vertex="{" ".join(f"{x:.6f}" for x in v.ravel())}"/>')
    lines.append("</asset><worldbody>")
    lines.append(f'<geom name="floor" type="plane" size="0 0 1" pos="0 0 {floor_z - 1e-4:.6f}"/>')
    children = {}
    for c, (p, j) in tree.items():
        children.setdefault(p, []).append((c, j))

    def body(gid, j, depth):
        pad = " " * depth
        out = [f'{pad}<body name="g{gid}" pos="0 0 0">']
        if j is None and not (mounted and gid == root_gid):
            out.append(f'{pad} <freejoint name="free{gid}"/>')
        else:
            kind = {"prismatic": "slide", "ball": "ball"}.get(j.type, "hinge")
            ax = np.asarray(j.axis if j.axis is not None else [0, 0, 1], float)
            ax = ax / max(np.linalg.norm(ax), 1e-12)
            org = np.asarray(j.origin if j.origin is not None else parts[groups[gid][0]].vertices.mean(0), float)
            axis = "" if kind == "ball" else f' axis="{ax[0]:.6f} {ax[1]:.6f} {ax[2]:.6f}"'
            out.append(f'{pad} <joint name="j{gid}" type="{kind}" pos="{org[0]:.6f} {org[1]:.6f} {org[2]:.6f}"{axis}/>')
        for i in groups[gid]:
            if gid in round_groups and j is not None:
                p0, p1, r = _revolve(parts[i].vertices, j.axis, j.origin if j.origin is not None
                                     else parts[groups[gid][0]].vertices.mean(0))
                out.append(f'{pad} <geom name="p{i}" type="cylinder" size="{r:.6f}" '
                           f'fromto="{p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f} {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}" '
                           f'mass="{max(masses[i], 1e-9):.9g}"/>')
            else:
                out.append(f'{pad} <geom name="p{i}" type="mesh" mesh="m{i}" mass="{max(masses[i], 1e-9):.9g}"/>')
        for c, jj in children.get(gid, []):
            out += body(c, jj, depth + 1)
        out.append(f"{pad}</body>")
        return out

    lines += body(root_gid, None, 0)
    for gid in range(len(groups)):              # groups not in the tree: loose, free bodies
        if gid != root_gid and gid not in tree:
            lines += body(gid, None, 0)
    lines.append("</worldbody>")
    if excludes:
        lines.append("<contact>" + "".join(f'<exclude body1="g{a}" body2="g{b}"/>' for a, b in sorted(excludes)) + "</contact>")
    lines.append("<actuator>")
    for c, (_, j) in tree.items():
        if j.type != "ball":                     # a ball joint has no scalar servo: held by damping instead
            lines.append(f'<position name="a{c}" joint="j{c}" kp="1" ctrllimited="false"/>')
    lines.append("</actuator></mujoco>")
    return "\n".join(lines)


class Sim:
    def __init__(self, design, parts, rows, masses, free_standing=True):
        import mujoco
        from ppbench.v2 import analysis
        from ppbench.v2.metrics_v3 import kinematic_moving_set
        self.mj = mujoco
        self.parts = parts
        idx = {p.id: i for i, p in enumerate(parts)}
        movers = [j for j in design.joints if j.type in ("revolute", "continuous", "prismatic", "ball")
                  and j.parent in idx and j.child in idx and j.parent != j.child]
        # connected parts are one solid, except across a moving joint: whatever moves with the joint's child
        # (declared joints first, contact only for parts no joint names) is cut loose from the rest
        cut = set()
        for j in movers:
            mset, _ = kinematic_moving_set(parts, rows, design.joints, j)
            for r in rows:
                if (r["a"] in mset) != (r["b"] in mset):
                    cut.add(frozenset((r["a"], r["b"])))
            cut.add(frozenset((j.parent, j.child)))
        edges = [(r["i"], r["k"]) for r in rows if frozenset((r["a"], r["b"])) not in cut]
        edges += [(idx[j.parent], idx[j.child]) for j in design.joints
                  if j.type == "fixed" and j.parent in idx and j.child in idx]
        comps = analysis.components(len(parts), edges)
        groups, gid_of = [], {}
        for c in comps:
            for i in c:
                gid_of[parts[i].id] = len(groups)
            groups.append(sorted(c))
        self.groups, self.gid_of = groups, gid_of
        gmass = [sum(masses[i] for i in gg) for gg in groups]
        moving = [j for j in movers if gid_of[j.parent] != gid_of[j.child]]
        # the base: the heaviest body that stands on the floor or hangs directly on one that does (the star base
        # above its casters, the chassis above its wheels), never a heavy top carried by joints
        zmin = float(min(p.vertices[:, 2].min() for p in parts))
        band = max(0.005, 0.01 * float(np.ptp(np.vstack([p.vertices for p in parts])[:, 2])))
        grounded = {gid for gid, gg in enumerate(groups) if min(parts[i].vertices[:, 2].min() for i in gg) <= zmin + band}
        near = set(grounded)
        for j in moving:
            a, b = gid_of[j.parent], gid_of[j.child]
            if a in grounded or b in grounded:
                near |= {a, b}
        self.root = max(near or range(len(groups)), key=lambda g: gmass[g])
        tree, seen, frontier = {}, {self.root}, [self.root]
        while frontier:                                  # BFS over the joint graph; extra joints closing a loop are dropped
            nxt = []
            for u in frontier:
                for j in moving:
                    a, b = gid_of[j.parent], gid_of[j.child]
                    for p_, c_ in ((a, b), (b, a)):
                        if p_ == u and c_ not in seen:
                            tree[c_] = (p_, j)
                            seen.add(c_)
                            nxt.append(c_)
            frontier = nxt
        self.tree = tree
        self.joint_of_group = {c: j for c, (_, j) in tree.items()}
        self.dropped_loops = len(moving) - len(tree)
        zmin = float(min(p.vertices[:, 2].min() for p in parts))
        # a body that turns on a roughly horizontal hinge and touches the floor is a wheel, roller or caster: it
        # collides as the cylinder it sweeps about that hinge, not as its faceted hull (a 24-sided tyre resting on a
        # facet has to be lifted over every edge to roll)
        round_groups = set()
        for c, (_, j) in tree.items():
            if j.type in ("revolute", "continuous") and j.axis is not None and c in grounded:
                a = np.asarray(j.axis, float)
                if abs(a[2]) / max(np.linalg.norm(a), 1e-12) < 0.5 and _round_about(
                        np.vstack([parts[i].vertices for i in groups[c]]), a,
                        j.origin if j.origin is not None else np.vstack([parts[i].vertices for i in groups[c]]).mean(0)):
                    round_groups.add(c)
        self.round_groups = round_groups
        # a moving part collides with the body it hangs on (a door with its cabinet), except a rolling body with its
        # axle carrier: its sweep is the cylinder itself, and a filled-in wheel arch would only brake it
        excludes = {(min(c, tree[c][0]), max(c, tree[c][0])) for c in round_groups}
        # the design's own interpenetrations are 1.2's finding; in the engine they would only throw the parts apart
        for r in rows:
            ga, gb = gid_of[r["a"]], gid_of[r["b"]]
            if ga != gb and r.get("frac_of_smaller", 0.0) > OVERLAP_EXCLUDE_FRAC:
                excludes.add((min(ga, gb), max(ga, gb)))
        self.n_excluded = len(excludes)
        # an object that is not free-standing (held, mounted: the task's reference does not stand on its own) is
        # operated with its base fixed in place, as P.1 and 1.3 treat it
        self.mounted = free_standing is False
        xml = compile_mjcf(parts, groups, design.joints, masses, self.root, tree, zmin, excludes, self.mounted, round_groups)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        # convex hulls of concave parts (a hollow cabinet, an L-shaped frame) can overlap a neighbour the real shapes
        # do not: a body pair already penetrating in the engine at t = 0 is excluded as well, then the model rebuilt.
        # Resting contact is kept, and contact made later, by motion, blocks
        mujoco.mj_forward(self.model, self.data)
        V0 = np.vstack([p.vertices for p in parts])
        tol = HULL_PEN_FRAC * float(np.linalg.norm(V0.max(0) - V0.min(0)))
        more = set()
        for k in range(self.data.ncon):
            con = self.data.contact[k]
            b1, b2 = self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2]
            if b1 == 0 or b2 == 0 or con.dist > -tol:
                continue
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1)
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2)
            a, b = int(n1[1:]), int(n2[1:])
            more.add((min(a, b), max(a, b)))
        if more - excludes:
            excludes |= more
            self.n_excluded = len(excludes)
            xml = compile_mjcf(parts, groups, design.joints, masses, self.root, tree, zmin, excludes, self.mounted,
                               round_groups)
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
        V = np.vstack([p.vertices for p in parts])
        self.diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        self.total_mass = float(self.model.body_subtreemass[0])
        self.weight = self.total_mass * 9.81
        self.root_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"g{self.root}")
        # servo gains: stiff against the WHOLE object's weight (the base may carry a heavy top on its joints),
        # with joint-space armature so every servo has the same natural frequency SERVO_W. A ball joint has no
        # scalar servo: it is held by the same damping on its three rotational dofs, and released like the rest.
        M = max(self.total_mass, 1e-9)
        self.jid, self.aid, self.ndof = {}, {}, {}
        self.kp0, self.damp0, self.arm0, self.arm_free = {}, {}, {}, {}
        for c in tree:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"j{c}")
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"g{c}")
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a{c}")
            m = max(float(self.model.body_subtreemass[bid]), 1e-9)
            Vc = np.vstack([parts[i].vertices for i in groups[c]])
            L = max(float(np.linalg.norm(Vc.max(0) - Vc.min(0))), 1e-3)
            jt = self.model.jnt_type[jid]
            slide = jt == mujoco.mjtJoint.mjJNT_SLIDE
            kp = 100.0 * M * 9.81 * (1.0 / L if slide else L)
            nd = 3 if jt == mujoco.mjtJoint.mjJNT_BALL else 1
            self.jid[c], self.aid[c], self.ndof[c] = jid, (aid if aid >= 0 else None), nd
            self.kp0[c] = kp
            self.damp0[c] = 2.0 * kp / SERVO_W * (5.0 if nd == 3 else 1.0)
            self.arm0[c] = kp / SERVO_W ** 2
            self.arm_free[c] = 0.01 * (m if slide else m * L * L)
        self.grounded = grounded
        self.set_free(set())

    # -- helpers
    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)

    def q(self, c):
        return float(self.data.qpos[self.model.jnt_qposadr[self.jid[c]]])

    def tilt(self):
        R = self.data.xmat[self.root_bid].reshape(3, 3)
        return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))

    def root_xy(self):
        return self.data.xpos[self.root_bid][:2].copy()

    def ok(self):
        return bool(np.all(np.isfinite(self.data.qpos))) and float(np.abs(self.data.qvel).max(initial=0)) < 1e4

    def set_free(self, cs):
        """Release the joints in cs (no servo, no damping); hold every other one."""
        for c in self.tree:
            free = c in cs
            if self.aid[c] is not None:
                g = 0.0 if free else self.kp0[c]
                self.model.actuator_gainprm[self.aid[c], 0] = g
                self.model.actuator_biasprm[self.aid[c], 1] = -g
            d0 = self.model.jnt_dofadr[self.jid[c]]
            for dof in range(d0, d0 + self.ndof[c]):
                self.model.dof_damping[dof] = 0.0 if free else self.damp0[c]
                self.model.dof_armature[dof] = self.arm_free[c] if free else self.arm0[c]

    def servos(self):
        return [c for c in self.tree if self.aid[c] is not None]

    def ancestors(self, gid):
        out = [gid]
        while gid in self.tree:
            gid = self.tree[gid][0]
            out.append(gid)
        return out

    def path_joints(self, gid, stop):
        """Joints on the tree path between body gid and the nearest body in `stop` (as child gids)."""
        best = None
        a = self.ancestors(gid)
        for s_ in stop:
            if s_ == gid:
                return []
            b = self.ancestors(s_)
            common = next((x for x in a if x in b), None)
            if common is None:
                continue
            path = a[:a.index(common)] + b[:b.index(common)]
            if best is None or len(path) < len(best):
                best = path
        return best or []

    def run(self, T, ctrl=None, force=None):
        n = int(T / self.model.opt.timestep)
        max_tilt = 0.0
        for k in range(n):
            if ctrl:
                for c, f in ctrl.items():
                    self.data.ctrl[self.aid[c]] = f(k / max(n - 1, 1))
            if force is not None:
                self.data.xfrc_applied[self.root_bid, :3] = force
            self.mj.mj_step(self.model, self.data)
            if k % 20 == 0:
                if not self.ok():
                    return None
                max_tilt = max(max_tilt, self.tilt())
        self.data.xfrc_applied[:] = 0
        return max_tilt

    def settle(self):
        self.reset()
        self.set_free(set())
        for c in self.servos():
            self.data.ctrl[self.aid[c]] = self.q(c)
        xy0 = self.root_xy()
        t = self.run(SETTLE_T)
        if t is None:
            return False, {"reason": "simulation diverged (interpenetrating bodies)"}
        moved = float(np.linalg.norm(self.root_xy() - xy0))
        ok = t < SETTLE_TILT_DEG and moved < 0.3 * self.diag
        return ok, {"settle_tilt_deg": round(t, 1), "settle_slide_frac": round(moved / max(self.diag, 1e-9), 3)}

    # -- tests
    def actuate(self, c):
        """Drive joint c through its range; best of the two directions."""
        j = self.joint_of_group[c]
        slide = j.type == "prismatic"
        lims = list(j.limits) if j.limits is not None and len(j.limits) == 2 else None
        if j.type == "continuous":
            targets = [2 * math.pi, -2 * math.pi]
        elif lims and abs(lims[1] - lims[0]) > 1e-6:
            targets = [t for t in lims if abs(t) > 1e-6] or [lims[1]]
        elif slide:
            ext = self.parts[self.groups[c][0]].vertices
            L = 0.5 * float(np.ptp(ext @ (np.asarray(j.axis, float) / max(np.linalg.norm(j.axis), 1e-12))))
            targets = [L, -L]
        else:
            targets = [math.pi / 2, -math.pi / 2]
        best = {"achieved": 0.0, "upright": False, "score": 0.0}
        for tgt in targets:
            ok, info = self.settle()
            if not ok:
                return {"score": 0.0, **info}
            q0 = self.q(c)
            hold = {k: (lambda _t, v=self.q(k): v) for k in self.servos() if k != c}
            ctrl = {**hold, c: (lambda s, a=q0, b=q0 + tgt: a + (b - a) * min(1.0, s * (ACT_T + HOLD_T) / ACT_T))}
            t = self.run(ACT_T + HOLD_T, ctrl)
            if t is None:
                continue
            ach = min(1.0, abs(self.q(c) - q0) / max(abs(tgt), 1e-9))
            up = t < UPRIGHT_TILT_DEG
            s = ach * float(up)
            if s >= best["score"]:
                best = {"achieved": round(ach, 3), "upright": up, "max_tilt_deg": round(t, 1),
                        "target": round(tgt, 3), "score": s}
        return best

    def roll(self, wheel_cs, free_cs):
        """wheel_cs: the wheels' own joints (their horizontal axes set the push direction); free_cs: every joint
        released for the push (the wheels plus any swivel between them and the base)."""
        ok, info = self.settle()
        if not ok:
            return {"score": 0.0, **info}
        axes = []
        for c in wheel_cs:
            if self.joint_of_group[c].axis is None or self.joint_of_group[c].type == "ball":
                continue
            a = np.asarray(self.joint_of_group[c].axis, float)
            a = a / max(np.linalg.norm(a), 1e-12)
            if abs(a[2]) < 0.5:                      # a wheel rolls about a roughly horizontal axis
                axes.append(a)
        if not axes:
            return {"score": 0.0, "reason": "no wheel turns about a horizontal axis"}
        A = np.asarray(axes)
        a = A[0] * np.sign(A @ A[0])[:, None]
        a = a.mean(0) if len(A) else np.array([1.0, 0, 0])
        d = np.array([-a[1], a[0], 0.0])
        if np.linalg.norm(d) < 1e-6:
            return {"score": 0.0, "reason": "wheel axes are vertical"}
        d /= np.linalg.norm(d)
        free = set(free_cs)
        hold = {k: (lambda _t, v=self.q(k): v) for k in self.servos() if k not in free}
        best = 0.0
        det = {}
        for sgn in (1, -1):
            ok, _ = self.settle()
            self.set_free(free)
            xy0 = self.root_xy()
            t = self.run(ROLL_T, hold, force=sgn * ROLL_PUSH * self.weight * d)
            self.set_free(set())
            if t is None:
                continue
            dist = float(np.dot(self.root_xy() - xy0, sgn * d[:2]))
            ideal = 0.5 * ROLL_PUSH * 9.81 * ROLL_T ** 2
            s = min(1.0, max(0.0, dist / (ROLL_FULL * ideal))) * float(t < UPRIGHT_TILT_DEG)
            if s >= best:
                best, det = s, {"distance_m": round(dist, 4), "frictionless_m": round(ideal, 4),
                                "max_tilt_deg": round(t, 1), "n_wheels": len(axes), "n_joints_free": len(free)}
        return {"score": best, **det}


NOT_ROLLING = {("steering", "wheel"), ("hand", "wheel"), ("scroll", "wheel"), ("fly", "wheel"), ("gear", "wheel")}


def is_rolling(phrase):
    """A rolling element the object moves on. A steering wheel, handwheel or scroll wheel is turned, not rolled on."""
    from ppbench.v2.lexicon import tokens
    tk = tokens(phrase or "")
    return bool(tk) and tk[-1] in ROLL_HEADS and tuple(tk[-2:]) not in NOT_ROLLING


def simulate(design, task, parts, rows, ctx, masses, free_standing=True):
    """masses: per part, kg (voxel volume x the declared material's density, else DEFAULT_DENSITY)."""
    """One row per claimed motion K. Returns (score or None, rows, info)."""
    ks = task.raw.get("required_kinematics", [])
    if not ks:
        return None, [], {"reason": "the task claims no motion"}
    try:
        sim = Sim(design, parts, rows, masses, free_standing)
    except Exception as e:  # noqa: BLE001
        return 0.0, [], {"reason": f"could not compile: {type(e).__name__}: {str(e)[:200]}"}
    ok, sinfo = sim.settle()
    out = []
    HINGE, SLIDE, ROT = ("revolute", "continuous"), ("prismatic",), ("revolute", "continuous", "ball")
    for k in ks:
        role, rel = k.get("moving_part"), k.get("relative_to")
        ids = ctx.roles(role, broad=False) or ctx.roles(role)
        rel_ids = (ctx.roles(rel, broad=False) or ctx.roles(rel)) if rel else []
        stop = {sim.gid_of[parts[i].id] for i in rel_ids} or {sim.root}
        rolling = is_rolling(role) and k.get("joint_type") != "prismatic"
        want = SLIDE if k.get("joint_type") == "prismatic" else (ROT if rolling else HINGE) \
            if k.get("joint_type") in HINGE + SLIDE else None
        bodies = sorted({sim.gid_of[parts[i].id] for i in ids})
        paths = {b: sim.path_joints(b, stop) for b in bodies}
        cands = sorted({c for b, pj in paths.items() for c in pj
                        if want is None or sim.joint_of_group[c].type in want})
        row = {"claim": k["id"], "moving_part": role, "relative_to": rel, "motion": k.get("joint_type"),
               "n_parts": len(ids), "n_joints": len(cands)}
        if not ok:
            row.update({"test": "none", "score": 0.0, "reason": "the object does not stand", **sinfo})
        elif not cands:
            row.update({"test": "none", "score": 0.0,
                        "reason": "no part with this role" if not ids else
                        f"no {k.get('joint_type') or ''} joint moves it relative to {rel or 'the base'}".replace("  ", " ")})
        elif rolling:
            # pushing a wheeled object releases every hinge between a ground-touching body and the base: all its
            # wheels and their swivels, not only the ones this claim names
            free = sorted({c for b in sim.grounded for c in sim.path_joints(b, {sim.root})
                           if sim.joint_of_group[c].type in ROT})
            row.update({"test": "roll", **sim.roll(cands, free)})
        else:
            res = [sim.actuate(g) for g in cands[:4] if sim.aid[g] is not None] or [{"score": 0.0, "reason": "only ball joints"}]
            b = max(res, key=lambda r: r["score"])
            row.update({"test": "actuate", **b, "n_tested": len(res)})
        out.append(row)
    return float(np.mean([r["score"] for r in out])), out, {
        "stands": ok, **sinfo, "n_bodies": len(sim.groups), "n_joints": len(sim.tree), "mounted": sim.mounted,
        "n_round_bodies": len(sim.round_groups),
        "n_overlapping_pairs_excluded": sim.n_excluded,
        "loops_dropped": sim.dropped_loops, "mass_kg": round(sim.total_mass, 4)}
