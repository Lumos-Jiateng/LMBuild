"""Spec v3.8 (2026-10-02): one grounding rule for every system -- a part's declared role is a claim that has to
be checked against the geometry, the same way an unnamed part is found.

Why. Under v3.7, a design whose parts carry names (at least half of them) was grounded by name alone: a box
called "engine" anywhere counted as the engine, while a design without names had to put each part where the
reference has that component (`rules.geometric_match`). The test was asymmetric. It also let a system that is
handed the component list as its part schema (Cube3D+CubePart) collect component credit for its labels.

Rule, identical for all systems (agents, generators, any tier):

* a part whose declared role maps to component c (task lexicon, `rules.Grounder`) is checked with the
  unnamed-part test against the reference's parts of c: at least half of its surface within 5 % of the
  normalised diagonal (TAU_GROUND) of them, after the same yaw alignment 2.1 uses.
    - passes                         -> grounded to c, verified (factor 1.0)
    - fails, but it lies on another
      component c' by the same test  -> grounded to c' by geometry (factor 1.0): the part is there, the name is wrong
    - fails, and lies on none        -> kept as c, unverified (factor UNVERIFIED = 0.5): a part with that role
                                        exists, but not where the object has it
* a part whose role maps to a component the reference shows no geometry for (an internal module the reference
  mesh does not segment, a claim part the reference lacks, an LDraw reference) cannot be checked against the
  reference. Fallback, also identical for all systems:
    - an internal component (sheet "visible": false) whose part is enclosed in the design (no ray from outside
      reaches it, `rules.interior_parts`)       -> verified by placement (factor 1.0)
    - otherwise                                 -> unverifiable (factor UNVERIFIED = 0.5)
* a part without a usable role is grounded by geometry exactly as in v3.7.

UNVERIFIED = 0.5 is the value v3.7 already uses for "there, but not doing its job" (a floating component, an
unnamed interior module). Nothing here depends on which system made the design.

`ground(tid, d, family)` returns the per-part grounding; `verified_roles(task_id, key)` loads a main-setting
design by its record key and returns {part_id: factor} for Level 4 (P.3 presence).
"""
from __future__ import annotations

import json

import numpy as np

from ppbench.v2.afford import rules
from ppbench.v2.task import RESULTS

SPEC = "v3.8"
UNVERIFIED = 0.5


def _frac_on(ref, Dr, DL, idx, cid):
    from scipy.spatial import cKDTree
    Q = Dr[DL == idx]
    pts = ref["comp_pts"].get(cid) if ref is not None else None
    if pts is None or len(pts) < 5 or len(Q) < 5:
        return None
    dist, _ = cKDTree(pts).query(Q)
    return float((dist < rules.TAU_GROUND).mean())


def aligned_samples(tid, d):
    """The design's surface samples in the reference frame (normalised, best of four yaws), as 2.1 does."""
    ref = rules.reference_geometry(tid)
    D, DL = rules.sample_labelled(d.parts, seed=0)
    Dn, _ = rules.normalise(D) if len(D) else (D, 0.0)
    R = np.eye(3)
    if ref is not None and len(D):
        best = None
        for yaw in (0, 90, 180, 270):
            Rz = rules.rotz(yaw)
            f = rules.fscore(Dn @ Rz.T, ref["P"], rules.TAU)[0]
            if best is None or f > best[0]:
                best = (f, Rz)
        R = best[1]
    return ref, (Dn @ R.T if len(D) else Dn), DL


def ground(tid, d, family, ref=None, Dr=None, DL=None, interior=None):
    """Per design part: {"claimed", "comp", "factor", "how"}.

    how: "role-verified" | "role-moved" (geometry overrides the name) | "role-unverified" (reference has the
    component, the part is not there) | "role-interior" (no reference geometry; internal component, enclosed
    part) | "role-unverifiable" (no reference geometry) | "geometry" (unnamed, found by position) | None."""
    t, lex, comps, g = rules.context(tid)
    sheet = rules.load_sheet(tid)
    visible = {c["id"]: c["visible"] for c in sheet["components"]}
    if Dr is None:
        ref, Dr, DL = aligned_samples(tid, d)
    named = rules.roles_declared(d)
    if interior is None:
        from ppbench.v2.afford import kin
        interior, _ = rules.interior_parts(kin.Scene(d))
    out = {}
    for i, p in enumerate(d.parts):
        c = g.match(p.role) if named else None
        if c is None:
            gm = rules.geometric_match(ref, Dr, DL, i) if ref is not None else None
            out[p.id] = {"claimed": None, "comp": gm, "factor": 1.0 if gm else 0.0, "how": "geometry" if gm else None}
            continue
        has_geom = ref is not None and c in ref["comp_pts"] and len(ref["comp_pts"][c]) >= 5
        if has_geom:
            f = _frac_on(ref, Dr, DL, i, c)
            if f is not None and f >= 0.5:
                out[p.id] = {"claimed": c, "comp": c, "factor": 1.0, "how": "role-verified", "frac": f}
                continue
            gm = rules.geometric_match(ref, Dr, DL, i)
            if gm is not None and gm != c:
                out[p.id] = {"claimed": c, "comp": gm, "factor": 1.0, "how": "role-moved", "frac": f}
            else:
                out[p.id] = {"claimed": c, "comp": c, "factor": UNVERIFIED, "how": "role-unverified", "frac": f}
            continue
        if visible.get(c) is False and p.id in interior:
            out[p.id] = {"claimed": c, "comp": c, "factor": 1.0, "how": "role-interior"}
        else:
            out[p.id] = {"claimed": c, "comp": c, "factor": UNVERIFIED, "how": "role-unverifiable"}
    return out


def verified_roles(task_id, key):
    """{part_id: factor} for the design behind a main-setting record key (eval_v34 holds the item)."""
    rec = json.loads((RESULTS / task_id / rules.MAIN_EVAL / f"{key}.json").read_text())
    item = rec["item"]
    family = "domain" if "__ext__" in key else "llm"
    d = rules.core.load(rules.task(task_id), item)
    if d is None or not d.parts:
        return {}
    return role_factors(ground(task_id, d, family))


def role_factors(gr):
    """For name-based role lookups (Level 4): a part's name counts in full when its role was verified, by the
    reference or by an enclosed internal placement; at UNVERIFIED when the reference shows that component
    elsewhere or not at all (a 'role-moved' part is not where its name says). Parts without a role are 1.0 --
    name lookups never find them."""
    return {pid: (1.0 if v["how"] in ("role-verified", "role-interior") or v["claimed"] is None else UNVERIFIED)
            for pid, v in gr.items()}
