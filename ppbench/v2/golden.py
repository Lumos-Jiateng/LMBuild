"""Golden scenes for the v2 evaluator: one hand-built design per check, each built
to move exactly that check and leave the control untouched.

The control is a box "table" standing on four legs, every part touching the
next, declared roles and materials, a feasible build order. Each variant breaks
one thing. A metric that silently stops responding shows up here first.

    .venv_eval/bin/python -m ppbench.v2.golden
"""
from __future__ import annotations

import copy
import math
import sys

import numpy as np

from ppbench.v2 import analysis, voxel
from ppbench.v2.design import Design, Joint, Part


def box(pid, centre, size, role=None, material="ABS"):
    import trimesh
    m = trimesh.creation.box(extents=size)
    return Part(pid, np.asarray(m.vertices) + centre, np.asarray(m.faces), role, material, {"kind": "golden"})


def control():
    parts = [box("top", [0, 0, 0.42], [0.6, 0.6, 0.04], "seat")]
    for k, (x, y) in enumerate([(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]):
        parts.append(box(f"leg{k}", [x, y, 0.2], [0.04, 0.04, 0.4], "leg"))
    seq = [{"part": f"leg{k}", "direction": [0, 0, -1]} for k in range(4)] + [{"part": "top", "direction": [0, 0, -1]}]
    return Design("golden", "A", parts, [], seq, {"scale_mode": "metric",
                                                  "declares": {"roles": True, "materials": True, "joints": True, "sequence": True}})


def scenes():
    out = [("control", control(), {})]

    d = control()
    d.parts.append(box("floater", [0, 0, 0.8], [0.1, 0.1, 0.1], "seat"))
    d.sequence.append({"part": "floater", "direction": [0, 0, -1]})
    out.append(("floating_part", d, {"n_components": ("gt", 1), "floating": ("contains", "floater")}))

    d = control()
    d.parts.append(box("intruder", [0, 0, 0.42], [0.3, 0.3, 0.04], "seat"))
    d.sequence.append({"part": "intruder", "direction": [0, 0, -1]})
    out.append(("interpenetration", d, {"n_colliding": ("gt", 0)}))

    d = control()
    # the top still rests on all four legs (it starts at x = -0.25) but overhangs far to +X
    d.parts[0] = box("top", [0.55, 0, 0.42], [1.6, 0.6, 0.04], "seat")
    out.append(("com_outside_support", d, {"tilt": ("lt", 0.0)}))

    d = control()
    d.joints = [Joint("far", "revolute", "leg0", "top", [0, 0, 1], [2.0, 2.0, 2.0], [0, 1.0])]
    out.append(("joint_origin_off_boundary", d, {"joint_issue": ("gt", 0)}))

    d = control()
    # a flap hanging under the top's front edge, hinged about X where the two meet. Its declared
    # range [0, pi] swings it under the top and then up through it: blocked past about 90 degrees.
    d.parts.append(box("flap", [0, -0.28, 0.30], [0.4, 0.04, 0.2], "seat"))
    d.joints = [Joint("hinge", "revolute", "top", "flap", [1, 0, 0], [0, -0.28, 0.40], [0, math.pi])]
    out.append(("blocked_sweep", d, {"sweep_blocked": ("eq", True)}))

    d = control()
    d.sequence = [{"part": "top", "direction": [0, 0, -1]}] + [{"part": f"leg{k}", "direction": [0, 0, -1]} for k in range(4)]
    out.append(("unsupported_first_step", d, {"supported_prefix_frac": ("lt", 1.0)}))
    return out


def measure(d: Design):
    occs = analysis.occupancies(d.parts)
    rep, _, rows = analysis.summarize(d.parts, d.joints, occs=occs)
    out = {"n_components": rep["connected_groups"], "floating": [x for g in rep["not_connected_to_main_group"] for x in g],
           "n_colliding": rep["n_collisions"], "tilt": rep["stability"]["critical_tilt_deg"],
           "joint_issue": len(rep["joint_issues"])}
    # sweep: rotate the flap and look for new overlap with static parts
    from ppbench.v2.design import axis_angle
    out["sweep_blocked"] = False
    for j in d.joints:
        if j.child not in {p.id for p in d.parts} or j.type == "fixed":
            continue
        ci = next(i for i, p in enumerate(d.parts) if p.id == j.child)
        base = sum(voxel.overlap_cells(occs[ci], occs[s]) for s in range(len(d.parts)) if s != ci)
        for a in np.linspace(j.limits[0], j.limits[1], 8):
            v2 = (d.parts[ci].vertices - j.origin) @ axis_angle(j.axis, a).T + j.origin
            o2 = voxel.voxelize(v2, d.parts[ci].faces)
            inc = sum(voxel.overlap_cells(o2, occs[s]) for s in range(len(d.parts)) if s != ci) - base
            if inc * voxel.RES ** 3 > 1e-6:
                out["sweep_blocked"] = True
    # supported prefix
    idx = {p.id: i for i, p in enumerate(d.parts)}
    pairs = {(r["i"], r["k"]) for r in rows} | {(r["k"], r["i"]) for r in rows}
    zmin = min(p.vertices[:, 2].min() for p in d.parts)
    placed, sup = [], []
    for st in d.sequence or []:
        placed.append(idx[st["part"]])
        comps = analysis.components(len(d.parts), [(a, b) for a, b in pairs if a in placed and b in placed])
        comps = [[x for x in c if x in placed] for c in comps]
        comps = [c for c in comps if c]
        sup.append(float(all(any(d.parts[x].vertices[:, 2].min() <= zmin + 0.01 for x in c) for c in comps)))
    out["supported_prefix_frac"] = float(np.mean(sup)) if sup else 1.0
    return out


LEXICON_CASES = [   # names that real episodes produced and an earlier table got wrong
    ("backrest cushion", "backrest"), ("backrest support bar (bolted under seat)", "backrest"),
    ("chair base (5-star swivel base)", "base"), ("gas lift column and seat plate", "gas lift cylinder"),
    ("seat height adjuster support", "gas lift cylinder"), ("seat back", "backrest"), ("back leg", "base"),
    ("swivel caster wheel", "caster wheel"), ("caster swivel fork", "caster"), ("left armrest pad", "armrest"),
]


def check_lexicon():
    from ppbench.v2.lexicon import Lexicon
    lex = Lexicon("swivel_office_chair")
    ok = True
    for raw, want in LEXICON_CASES:
        got = lex.canon(raw)
        if got != want:
            ok = False
            print(f"FAIL lexicon {raw!r} -> {got!r}, expected {want!r}")
    # strict matching: a caster fork is not a wheel, and "casters" presence accepts either
    if lex.satisfies("caster fork", "caster wheel", broad=False) or not lex.satisfies("casters", "caster wheel"):
        ok = False
        print("FAIL lexicon broad/strict claim matching")
    print(f"{'ok ' if ok else 'FAIL'} lexicon ({len(LEXICON_CASES)} names)")
    return ok


def check_bundle():
    """A Y-up bundle whose seat faces +Z must come out Z-up with the seat facing -Y, grounded,
    with joint axes and origins moved by the same transform."""
    import json
    import tempfile
    from pathlib import Path
    import trimesh
    from ppbench.v2.external import bundle_design, map_material
    d = Path(tempfile.mkdtemp(dir="/dev/shm"))
    (d / "parts").mkdir()
    # seat slab at y in [0.4, 0.45] (Y up); a backrest at the -Z side, so the seat faces +Z
    trimesh.creation.box(extents=[0.5, 0.05, 0.5]).apply_translation([0, 0.425, 0]).export(d / "parts" / "seat.glb")
    trimesh.creation.box(extents=[0.5, 0.5, 0.05]).apply_translation([0, 0.7, -0.25]).export(d / "parts" / "back.glb")
    (d / "bundle.json").write_text(json.dumps({"up": "y", "front": "+z", "units": "m",
        "parts": [{"id": "seat", "file": "parts/seat.glb", "name": "seat", "material": "Stainless steel frame"},
                  {"id": "back", "file": "parts/back.glb", "name": "backrest", "material": "leather"}],
        "joints": [{"id": "tilt", "type": "revolute", "parent": "seat", "child": "back", "axis": [1, 0, 0],
                    "origin": [0, 0.45, -0.25], "limits": [0, 0.3]}]}))
    des = bundle_design(d, "golden")
    seat, back = des.part("seat"), des.part("back")
    ok = True
    if abs(des.parts[0].vertices[:, 2].min() - 0.0) > 1e-6 and abs(min(p.vertices[:, 2].min() for p in des.parts)) > 1e-6:
        ok = False; print("FAIL bundle: not grounded")
    if not back.vertices[:, 1].mean() > seat.vertices[:, 1].mean():
        ok = False; print("FAIL bundle: backrest should be at +Y (behind a seat facing -Y)")
    j = des.joints[0]
    if abs(abs(j.axis[0]) - 1) > 1e-6 or not (j.origin[1] > 0.2 and abs(j.origin[2] - 0.05) < 1e-6):
        ok = False; print(f"FAIL bundle: joint axis/origin not transformed: {j.axis} {j.origin}")
    if map_material("Stainless steel frame") != "stainless_steel" or map_material("leather") != "cotton_fabric":
        ok = False; print("FAIL bundle: material mapping")
    print(f"{'ok ' if ok else 'FAIL'} bundle adapter frame/material conversion")
    return ok


def check_moving_set():
    """A wheel in its arch: the joint must move the wheel, not the car.

    `voxel.touching` is a one-cell test, so a wheel flush against its fender "touches" it with
    zero shared volume. A flood fill over contacts therefore walks wheel -> fender -> body and
    calls 95% of the car the moving side, which then sweeps freely because the block threshold
    scales with the moving volume. The declared joint graph is what decides now, and contact
    only places the parts no joint mentions -- static side first.

    chassis --axle--> wheel        fender touches both and stays put
                      hubcap       touches only the wheel and rides along
    """
    from ppbench.v2.evaluate import moving_set as contact_flood
    from ppbench.v2.metrics_v3 import joint_quality, kinematic_moving_set
    parts = [box("chassis", [0, 0, 0.40], [1.2, 0.70, 0.20], "chassis"),
             box("wheel", [-0.40, -0.30, 0.15], [0.30, 0.10, 0.30], "wheel"),
             box("fender", [-0.40, -0.375, 0.275], [0.40, 0.05, 0.35], "fender"),
             box("hubcap", [-0.40, -0.225, 0.15], [0.08, 0.05, 0.08], "hubcap")]
    axle = Joint("axle", "continuous", "chassis", "wheel", [0, 1, 0], [-0.40, -0.30, 0.15], None)
    joints = [axle]
    occs = analysis.occupancies(parts)
    rows = analysis.pair_table(parts, occs, frozenset([frozenset((axle.parent, axle.child))]))
    idx = {p.id: i for i, p in enumerate(parts)}
    vol_total = float(sum(o.volume for o in occs))
    ok = True

    def want(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"FAIL moving set: {msg}")

    want({r["deep_m3"] for r in rows} == {0.0}, "the scene should hold no interpenetration, only flush contact")
    want({"wheel", "fender"} in [{r["a"], r["b"]} for r in rows], "the wheel should touch the fender (the trap this guards)")
    mset, looped = kinematic_moving_set(parts, rows, joints, axle)
    want(mset == {"wheel", "hubcap"}, f"the axle should move the wheel and its hubcap, got {sorted(mset)}")
    want(not looped, "one joint cannot be a loop")
    want("fender" in contact_flood(parts, rows, joints, axle)[0],
         "the contact flood no longer reaches the fender: this scene has stopped testing the regression")
    q = joint_quality(axle, parts, occs, rows, joints, idx, voxel.RES, vol_total)
    want(q["moving_volume_frac"] < 0.10, f"the axle should move a small fraction of the car, got {q['moving_volume_frac']:.2f}")
    want(q["welded_by_contact"] and not q["declared_loop"], "contact-welded, but not a declared loop")

    looped_joints = joints + [Joint("weld", "fixed", "chassis", "wheel", None, None, None)]
    want(kinematic_moving_set(parts, rows, looped_joints, axle)[1], "a redundant fixed joint is a declared loop")

    # a small wheel is still a wheel: effectiveness asks how the object is partitioned, not how big the
    # moving side is. A 0.5% floor in volume fails a real skateboard wheel (0.3% of the board).
    tiny = [parts[0], box("wheel", [-0.40, -0.30, 0.03], [0.06, 0.04, 0.06], "wheel")]
    tocc = analysis.occupancies(tiny)
    trows = analysis.pair_table(tiny, tocc, frozenset([frozenset(("chassis", "wheel"))]))
    taxle = Joint("axle", "continuous", "chassis", "wheel", [0, 1, 0], [-0.40, -0.30, 0.03], None)
    tq = joint_quality(taxle, tiny, tocc, trows, [taxle], {p.id: i for i, p in enumerate(tiny)},
                       voxel.RES, float(sum(o.volume for o in tocc)))
    want(tq["moving_volume_frac"] < 0.005 and tq["effectiveness"] == 1.0,
         f"a wheel at {tq['moving_volume_frac']:.4f} of the volume is still a proper subassembly")
    print(f"{'ok ' if ok else 'FAIL'} moving set (wheel in its arch)")
    return ok


def check_v35():
    """Spec v3.5 P.1 (executable sequence) and the P.2 claim parser."""
    from ppbench.v2 import spec_v33 as V33, spec_v35 as S

    def run(d):
        parts = d.parts
        occs = analysis.occupancies(parts)
        rows = V33._pairs(parts, occs, d.joints)
        w, _ = V33._masses(parts, occs)
        return S.p1_sequence(d, parts, occs, rows, V33.joint_edges(parts, occs, d.joints), w, True)["metrics"]

    ok = True
    m = run(control())
    good = m["executable_frac"] == 1.0 and m["strict_one_body"]["executable_frac"] < 1.0 and 0 < m["state_robustness"] < 1
    print(f"{'ok ' if good else 'FAIL'} v35 legs then top: executable, lone legs are vulnerable (0<S<1) E={m['executable_frac']} S={m['state_robustness']:.2f}")
    ok &= good
    d = control()
    d.sequence = d.sequence[-1:] + d.sequence[:-1]
    m = run(d)
    good = m["executable_frac"] == 0.0 and (m["first_failure"] or {}).get("reason") == "floating" and m["oracle_executable_frac"] == 1.0
    print(f"{'ok ' if good else 'FAIL'} v35 top first: floating at step 0, an order exists           {m['first_failure']}")
    ok &= good
    parts = [box("base", [0, 0, 0.01], [0.4, 0.4, 0.02], "base"),
             box("wall_n", [0, 0.19, 0.12], [0.4, 0.02, 0.2], "wall"), box("wall_s", [0, -0.19, 0.12], [0.4, 0.02, 0.2], "wall"),
             box("wall_e", [0.19, 0, 0.12], [0.02, 0.36, 0.2], "wall"), box("wall_w", [-0.19, 0, 0.12], [0.02, 0.36, 0.2], "wall"),
             box("lid", [0, 0, 0.23], [0.4, 0.4, 0.02], "lid"), box("cube", [0, 0, 0.07], [0.1, 0.1, 0.1], "cube")]
    seq = [{"part": p.id, "direction": [0, 0, -1]} for p in parts]
    d = Design("golden", "A", parts, [], seq, {"scale_mode": "metric", "declares": {"sequence": True}})
    m = run(d)
    good = (m["first_failure"] or {}).get("reason") == "blocked insertion" and abs(m["executable_frac"] - 6 / 7) < 1e-9 \
        and m["oracle_executable_frac"] == 1.0
    print(f"{'ok ' if good else 'FAIL'} v35 cube after the lid: blocked insertion, an order exists  {m['first_failure']}")
    ok &= good
    g = S.claim_material_groups("Tires are constructed from synthetic rubber, natural rubber, fabric, and wire.")
    subj = S.claim_subject("Frames and sashes are commonly made of wood, aluminum, PVC, or composite materials.")
    good = set().union(*g) == {"rubber", "textile", "heavy_metal"} and subj == ["Frames", "sashes"]
    print(f"{'ok ' if good else 'FAIL'} v35 claim parser: classes and subjects                        {sorted(set().union(*g))} {subj}")
    ok &= good
    return ok


def check_sim_v35():
    """Spec v3.5 P.3 simulation: a cart on free wheels rolls, the same cart on fixed wheels does not, a lid opens."""
    from ppbench.v2 import sim_v35 as M
    from ppbench.v2 import spec_v33 as V33
    import trimesh

    def cyl(pid, centre, r, w, role):
        m = trimesh.creation.cylinder(radius=r, height=w, sections=24)
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
        return Part(pid, np.asarray(m.vertices) + centre, np.asarray(m.faces), role, "ABS", {"kind": "golden"})

    def cart(jtype):
        parts = [box("body", [0, 0, 0.12], [0.5, 0.3, 0.08], "body")]
        joints = []
        for k, (x, y) in enumerate([(-0.18, -0.17), (0.18, -0.17), (-0.18, 0.17), (0.18, 0.17)]):
            parts.append(cyl(f"w{k}", [x, y, 0.05], 0.05, 0.03, "wheel"))
            joints.append(Joint(f"j{k}", jtype, "body", f"w{k}", [0, 1, 0], [x, y, 0.05]))
        return Design("golden", "A", parts, joints, None, {"scale_mode": "metric"})

    def wheels_axis_y(d):                       # cylinder built along x after the rotation: turn it to y
        for p in d.parts[1:]:
            c = p.vertices.mean(0)
            v = p.vertices - c
            p.vertices = np.c_[v[:, 1], v[:, 0], v[:, 2]] + c
        return d

    ok = True
    for jtype, want in (("continuous", True), ("fixed", False)):
        d = wheels_axis_y(cart(jtype))
        occs = analysis.occupancies(d.parts)
        rows = V33._pairs(d.parts, occs, d.joints)
        sim = M.Sim(d, d.parts, rows, [o.volume * 1000 for o in occs])
        wheels = [c for c in sim.tree if sim.joint_of_group[c].type == "continuous"]
        r = sim.roll(wheels, wheels) if wheels else {"score": 0.0, "reason": "no wheel joint"}
        good = (r["score"] > 0.9) == want
        print(f"{'ok ' if good else 'FAIL'} sim cart, {jtype} wheels: {'rolls' if want else 'does not roll':14s}        {r}")
        ok &= good
    parts = [box("box", [0, 0, 0.1], [0.3, 0.3, 0.2], "box"), box("lid", [0, 0, 0.21], [0.3, 0.3, 0.02], "lid")]
    d = Design("golden", "A", parts, [Joint("h", "revolute", "box", "lid", [1, 0, 0], [0, -0.15, 0.2], [0, 1.6])],
               None, {"scale_mode": "metric"})
    occs = analysis.occupancies(d.parts)
    sim = M.Sim(d, d.parts, V33._pairs(d.parts, occs, d.joints), [o.volume * 1000 for o in occs])
    r = sim.actuate(next(iter(sim.tree)))
    good = r["score"] > 0.9
    print(f"{'ok ' if good else 'FAIL'} sim hinged lid opens without tipping the box                {r}")
    ok &= good
    # contact made by motion still blocks: a flap on a vertical hinge swings into a wall 2 cm away (both ways)
    parts = [box("base", [0, 0, 0.01], [0.6, 0.6, 0.02], "base"), box("post", [0, 0, 0.17], [0.04, 0.04, 0.3], "post"),
             box("flap", [0.12, 0, 0.17], [0.2, 0.02, 0.2], "flap"),
             box("wall_a", [0.12, 0.05, 0.17], [0.3, 0.02, 0.3], "wall"), box("wall_b", [0.12, -0.05, 0.17], [0.3, 0.02, 0.3], "wall")]
    d = Design("golden", "A", parts, [Joint("h", "revolute", "post", "flap", [0, 0, 1], [0.02, 0, 0.17], [-1.5, 1.5])],
               None, {"scale_mode": "metric"})
    occs = analysis.occupancies(d.parts)
    sim = M.Sim(d, d.parts, V33._pairs(d.parts, occs, d.joints), [o.volume * 1000 for o in occs])
    r = sim.actuate(next(iter(sim.tree)))
    good = r.get("achieved", 1.0) < 0.5
    print(f"{'ok ' if good else 'FAIL'} sim flap between two walls is blocked by them                {r}")
    return ok & good


def main():
    base = measure(control())
    ok_all = check_lexicon() & check_bundle() & check_moving_set() & check_v35() & check_sim_v35()
    for name, d, expect in scenes():
        m = measure(d)
        fails = []
        for key, (op, val) in expect.items():
            v = m[key]
            good = {"gt": lambda: v > val, "lt": lambda: v < val, "eq": lambda: v == val, "contains": lambda: val in v}[op]()
            if not good:
                fails.append(f"{key}={v!r} expected {op} {val!r}")
        # every check not named in `expect` must match the control
        for key in ("n_components", "n_colliding", "joint_issue", "sweep_blocked", "supported_prefix_frac"):
            if key not in expect and m[key] != base[key] and not (name == "floating_part" and key == "supported_prefix_frac"):
                fails.append(f"{key} moved: {m[key]!r} vs control {base[key]!r}")
        if name != "com_outside_support" and m["tilt"] <= 0:
            fails.append(f"tilt {m['tilt']:.1f} <= 0")
        status = "ok " if not fails else "FAIL"
        ok_all &= not fails
        print(f"{status} {name:28s} {'; '.join(fails)}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
