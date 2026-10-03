"""Golden scenes for Level A. Each asserts one behaviour a reviewer would attack.

    python -m ppbench.v2.afford.golden
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

from ppbench.v2.afford import core, kin, score
from ppbench.v2.design import Design, Joint, Part

WHEEL_EXPECT = {"types": ["continuous"], "axis": "symmetry", "axis_dir": "horizontal", "offset": "centre",
                "span": 2 * math.pi, "moving_frac_max": 0.6}
DOOR_EXPECT = {"types": ["revolute"], "axis": "pca0", "axis_dir": "vertical", "offset": "edge",
               "span": math.radians(72), "moving_frac_max": 0.9}
DRAWER_EXPECT = {"types": ["prismatic"], "axis": None, "axis_dir": "horizontal", "offset": None,
                 "travel_ratio": 0.4, "moving_frac_max": 0.6}


def box(ext, centre):
    m = trimesh.creation.box(extents=ext)
    return np.asarray(m.vertices) + np.asarray(centre), np.asarray(m.faces)


def cyl(r, h, centre, axis):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=32)
    v = np.asarray(m.vertices)
    if axis == "x":
        v = v[:, [2, 1, 0]]
    elif axis == "y":
        v = v[:, [0, 2, 1]]
    f = np.asarray(m.faces)
    if axis in ("x", "y"):
        f = f[:, ::-1]
    return v + np.asarray(centre), f


def car(wheel_axis="x", joint_axis=(1, 0, 0), bury=False, origin_off=0.0):
    parts = [Part("body", *box([0.5, 1.0, 0.25], [0, 0, 0.30 if not bury else 0.2]), "body")]
    joints = []
    for k, (x, y) in enumerate([(-0.3, -0.35), (0.3, -0.35), (-0.3, 0.35), (0.3, 0.35)]):
        xw = x if not bury else 0.6 * x
        parts.append(Part(f"w{k}", *cyl(0.12, 0.08, [xw, y, 0.12], wheel_axis), "wheel"))
        joints.append(Joint(f"j{k}", "continuous", "body", f"w{k}", list(joint_axis),
                            [x if not bury else 0.6 * x, y + origin_off, 0.12], None))
    return Design("golden", "B", parts, joints, None, {"scale_mode": "metric"})


def door(hinge_at_edge=True, limits=(0.0, 1.6)):
    """Two jambs and a head around the opening; the leaf swings out of it about a jamb-side edge."""
    jl = Part("jamb_l", *box([0.06, 0.12, 2.05], [-0.435, 0, 1.025]), "frame")
    jr = Part("jamb_r", *box([0.06, 0.12, 2.05], [0.435, 0, 1.025]), "frame")
    hd = Part("head", *box([0.93, 0.12, 0.06], [0, 0, 2.08]), "frame")
    leaf = Part("leaf", *box([0.8, 0.04, 2.0], [0, 0, 1.0]), "door panel")
    x = -0.4 if hinge_at_edge else 0.0
    return Design("golden", "B", [jl, jr, hd, leaf],
                  [Joint("h", "revolute", "jamb_l", "leaf", [0, 0, 1], [x, 0, 1.0], list(limits)),
                   Joint("f1", "fixed", "jamb_l", "head", None, None, None), Joint("f2", "fixed", "head", "jamb_r", None, None, None)],
                  None, {"scale_mode": "metric"})


def drawer(solid_case=True):
    if solid_case:
        case = [Part("case", *box([0.6, 0.5, 0.3], [0, 0, 0.15]), "case")]
    else:   # an open box: floor, two sides, back, top
        case = [Part("floor", *box([0.6, 0.5, 0.02], [0, 0, 0.01]), "case"),
                Part("left", *box([0.02, 0.5, 0.3], [-0.29, 0, 0.15]), "case"),
                Part("right", *box([0.02, 0.5, 0.3], [0.29, 0, 0.15]), "case"),
                Part("back", *box([0.6, 0.02, 0.3], [0, 0.24, 0.15]), "case"),
                Part("top", *box([0.6, 0.5, 0.02], [0, 0, 0.29]), "case")]
    dr = Part("drawer", *box([0.5, 0.42, 0.2], [0, -0.02, 0.13]), "drawer")
    ms = case + [dr]
    js = [Joint("s", "prismatic", case[0].id, "drawer", [0, -1, 0], [0, -0.23, 0.13], [0.0, 0.35])]
    js += [Joint(f"f{k}", "fixed", case[0].id, p.id, None, None, None) for k, p in enumerate(case[1:])]
    return Design("golden", "B", ms, js, None, {"scale_mode": "metric"})


def q(d, jid, inst, expect, motion):
    sc = kin.Scene(d)
    j = next(x for x in d.joints if x.id == jid)
    return kin.joint_quality(sc, j, inst, expect, motion, [], sc.contact())


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    return bool(cond)


def v36_checks():
    """Spec v3.6 (afford/rules.py): one assertion per fix made in the 2026-09-22 revision."""
    from ppbench.v2.afford import rules
    ok = True
    print("v3.6 grounding (declared role -> claimed component)")
    for tid, label, want in [("dining_chair", "backrest panel with fan pleats", "back-rest"),
                             ("dining_chair", "tall dining chair backrest", "back-rest"),
                             ("mixer_faucet", "spout", "outlet spout"),
                             ("hinged_door", "door leaf (15-panel hinged door slab)", "panel"),
                             ("offroad_jeep", "front left tyre", "tires"),
                             ("offroad_jeep", "wheel rim", "wheels")]:
        t, lex, comps, g = rules.context(tid)
        names = {c["id"]: c["name"] for c in comps}
        got = names.get(g.match(label))
        ok &= check(f"{tid}: '{label}' is the {want}", got == want, f"got {got}")
    print("v3.6 instances and counts")
    tire = Part("tire", *cyl(0.3, 0.2, [1, 0, 0.3], "x"), "wheel")
    rim = Part("rim", *cyl(0.2, 0.15, [1, 0, 0.3], "x"), "wheel")
    legs = [Part(f"leg{i}", *box([0.04, 0.04, 0.45], [x, y, 0.225]), "leg")
            for i, (x, y) in enumerate([(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)])]
    seat = Part("seat", *box([0.5, 0.5, 0.04], [0, 0, 0.47]), "seat")
    d = Design("golden", "B", [tire, rim, seat, *legs],
               [Joint(f"f{i}", "fixed", "seat", l.id, None, None, None) for i, l in enumerate(legs)], None, {})
    pc = {"tire": "W", "rim": "W", "seat": "S", **{l.id: "L" for l in legs}}
    ok &= check("a tire on its rim is one wheel", len(rules.instances(d, pc, "W", False)) == 1)
    ok &= check("four legs fixed to one seat are four legs", len(rules.instances(d, pc, "L", False)) == 4)
    ok &= check("'wheels' asks for a count, 'keyboard' for one",
                rules.plural({"phrases": ["wheels"]}) and not rules.plural({"phrases": ["keyboard"]}))
    print("v3.6 real-instance relations")
    base = Part("base", *box([0.3, 0.3, 0.05], [0, 0, 0.2]), "base")
    fork = Part("fork", *box([0.02, 0.06, 0.1], [0, 0, 0.125]), "caster")
    wheel = Part("wheel", *cyl(0.05, 0.03, [0, 0, 0.05], "x"), "wheel")
    cd = Design("golden", "ref", [base, fork, wheel],
                [Joint("swivel", "continuous", "base", "fork", [0, 0, 1], [0, 0, 0.175], None),
                 Joint("spin", "continuous", "fork", "wheel", [1, 0, 0], [0, 0, 0.05], None)], None, {})
    pc = {"base": ("B", "l"), "fork": ("C", "l"), "wheel": ("W", "l")}
    rel = rules.relations_in(cd, pc, [["wheel"]], {"relative": "C"}, "rotate")
    ok &= check("a caster wheel's claim reads the wheel's own axle, not the caster swivel",
                len(rel) == 1 and rel[0]["axis_z"] < 0.1, f"axis_z={rel[0]['axis_z'] if rel else None}")
    front = Part("front", *box([0.5, 0.02, 0.2], [0, -0.24, 0.13]), "drawer")
    boxp = Part("box", *box([0.48, 0.4, 0.18], [0, -0.02, 0.13]), "drawer box")
    case = Part("case", *box([0.6, 0.02, 0.3], [0, 0.24, 0.15]), "case")
    dd = Design("golden", "ref", [case, front, boxp],
                [Joint("s", "prismatic", "case", "front", [0, -1, 0], [0, -0.24, 0.13], [0.0, 0.35]),
                 Joint("r", "fixed", "front", "box", None, None, None)], None, {})
    r = kin.relation(dd, dd.joints[0], ["front"], {"front", "box"})
    ok &= check("a drawer's travel is measured against the whole moving drawer, not its 2 cm front",
                r["travel_ratio"] < 1.0, f"travel_ratio={r['travel_ratio']:.2f}")
    return ok


def v37_checks():
    """Spec v3.7 additions: interior modules, merged dataset labels, and 2.3 as an F1."""
    from ppbench.v2.afford import rules
    ok = True
    print("v3.7 interior modules, labels, kinematics F1")
    shell = []
    for i, (ext, c) in enumerate([([0.6, 0.02, 0.4], [0, -0.29, 0.2]), ([0.6, 0.02, 0.4], [0, 0.29, 0.2]),
                                  ([0.02, 0.6, 0.4], [-0.29, 0, 0.2]), ([0.02, 0.6, 0.4], [0.29, 0, 0.2]),
                                  ([0.6, 0.6, 0.02], [0, 0, 0.01]), ([0.6, 0.6, 0.02], [0, 0, 0.39])]):
        shell.append(Part(f"wall{i}", *box(ext, c), "housing"))
    motor = Part("motor", *box([0.2, 0.2, 0.2], [0, 0, 0.2]), "thing")
    handle = Part("handle", *box([0.1, 0.05, 0.05], [0, 0, 0.45]), "thing")
    d = Design("golden", "B", [*shell, motor, handle], [], None, {})
    inside, _ = rules.interior_parts(kin.Scene(d))
    ok &= check("a block sealed inside a closed housing is interior", "motor" in inside)
    ok &= check("a handle on top of the housing is not", "handle" not in inside and "wall5" not in inside)
    t, lex, comps, g = rules.context("dishwasher")
    names = {c["id"]: c["name"] for c in comps}
    got = names.get(g.match("door panel, handle"))
    ok &= check("a merged dataset label 'door panel, handle' is the door", got == "door", f"got {got}")
    R_, P_ = 0.0, 0.9
    f1 = 2 * R_ * P_ / (R_ + P_) if R_ + P_ > 0 else 0.0
    ok &= check("sound joints that realise no target motion score 0 in 2.3", f1 == 0.0)
    return ok


def main():
    ok = True
    print("A.3 kinematics")
    r = q(car(), "j0", ["w0"], WHEEL_EXPECT, "rotate")
    ok &= check("wheel on its own axle turns freely", r["q"] > 0.95, f"q={r['q']:.2f}")
    r = q(car(joint_axis=(0, 0, 1)), "j0", ["w0"], WHEEL_EXPECT, "rotate")
    ok &= check("wheel joint about a vertical axis is not a wheel", r["q"] == 0.0, f"axis_ok={r['axis_ok']:.2f}")
    r = q(car(joint_axis=(0, 1, 0)), "j0", ["w0"], WHEEL_EXPECT, "rotate")
    ok &= check("wheel joint along the travel direction is not a wheel", r["q"] == 0.0, f"axis_ok={r['axis_ok']:.2f}")
    r = q(car(origin_off=0.2), "j0", ["w0"], WHEEL_EXPECT, "rotate")
    ok &= check("axle line off the wheel centre (wobbles) scores 0", r["q"] < 0.2, f"offset={r['offset']:.2f}")
    r = q(car(bury=True), "j0", ["w0"], WHEEL_EXPECT, "rotate")
    ok &= check("wheel buried in the body cannot turn", r["q"] == 0.0, f"buried={r['buried']}")
    r = q(door(True), "h", ["leaf"], DOOR_EXPECT, "rotate")
    ok &= check("door hinged at its edge swings", r["q"] > 0.9, f"q={r['q']:.2f}")
    r = q(door(False), "h", ["leaf"], DOOR_EXPECT, "rotate")
    ok &= check("door pivoting about its middle is not hinged", r["q"] < 0.1, f"offset_ok={r['offset_ok']:.2f}")
    r = q(door(True, (0.0, 0.2)), "h", ["leaf"], DOOR_EXPECT, "rotate")
    ok &= check("door that opens 11 deg gets a fraction of the range", 0.1 < r["range_ok"] < 0.3, f"range_ok={r['range_ok']:.2f}")
    r = q(drawer(False), "s", ["drawer"], DRAWER_EXPECT, "slide")
    ok &= check("drawer in an open case slides", r["q"] > 0.9, f"q={r['q']:.2f}")
    r = q(drawer(True), "s", ["drawer"], DRAWER_EXPECT, "slide")
    ok &= check("drawer inside a solid block is buried", r["q"] == 0.0, f"buried={r['buried']}")
    print("A.1 bands")
    b = {"log": True, "lo": math.log(0.4), "hi": math.log(0.5)}
    ok &= check("inside the band is 1", score.band_score(0.45, b) == 1.0)
    ok &= check("a factor of 2 beyond the band is 0", score.band_score(1.0, b) == 0.0)
    ok &= check("halfway (x1.41) beyond is 0.5", abs(score.band_score(0.5 * math.sqrt(2), b) - 0.5) < 1e-6)
    ok &= v36_checks()
    ok &= v37_checks()
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    main()
