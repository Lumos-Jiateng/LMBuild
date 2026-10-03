"""Annotation arm for external generators: the generator's geometry is fixed, an agent declares the rest.

External generators (PartCrafter, PartPacker, Cube3D + CubePart, Particulate, PhysX-Anything, ...) emit part
meshes and at most some joints, roles or materials, so most rule-based dimensions are `skipped` for them. This arm
asks: given exactly this geometry, how well can the object be completed as a functional, buildable product? An
annotator agent sees the generated parts and uses only the declaration tools of the tool protocol:
  set_part_info (role, material), add_joint / remove_joint, set_assembly_sequence, plus get_scene, measure,
  list_materials, check, render and submit. Tools that change geometry are refused.
Whatever the generator declared itself (roles, materials, joints) is loaded first and can be kept or corrected;
the trace records every change. Scores of this arm belong to the pair generator + annotator and are reported as
their own rows next to the generator's raw rows, never replacing them.

Frames: the annotator works in the frame the evaluator uses (`evaluate.prepare`: oracle scale and yaw for unitless
outputs, grounded), so joint origins it measures are the ones scored. The exported design is mapped back to the
generator's raw frame, so the evaluator's own `prepare` reproduces the annotator's frame exactly.

    .venv_eval/bin/python -m ppbench.v2.session init-annotate --state S.json --task swivel_office_chair \
        --from partcrafter_np15_s0 --system claude-sonnet-5 --out results/v2/swivel_office_chair/annotated/partcrafter_np15_s0__claude-sonnet-5
    (then prompt / call / finish as for tool runs)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ppbench.v2.design import Design, Joint, Part, axis_angle
from ppbench.v2.env import BUDGETS, TOOLS, AssemblyEnv

ANNOTATE_PROTOCOL = "v2.2-annotate"
DISABLED = {"place_part", "move_part", "attach", "remove_part", "create_part", "list_parts", "inspect_part", "view_catalog_sheet"}
ANNOTATE_TOOLS = [t for t in TOOLS if t[0] not in DISABLED]
ANNOTATE_BUDGETS = dict(BUDGETS, max_tool_calls=160, max_check_calls=6, max_render_calls=4)
LINEAR_JOINTS = {"prismatic"}

SYSTEM_ANNOTATE = """You are completing a 3D object that another system generated from the prompt and photo below. Its geometry
is fixed: you cannot add, move, resize or delete parts, and you cannot create new ones. Your job is to declare, as a product
engineer would, what the object is made of and how it works, so it can be judged as a real, functional, buildable object:

1. set_part_info for EVERY part: a role (what the part is, in plain words: seat, backrest, armrest, gas lift cylinder,
   caster wheel, five-star base, ...) and a material from list_materials.
2. add_joint for every connection the real object needs between these parts: fixed joints where parts are rigidly attached,
   and moving joints (revolute, continuous, prismatic, cylindrical, ball) with axis, origin and limits where the real object
   moves. A joint connects two parts only if its origin lies within 2 cm of BOTH parts' surfaces: use measure to find the
   closest points and put the origin there.
3. set_assembly_sequence: the order in which a person would assemble these parts, every part exactly once, with the
   straight-line direction each part is inserted along.

Declare only what this geometry supports. If a part the real object needs is missing or fused into another part, do not
relabel an unrelated part to stand in for it: give each part the role it actually has. Some roles, materials or joints may
already be declared by the generator: keep them if they are right and correct them if not.

World frame: metres, +Z up, the ground is z = 0, the front of the object faces -Y. Parts are named part_00, part_01, ...;
get_scene lists their boxes, render shows them (each part in its own colour). check reports connectivity, collisions,
stability and missing declarations. When everything is declared, call submit."""


def tool_docs_annotate() -> str:
    lines = []
    for name, args, desc in ANNOTATE_TOOLS:
        a = ", ".join(f"{k}: {v}" for k, v in args.items())
        lines.append(f"- {name}({a}): {desc}")
    return "\n".join(lines)


def _frame(raw: Design, prepared: Design, meta: dict):
    """(c, s, R) with prepared = R @ ((raw - c) * s), recovered from evaluate.prepare's rule and verified on the vertices."""
    s = float(meta.get("oracle_scale", 1.0))
    V = np.vstack([p.vertices for p in raw.parts])
    if "oracle_scale" in meta:
        c = np.r_[(V.max(0)[:2] + V.min(0)[:2]) / 2, V.min(0)[2]]
    else:
        c = np.zeros(3)
    R = axis_angle([0, 0, 1], math.radians(meta.get("oracle_yaw_deg", 0) or 0))
    P = np.vstack([p.vertices for p in prepared.parts])
    fwd = ((V - c) * s) @ R.T
    if not np.allclose(fwd, P, atol=1e-5 * max(1.0, float(np.abs(P).max()))):
        raise RuntimeError("could not reproduce evaluate.prepare's frame for this design")
    return c, s, R


class AnnotationEnv(AssemblyEnv):
    """AssemblyEnv over fixed generated parts (source kind 'fixed', meshes in <workdir>/fixed_parts.npz)."""

    def __init__(self, task, condition_id, rounds=1, state=None, workdir=None):
        super().__init__(task, "A", condition_id, rounds, budgets=ANNOTATE_BUDGETS, state=state, workdir=workdir)
        self.s.setdefault("mode", "annotate")
        self._fixed = None

    def _fixed_meshes(self):
        if self._fixed is None:
            z = np.load(self.workdir / "fixed_parts.npz")
            self._fixed = {k[2:]: (z[k], z["f_" + k[2:]]) for k in z.files if k.startswith("v_")}
        return self._fixed

    def _local(self, source):
        if source["kind"] == "fixed":
            return self._fixed_meshes()[source["id"]]
        return super()._local(source)

    def call(self, name, args):
        if name in DISABLED:
            c = self.s["counters"]
            c["tool_calls"] += 1
            c["errors"] += 1
            res = {"ok": False, "error": f"{name} is not available here: the geometry is fixed. Declare roles and materials "
                                         "(set_part_info), joints (add_joint) and the assembly sequence (set_assembly_sequence)."}
            self.s["trace"].append({"t": round(time.time() - self.s["started"], 2), "round": self.s["round"], "tool": name,
                                    "args": args, "ok": False, "error": res["error"], "wall_s": 0.0})
            return res
        return super().call(name, args)

    @classmethod
    def from_state(cls, st, task, workdir=None):
        return cls(task, st["condition_id"], st["rounds"], state=st, workdir=workdir)


def create(task, ext, out: Path, condition_id: str = "name_only+image"):
    """Build the annotation env for one external design; returns (env, info). `ext` is a discovered key of the chair
    demo layout, or an item dict {key, kind: external, system, prefix, path} (core-set layout, ppbench.v2.core_externals)."""
    from ppbench.v2 import evaluate as E
    from ppbench.v2.report import discover, load_design
    item = ext if isinstance(ext, dict) else next(i for i in discover(task.id) if i["key"] == ext and i["kind"] == "external")
    ext_key = item["key"]
    raw = load_design(task, item)
    prepared = load_design(task, item)
    notes = E.prepare(prepared, task.reference())
    c, s, R = _frame(raw, prepared, prepared.meta)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = AnnotationEnv(task, condition_id, rounds=1, workdir=out)
    arrays, ids = {}, {}
    for k, (pr, pp) in enumerate(zip(raw.parts, prepared.parts)):
        iid = f"part_{k:02d}"
        ids[pr.id] = iid
        arrays["v_" + iid] = np.asarray(pp.vertices, np.float64)
        arrays["f_" + iid] = np.asarray(pp.faces, np.int64)
        env.s["instances"][iid] = {"source": {"kind": "fixed", "id": iid, "generator_part": pr.id},
                                   "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
                                   "role": pr.role, "material": pr.material}
    np.savez_compressed(out / "fixed_parts.npz", **arrays)
    for j in raw.joints:   # generator joints, mapped into the annotator's frame
        if j.parent not in ids or j.child not in ids:
            continue
        o = None if j.origin is None else (R @ ((np.asarray(j.origin) - c) * s)).round(6).tolist()
        a = None if j.axis is None else (R @ np.asarray(j.axis, float)).round(6).tolist()
        lim = None if j.limits is None else ([float(x) * s for x in j.limits] if j.type in LINEAR_JOINTS else list(j.limits))
        jid = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in j.id)[:48] or f"j{len(env.s['joints'])}"
        env.s["joints"][jid] = {"id": jid, "type": j.type, "parent": ids[j.parent], "child": ids[j.child],
                                "axis": a, "origin": o, "limits": lim}
    info = {"generator_key": ext_key, "generator_system": item["system"], "generator_meta": raw.meta, "prepare_notes": notes,
            "frame": {"c": c.tolist(), "s": s, "yaw_deg": prepared.meta.get("oracle_yaw_deg", 0) or 0},
            "part_ids": ids, "generator_declared": {"roles": sum(1 for p in raw.parts if p.role),
                                                    "materials": sum(1 for p in raw.parts if p.material),
                                                    "joints": len(env.s["joints"])}}
    env.s["protocol"] = ANNOTATE_PROTOCOL
    return env, info


def prompt_text(env, st) -> str:
    cond = env.task.condition(env.s["condition_id"])
    g = st["annotation"]
    task_lines = [f"TASK THE GENERATOR WAS GIVEN: {cond['text']}"]
    if cond.get("image"):
        task_lines.append(f"The condition image {cond['image']['id']} is at {cond['image']['path']} (open it to look at it).")
    lines = [SYSTEM_ANNOTATE, "", "TOOLS", tool_docs_annotate(), "", *task_lines,
             "",
             f"THE GIVEN OBJECT: generated by {g['generator_system']} ({g['generator_meta'].get('family', 'external generator')}); "
             f"{len(env.s['instances'])} parts. The generator already declared {g['generator_declared']['roles']} roles, "
             f"{g['generator_declared']['materials']} materials and {g['generator_declared']['joints']} joints.",
             "Start with get_scene and render to see the parts."]
    return "\n".join(lines)


def export(env, st) -> Design:
    """The annotated design in the generator's raw frame, with the generator's meta and the annotation record."""
    g = st["annotation"]
    c = np.asarray(g["frame"]["c"])
    s = float(g["frame"]["s"])
    R = axis_angle([0, 0, 1], math.radians(g["frame"]["yaw_deg"]))
    inv = lambda p: (np.asarray(p, float) @ R) / s + c          # R orthonormal: (R^T p) as row vectors is p @ R
    parts = []
    for p in env.parts():
        parts.append(Part(p.id, inv(p.vertices), p.faces, p.role, p.material, p.source))
    joints = []
    for j in env.joints():
        o = None if j.origin is None else inv(j.origin).round(6).tolist()
        a = None if j.axis is None else (np.asarray(j.axis, float) @ R).round(6).tolist()
        lim = None if j.limits is None else ([float(x) / s for x in j.limits] if j.type in LINEAR_JOINTS else list(j.limits))
        joints.append(Joint(j.id, j.type, j.parent, j.child, a, o, lim))
    seq = None
    if env.s["sequence"]:
        seq = [{"part": st_["part"], "direction": None if st_["direction"] is None else
                (np.asarray(st_["direction"], float) @ R).round(6).tolist()} for st_ in env.s["sequence"]]
    gm = dict(g["generator_meta"])
    c_ = env.s["counters"]
    meta = {**gm, "declares": {"roles": True, "materials": True, "joints": True, "sequence": True},
            "annotation": {"annotator": st["system"], "generator": g["generator_system"], "generator_key": g["generator_key"],
                           "protocol": ANNOTATE_PROTOCOL, "counters": c_, "submitted_final": env.s["done"],
                           "generator_declared": g["generator_declared"], "transport": st.get("transport")},
            "family": f"{gm.get('family', 'external generator')} + annotator {st['system']}",
            "input": gm.get("input"), "wall_s": round(time.time() - env.s["started"], 1)}
    return Design(f"{g['generator_system']}+{st['system']}", "external", parts, joints, seq, meta)
