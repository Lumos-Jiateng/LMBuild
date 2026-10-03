"""Category statistics from the Artiverse population of the task's object type.

The reference is one realisation; the population says what is typical. For
the office chair this reads every `swivel_chair` object whose geometry is on
disk (both Artiverse roots), excluding the task's own reference so a prior is
never computed from the answer, and records per object:

* extents in the world frame (height, and the two horizontal extents sorted);
* canonical role counts (through the task lexicon; unmatched labels kept);
* the material class of every part, by canonical role;
* joints: child role, parent role, type, and whether the axis is vertical.

    .venv_eval/bin/python -m ppbench.v2.population swivel_office_chair swivel_chair
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

from ppbench.v2 import glb
from ppbench.v2.lexicon import Lexicon
from ppbench.v2.task import ARTIVERSE, RESULTS, Task, to_zup


def object_stats(d, lex):
    mid = d.name
    seg = glb.read_nodes(d / f"{mid}.segmented.glb")
    table = glb.node_table(d / f"{mid}.segmented.glb")
    mats = json.loads((d / "material.json").read_text()) if (d / "material.json").exists() else {}
    art = json.loads((d / f"{mid}.articulations.json").read_text()) if (d / f"{mid}.articulations.json").exists() else {}
    if not seg:
        return None
    allv = to_zup(np.vstack([s["vertices"] for s in seg]))
    ext = allv.max(0) - allv.min(0)
    labels = {}
    by_index = {n["index"]: n for n in table}
    for n in table:
        ex = n["extras"]
        if ex.get("id") is not None:
            labels[ex["id"]] = ex.get("label") or ex.get("name") or n["name"]
    # a virtual group (e.g. "support" = seat + backrest + column) takes the role of its
    # most specific member, in the order the task's claims care about
    order = ["seat", "backrest", "armrest", "lever", "caster wheel", "caster", "gas lift cylinder", "base"]
    for n in table:
        ex = n["extras"]
        if ex.get("type") == "ArticulatedPartVirtual" and lex.canon(labels.get(ex.get("id"), "")) is None:
            mem = [lex.canon(labels.get(by_index[c]["extras"].get("id"), "")) for c in n["children"] if c in by_index]
            mem = [m for m in mem if m]
            if mem:
                labels[ex["id"]] = min(mem, key=lambda m: order.index(m) if m in order else 99)
    roles, unmatched, mat_by_role = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    for s in seg:
        pid = s["extras"].get("id")
        lab = labels.get(pid, s["name"])
        c = lex.canon(lab)
        if c is None:
            unmatched[str(lab)] += 1
            continue
        roles[c] += 1
        m = (mats.get(str(pid)) or {}).get("material")
        if m:
            mat_by_role[c][m] += 1
    joints = []
    for a in art.get("articulations") or []:
        ax = to_zup(a.get("axis") or [0, 0, 0])
        nrm = np.linalg.norm(ax)
        joints.append({"type": a.get("type"), "child": lex.canon(labels.get(a["pid"], "")),
                       "parent": [lex.canon(labels.get(b, "")) for b in a.get("base") or []],
                       "axis_vertical_cos": float(abs(ax[2]) / nrm) if nrm > 0 else None,
                       "range": [a.get("rangeMin"), a.get("rangeMax")],
                       "prismatic_range": [a.get("prismatic_rangeMin"), a.get("prismatic_rangeMax")]})
    return {"model_id": mid, "extent_m": {"height": float(ext[2]), "horizontal_sorted": sorted(map(float, ext[:2]), reverse=True)},
            "roles": dict(roles), "unmatched": dict(unmatched), "materials_by_role": {k: dict(v) for k, v in mat_by_role.items()},
            "joints": joints, "n_parts": len(seg)}


def build(task_id: str, category: str):
    task = Task(task_id)
    lex = Lexicon(task_id)
    ref_mid = task.raw["reference"]["record"].split("-", 1)[1]
    seen, rows = set(), []
    for root in ARTIVERSE:
        for d in sorted((root / category).glob("*/*")):
            if d.name in seen or d.name == ref_mid or not (d / f"{d.name}.segmented.glb").exists():
                continue
            seen.add(d.name)
            try:
                r = object_stats(d, lex)
            except Exception as e:  # a broken file is recorded, not fatal
                r = {"model_id": d.name, "error": repr(e)[:200]}
            if r:
                rows.append(r)
    ok = [r for r in rows if "error" not in r]
    agg = {"category": category, "n_objects": len(ok), "n_errors": len(rows) - len(ok), "excluded_reference": ref_mid}
    H = np.array([r["extent_m"]["height"] for r in ok])
    W = np.array([r["extent_m"]["horizontal_sorted"][0] for r in ok])
    D = np.array([r["extent_m"]["horizontal_sorted"][1] for r in ok])
    q = lambda a: {"p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
                   "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95))}
    agg["extent_m"] = {"height": q(H), "width": q(W), "depth": q(D)}
    role_presence = collections.Counter()
    role_counts = collections.defaultdict(list)
    mat = collections.defaultdict(collections.Counter)
    jt = collections.defaultdict(collections.Counter)
    vert = collections.defaultdict(list)
    for r in ok:
        for k, v in r["roles"].items():
            role_presence[k] += 1
            role_counts[k].append(v)
        for k, c in r["materials_by_role"].items():
            mat[k].update(c)
        for j in r["joints"]:
            jt[j["child"]][j["type"]] += 1
            if j["axis_vertical_cos"] is not None:
                vert[j["child"]].append(j["axis_vertical_cos"])
    agg["role_presence_frac"] = {k: v / len(ok) for k, v in role_presence.most_common()}
    agg["role_count_median"] = {k: float(np.median(v)) for k, v in role_counts.items()}
    agg["material_class_by_role"] = {k: dict(v) for k, v in mat.items()}
    agg["joint_types_by_child_role"] = {str(k): dict(v) for k, v in jt.items()}
    agg["joint_axis_vertical_cos_median"] = {str(k): float(np.median(v)) for k, v in vert.items()}
    unm = collections.Counter()
    for r in ok:
        unm.update(r["unmatched"])
    agg["unmatched_labels"] = dict(unm.most_common(40))
    out = RESULTS / task_id / "population.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"aggregate": agg, "objects": rows}, indent=1))
    print(json.dumps(agg, indent=1)[:4000])


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
