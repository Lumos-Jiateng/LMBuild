"""v3.2 dimension layout: the v3.1 numbers, re-partitioned, plus one new dimension.

Moves (2026-09-20, the user's request):

    v3.1 1.1 integrity   -> 1.1 connectivity  +  1.2 collision      (split, no term is redefined)
    v3.1 1.2 stability   -> 1.3 stability
    v3.1 2.2 kinematics  -> 2.3 kinematics
    v3.1 2.3 operability -> 4.3 operability
    NEW                     2.2 part-level alignment
    v3.1 4.3 functional completeness: its part-presence half feeds 2.2, its connection half stays a
        reported metric inside 2.2; it is no longer a dimension of its own.

Nine of the eleven dimensions are therefore a rename or an exact re-partition of numbers metrics_v3
already computed, so nothing is re-voxelized or re-swept: `upgrade` derives a v3.2 record from a
stored v3.1 record. Only 2.2 is computed here, and it needs no geometry -- the design's declared
roles, the task's own cited claims (required parts P, subsystem parts F, kinematics K) and the
floating-part list 1.1 already recorded.

2.2 answers "is the part an affordance needs there at all", which is exactly what every P claim's own
verification line asks for ("a part declared with this role exists and is connected to the assembly"),
so a role that is present but floats off the assembly scores half.

    .venv_eval/bin/python -m ppbench.v2.spec_v32 build [--tasks all] [--workers 16] [--force]
    .venv_eval/bin/python -m ppbench.v2.spec_v32 tables [--condition name_only+image] [--rounds 3]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from ppbench.v2.evaluate import _dim, _mean
from ppbench.v2.lexicon import Lexicon
from ppbench.v2.metrics_v3 import CONNECTED_PARTS_FAIL, LARGEST_COMPONENT_FAIL, PEN_FRAC_FAIL
from ppbench.v2.task import RESULTS, Task

SPEC_VERSION = "v3.2"
DIMS = ["1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "3.1", "3.2", "4.1", "4.2", "4.3"]
NAMES = {"1.1": "connectivity", "1.2": "collision", "1.3": "stability",
         "2.1": "geometry", "2.2": "part alignment", "2.3": "kinematics",
         "3.1": "decomposition", "3.2": "aesthetics", "4.1": "material",
         "4.2": "assembly sequence", "4.3": "operability"}
RENAMED = {"1.3": "1.2", "2.1": "2.1", "2.3": "2.2", "3.1": "3.1", "3.2": "3.2",
           "4.1": "4.1", "4.2": "4.2", "4.3": "2.3"}          # v3.2 id -> v3.1 id, carried unchanged
STRUCTURE = ("1.1", "1.2", "1.3")                              # a fail here is not buildable
CRITICAL = ("1.1", "1.2", "1.3", "2.1", "2.3", "4.1", "4.2")   # the v3.1 critical set, under the new ids
MESH_SUFFIX = (".glb", ".obj", ".ply", ".stl")


# ---------------------------------------------------------------- 1.1 / 1.2: split, never redefined
def split_integrity(d11: dict):
    """v3.1 1.1 was mean(collision_free_pair_rate, (largest+connected)/2, whole_body_rate).
    Connectivity keeps the second and third term, collision keeps the first. Same numbers, two dimensions."""
    m = d11.get("metrics") or {}
    largest, conn = m.get("largest_component_volume_frac"), m.get("connected_part_frac")
    whole, cfree = m.get("whole_body_rate"), m.get("collision_free_pair_rate")
    pen = m.get("pen_volume_frac")
    shells = m.get("shell_only_parts") or []
    if d11.get("status") == "fail" and not m:      # the empty / unloadable design records carry no metrics
        z = _dim("computed", "fail", 0.0, {}, d11.get("notes") or [])
        return z, dict(z)
    s_conn = _mean([_mean([largest, conn]), whole])
    st_conn = ("fail" if (largest is not None and largest < LARGEST_COMPONENT_FAIL)
               or (conn is not None and conn < CONNECTED_PARTS_FAIL)
               else ("degraded" if shells else "pass"))
    conn_dim = _dim("computed", st_conn, s_conn, {
        "n_parts": m.get("n_parts"), "n_components": m.get("n_components"),
        "largest_component_volume_frac": largest, "connected_part_frac": conn,
        "whole_body_rate": whole, "floating_parts": m.get("floating_parts") or [],
        "shell_only_parts": shells},
        ["score = mean((largest_component_volume_frac + connected_part_frac) / 2, whole_body_rate)"]
        + (["volumes of shell-only parts (open meshes) are under-counted"] if shells else []))
    st_coll = ("fail" if (pen is not None and pen > PEN_FRAC_FAIL)
               else ("degraded" if (m.get("n_colliding_pairs") or 0) else "pass"))
    coll_dim = _dim("computed", st_coll, cfree, {
        "collision_free_pair_rate": cfree, "n_touching_pairs": m.get("n_touching_pairs"),
        "n_colliding_pairs": m.get("n_colliding_pairs"), "pen_volume_frac": pen,
        "pen_volume_cm3": m.get("pen_volume_cm3"), "voxel_res_m": m.get("voxel_res_m"),
        # graded alternative, reported only: the spec prefers volume over a pair count, but scoring it
        # would silently redefine the dimension, so the score stays the v3.1 pair rate until asked.
        "graded_penetration_score": None if pen is None else max(0.0, 1 - pen / PEN_FRAC_FAIL)},
        ["score = collision_free_pair_rate (the v3.1 term, unchanged); penetration volume is reported"])
    return conn_dim, coll_dim


# ---------------------------------------------------------------- 2.2 part-level alignment (new)
def design_roles(item: dict):
    """[(part id, declared role)] for a stored design, without loading a single mesh."""
    p = Path(item["path"])
    if item.get("kind", "tool") in ("tool", "annotated"):
        rec = json.loads((p / "design.json").read_text())
        return [(x.get("id"), x.get("role")) for x in rec.get("parts", [])]
    if p.suffix == ".ldr" or not p.exists():                       # brick baselines: bricks, no roles
        return []
    if (p / "bundle" / "bundle.json").exists():                    # articulated bundles: the producer's names
        spec = json.loads((p / "bundle" / "bundle.json").read_text())
        return [(str(x["id"]), x.get("name")) for x in spec.get("parts", []) if x.get("file")]
    if (p / "parts").is_dir():                                     # one file per part; CubePart got the schema
        files = sorted(f for f in (p / "parts").iterdir() if f.suffix.lower() in MESH_SUFFIX)
        named = (item.get("prefix") or item["system"]).startswith("cubepart")
        return [(f"part_{k:02d}", re.sub(r"^\d+_", "", f.stem).replace("_", " ") if named else None)
                for k, f in enumerate(files)]
    return []


def part_alignment(task: Task, lex: Lexicon, roles, floating, declares) -> dict:
    """Are the parts an affordance needs there at all? One term per claim group, missing parts named."""
    canon = {pid: lex.canon(r) for pid, r in roles}
    floating = set(floating or [])
    groups = {
        "required_parts": [p["part"] for p in task.required_parts],
        "subsystem_parts": sorted({n for f in task.subsystems for n in f.get("parts", [])}),
        "kinematic_parts": sorted({k[w] for k in task.kinematics for w in ("moving_part", "relative_to")
                                   if k.get(w)}),
    }
    if not any(r for _, r in roles) and not (declares or {}).get("roles"):
        return _dim("computed", "skipped", None,
                    {"declared_roles": False, "n_parts": len(roles),
                     "claims": {g: len(v) for g, v in groups.items()}},
                    ["no part carries a role: which part is which cannot be located, so presence is not measurable"])
    rows, out = {}, {}
    for g, phrases in groups.items():
        terms, missing, floaters = [], [], []
        for ph in phrases:
            hits = [pid for pid, _ in roles if lex.satisfies(ph, canon[pid])]
            if not hits:
                terms.append(0.0)
                missing.append(ph)
            elif all(h in floating for h in hits):
                terms.append(0.5)                 # the claim's own check is "exists AND is connected"
                floaters.append(ph)
            else:
                terms.append(1.0)
        rows[g] = {"n": len(phrases), "score": (sum(terms) / len(terms)) if terms else None,
                   "missing": missing, "present_but_floating": floaters}
        out[g] = rows[g]["score"]
    score = _mean(list(out.values()))
    unrecognised = sorted({r for pid, r in roles if r and canon[pid] is None})
    return _dim("computed", "skipped" if score is None else ("fail" if score == 0 else "pass"), score, {
        "declared_roles": True, "n_parts": len(roles),
        "required_part_coverage": out.get("required_parts"),
        "subsystem_part_coverage": out.get("subsystem_parts"),
        "kinematic_part_coverage": out.get("kinematic_parts"),
        "groups": rows, "unrecognised_roles": unrecognised[:20],
        "unrecognised_role_frac": (len(unrecognised) / max(len([r for _, r in roles if r]), 1))},
        ["score = mean of the three claim-group coverages; a role that exists but floats off the assembly scores 0.5",
         "matching is the task's own auto lexicon plus its authored bridges, never a hand-written per-object predicate"])


# ---------------------------------------------------------------- record
def upgrade(rec: dict, task: Task, lex: Lexicon) -> dict:
    d31 = rec.get("dims") or {}
    D = {}
    conn, coll = split_integrity(d31.get("1.1") or {})
    D["1.1"], D["1.2"] = conn, coll
    for new, old in RENAMED.items():
        if old in d31:
            D[new] = dict(d31[old])
    try:
        roles = design_roles(rec["item"])
    except (OSError, json.JSONDecodeError, KeyError):
        roles = []
    D["2.2"] = part_alignment(task, lex, roles, (conn.get("metrics") or {}).get("floating_parts"),
                              (rec.get("meta") or {}).get("declares"))
    # 4.3 operability locates parts by role, exactly as 2.2 does: on an output that carries no role at
    # all (the mesh generators), its predicates cannot resolve and a 0 would read as "unusable" when the
    # truth is "not expressible". Same rule as 2.2 and as 4.1 on a design with no material.
    if D.get("4.3") and D["4.3"].get("score") is not None and not any(r for _, r in roles) \
            and not ((rec.get("meta") or {}).get("declares") or {}).get("roles"):
        D["4.3"] = dict(D["4.3"], status="skipped", score=None,
                        metrics={**(D["4.3"].get("metrics") or {}), "score_when_roleless_v31": D["4.3"]["score"]},
                        notes=(D["4.3"].get("notes") or []) + ["no part carries a role: capabilities cannot be located"])
    # the connection half of the retired 4.3 keeps its numbers, as a reported metric of 2.2
    old43 = (d31.get("4.3") or {}).get("metrics") or {}
    if old43:
        D["2.2"].setdefault("metrics", {})["subsystem_connection"] = old43.get("connection")
        D["2.2"]["metrics"]["subsystem_presence_v31"] = old43.get("presence")
    out = {k: rec[k] for k in ("system", "tier", "meta", "notes", "item") if k in rec}
    crit = [k for k in CRITICAL if (D.get(k) or {}).get("status") == "fail"]
    scored = [v["score"] for v in D.values() if v.get("score") is not None]
    out.update({"spec": SPEC_VERSION, "derived_from": rec.get("spec"), "dims": D,
                "headline": {"critical_fail": crit,
                             "buildable": not any(k in crit for k in STRUCTURE),
                             "overall": (sum(scored) / len(scored)) if scored else None,
                             "n_scored_dims": len(scored), "gate_pending": True},
                "stamp": f"{rec.get('stamp')}|{SPEC_VERSION}"})
    return out


def build_task(args):
    task_id, force = args
    base = RESULTS / task_id
    src, dst = base / "eval_v3", base / "eval_v32"
    if not src.is_dir():
        return task_id, 0, 0, 0
    task = Task(task_id, snapshot=base / "task_snapshot_core.json")
    lex = Lexicon(task_id, task)
    dst.mkdir(parents=True, exist_ok=True)
    ok = cached = bad = 0
    for f in sorted(src.glob("*.json")):
        o = dst / f.name
        try:
            rec = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            bad += 1
            continue
        want = f"{rec.get('stamp')}|{SPEC_VERSION}"
        if not force and o.exists():
            try:
                if json.loads(o.read_text()).get("stamp") == want:
                    cached += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        try:
            o.write_text(json.dumps(upgrade(rec, task, lex), indent=1, default=float))
            ok += 1
        except Exception:
            bad += 1
    return task_id, ok, cached, bad


def build(tasks=None, workers=16, force=False):
    from concurrent.futures import ProcessPoolExecutor
    ids = tasks or [t["task_id"] for t in json.loads(
        (__import__("ppbench.v2.task", fromlist=["CORE"]).CORE).read_text())]
    tot = defaultdict(int)
    with ProcessPoolExecutor(workers) as ex:
        for task_id, ok, cached, bad in ex.map(build_task, [(i, force) for i in ids]):
            tot["ok"] += ok
            tot["cached"] += cached
            tot["bad"] += bad
            print(f"{task_id:28s} written {ok:5d}  cached {cached:5d}  failed {bad:3d}", flush=True)
    print(f"{'TOTAL':28s} written {tot['ok']:5d}  cached {tot['cached']:5d}  failed {tot['bad']:3d}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args(argv)
    build(None if ns.tasks == "all" else ns.tasks.split(","), ns.workers, ns.force)


if __name__ == "__main__":
    main()
