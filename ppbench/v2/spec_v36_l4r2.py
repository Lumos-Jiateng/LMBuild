"""R.2 (dim P.2, material plausibility) v3.6 -- coverage-complete scoring, one rule for every system (2026-10-02).

Why. Under v3.5 a part whose *name* gave no basis (no wiki claim, Artiverse prior or judged function for that name)
was dropped from the mean, and a design with no such part at all was "skipped" (removed from the task mean). Agents
name parts by role and reach 98-100 % part coverage; a system with generic names ("Base Body", "Button") was scored on
a few parts of some tasks and skipped on the rest -- PhysX-Anything 109/200 tasks, 290/1,018 parts -- which inflated
its mean (60.5; 33.0 if skipped tasks counted as 0). The asymmetry came from naming, not from materials.

Rule (v3.6), applied identically to every design that declares at least one library material:
  1. role by name, exactly as v3.5 (wiki claims + quote-verified Wikipedia materials, Artiverse category prior,
     reference parts in the same place, then the judged function of the part name);
  2. if the name gives no basis, role by place: the Level 2 geometric grounding (ppbench.v2.afford.rules
     geometric_match -- at least half the part's surface within 5 % of the diagonal of one claimed component's
     reference parts, after the same normalisation and best yaw as A.1) gives the component, and the component's
     name is used as the role (same three evidence sources as step 1);
  3. P.2 = (1/N) sum_i a_i over ALL N parts of the design; a part with no basis after steps 1-2 has a_i = 0
     ("not shown to be plausible"), as a part without a library material already had. No part and no task is
     dropped, so P.2 is never "skipped" for a design that declares materials.
Designs that declare no library material are unchanged (0 by the owner's rule; the tables show generators as N/A).
For full-coverage systems steps 2-3 change nothing; agents' v3.5 coverage is 0.98-1.00.

    python -m ppbench.v2.spec_v36_l4r2 build [--tasks all] [--workers 48] [--force]
writes results/v2/<task>/eval_v38p2/<key>.json (dims: P.2 only) for the r1main scope of the 30 reported systems.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ppbench.v2 import spec_v35 as S
from ppbench.v2.evaluate import _dim
from ppbench.v2.task import RESULTS

SPEC_VERSION = "v3.6-r2"
OUT = "eval_v38p2"
SYSTEMS = ["gpt-6-astra", "gpt-5.6-sol", "claude-opus-5", "claude-fable-5.1", "claude-sonnet-5", "claude-haiku-4.5",
           "qwen3.5-27b", "qwen3.5-35b-a3b", "gpt-oss-120b", "gemma-4-31b-it", "qwen3-vl-32b-instruct",
           "qwen3-vl-30b-a3b-instruct", "qwen3-vl-8b-instruct", "qwen3-4b-instruct-2507", "internvl3.5-38b",
           "internvl3.5-8b", "minicpm-v-4.5", "ernie-4.5-vl-28b-a3b", "ministral-3-8b-instruct-2512",
           "partcrafter_np15", "particulate-partcrafter", "partcrafter_np8", "particulate-partcrafter-np8",
           "partpacker", "particulate-partpacker", "physx-anything", "cubepart_core", "particulate-cube3d",
           "brickgpt", "legoace"]


EXTRA_FUNC_MATS = S.FUNC_MATS.parent / "part_function_materials_v36_extra.json"
_XFM = None


def judged_classes(task_id, name):
    """v3.5's judged-function lexicon, then the v3.6 extra table: the same prompt and model over part names the
    v3.5 table never saw (it was built from tool designs only, so generator part names were missing)."""
    global _XFM
    A, func = S.judged_classes(task_id, name)
    if A:
        return A, func
    if _XFM is None:
        _XFM = json.loads(EXTRA_FUNC_MATS.read_text()).get("tasks", {}) if EXTRA_FUNC_MATS.exists() else {}
    rec = (_XFM.get(task_id) or {}).get(S.role_key(name)) or {}
    return set(rec.get("classes") or []), rec.get("function")


def geometric_components(task_id, parts):
    """Component name per design part by where it lies (Level 2 grounding), or None."""
    from ppbench.v2.afford import rules as RL
    ref = RL.reference_geometry(task_id)
    if ref is None or not parts:
        return [None] * len(parts)
    D, DL = RL.sample_labelled(parts, seed=0)
    if not len(D):
        return [None] * len(parts)
    Dn, _ = RL.normalise(D)
    best = None
    for yaw in (0, 90, 180, 270):
        Rz = RL.rotz(yaw)
        f = RL.fscore(Dn @ Rz.T, ref["P"], RL.TAU)[0]
        if best is None or f > best[0]:
            best = (f, Rz)
    Dr = Dn @ best[1].T
    _, _, comps, _ = RL.context(task_id)
    name = {c["id"]: c["name"] for c in comps}
    out = []
    for i in range(len(parts)):
        cid = RL.geometric_match(ref, Dr, DL, i)
        out.append(name.get(cid) if cid else None)
    return out


def p2_material_v36(design, task, lex, ref):
    from ppbench.v2 import materials
    parts = design.parts
    res_m = [materials.resolve(p.material) for p in parts]
    cls = [r[1] if r else None for r in res_m]
    if not any(cls):
        return _dim("declared+computed", "fail", 0.0,
                    {"declared": any(p.material for p in parts), "n_parts_with_material": 0},
                    ["no part carries a library material: scored 0 (owner's rule, 2026-09-22)"])
    canon = [lex.canon(p.role) for p in parts]
    wreqs = S.wiki_material_requirements(task, lex)
    wiki_by_role = defaultdict(set)
    for w in wreqs:
        for r in (w["roles"] or []):
            wiki_by_role[r] |= set(w["allowed"])
    ref_at = [set() for _ in parts]                      # reference parts in the same place (as v3.5)
    rparts = [p for p in (ref or {}).get("parts", []) if p.get("material_class")]
    if rparts:
        dc, dd = S._frame([p.vertices for p in parts])
        rc, rd = S._frame([p["vertices"] for p in ref["parts"]])
        dsets = [S._norm_cells(p.vertices, p.faces, dc, dd, S.MAT_GRID) for p in parts]
        for rp in rparts:
            rs = S._norm_cells(rp["vertices"], rp["faces"], rc, rd, S.MAT_GRID)
            acc = {rp["material_class"]} | S.ref_exterior(task, rp) | S.prior_role(task.id, lex.canon(rp.get("role_raw")))
            for i, ds in enumerate(dsets):
                if ds and len(rs & ds) >= S.REF_OVERLAP_MIN * len(ds):
                    ref_at[i] |= acc

    def basis_for(role_name, c, i):
        A = set(wiki_by_role.get(c, set())) | S.prior_role(task.id, c) | (ref_at[i] if i is not None else set())
        if A:
            return A, "evidence", None
        A, func = judged_classes(task.id, role_name)
        return (A, "judged", func) if A else (set(), None, None)

    geo = None
    rows_, scores, src_count = [], [], defaultdict(int)
    for i, p in enumerate(parts):
        A, src, func = basis_for(p.role or p.id, canon[i], i)
        via = "name"
        if not A:
            if geo is None:
                try:
                    geo = geometric_components(task.id, parts)
                except Exception:
                    geo = [None] * len(parts)
            comp = geo[i]
            if comp:
                A, src, func = basis_for(comp, lex.canon(comp), None)
                via = "place"
        if not A:
            a, src = 0.0, "no basis"
        else:
            a = S._agree(cls[i], A)
        scores.append(a)
        src_count[f"{src} ({via})" if A else "no basis"] += 1
        rows_.append({"part": p.id, "role": p.role, "material": p.material, "class": cls[i], "accepted": sorted(A),
                      "basis": src, "role_via": via if A else None, "function": func, "score": a})
    score = float(np.mean(scores))
    n_basis = sum(1 for r in rows_ if r["accepted"])
    return _dim("declared+computed", "pass" if score >= 0.8 else ("fail" if score == 0 else "degraded"), score, {
        "n_parts": len(parts), "n_with_basis": n_basis, "coverage": n_basis / len(parts),
        "basis": dict(src_count), "n_without_material": sum(1 for c_ in cls if not c_),
        "parts": rows_[:120], "design_classes": sorted({c_ for c_ in cls if c_})}, [
        "P.2 v3.6 = mean over ALL parts of a_i; a_i = 1 accepted class, 0.5 same family, 0 otherwise, without a "
        "library material, or without any basis",
        "basis: role by name (v3.5 evidence, then judged function); if none, role by place (Level 2 geometric "
        "grounding to a claimed component)"])


def _lex_stamp():
    """The judged-function lexicon grows when missing part names are judged; a record is stale when it grows."""
    return "-".join(str(p.stat().st_size) if p.exists() else "none" for p in (S.FUNC_MATS, EXTRA_FUNC_MATS))


def _one(args):
    task_id, name, force = args
    src, dst = RESULTS / task_id / "eval_v35" / name, RESULTS / task_id / OUT / name
    try:
        rec = json.loads(src.read_text())
    except (OSError, ValueError):
        return "bad"
    want = f"{rec.get('stamp')}|{SPEC_VERSION}|{_lex_stamp()}"
    if not force and dst.exists():
        try:
            if json.loads(dst.read_text()).get("stamp") == want:
                return "cached"
        except (OSError, ValueError):
            pass
    item = rec.get("item") or {}
    try:
        task, design = S.load(task_id, item)
        if not design.parts:
            dim = _dim("computed", "fail", 0.0, {"n_parts": 0}, ["empty design"])
        else:
            try:
                ref = task.reference()
            except Exception:
                ref = None
            dim = p2_material_v36(design, task, S.Lexicon(task.id, task), ref)
        out = {k: rec[k] for k in ("system", "tier", "meta", "item") if k in rec}
        out.update({"spec": SPEC_VERSION, "derived_from": rec.get("spec"), "stamp": want,
                    "dims": {"P.2": dim}, "v35_P.2": (rec.get("dims") or {}).get("P.2", {}).get("score")})
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1, default=float))
        return "ok"
    except Exception:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.with_suffix(".error.txt").write_text(traceback.format_exc())
        return "error"


def build(tasks=None, workers=48, force=False, keys_file=None):
    ids = tasks or [x["task_id"] for x in json.loads(__import__("ppbench.v2.task", fromlist=["CORE"]).CORE.read_text())]
    jobs = []
    if keys_file:                      # 2026-10-02: explicit (task, key) list, e.g. the release export scope
        jobs = [(t, f"{k}.json", force) for t, k in (l.rstrip("\n").split("\t") for l in open(keys_file) if l.strip())
                if (RESULTS / t / "eval_v35" / f"{k}.json").exists()]
        ids = []
    for t in ids:
        d = RESULTS / t / "eval_v35"
        if d.is_dir():
            jobs += [(t, f.name, force) for f in sorted(d.glob("*.json"))
                     if f.name.split("__")[0] in SYSTEMS and S.in_scope(f.name, "r1main")]
    jobs.sort(key=lambda j: hash(j) % 997)
    print(f"{len(jobs)} records over {len(ids)} tasks", flush=True)
    tot, t0 = defaultdict(int), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=2), 1):
            tot[r] += 1
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)}  {dict(tot)}  {time.time() - t0:.0f}s", flush=True)
    print(f"TOTAL {dict(tot)}  {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keys-file", default=None)
    a = ap.parse_args(argv)
    build(None if a.tasks == "all" else a.tasks.split(","), a.workers, a.force, a.keys_file)


if __name__ == "__main__":
    main()
