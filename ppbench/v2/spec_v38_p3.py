"""Spec v3.8 for P.3 Operability (2026-10-02): presence counts verified roles; everything else as v3.5.

v3.5's `part_present` link counted a part by its declared name alone. Under v3.8 a name is a claim checked the
way Level 2 checks it (ppbench/v2/afford/grounding_v38.py, one rule for every system): a part counts 1.0 toward
presence when its role is verified -- it lies on the reference's part of that component, or, for an internal
component the reference does not show, it is enclosed in the design -- and UNVERIFIED (0.5) when the reference
shows the component elsewhere or not at all.

    part_present = min(sum of the matching parts' factors, need) / need          (v3.5: min(count, need) / need)

Chains, co-location, reach, clearance, grip, surface, ground contact, joints and the MuJoCo simulation are
already geometric or physical tests and are unchanged. P.1 and P.2 are copied from the v3.5 record. Records go
to eval_v38p3/<key>.json.

    python -m ppbench.v2.spec_v38_p3 build --keys-file F [--workers 48] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from ppbench.v2 import operability as OP
from ppbench.v2 import spec_v33 as V33
from ppbench.v2 import spec_v35 as S
from ppbench.v2.task import RESULTS

SPEC_VERSION = "v3.8-p3"
OUT = "eval_v38p3"
_FACTORS = {}          # the design being scored: part id -> factor (one design per call, set by _score)


def _part_present_v38(ctx, r):
    ids = ctx.roles(r["role"])
    need = int(r.get("min_count", 1))
    got = sum(_FACTORS.get(ctx.parts[i].id, 1.0) for i in ids)
    return min(got, need) / need, {"n": len(ids), "need": need, "verified_sum": got, "parts": ctx.names(ids)}


def _score(task_id, key, item):
    from ppbench.v2.afford import grounding_v38 as G
    task, design = S.load(task_id, item)
    family = "domain" if "__ext__" in key else "llm"
    _FACTORS.clear()
    if design.parts:
        _FACTORS.update(G.role_factors(G.ground(task_id, design, family)))
    v35 = RESULTS / task_id / "eval_v35" / f"{key}.json"
    v34 = RESULTS / task_id / "eval_v34" / f"{key}.json"
    rec = json.loads((v35 if v35.exists() else v34).read_text())
    if not design.parts:
        return rec, None
    fs = V33.get_anchors(task_id, task).get("free_standing")
    parts, occs, rows, jedges, weights = S.geometry(design, task)
    per_joint = ((rec.get("dims") or {}).get("2.3") or {}).get("metrics", {}).get("per_joint") or []
    saved = OP.PREDICATES["part_present"]
    OP.PREDICATES["part_present"] = _part_present_v38
    try:
        p3 = S.p3_operability(design, task, parts, occs, rows, per_joint, free_standing=fs, weights=weights)
    finally:
        OP.PREDICATES["part_present"] = saved
    p3.setdefault("metrics", {})["role_factors"] = {
        "n_parts": len(_FACTORS), "n_unverified": sum(1 for v in _FACTORS.values() if v < 1)}
    return rec, p3


def _one(args):
    task_id, key, force = args
    os.environ.setdefault("TMPDIR", "/tmp")
    dst = RESULTS / task_id / OUT / f"{key}.json"
    try:
        src = RESULTS / task_id / "eval_v34" / f"{key}.json"
        if not src.exists():
            return "no-item"
        base = json.loads(src.read_text())
        want = f"{base.get('stamp')}|{SPEC_VERSION}"
        if not force and dst.exists():
            try:
                if json.loads(dst.read_text()).get("stamp") == want:
                    return "cached"
            except (OSError, ValueError):
                pass
        item = base.get("item") or {}
        rec, p3 = _score(task_id, key, item)
        D = {k: v for k, v in (rec.get("dims") or {}).items() if k in ("P.1", "P.2", "P.3")}
        if p3 is not None:
            D["P.3_v35"] = {"score": (D.get("P.3") or {}).get("score")}
            D["P.3"] = p3
        out = {k: rec[k] for k in ("system", "tier", "item") if k in rec}
        out.update({"spec": SPEC_VERSION, "derived_from": rec.get("spec"), "dims": D, "stamp": want})
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1, default=float))
        return "ok"
    except Exception:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.with_suffix(".error.txt").write_text(traceback.format_exc())
        return "error"


def build(pairs, workers=48, force=False):
    jobs = [(t, k, force) for t, k in pairs]
    jobs.sort(key=lambda j: hash(j) % 997)
    print(f"{SPEC_VERSION}: {len(jobs)} designs", flush=True)
    tot, t0 = defaultdict(int), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=2), 1):
            tot[r] += 1
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {dict(tot)}  {time.time() - t0:.0f}s", flush=True)
    print(f"TOTAL {dict(tot)}  {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--keys-file", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    pairs = [tuple(l.rstrip("\n").split("\t")) for l in open(a.keys_file) if l.strip()]
    build(pairs, a.workers, a.force)


if __name__ == "__main__":
    main()
