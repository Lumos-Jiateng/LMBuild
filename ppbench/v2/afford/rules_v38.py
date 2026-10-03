"""Level 2 (Affordance), spec v3.8 (2026-10-02): 2.2 Parts with verified grounding; 2.1 and 2.3 unchanged.

v3.7's 2.2 grounded a named design by its names alone (see grounding_v38 for the rule that replaces it). v3.8
changes only 2.2:

    2.2 = 2/3 visible components + 1/3 internal components      (count-aware, as v3.7)
    instance credit = attachment x verification
        attachment   1.0, or 0.5 when every part of the instance floats                  (v3.7)
        verification 1.0 when a part of the instance is verified (reference geometry, or an enclosed part
                     for an internal component) or was found by geometry; 0.5 otherwise  (v3.8)
    unnamed interior module = 0.5 for one unmet internal component                         (v3.7)

2.1 and 2.3 are copied from the v3.7 record of the same design (same code, same stamp); a design without one is
scored by rules.score_design first. Records go to eval_v38l2/<key>.json.

    python -m ppbench.v2.afford.rules_v38 score --keys-file F [--workers 48] [--force]
    (F: one "task<TAB>key" per line; keys as in results/scores/all_scores.csv)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ppbench.v2.afford import core, grounding_v38 as G, kin, rules
from ppbench.v2.task import RESULTS

SPEC = "v3.8"
OUT = "eval_v38l2"


def score22(tid, d, family, item_key):
    sheet = rules.load_sheet(tid)
    scene = kin.Scene(d)
    interior, _ = rules.interior_parts(scene)
    ref, Dr, DL = G.aligned_samples(tid, d)
    gr = G.ground(tid, d, family, ref, Dr, DL, interior)
    part_comp = {pid: v["comp"] for pid, v in gr.items()}
    fac = {pid: v["factor"] for pid, v in gr.items()}
    floating, fsrc = rules._unattached(tid, item_key, d)
    per22, vis, intl = {}, [], []
    anon = [p.id for p in d.parts if part_comp[p.id] is None and p.id in interior
            and (p.source or {}).get("kind") != "ldraw"]
    anon_groups = rules._union_groups(d, anon) if anon else []
    anon_left = len(anon_groups)
    hows = defaultdict(int)
    for v in gr.values():
        hows[v["how"] or "none"] += 1
    for c in sheet["components"]:
        if c.get("motion_only"):
            continue
        insts = rules.instances(d, part_comp, c["id"], c["singular"])
        m = c["count_min"]
        vals = sorted([(0.5 if all(x in floating for x in grp) else 1.0) * max(fac[x] for x in grp)
                       for grp in insts], reverse=True)[:m]
        src = "named or placed" if vals else None
        if not c["visible"] and not vals and anon_left > 0:
            vals, anon_left, src = [rules.UNNAMED_INTERIOR], anon_left - 1, "unnamed interior module"
        s = float(sum(vals) / m)
        per22[c["name"]] = {"score": s, "n": len(insts), "need": m, "visible": c["visible"], "evidence": src,
                            "verified": [max(fac[x] for x in grp) for grp in insts][:m]}
        (vis if c["visible"] else intl).append(s)
    groups = {"visible": float(np.mean(vis)) if vis else None, "internal": float(np.mean(intl)) if intl else None}
    live = {k: v for k, v in groups.items() if v is not None}
    s22 = sum(rules.W22[k] * v for k, v in live.items()) / sum(rules.W22[k] for k in live) if live else None
    return {"score": s22, "status": "pass" if s22 is not None else "skipped",
            "metrics": {"groups": groups, "components": per22, "roles_declared": rules.roles_declared(d),
                        "grounding": dict(hows), "interior_parts": len(interior),
                        "unnamed_interior_modules": len(anon_groups), "floating_source": fsrc,
                        "n_parts": len(d.parts)},
            "notes": [f"v3.8: instance credit = attachment x verification (unverified role {G.UNVERIFIED}); "
                      f"weights {rules.W22}"]}


def _one(args):
    tid, key, force = args
    os.environ.setdefault("TMPDIR", "/tmp")
    out = RESULTS / tid / OUT / f"{key}.json"
    try:
        src = RESULTS / tid / rules.MAIN_EVAL / f"{key}.json"
        if not src.exists():
            return "no-item"
        item = json.loads(src.read_text())["item"]
        family = "domain" if "__ext__" in key else "llm"
        sheet = rules.load_sheet(tid)
        stamp = f"{SPEC}|{rules._stamp(item)}|{sheet['stamp']}"
        if not force and out.exists():
            try:
                if json.loads(out.read_text()).get("stamp_all") == stamp:
                    return "cached"
            except (OSError, ValueError):
                pass
        v37 = RESULTS / tid / rules.OUT / f"{key}.json"
        base = json.loads(v37.read_text()) if v37.exists() else None
        if base is None or base.get("stamp_all") != f"{rules.SPEC}|{rules._stamp(item)}|{sheet['stamp']}":
            base = rules.score_design(tid, item, family)
        rec = {k: base[k] for k in ("task", "key", "system", "tier", "family", "sheet") if k in base}
        rec.update({"spec": SPEC, "derived_from": rules.SPEC, "dims": dict(base["dims"])})
        if not base.get("empty"):
            d = core.load(rules.task(tid), item)
            rec["dims"]["2.2"] = score22(tid, d, family, item.get("key", key))
            rec["dims"]["2.2_v37"] = {"score": base["dims"]["2.2"].get("score")}
        else:
            rec["empty"] = True
        rec["stamp_all"] = stamp
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, default=float))
        return "ok"
    except Exception:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".error.txt").write_text(traceback.format_exc())
        return "error"


def build(pairs, workers=48, force=False):
    jobs = [(t, k, force) for t, k in pairs]
    jobs.sort(key=lambda j: hash(j[1]) % 97)
    print(f"score {SPEC} 2.2: {len(jobs)} designs", flush=True)
    tot, t0 = defaultdict(int), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=1), 1):
            tot[r] += 1
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} {dict(tot)} {time.time() - t0:.0f}s", flush=True)
    print(f"SCORE TOTAL {dict(tot)} {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score"])
    ap.add_argument("--keys-file", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    pairs = [tuple(l.rstrip("\n").split("\t")) for l in open(a.keys_file) if l.strip()]
    build(pairs, a.workers, a.force)


if __name__ == "__main__":
    main()
