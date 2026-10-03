"""Level 3 calibration: fit the judge and computed halves of 3.1-3.3 to the human study.

The VLM judges rarely leave the middle of their 1-10 scale, so raw Level 3 scores sat in a 40-70 band while the
human raters (human_evaluation/, v2) used the whole 1-5 scale. Each half of each dimension is mapped onto the
human scale by a clipped linear map fitted on human-rated items:

    cal(x) = clip((x - lo) / (hi - lo), 0, 1)

with lo, hi chosen so that on the fitting items cal(x) has the human mean and standard deviation (moment matching;
least squares was rejected because it shrinks the spread back towards the middle). The calibrated halves are then
blended with an integer judge:computed ratio.

Two versions:
  v3.4.5 (P20, current)  fitted on the FIT objects only (a fixed, pre-declared split of the 15 rated objects); the
                         ratio is chosen on the FIT objects by leave-one-object-out; the TEST objects are held out
                         for validation (`heldout`).
  v3.4.4 (P19)           fitted on all 15 objects with the ratios fixed in WEIGHTS_V344.

    .venv_eval/bin/python -m ppbench.v2.calibrate_l3 fit [--version v345|v344]
    .venv_eval/bin/python -m ppbench.v2.calibrate_l3 heldout    # validation on the TEST objects + 50 random splits
    .venv_eval/bin/python -m ppbench.v2.calibrate_l3 cv         # leave-one-object-out ratio table, all objects
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics as st
import time
from pathlib import Path

from ppbench.v2.task import RESULTS

ROOT = Path(__file__).resolve().parents[2]
HUMAN = ROOT / "human_evaluation"
CALIBS = {"v344": ROOT / "docs/evaluation/l3_calibration_v344.json",
          "v345": ROOT / "docs/evaluation/l3_calibration_v345.json"}
CALIB = CALIBS["v345"]
HELDOUT = ROOT / "docs/evaluation/l3_calibration_v345_heldout.json"
JUDGES = ("gemma-3-27b", "qwen2.5-vl-32b")
DIMS = ("3.1", "3.2", "3.3")
WEIGHTS_V344 = {"3.1": (1, 1), "3.2": (1, 0), "3.3": (3, 1)}   # P19; 3.2 has no computed half since v3.4.1 (P15)
V343 = {"3.1": (3, 7), "3.2": (1, 0), "3.3": (7, 3)}           # the uncalibrated v3.4.3 blend, for comparison
RATIOS = [(1, 0), (4, 1), (3, 1), (2, 1), (3, 2), (1, 1), (2, 3), (1, 2), (1, 3), (0, 1)]

# The 15 rated objects in four categories. Split rule, declared before any result was looked at (P20): alphabetical
# order within each category, positions 1 and 3 -> FIT, positions 2 and 4 -> TEST.
CATEGORIES = {"vehicles": ["offroad_jeep", "passenger_car", "wheelchair", "wheeled_excavator"],
              "mechanisms": ["bench_drill_press", "oscillating_steam_engine", "scissor_car_jack", "stepladder"],
              "furniture": ["bed_frame", "chest_of_drawers", "dining_table", "swivel_office_chair"],
              "appliances": ["desk_lamp", "oscillating_fan", "refrigerator"]}
FIT = sorted(t for ts in CATEGORIES.values() for i, t in enumerate(sorted(ts)) if i % 2 == 0)
TEST = sorted(t for ts in CATEGORIES.values() for i, t in enumerate(sorted(ts)) if i % 2 == 1)


def cal(x, lohi):
    if x is None or lohi is None:
        return None
    lo, hi = lohi
    return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))


def blend(dim, judge_mean, computed, calib):
    """Calibrated score of one dimension, or None when neither half exists. Returns (score, J', C', how)."""
    c = calib["dims"][dim]
    a, b = c["weights"]
    jc = cal(judge_mean, c.get("judge"))
    cc = cal(computed, c.get("computed")) if b else None
    if a == 0 and cc is not None:
        return cc, jc, cc, "calibrated computed only, by design"
    if jc is not None and cc is not None:
        return (a * jc + b * cc) / (a + b), jc, cc, f"{a}:{b} calibrated judge : calibrated computed"
    if jc is not None:
        return jc, jc, cc, "calibrated judge only" + ("" if b else ", by design")
    if cc is not None:
        return cc, jc, cc, "calibrated computed only (judge did not return a score)"
    return None, jc, cc, "neither half available"


def _moments(xs, hs):
    mx, sx, mh, sh = st.mean(xs), st.pstdev(xs), st.mean(hs), st.pstdev(hs)
    b = sh / sx
    a = mh - b * mx
    return [-a / b, (1 - a) / b]


def human_items():
    """One row per (design, dim) of the human study: the human mean on [0, 1] and the raw judge / computed halves."""
    key = {d["id"]: d for d in json.loads((HUMAN / "key.json").read_text())["designs"]}
    ratings = {}
    for p in sorted((HUMAN / "annotations").glob("A*.json")):
        for r in json.loads(p.read_text()).values():
            ratings.setdefault((r["design"], r["dim"]), []).append(r["score"])
    jcache, rows = {}, []
    for (did, dim), ss in sorted(ratings.items()):
        d = key[did]
        items = {}
        for m in JUDGES:
            p = RESULTS / d["task"] / "judge_l3" / f"{m}.json"
            if p not in jcache:
                jcache[p] = json.loads(p.read_text())["items"] if p.exists() else {}
            r = jcache[p].get(d["key"], {}).get(dim) or {}
            if r.get("judge_score") is not None:
                items[m] = r["judge_score"]
        if len(items) < len(JUDGES):
            continue
        rec = RESULTS / d["task"] / "eval_v34" / f"{d['key']}.json"
        if not rec.exists():   # the release ships these records apart from the evaluator's working directories
            rec = ROOT / "results" / "human_study" / "eval_v34" / d["task"] / f"{d['key']}.json"
        comp = ((json.loads(rec.read_text())["dims"].get(dim) or {}).get("metrics") or {}).get("computed_score") \
            if rec.exists() else None
        rows.append({"dim": dim, "task": d["task"], "system": d["system"], "key": d["key"],
                     "human": st.mean((s - 1) / 4 for s in ss), "n_raters": len(ss),
                     "judge": st.mean(items.values()), "computed": comp})
    return rows


def fit_dim(rows, weights):
    j = _moments([r["judge"] for r in rows], [r["human"] for r in rows])
    cr = [r for r in rows if r["computed"] is not None]
    c = _moments([r["computed"] for r in cr], [r["human"] for r in cr]) if weights[1] and cr else None
    return {"weights": list(weights), "judge": j, "computed": c}


def predictor(rows, dim, weights):
    c = {"dims": {dim: fit_dim(rows, weights)}}
    return lambda r: blend(dim, r["judge"], r["computed"], c)[0]


def choose_ratio(rows, dim):
    """Integer judge:computed ratio with the lowest leave-one-object-out MAE on `rows`. Returns (ratio, table)."""
    if not any(r["computed"] is not None for r in rows):
        return (1, 0), {"1:0": None}
    table = {}
    for w in RATIOS:
        err = []
        for t in sorted({r["task"] for r in rows}):
            f = predictor([r for r in rows if r["task"] != t], dim, w)
            err += [abs(f(r) - r["human"]) for r in rows if r["task"] == t]
        table[f"{w[0]}:{w[1]}"] = st.mean(err)
    best = min(RATIOS, key=lambda w: table[f"{w[0]}:{w[1]}"])
    return best, table


def fit(version="v345"):
    rows = human_items()
    out = {"spec": {"v344": "v3.4.4", "v345": "v3.4.5"}[version], "fitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "method": "per half: clipped linear map, lo/hi matching the human mean and sd on the fitting items; "
                     "blend = (a*J' + b*C')/(a+b); one half missing -> the other calibrated half",
           "source": {"human": "human_evaluation/annotations/A*.json (latest rating per rater, mean over raters, (s-1)/4)",
                      "judge": "results/v2/<task>/judge_l3/{gemma-3-27b,qwen2.5-vl-32b}.json, mean of the two",
                      "computed": "results/v2/<task>/eval_v34/<key>.json computed_score"},
           "dims": {}}
    if version == "v345":
        out["split"] = {"rule": "alphabetical within category; positions 1, 3 -> fit; 2, 4 -> test",
                        "categories": CATEGORIES, "fit": FIT, "test": TEST}
    for dim in DIMS:
        rs = [r for r in rows if r["dim"] == dim and (version == "v344" or r["task"] in FIT)]
        if version == "v344":
            w, table = WEIGHTS_V344[dim], None
        else:
            w, table = choose_ratio(rs, dim)
        c = fit_dim(rs, w)
        c.update({"n_items": len(rs), "n_items_with_computed": sum(r["computed"] is not None for r in rs),
                  "human_mean": st.mean(r["human"] for r in rs), "human_sd": st.pstdev(r["human"] for r in rs)})
        if table:
            c["ratio_selection_loo_mae"] = table
        out["dims"][dim] = c
    CALIBS[version].write_text(json.dumps(out, indent=1))
    print(json.dumps(out["dims"], indent=1))


def _spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / ((sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5)


def agreement(pred, rows):
    """Item and system agreement of predictions with the human means on `rows`."""
    h = [r["human"] for r in rows]
    ag = tot = 0
    for t in {r["task"] for r in rows}:
        q = [(p, r["human"]) for p, r in zip(pred, rows) if r["task"] == t]
        for (a, b), (c, d) in itertools.combinations(q, 2):
            if b != d:
                tot, ag = tot + 1, ag + ((a - c) * (b - d) > 0)
    sy = sorted({r["system"] for r in rows})
    hs = [st.mean(r["human"] for r in rows if r["system"] == s) for s in sy]
    ps = [st.mean(p for p, r in zip(pred, rows) if r["system"] == s) for s in sy]
    return {"mae": st.mean(abs(x - y) for x, y in zip(pred, h)), "spearman": _spearman(pred, h),
            "pairwise": ag / tot, "system_mae": st.mean(abs(x - y) for x, y in zip(ps, hs)),
            "system_spearman": _spearman(ps, hs),
            "per_system": {s: {"human": a, "pred": b} for s, a, b in zip(sy, hs, ps)}}


def _v343(dim):
    a, b = V343[dim]
    return lambda r: r["judge"] if (not b or r["computed"] is None) else (a * r["judge"] + b * r["computed"]) / (a + b)


def heldout(n_random=50, seed=0):
    """Fit on FIT, score TEST; then the same protocol on n_random stratified random object splits."""
    rows = human_items()
    res = {"fit": FIT, "test": TEST, "fixed_split": {}, "random_splits": {"n": n_random, "seed": seed},
           "inter_annotator": {}}
    for dim in DIMS:
        rs = [r for r in rows if r["dim"] == dim]
        tr, te = [r for r in rs if r["task"] in FIT], [r for r in rs if r["task"] in TEST]
        w, _ = choose_ratio(tr, dim)
        res["fixed_split"][dim] = {"n_fit": len(tr), "n_test": len(te), "ratio": list(w),
                                   "v3.4.3": agreement([_v343(dim)(r) for r in te], te),
                                   "calibrated": agreement([predictor(tr, dim, w)(r) for r in te], te)}
    rng = random.Random(seed)
    acc = {}
    for _ in range(n_random):
        fit_set = set()
        for ts in CATEGORIES.values():
            k = len(ts) // 2 + (len(ts) % 2 and rng.random() < 0.5)
            fit_set |= set(rng.sample(sorted(ts), k))
        for dim in DIMS:
            rs = [r for r in rows if r["dim"] == dim]
            tr, te = [r for r in rs if r["task"] in fit_set], [r for r in rs if r["task"] not in fit_set]
            w, _ = choose_ratio(tr, dim)
            for name, f in (("v3.4.3", _v343(dim)), ("calibrated", predictor(tr, dim, w))):
                a = agreement([f(r) for r in te], te)
                for k in ("mae", "spearman", "pairwise", "system_mae", "system_spearman"):
                    acc.setdefault(dim, {}).setdefault(name, {}).setdefault(k, []).append(a[k])
            acc[dim].setdefault("ratios", []).append(f"{w[0]}:{w[1]}")
    for dim, v in acc.items():
        res["random_splits"][dim] = {name: {k: {"mean": st.mean(x), "sd": st.pstdev(x)} for k, x in m.items()}
                                     for name, m in v.items() if name != "ratios"}
        res["random_splits"][dim]["ratio_counts"] = {r: v["ratios"].count(r) for r in sorted(set(v["ratios"]))}
    # inter-annotator agreement on the double-rated items (the human ceiling)
    key = {d["id"]: d for d in json.loads((HUMAN / "key.json").read_text())["designs"]}
    per = {}
    for p in sorted((HUMAN / "annotations").glob("A*.json")):
        for r in json.loads(p.read_text()).values():
            per.setdefault((r["design"], r["dim"]), []).append(r["score"])
    for dim in DIMS:
        dbl = [v for (d, dd), v in per.items() if dd == dim and len(v) >= 2 and d in key]
        res["inter_annotator"][dim] = {"n": len(dbl), "spearman": _spearman([v[0] for v in dbl], [v[1] for v in dbl]),
                                       "mae": st.mean(abs(v[0] - v[1]) / 4 for v in dbl)}
    HELDOUT.write_text(json.dumps(res, indent=1))
    for dim in DIMS:
        f = res["fixed_split"][dim]
        print(f"{dim} ratio {f['ratio']}  " + "  ".join(
            f"{k} {f['v3.4.3'][k]:.3f}->{f['calibrated'][k]:.3f}"
            for k in ("mae", "spearman", "pairwise", "system_mae", "system_spearman")))


def cv():
    """Leave-one-object-out over all 15 objects, for every candidate ratio (the v3.4.4 table)."""
    rows = human_items()
    for dim in DIMS:
        rs = [r for r in rows if r["dim"] == dim]
        print(f"\n{dim}  n={len(rs)}")
        for w in (RATIOS if any(r["computed"] is not None for r in rs) else [(1, 0)]):
            pred = {}
            for t in sorted({r["task"] for r in rs}):
                f = predictor([r for r in rs if r["task"] != t], dim, w)
                for r in rs:
                    if r["task"] == t:
                        pred[r["task"], r["key"]] = f(r)
            a = agreement([pred[r["task"], r["key"]] for r in rs], rs)
            print(f"  {w[0]}:{w[1]}  MAE {a['mae']:.3f}  rho {a['spearman']:.2f}  pairwise {a['pairwise']:.2f}  "
                  f"system MAE {a['system_mae']:.3f}  system rho {a['system_spearman']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "heldout", "cv"])
    ap.add_argument("--version", default="v345", choices=["v344", "v345"])
    ns = ap.parse_args()
    if ns.cmd == "fit":
        fit(ns.version)
    else:
        {"heldout": heldout, "cv": cv}[ns.cmd]()
