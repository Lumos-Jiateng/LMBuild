"""A.1 Geometry, A.2 Parts, A.3 Kinematics: one record per design, 0-100 each.

Every design of every system is scored on every task (A.1, A.2) and on every task with at least one
scored kinematic claim (A.3): which questions are asked depends on the task only, never on what the
design happened to declare, so the denominators are the same for every row. An empty design answers
every question with 0.

Presence is soft. For component c the judges' mean probability that segment s is c, p_s(c), is the
evidence; a component that real instances have m of (four wheels) has m slots, filled by the m most
probable segments, and a missing wheel is an empty slot. A part floating off the assembly counts half.

    A.2 = mean over scored components of  mean over slots  p_s(c) * attached_s
    A.1 = mean over {object envelope} + scored components of
              envelope:  mean(size vs real instances [metric designs only], proportions)
              component: mean over slots  p_s(c) * mean(size, shape, orientation, position aspects)
    A.3 = mean over scored claims of  mean over slots of the moving component  p_s(c) * q(best joint)

Each aspect is the mean of its descriptors' band scores: 1 inside the band real instances span (widened),
falling linearly to 0 at a factor of 2 beyond it for sizes and ratios, and 0.25 beyond it for positions
and orientations. The reference object is scored against a sheet calibrated without it (leave-one-out).
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ppbench.v2.afford import calib, core, ground, kin
from ppbench.v2.lexicon import Lexicon
from ppbench.v2.task import RESULTS

INSTANCE_MIN_P = 0.35


def band_score(x, b):
    if b is None or x is None or not np.isfinite(x):
        return None
    if b.get("log"):
        v = math.log(max(x, 1e-9))
        d = max(b["lo"] - v, v - b["hi"], 0.0)
        return float(max(0.0, 1 - d / math.log(2)))
    d = max(b["lo"] - x, x - b["hi"], 0.0)
    return float(max(0.0, 1 - d / 0.25))


def desc_scores(dd, bands, is_metric):
    """Aspect -> score for one instance's descriptors."""
    out = {}
    app = bands.get("_applies", {})
    for asp, keys in core.ASPECTS.items():
        vals = []
        for k in keys:
            if k not in dd or k not in bands:
                continue
            if k == "abs_size" and not is_metric:
                continue
            if k in ("sym_axis_z", "long_axis_z", "thin_axis_z") and not app.get(k):
                continue
            b = bands[k]
            if b and b.get("binary"):
                f = b["freq"]
                if f >= 0.75:
                    vals.append(float(dd[k]))
                elif f <= 0.25:
                    vals.append(1.0 - float(dd[k]))
                continue
            s = band_score(dd[k], b)
            if s is not None:
                vals.append(s)
        if vals:
            out[asp] = float(np.mean(vals))
    return out


def unattached(tid, key, d):
    """Parts off the main connected group, from the frozen Level 1 record when there is one."""
    rec = RESULTS / tid / "eval_v34" / f"{key}.json"
    if rec.exists():
        try:
            m = json.loads(rec.read_text())["dims"]["1.1"].get("metrics") or {}
            if "unattached_parts" in m:
                return set(m["unattached_parts"]), "level1"
        except (OSError, ValueError, KeyError):
            pass
    try:
        from ppbench.v2 import analysis
        from ppbench.v2.spec_v33 import _pairs, connectivity
        occs = analysis.occupancies(d.parts)
        rows = _pairs(d.parts, occs, d.joints)
        dim, main = connectivity(d.parts, occs, rows, d.joints)
        return set((dim.get("metrics") or {}).get("unattached_parts") or []), "computed"
    except Exception:
        return set(), "unknown"


def slots_for(comp, seg_p, segs):
    """(probability, part ids) per slot of a component."""
    ci = comp["_opt"]
    m = comp["count_min"]
    ranked = sorted(((float(seg_p[s["i"]][ci]) if s["i"] in seg_p else 0.0, s) for s in segs), key=lambda t: -t[0])
    if comp["singular"]:
        members = [s for p, s in ranked if p >= INSTANCE_MIN_P]
        if not members and ranked:
            members = [ranked[0][1]]
        p = ranked[0][0] if ranked else 0.0
        return [(p, [x for s in members for x in s["parts"]], [s["i"] for s in members])] + [(0.0, [], [])] * (m - 1)
    out = [(p, list(s["parts"]), [s["i"]]) for p, s in ranked[:m]]
    return out + [(0.0, [], [])] * (m - len(out))


def score_item(tid, manifest, sheet, comps_all, loo_sheet=None):
    sh = loo_sheet or sheet
    item = manifest["item"]
    t = core.task(tid)
    comps = [dict(c) for c in sh["components"]]
    opt_ids = [c["id"] for c in ground.options(comps_all)]
    for c in comps:
        c["_opt"] = opt_ids.index(c["id"]) if c["id"] in opt_ids else None
    scored = [c for c in comps if c["scored"] and c["_opt"] is not None]
    claims = [k for k in sh["kinematics"] if k["scored"]]
    rec = {"spec": core.SPEC, "task": tid, "key": item["key"], "system": item["system"], "family": item.get("family"),
           "kind": item["kind"], "stamp": manifest["stamp"], "n_components_scored": len(scored),
           "n_claims_scored": len(claims), "metric": manifest.get("metric")}
    if manifest["n_parts"] == 0 or not manifest["segments"]:
        rec.update({"A1": 0.0, "A2": 0.0, "A3": 0.0 if claims else None, "empty": True})
        return rec
    d = core.load(t, {**item, "ref": next((i["ref"] for i in core.population_items(t) if i["key"] == item["key"]), None)}
                  if item["kind"] == "population" else item)
    is_metric = core.metric(d) and (d.meta or {}).get("scale_mode") != "prior_scaled"
    segs = manifest["segments"]
    n_opts = len(ground.options(comps_all))
    lego_pop = item["kind"] == "population" and not any(s["png"] for s in segs)
    if lego_pop:            # LEGO real instances: every brick has its LDraw name, which grounds it
        lex = Lexicon(tid, t)
        seg_p = {}
        for s in segs:
            p = np.zeros(n_opts + 1)
            labs = [calib.comp_of_label(l, comps_all, lex) for l in (s.get("labels") or [])]
            hits = [opt_ids.index(x) for x in labs if x in opt_ids]
            if hits:
                for h in hits:
                    p[h] += 1
                p /= p.sum()
            else:
                p[-1] = 1.0
            seg_p[s["i"]] = p
        used = ["labels"]
    else:
        seg_p, used = calib.judge_probs(tid, item["key"], n_opts)
    rec["judges"] = used
    rec["n_unjudged"] = sum(1 for s in segs if s["i"] not in seg_p)
    floating, fsrc = unattached(tid, item["key"], d) if item["kind"] != "population" else (set(), "n/a")
    byid = {p.id: p for p in d.parts}
    obj = core.object_stats(d)
    # ---------------- A.2
    a2 = {}
    comp_slots = {}
    for c in scored:
        sl = slots_for(c, seg_p, segs)
        comp_slots[c["id"]] = sl
        vals = []
        for p, pids, _ in sl:
            att = 0.5 if (pids and all(x in floating for x in pids)) else 1.0
            vals.append(p * att)
        a2[c["id"]] = float(np.mean(vals))
    rec["A2"] = 100 * float(np.mean(list(a2.values()))) if a2 else None
    rec["A2_components"] = a2
    # ---------------- A.1
    env = sh["envelope"]
    size = [band_score(obj[k], env.get(k)) for k in ("height", "long", "wide")] if is_metric else []
    size = [s for s in size if s is not None]
    prop = [band_score(obj["height"] / max(obj["long"], 1e-9), env.get("h_over_long")),
            band_score(obj["wide"] / max(obj["long"], 1e-9), env.get("wide_over_long"))]
    prop = [s for s in prop if s is not None]
    g_env = {}
    if size:
        g_env["size"] = float(np.mean(size))
    if prop:
        g_env["proportion"] = float(np.mean(prop))
    a1 = {"envelope": float(np.mean(list(g_env.values()))) if g_env else None}
    a1_aspects = {}
    for c in scored:
        b = sh["bands"].get(c["id"])
        if not b:
            continue
        vals, asp_acc = [], defaultdict(list)
        for p, pids, _ in comp_slots[c["id"]]:
            if not pids or p <= 0:
                vals.append(0.0)
                continue
            dd = core.instance_desc([byid[x] for x in pids if x in byid], obj, is_metric)
            if not dd:
                vals.append(0.0)
                continue
            ds = desc_scores(dd, b, is_metric)
            for k, v in ds.items():
                asp_acc[k].append(v)
            vals.append(p * (float(np.mean(list(ds.values()))) if ds else 0.0))
        a1[c["id"]] = float(np.mean(vals))
        a1_aspects[c["id"]] = {k: float(np.mean(v)) for k, v in asp_acc.items()}
    q = [v for v in a1.values() if v is not None]
    rec["A1"] = 100 * float(np.mean(q)) if q else None
    rec["A1_questions"] = a1
    rec["A1_envelope"] = g_env
    rec["A1_aspects"] = a1_aspects
    rec["object"] = {k: obj[k] for k in ("height", "long", "wide", "diag")}
    # ---------------- A.3
    if claims:
        scene = kin.Scene(d)
        contact = scene.contact() if d.joints else {}
        byc = {c["id"]: c for c in comps}
        a3, detail = {}, {}
        for k in claims:
            cm = byc.get(k["moving"])
            if cm is None or cm["_opt"] is None:
                continue
            sl = comp_slots.get(cm["id"]) or slots_for(cm, seg_p, segs)
            rel_parts = []
            if k.get("relative") and byc.get(k["relative"]) and byc[k["relative"]]["_opt"] is not None:
                ri = byc[k["relative"]]["_opt"]
                rel_parts = [x for s in segs if s["i"] in seg_p and int(np.argmax(seg_p[s["i"]])) == ri
                             and seg_p[s["i"]][ri] >= 0.5 for x in s["parts"]]
            vals, dets = [], []
            for p, pids, sids in sl:
                if not pids or p <= 0:
                    vals.append(0.0)
                    dets.append(None)
                    continue
                best = None
                for j in d.joints:
                    if j.type == "fixed":
                        continue
                    try:
                        r = kin.joint_quality(scene, j, pids, k["expect"], k["motion_kind"], rel_parts, contact)
                    except Exception as e:
                        r = {"joint": j.id, "q": 0.0, "error": str(e)[:120]}
                    if r and (best is None or r["q"] > best["q"]):
                        best = r
                vals.append(p * (best["q"] if best else 0.0))
                dets.append({"p": p, "segments": sids, "best": best})
            a3[k["id"]] = float(np.mean(vals)) if vals else 0.0
            detail[k["id"]] = dets
        rec["A3"] = 100 * float(np.mean(list(a3.values()))) if a3 else None
        rec["A3_claims"] = a3
        rec["A3_detail"] = detail
    else:
        rec["A3"] = None
    rec["floating_source"] = fsrc
    return rec


def _one(args):
    tid, mf, force = args
    try:
        m = json.loads(Path(mf).read_text())
    except (OSError, ValueError):
        return "bad"
    key = m["item"]["key"]
    out = RESULTS / tid / "eval_A" / f"{key}.json"
    sheet_p = RESULTS / tid / "afford" / "sheet.json"
    if not sheet_p.exists():
        return "nosheet"
    sheet = json.loads(sheet_p.read_text())
    loo = None
    if key == "pop__reference":
        lp = RESULTS / tid / "afford" / "sheet_loo_reference.json"
        loo = json.loads(lp.read_text()) if lp.exists() else None
    elif m["item"]["kind"] == "population":
        return "skip"
    stamp = f"{m['stamp']}|{sheet.get('stamp', '')}|{_judge_stamp(tid, key)}"
    if not force and out.exists():
        try:
            if json.loads(out.read_text()).get("stamp_all") == stamp:
                return "cached"
        except (OSError, ValueError):
            pass
    t = core.task(tid)
    comps_all = core.components(t, Lexicon(tid, t))
    try:
        rec = score_item(tid, m, sheet, comps_all, loo)
        rec["stamp_all"] = stamp
        rec["loo"] = bool(loo)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, default=float))
        return "ok"
    except Exception:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".error.txt").write_text(traceback.format_exc())
        return "error"


def _judge_stamp(tid, key):
    s = []
    for model in ground.JUDGES:
        ap = RESULTS / tid / "afford" / "ground" / f"{key}.{model}.json"
        if ap.exists():
            s.append(f"{model}:{ap.stat().st_mtime:.0f}")
    return ",".join(s)


def build(tids, workers=64, force=False):
    jobs = []
    for tid in tids:
        for mf in sorted((RESULTS / tid / "afford" / "items").glob("*.json")):
            jobs.append((tid, str(mf), force))
    print(f"score: {len(jobs)} items", flush=True)
    tot, t0 = defaultdict(int), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=1), 1):
            tot[r] += 1
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)} {dict(tot)} {time.time() - t0:.0f}s", flush=True)
    print(f"SCORE TOTAL {dict(tot)} {time.time() - t0:.0f}s", flush=True)


def load_records(tids):
    rows = []
    for tid in tids:
        for f in sorted((RESULTS / tid / "eval_A").glob("*.json")):
            try:
                rows.append(json.loads(f.read_text()))
            except (OSError, ValueError):
                pass
    return rows


def tables(tids):
    from ppbench.v2.afford import report
    report.write(load_records(tids), tids)
