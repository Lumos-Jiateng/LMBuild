"""Task sheets: what real instances of the category say each question should accept.

For every task the sheet records, from the population of real instances (afford/core.population_items):

* per component: in how many instances it is a separately modelled part (visibility), how many it has
  (count), and the band of each geometric descriptor over the instances that have it;
* per kinematic claim: whether any real instance realises it, and, from those that do, the joint type,
  the axis relation to the moving part, where the axis line sits on the part, the range, and how much of
  the object moves;
* the object envelope: overall size (metric instances only; PartNeXt meshes are scaled to a curated
  height prior, so they give proportions but not size) and proportions, widened by the typical
  dimensions Wikipedia quotes for the object (afford/wiki.py).

Calibration rules (the real object must be able to pass):
* a component is scored when at least 40% of the instances model it (one instance: when it does). A
  claimed part no real instance models separately -- a magnetron, a transfer case, refrigerant -- is
  reported as declared-only, never scored, so it cannot zero every design and the real object alike;
* a kinematic claim is scored when at least one real instance realises it with a compatible joint;
  its moving component must itself be scored.

Population instances are grounded by their dataset labels through the task lexicon; a label that does
not resolve falls back to the visual judges' answers on that segment, as for a design.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from ppbench.v2.afford import core, ground
from ppbench.v2.lexicon import Lexicon, tokens
from ppbench.v2.task import RESULTS

VISIBLE_MIN = 0.4
MOTION_TYPES = {"rotate": {"revolute", "continuous", "cylindrical"}, "slide": {"prismatic", "cylindrical"},
                "ball": {"ball"}}


def motion_of(claim):
    jt = (claim.get("joint_type") or "").lower()
    if jt in ("revolute", "continuous"):
        return "rotate"
    if jt in ("prismatic",):
        return "slide"
    if jt == "ball":
        return "ball"
    if jt == "cylindrical":
        return "rotate"
    return None      # planar / unknown: not a joint the protocol can declare


def comp_of_label(label, comps, lex):
    """A dataset label -> component id, through the lexicon (and the component's own phrases)."""
    if not label:
        return None
    c = lex.canon(label)
    tk = tuple(tokens(label))
    for comp in comps:
        canons = [comp["canon"], *comp.get("merged", [])]
        if c is not None and c in canons:
            return comp["id"]
        if tk and any(tuple(tokens(ph)) == tk for ph in comp["phrases"]):
            return comp["id"]
    # lexicon miss: a label whose head noun is a component's head noun and whose modifiers are a subset
    if tk:
        for comp in comps:
            for ph in comp["phrases"]:
                pt = tokens(ph)
                if pt and pt[-1] == tk[-1] and set(tk[:-1]) <= set(pt[:-1]) | {"wheel", "rim"}:
                    return comp["id"]
    return None


def judge_probs(tid, key, n_opts):
    """Mean answer distribution of the judges per segment index."""
    out, used = {}, []
    for model in ground.JUDGES:
        ap = RESULTS / tid / "afford" / "ground" / f"{key}.{model}.json"
        if not ap.exists():
            continue
        a = json.loads(ap.read_text())
        if a.get("prompt") != ground.PROMPT_VERSION:
            continue
        used.append(model)
        for si, ans in a["answers"].items():
            p = ground.distribution(ans, a.get("n_opts", n_opts))
            if p is None:
                continue
            out.setdefault(int(si), []).append(np.asarray(p))
    return {k: np.mean(v, 0) for k, v in out.items()}, used


def pop_grounding(tid, comps, lex, manifest, design):
    """Per-part component id for a real instance: labels first, judges for labels that do not resolve."""
    ids = [c["id"] for c in comps]
    n_opts = len(ground.options(comps))
    probs, _ = judge_probs(tid, manifest["item"]["key"], n_opts)
    part_comp = {}
    seg_of = {}
    for s in manifest["segments"]:
        for pid in s["parts"]:
            seg_of[pid] = s["i"]
    for p in design.parts:
        cid = comp_of_label(p.role, comps, lex)
        src = "label"
        if cid is None and seg_of.get(p.id) in probs:
            pr = probs[seg_of[p.id]]
            k = int(np.argmax(pr))
            if k < len(ids) and pr[k] >= 0.5:
                cid, src = ids[k], "judge"
        part_comp[p.id] = (cid, src)
    return part_comp


def instances(design, part_comp, cid, singular):
    """Part-id groups that are instances of component cid (one group when the component is singular)."""
    pids = [p.id for p in design.parts if part_comp.get(p.id, (None,))[0] == cid]
    if not pids:
        return []
    return [pids] if singular else [[x] for x in pids]


def band(vals, log=False, unit_margin=0.1):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    if log:
        v = np.log(np.maximum(v, 1e-9))
        lo, hi = float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))
        w = max(math.log(1.25), 0.15 * (hi - lo))
        lo, hi = lo - w, hi + w
        return {"log": True, "lo": lo, "hi": hi, "n": int(len(v))}
    lo, hi = float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))
    return {"log": False, "lo": max(0.0, lo - unit_margin), "hi": min(1.0, hi + unit_margin), "n": int(len(v))}


def build_task(tid, exclude=None):
    t = core.task(tid)
    lex = Lexicon(tid, t)
    comps = core.components(t, lex)
    claims = core.kin_claims(t, lex, comps)
    mdir = RESULTS / tid / "afford" / "items"
    pops = []
    for mf in sorted(mdir.glob("pop__*.json")):
        m = json.loads(mf.read_text())
        if exclude and m["item"]["key"] == exclude:
            continue
        item = next((i for i in core.population_items(t) if i["key"] == m["item"]["key"]), None)
        if item is None:
            continue
        try:
            d = core.load(t, item)
        except Exception:
            continue
        if not d.parts:
            continue
        pops.append((m, item, d, pop_grounding(tid, comps, lex, m, d)))
    npop = len(pops)
    # --- component visibility and counts
    stats = {c["id"]: {"present": 0, "counts": []} for c in comps}
    for m, item, d, pc in pops:
        cnt = defaultdict(int)
        for pid, (cid, _) in pc.items():
            if cid:
                cnt[cid] += 1
        for cid, n in cnt.items():
            stats[cid]["present"] += 1
            stats[cid]["counts"].append(n)
    sheet_comps = []
    for c in comps:
        s = stats[c["id"]]
        vis = s["present"] / max(npop, 1)
        scored = (vis >= VISIBLE_MIN) if npop >= 2 else (s["present"] >= 1)
        counts = s["counts"]
        cmin = int(max(1, math.floor(np.quantile(counts, 0.25)))) if counts else 1
        cmin = min(cmin, 8)
        singular = (np.median(counts) <= 1.0) if counts else True
        sheet_comps.append({**c, "visible_frac": vis, "n_present": s["present"], "scored": bool(scored),
                            "count_min": cmin, "singular": bool(singular), "counts": counts})
    byc = {c["id"]: c for c in sheet_comps}
    # --- descriptor bands, per component, over the instances of the real objects
    desc = defaultdict(lambda: defaultdict(list))
    obj = {"height": [], "long": [], "wide": [], "h_over_long": [], "wide_over_long": []}
    for m, item, d, pc in pops:
        os_ = core.object_stats(d)
        is_metric = (d.meta or {}).get("scale_mode") == "metric"
        if is_metric:
            obj["height"].append(os_["height"]); obj["long"].append(os_["long"]); obj["wide"].append(os_["wide"])
        obj["h_over_long"].append(os_["height"] / max(os_["long"], 1e-9))
        obj["wide_over_long"].append(os_["wide"] / max(os_["long"], 1e-9))
        byid = {p.id: p for p in d.parts}
        for c in sheet_comps:
            if not c["scored"]:
                continue
            for inst in instances(d, pc, c["id"], c["singular"]):
                dd = core.instance_desc([byid[x] for x in inst], os_, is_metric)
                if dd:
                    for k, v in dd.items():
                        if not k.startswith("_"):
                            desc[c["id"]][k].append(v)
    bands = {}
    for cid, dd in desc.items():
        b = {}
        for k, vals in dd.items():
            if k == "ground":
                f = float(np.mean(vals))
                b[k] = {"binary": True, "freq": f, "n": len(vals)}
            elif k in ("sym_axis_z", "long_axis_z", "thin_axis_z"):
                b[k] = band(vals, unit_margin=0.15)
            else:
                b[k] = band(vals, log=k in core.LOG_DESC, unit_margin=0.1)
        med = {k: float(np.median(v)) for k, v in dd.items()}
        b["_applies"] = {"sym_axis_z": med.get("roundness", 0) >= 0.6,
                         "long_axis_z": med.get("ratio_mid", 1) <= 0.7 and med.get("roundness", 0) < 0.6,
                         "thin_axis_z": (med.get("ratio_min", 1) / max(med.get("ratio_mid", 1), 1e-9)) <= 0.6
                                        and med.get("roundness", 0) < 0.6}
        b["_median"] = med
        bands[cid] = b
    from ppbench.v2.afford import wiki
    wsize = wiki.sizes(tid) if core.medium(t) == "real" else []
    env = {"height": band(obj["height"], log=True), "long": band(obj["long"], log=True),
           "wide": band(obj["wide"], log=True), "h_over_long": band(obj["h_over_long"], log=True),
           "wide_over_long": band(obj["wide_over_long"], log=True), "wikipedia": wsize,
           "n_metric": len(obj["height"]), "n": npop}
    for w in wsize:           # a typical dimension Wikipedia quotes widens the band to include it
        if w.get("target") != "object":
            continue
        dim = w.get("dim")
        if dim == "horiz":    # "width 81 cm" of a door is its long horizontal side: the nearer band decides
            mid = math.log(max(math.sqrt(w["min_m"] * max(w["max_m"], w["min_m"])), 1e-6))
            cand = [(abs((env[k]["lo"] + env[k]["hi"]) / 2 - mid), k) for k in ("long", "wide") if env.get(k)]
            dim = min(cand)[1] if cand else "long"
        w["mapped_to"] = dim
        b = env.get(dim)
        lo, hi = math.log(max(w["min_m"], 1e-6)), math.log(max(w["max_m"], w["min_m"], 1e-6))
        if b is None:
            env[dim] = {"log": True, "lo": lo - math.log(1.25), "hi": hi + math.log(1.25), "n": 0, "wiki": True}
        else:
            b["lo"], b["hi"], b["wiki"] = min(b["lo"], lo - math.log(1.25)), max(b["hi"], hi + math.log(1.25)), True
    # --- kinematic claims: evidence and relations from real instances
    kin = []
    for k in claims:
        mot = motion_of(k)
        row = {**k, "motion_kind": mot, "evidence": 0, "n_with_part": 0, "relations": [], "scored": False}
        cm = byc.get(k["moving"]) if k["moving"] else None
        if mot is None or cm is None:
            row["why_not"] = "no declarable motion" if mot is None else "moving part is the object itself or unnamed"
            kin.append(row)
            continue
        for m, item, d, pc in pops:
            insts = instances(d, pc, cm["id"], False)
            if not insts:
                continue
            row["n_with_part"] += 1
            rel = relations_in(d, pc, insts, k, mot)
            if rel:
                row["evidence"] += 1
                row["relations"] += rel
        row["scored"] = bool(cm["scored"] and row["evidence"] > 0)
        if not row["scored"]:
            row["why_not"] = ("moving part not modelled in real instances" if not cm["scored"]
                              else "no real instance realises this motion")
        row["expect"] = expectation(row["relations"], mot, k) if row["scored"] else None
        kin.append(row)
    sheet = {"spec": core.SPEC, "task": tid, "medium": core.medium(t), "n_population": npop,
             "population": [{"key": m["item"]["key"], "dataset": m["item"].get("dataset"),
                             "object_id": m["item"].get("object_id"), "n_parts": len(d.parts),
                             "grounded_by_judge": sum(1 for v in pc.values() if v[1] == "judge"),
                             "grounded_by_label": sum(1 for v in pc.values() if v[1] == "label" and v[0])}
                            for m, item, d, pc in pops],
             "components": sheet_comps, "bands": bands, "envelope": env, "kinematics": kin}
    import hashlib
    sheet["stamp"] = hashlib.sha1(json.dumps(sheet, sort_keys=True, default=float).encode()).hexdigest()[:12]
    sheet["excluded"] = exclude
    out = RESULTS / tid / "afford" / ("sheet.json" if not exclude else f"sheet_loo_{exclude.replace('pop__', '')}.json")
    out.write_text(json.dumps(sheet, indent=1, default=float))
    return sheet


def relations_in(d, pc, insts, claim, mot):
    """Joints of a real instance that move an instance of the claim's moving component."""
    from ppbench.v2.afford import kin
    out = []
    ok_types = MOTION_TYPES[mot]
    scene = kin.Scene(d)
    for inst in insts:
        for j in d.joints:
            if j.type == "fixed" or j.type not in ok_types:
                continue
            ms = kin.moving_set(d, j)
            if not (set(inst) & ms) or len(ms) >= len(d.parts):
                continue
            r = kin.relation(d, j, inst, ms, scene)
            if r:
                out.append(r)
                break
    return out


def expectation(rels, mot, claim):
    """What a design's joint for this claim must satisfy, from the real instances that realise it."""
    types = sorted({r["type"] for r in rels})
    sym = [r["sym_aligned"] for r in rels if r.get("sym_aligned") is not None]
    idx = [r["pca_index"] for r in rels]
    offs = [r["offset"] for r in rels if r.get("offset") is not None]
    exp = {"types": types, "n": len(rels)}
    if sym and np.mean(sym) >= 0.8:
        exp["axis"] = "symmetry"
    elif idx and len(set(idx)) == 1:
        exp["axis"] = f"pca{idx[0]}"
    else:
        exp["axis"] = None
    vz = [r["axis_z"] for r in rels]
    if vz and min(vz) >= 0.9:
        exp["axis_dir"] = "vertical"
    elif vz and max(vz) <= 0.25:
        exp["axis_dir"] = "horizontal"
    else:
        exp["axis_dir"] = None
    if mot == "rotate" and offs:
        if max(offs) <= 0.35:
            exp["offset"] = "centre"
        elif min(offs) >= 0.5:
            exp["offset"] = "edge"
        else:
            exp["offset"] = None
    else:
        exp["offset"] = None
    if mot == "rotate":
        spans = [r["span"] for r in rels if r.get("span") is not None]
        full = [s >= 2 * math.pi - 1e-3 for s in spans]
        if spans and np.mean(full) >= 0.5:
            exp["span"] = 2 * math.pi
        elif spans:
            exp["span"] = float(max(math.radians(10), 0.8 * np.quantile(spans, 0.25)))
        else:
            exp["span"] = math.radians(30)
    elif mot == "slide":
        tr = [r["travel_ratio"] for r in rels if r.get("travel_ratio") is not None]
        exp["travel_ratio"] = float(max(0.05, 0.8 * np.quantile(tr, 0.25))) if tr else 0.2
    mf = [r["moving_frac"] for r in rels]
    exp["moving_frac_max"] = float(max(0.6, 1.5 * np.median(mf))) if mf else 0.6
    return exp


def build(tids):
    for tid in tids:
        try:
            build_task(tid, exclude="pop__reference")
            s = build_task(tid)
            sc = sum(c["scored"] for c in s["components"])
            ks = sum(k["scored"] for k in s["kinematics"])
            print(f"{tid:26s} pop={s['n_population']:2d} comps {sc}/{len(s['components'])} scored, "
                  f"K {ks}/{len(s['kinematics'])} scored, wiki sizes {len(s['envelope']['wikipedia'])}", flush=True)
        except Exception as e:
            import traceback
            print(f"{tid:26s} FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
