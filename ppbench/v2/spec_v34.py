"""v3.4 Level 3: decomposition, aesthetics and structure alignment, judge-primary.

Written against docs/evaluation/metrics_feedback.md (2026-09-20 second sitting, P8-P13). Levels 1, 2
and 4 are carried through from the v3.3 record untouched; only Level 3 changes, and it gains a third
dimension.

    3.1 decomposition        0.70 * VLM judge + 0.30 * (granularity, consolidation, joint placement)
    3.2 aesthetics           1.00 * VLM judge, reference-free: the judge never sees the photograph and
                             no computed term uses the reference asset (P9; v3.4.1)
    3.3 structure alignment  0.70 * VLM judge + 0.30 * (part correspondence IoU, symmetry match)

part_correspondence_iou moved out of 3.1 into 3.3 (P10) -- it measures "did you put the reference's
structure where the reference put it", which was never a decomposition-quality term. Absolute mirror
symmetry left 3.2 and became a referenced term in 3.3 (P11): the 40 reference assets average 0.803
mirror IoU, so rewarding raw symmetry rewarded boxes over real products.

    .venv_eval/bin/python -m ppbench.v2.spec_v34 build [--tasks all] [--workers 24] [--force]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ppbench.v2.evaluate import _dim, _mean
from ppbench.v2.spec_v32 import CRITICAL, STRUCTURE
from ppbench.v2.task import RESULTS

SPEC_VERSION = "v3.4.3"   # v3.4.3 (2026-09-21, P17): consolidation is None when the design cannot declare materials

JUDGE_WEIGHT = {"3.1": 0.30, "3.2": 1.00, "3.3": 0.70}   # v3.4.2: 3.1 is 0.70 rule-based + 0.30 judge (owner, 2026-09-21); 3.2 judge-only
NAMES34 = {"3.1": "decomposition", "3.2": "aesthetics", "3.3": "structure alignment"}
from ppbench.v2.task import CORE


def _clip(x):
    return None if x is None else max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------- the computed thirds
def rule_31(m31, m11, declares=None):
    """granularity (part count vs the reference), consolidation, and joint placement.

    Consolidation counts rigidly joined parts of the same role and the same material, so a design with no
    materials can never show a violation. Until v3.4.2 that silence scored 1.0 (every mesh and brick
    baseline sat at 100); since v3.4.3 (P17) the term is None when the design does not declare materials,
    the same rule joint placement already followed.

    Joint placement is one-sided on purpose: protocol v2.2 asks agents to declare a joint at every
    place the object *moves*, not at every seam, so "every boundary carries a joint" would grade
    against a rule nobody was given. A design that declares no joints scores None here rather than
    1.0 -- silence must not buy a point, and the absence is already charged in 1.1 and 2.3."""
    gran = _clip(m31.get("granularity_score"))
    n_parts, consol = m31.get("n_parts"), m31.get("consolidation_violations")
    cons = None
    if n_parts and consol is not None and (declares is None or declares.get("materials")):
        cons = _clip(1.0 - consol / max(n_parts - 1, 1))
    njp = m11.get("n_declared_joint_pairs")
    place = None
    if njp:
        place = _clip(1.0 - (m11.get("n_declared_joints_rejected") or 0) / njp)
    return _mean([gran, cons, place]), {"granularity_score": gran, "consolidation_score": cons,
                                        "joint_placement_rate": place,
                                        "consolidation_violations": consol,
                                        "n_declared_joint_pairs": njp}


def rule_32(m32):
    """No computed half (v3.4.1, owner's review): proportion_score is measured against the task's reference
    asset, and 3.2 is to be judged on the design alone -- reference alignment is 3.3's job. It is still
    reported, for information only, and never enters the score, not even as a fallback."""
    return None, {"proportion_score_info_only": _clip(m32.get("proportion_score"))}


def rule_33(m31, m32, ref):
    """Correspondence to the reference's structure, and agreement with its own symmetry.

    symmetry_match = 1 - |sym - sym_ref| / max(sym_ref, 1 - sym_ref): matching a real product's
    asymmetry scores 1, and a perfectly symmetric box no longer outscores the ground truth."""
    iou = _clip(m31.get("part_correspondence_iou"))
    sym, sym_ref = m32.get("symmetry_iou"), (ref or {}).get("symmetry_iou")
    match = None
    if sym is not None and sym_ref is not None:
        match = _clip(1.0 - abs(sym - sym_ref) / max(sym_ref, 1.0 - sym_ref, 1e-6))
    return _mean([iou, match]), {"part_correspondence_iou": iou, "symmetry_match": match,
                                 "symmetry_iou": sym, "reference_symmetry_iou": sym_ref}


# ---------------------------------------------------------------- the blend
def blend(dim, judge, rule, rule_metrics, prior_status):
    """P13: 0.7/0.8 judge + the rest computed, but only when both halves exist. A missing half is
    reported, never silently replaced by zero."""
    w = JUDGE_WEIGHT[dim]
    js = [v for v in (judge or {}).values() if v is not None]
    jm = {k: v for k, v in (judge or {}).items()}
    if js and rule is not None:
        score, how = w * _mean(js) + (1 - w) * rule, f"{w:.2f} judge + {1-w:.2f} computed"
        status = "pass"
    elif js:
        how = ("judge only, by design" if w >= 1.0 else "judge only (no computed half for this task)")
        score, status = _mean(js), "pass"
    elif rule is not None:
        score, how, status = rule, "computed only (judge did not return a score)", "degraded"
    else:
        score, how, status = None, "neither half available", "skipped"
    if prior_status == "fail":
        status = "fail"            # an empty or unloadable design stays a fail, whatever the judge said
        score = 0.0 if score is None else score
    return _dim("judge+computed", status, score,
                {"judge_weight": w, "judge_per_model": jm, "judge_mean": _mean(js) if js else None,
                 "n_judges": len(js), "computed_score": rule, "computed": rule_metrics},
                [how])


def judge_scores(task_id, key, cache):
    """{dim: {model: judge_score}} from every judge file on disk for this task."""
    if task_id not in cache:
        per = defaultdict(dict)
        d = RESULTS / task_id / "judge_l3"
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                if p.name.endswith(".tmp"):
                    continue
                try:
                    j = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue
                for k, v in (j.get("items") or {}).items():
                    for dim, rec in v.items():
                        per[k].setdefault(dim, {})[p.stem] = rec.get("judge_score")
        cache[task_id] = per
    return cache[task_id].get(key, {})


def _ref_metrics(task_id, cache):
    if task_id not in cache:
        p = RESULTS / task_id / "reference_eval_v3.json"
        m = None
        try:
            m = (json.loads(p.read_text()).get("dims") or {}).get("3.2", {}).get("metrics")
        except (OSError, ValueError):
            pass
        cache[task_id] = m
    return cache[task_id]


def level3(rec33, task_id, jcache, rcache):
    D = rec33.get("dims") or {}
    m31 = (D.get("3.1") or {}).get("metrics") or {}
    m32 = (D.get("3.2") or {}).get("metrics") or {}
    m11 = (D.get("1.1") or {}).get("metrics") or {}
    key = (rec33.get("item") or {}).get("key", "")
    J = judge_scores(task_id, key, jcache)
    ref = _ref_metrics(task_id, rcache)

    r31, x31 = rule_31(m31, m11, (rec33.get("meta") or {}).get("declares"))
    r32, x32 = rule_32(m32)
    r33, x33 = rule_33(m31, m32, ref)
    st31 = (D.get("3.1") or {}).get("status")
    st32 = (D.get("3.2") or {}).get("status")
    return {"3.1": blend("3.1", J.get("3.1"), r31, x31, st31),
            "3.2": blend("3.2", J.get("3.2"), r32, x32, st32),
            "3.3": blend("3.3", J.get("3.3"), r33, x33, st32)}


def upgrade(rec33, dims3, note):
    D = dict(rec33.get("dims") or {})
    D.update(dims3)
    crit = [k for k in CRITICAL if (D.get(k) or {}).get("status") == "fail"]
    scored = [v["score"] for v in D.values() if v.get("score") is not None]
    out = {k: rec33[k] for k in ("system", "tier", "meta", "item") if k in rec33}
    out["notes"] = (rec33.get("notes") or []) + [note]
    out.update({"spec": SPEC_VERSION, "derived_from": rec33.get("spec"), "dims": D,
                "headline": {"critical_fail": crit,
                             "buildable": not any(k in crit for k in STRUCTURE),
                             "overall": (sum(scored) / len(scored)) if scored else None,
                             "n_scored_dims": len(scored), "gate_pending": True},
                "stamp": f"{rec33.get('stamp')}|{SPEC_VERSION}"})
    return out


# ---------------------------------------------------------------- build
def _task_job(args):
    task_id, force = args
    src = RESULTS / task_id / "eval_v33"
    dst = RESULTS / task_id / "eval_v34"
    if not src.is_dir():
        return task_id, {"no_src": 1}
    jcache, rcache, tot = {}, {}, defaultdict(int)
    for f in sorted(src.glob("*.json")):
        try:
            rec33 = json.loads(f.read_text())
        except (OSError, ValueError):
            tot["bad"] += 1
            continue
        want = f"{rec33.get('stamp')}|{SPEC_VERSION}"
        out = dst / f.name
        if not force and out.exists():
            try:
                if json.loads(out.read_text()).get("stamp") == want:
                    tot["cached"] += 1
                    continue
            except (OSError, ValueError):
                pass
        try:
            dims3 = level3(rec33, task_id, jcache, rcache)
            dst.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(upgrade(rec33, dims3, "Level 3 rescored under spec v3.4"),
                                      indent=1, default=float))
            tot["judged" if dims3["3.2"]["metrics"]["n_judges"] else "computed_only"] += 1
        except Exception:
            import traceback
            dst.mkdir(parents=True, exist_ok=True)
            (dst / (f.stem + ".error.txt")).write_text(traceback.format_exc())
            tot["error"] += 1
    return task_id, dict(tot)


def build(tasks=None, workers=24, force=False):
    ids = tasks or [t["task_id"] for t in json.loads(CORE.read_text())]
    tot, t0 = defaultdict(int), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for task_id, c in ex.map(_task_job, [(t, force) for t in ids]):
            for k, v in c.items():
                tot[k] += v
            print(f"{task_id:28s} {c}", flush=True)
    print(f"done {dict(tot)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()
    ts = None if ns.tasks == "all" else [x.strip() for x in ns.tasks.split(",") if x.strip()]
    build(ts, ns.workers, ns.force)
