"""Flatten the final evaluation records into one score table (one row per scored design).

The paper's twelve metrics (metrics v3.8, docs/evaluation/metrics_v38_changes.md) come from these record sets under
results/v2/<task>/:

    S.1-S.3  soundness      eval_v34l1/  dims 1.1 connectivity, 1.2 collision, 1.3 stability  (L1 v3.4, design-scaled)
    A.1-A.3  affordance     eval_v38l2/  dims 2.1 geometry, 2.2 parts, 2.3 kinematics         (L2 v3.8, names confirmed)
    D.1-D.3  design         eval_v345/   dims 3.1 decomposition, 3.2 aesthetics, 3.3 alignment (L3 v3.4.5, two judges)
    R.1      realization    eval_v35/    dim P.1 assembly sequence                             (L4 v3.5)
    R.2      realization    eval_v38p2/  dim P.2 material                                      (R.2 v3.6, full coverage)
    R.3      realization    eval_v38p3/  dim P.3 operability                                   (R.3 v3.8, verified roles)

    python scripts/export_scores.py [--results results/v2] [--out results/scores/all_scores.csv]

Scores are in [0, 1]; an empty cell means the dimension was not scored for that design (not applicable, or the record
set holds no record for it). A failed design is scored 0 by the evaluator and stays 0 here.

Two extra columns describe the 200-task main table (Tier B, name_only+image, round 1):
    task_set   orig50 (the first 50 tasks of benchmark/core_v2.json) or new150
    main200    the records the main table may score for a (system, task), ranked: for the original 50 the round-1
               checkpoint of the 3-round trajectory (1), and the final design when the episode ended inside round 1 (2);
               for the new 150 the one-round run (1) or its round-1 checkpoint (2); generators: image (1), name_only (2).
               Per metric the table takes the best-ranked record present in that metric's record set (status != "").
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = [("S.1", "eval_v34l1", "1.1"), ("S.2", "eval_v34l1", "1.2"), ("S.3", "eval_v34l1", "1.3"),
           ("A.1", "eval_v38l2", "2.1"), ("A.2", "eval_v38l2", "2.2"), ("A.3", "eval_v38l2", "2.3"),
           ("D.1", "eval_v345", "3.1"), ("D.2", "eval_v345", "3.2"), ("D.3", "eval_v345", "3.3"),
           ("R.1", "eval_v35", "P.1"), ("R.2", "eval_v38p2", "P.2"), ("R.3", "eval_v38p3", "P.3")]
SETS = sorted({sp for _, sp, _ in METRICS})
CLOSED = {"gpt-6-astra", "gpt-5.6-sol", "claude-opus-5", "claude-fable-5.1", "claude-sonnet-5", "claude-haiku-4.5"}
EXCLUDED = {"glm-4.6v-flash", "kimi-vl-a3b-instruct"}   # removed from the study (tool-use failure); records kept upstream


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def rows(results: Path):
    for tdir in sorted(p for p in results.iterdir() if p.is_dir() and not p.name.startswith("_")):
        keys = set()
        for spec in SETS:
            keys |= {Path(f).stem for f in glob.glob(str(tdir / spec / "*.json"))}
        for key in sorted(keys):
            recs = {s: _load(tdir / s / f"{key}.json") for s in SETS}
            base = recs["eval_v345"] or recs["eval_v35"] or recs["eval_v34l1"] or recs["eval_v38l2"] or {}
            it = base.get("item") or {}
            system = base.get("system") or it.get("system") or key.split("__")[0]
            if system in EXCLUDED or it.get("seed", 0) != 0:
                continue
            kind = it.get("kind") or ("external" if "__ext__" in key else "tool")
            if kind not in ("tool", "external"):
                continue
            row = {"task": tdir.name, "key": key, "system": system,
                   "family": "generator" if kind == "external" else ("closed" if system in CLOSED else "open"),
                   "tier": "ext" if kind == "external" else it.get("tier", base.get("tier")),
                   "condition": it.get("condition"),
                   "trajectory_rounds": "" if kind == "external" else it.get("trajectory_rounds", it.get("rounds")),
                   "checkpoint": it.get("checkpoint_round") or "final"}
            for name, spec, dim in METRICS:
                rec = recs[spec]
                d = ((rec or {}).get("dims") or {}).get(dim)
                if rec is None:                # no record in this set: the design was not scored here (tables count 0)
                    row[name], row[name + "_status"] = "", ""
                elif not isinstance(d, dict):  # record present, dimension absent: not applicable
                    row[name], row[name + "_status"] = "", "n/a"
                else:
                    row[name] = "" if d.get("score") is None else round(float(d["score"]), 6)
                    row[name + "_status"] = d.get("status") or "n/a"
            row["_meta"] = base.get("meta") or {}
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "v2"))
    ap.add_argument("--out", default=str(ROOT / "results" / "scores" / "all_scores.csv"))
    ns = ap.parse_args()
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "key", "system", "family", "tier", "condition", "trajectory_rounds", "checkpoint"]
    fields += [m for m, _, _ in METRICS] + [m + "_status" for m, _, _ in METRICS]
    fields += ["task_set", "main200"]
    order = [t["task_id"] for t in json.loads(Path(os.environ.get("PPB_CORE", ROOT / "benchmark" / "core_v2.json")).read_text())]
    orig = set(order[:50])
    allrows = list(rows(Path(ns.results)))
    by = {(r["task"], r["key"]): r for r in allrows}
    for r in allrows:
        r["task_set"] = "orig50" if r["task"] in orig else "new150"
        r["main200"] = ""
    for (task, key), r in by.items():           # one main-table record per (system, task)
        s_, fam = r["system"], r["family"]
        if fam == "generator":
            cand = [f"{s_}__ext__image__s0", f"{s_}__ext__name_only__s0"]
        elif task in orig:
            fin = by.get((task, f"{s_}__B__name_only+image__r3__s0"))
            ended = fin is not None and (fin["_meta"].get("rounds_completed", 0) or 0) == 0
            cand = [f"{s_}__B__name_only+image__r3__s0__round1"] + ([f"{s_}__B__name_only+image__r3__s0"] if ended else [])
        else:
            cand = [f"{s_}__B__name_only+image__r1__s0", f"{s_}__B__name_only+image__r1__s0__round1"]
        if key in cand:                         # rank of this record among the candidates (1 = preferred)
            r["main200"] = cand.index(key) + 1
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(allrows, key=lambda r: (r["task"], r["key"])):
            w.writerow(r)
            n += 1
    print(f"{n} rows -> {out}")


if __name__ == "__main__":
    main()
