"""v3.4.4 Level 3: the v3.4.3 halves, calibrated to the human study (P19, owner, 2026-09-23).

Levels 1, 2 and 4 and both raw halves of 3.1-3.3 are exactly the v3.4.3 ones (`spec_v34.level3`). What changes is
how the halves become a score: each half is put on the human scale by the clipped linear map fitted in
`calibrate_l3.py` (docs/evaluation/l3_calibration_v344.json), and the calibrated halves are blended with an integer
judge:computed ratio.

    J' = clip((judge_mean - lo_J) / (hi_J - lo_J), 0, 1)
    C' = clip((computed   - lo_C) / (hi_C - lo_C), 0, 1)
    3.1 decomposition        (1*J' + 1*C') / 2
    3.2 aesthetics           J'                      (no computed half, P15)
    3.3 structure alignment  (3*J' + 1*C') / 4

One half missing -> the other calibrated half (status degraded when only the computed half is left, as in P13).
An empty or unloadable design stays a fail. The raw halves stay in the record next to the calibrated ones.

    .venv_eval/bin/python -m ppbench.v2.spec_v344 build [--tasks all] [--workers 24] [--force] [--version v344|v345]

v3.4.5 (P20) is the same formula with the calibration fitted on the FIT objects only
(docs/evaluation/l3_calibration_v345.json, ratios chosen on those objects); records go to eval_v345/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from ppbench.v2 import spec_v34
from ppbench.v2.calibrate_l3 import CALIBS, blend
from ppbench.v2.evaluate import _dim
from ppbench.v2.spec_v32 import CRITICAL, STRUCTURE
from ppbench.v2.task import RESULTS

VERSIONS = {"v344": ("v3.4.4", "eval_v344"), "v345": ("v3.4.5", "eval_v345")}   # v3.4.5 = P20, fit on FIT objects
SPEC_VERSION = "v3.4.4"
NAMES = spec_v34.NAMES34


def load_calibration(version="v344"):
    raw = CALIBS[version].read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()[:12]


def calibrate(dim, d34, calib, sha):
    """Re-score one v3.4.3 Level 3 dimension from its stored halves."""
    m = d34.get("metrics") or {}
    jm, comp = m.get("judge_mean"), m.get("computed_score")
    score, jc, cc, how = blend(dim, jm, comp, calib)
    status = "pass" if jc is not None else ("degraded" if cc is not None else "skipped")
    if d34.get("status") == "fail":
        status = "fail"            # an empty or unloadable design stays a fail, whatever the judge said
        score = 0.0 if score is None else score
    c = calib["dims"][dim]
    return _dim("calibrated judge+computed", status, score,
                {"weights_judge_computed": c["weights"], "judge_calibrated": jc, "computed_calibrated": cc,
                 "anchors": {"judge": c.get("judge"), "computed": c.get("computed")}, "calibration_sha": sha,
                 "judge_per_model": m.get("judge_per_model"), "judge_mean": jm, "n_judges": m.get("n_judges"),
                 "computed_score": comp, "computed": m.get("computed"), "v343_score": d34.get("score")},
                [how])


def upgrade(rec33, dims3, sha, spec=SPEC_VERSION):
    D = dict(rec33.get("dims") or {})
    D.update(dims3)
    crit = [k for k in CRITICAL if (D.get(k) or {}).get("status") == "fail"]
    scored = [v["score"] for v in D.values() if v.get("score") is not None]
    out = {k: rec33[k] for k in ("system", "tier", "meta", "item") if k in rec33}
    out["notes"] = (rec33.get("notes") or []) + ["Level 3 rescored under spec v3.4",
                                                  f"Level 3 calibrated to the human study under spec {spec} ({sha})"]
    out.update({"spec": spec, "derived_from": rec33.get("spec"), "dims": D,
                "headline": {"critical_fail": crit,
                             "buildable": not any(k in crit for k in STRUCTURE),
                             "overall": (sum(scored) / len(scored)) if scored else None,
                             "n_scored_dims": len(scored), "gate_pending": True},
                "stamp": f"{rec33.get('stamp')}|{spec}|{sha}"})
    return out


def _task_job(args):
    task_id, force, version = args
    spec, dst_name = VERSIONS[version]
    src, dst = RESULTS / task_id / "eval_v33", RESULTS / task_id / dst_name
    if not src.is_dir():
        return task_id, {"no_src": 1}
    calib, sha = load_calibration(version)
    jcache, rcache, tot = {}, {}, defaultdict(int)
    for f in sorted(src.glob("*.json")):
        try:
            rec33 = json.loads(f.read_text())
        except (OSError, ValueError):
            tot["bad"] += 1
            continue
        want = f"{rec33.get('stamp')}|{spec}|{sha}"
        out = dst / f.name
        if not force and out.exists():
            try:
                if json.loads(out.read_text()).get("stamp") == want:
                    tot["cached"] += 1
                    continue
            except (OSError, ValueError):
                pass
        try:
            d34 = spec_v34.level3(rec33, task_id, jcache, rcache)
            dims3 = {d: calibrate(d, d34[d], calib, sha) for d in ("3.1", "3.2", "3.3")}
            dst.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(upgrade(rec33, dims3, sha, spec), indent=1, default=float))
            tot["judged" if dims3["3.2"]["metrics"]["n_judges"] else "computed_only"] += 1
        except Exception:
            import traceback
            dst.mkdir(parents=True, exist_ok=True)
            (dst / (f.stem + ".error.txt")).write_text(traceback.format_exc())
            tot["error"] += 1
    return task_id, dict(tot)


def build(tasks=None, workers=24, force=False, version="v344"):
    ids = tasks or [t["task_id"] for t in json.loads(spec_v34.CORE.read_text())]
    tot, t0 = defaultdict(int), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for task_id, c in ex.map(_task_job, [(t, force, version) for t in ids]):
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
    ap.add_argument("--version", default="v344", choices=sorted(VERSIONS))
    ns = ap.parse_args()
    ts = None if ns.tasks == "all" else [x.strip() for x in ns.tasks.split(",") if x.strip()]
    build(ts, ns.workers, ns.force, ns.version)
