"""One table of every core-set result: evaluator scores joined to what the run itself recorded.

`collect` walks results/v2/<task>/eval_core/<key>.json (written by `report collect-core`), joins each record to
its run directory (design meta, harness log, claude_run.json) and writes one row per evaluated design to
results/v2/_core/summary.json. A row carries the configuration (task, system, tier, condition, rounds, which
round the record is), the evaluator's per-dimension scores, and the harness facts (parts, joints, tool calls,
wall time, cost, why the episode stopped).

Averages are means over the rows that have a value: a dimension the evaluator could not measure for a task is
None in the record and is left out of the mean, never counted as zero (see evaluate._mean). Every table reports
n next to the mean, so a mean over 14 tasks is never read as a mean over 40.

    .venv_eval/bin/python -m ppbench.v2.summary collect [--tasks all] [--out results/v2/_core/summary.json]
    .venv_eval/bin/python -m ppbench.v2.summary tables [--summary ...]        # prints the paper tables
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from ppbench.v2.task import RESULTS

ROOT = Path(__file__).resolve().parents[2]
from ppbench.v2.task import CORE
SUMMARY = RESULTS / "_core" / "summary.json"
DIMS = ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1", "4.2"]     # dimensions with a graded score
DIM_NAMES = {"1.1": "integrity", "1.2": "stability", "2.1": "geometry", "2.2": "kinematics",
             "3.1": "decomposition", "3.2": "aesthetics (computed)", "4.1": "material", "4.2": "assembly sequence"}
MAIN = {"tier": "B", "condition": "name_only+image", "rounds": 3}
KEY_RE = re.compile(r"^(?P<system>.+?)__(?P<tier>[ABC])__(?P<cond>.+?)__r(?P<rounds>\d+)__s(?P<seed>\d+)(?:__round(?P<round>\d+))?$")


def core_tasks():
    return [t["task_id"] for t in json.loads(CORE.read_text())]


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _run_dir(task: str, key: str, ns: str) -> Path:
    return RESULTS / task / "runs" / ns / key.split("__round")[0]


def row_for(task: str, f: Path):
    """One row from an eval_core record, joined to its run directory. None when the key is not a tool-arm run."""
    rec = _read(f)
    if not rec:
        return None
    m = KEY_RE.match(f.stem)
    if not m:
        return None       # externals and the reference are not part of the model comparison
    item = rec.get("item") or {}
    ns = (item.get("namespace") or (re.search(r"/runs/([^/]+)/", str(item.get("path", ""))) or [None, None])[1] or "core_v2.4")
    d = _run_dir(task, f.stem, ns)
    meta = (rec.get("meta") or {})
    loop = meta.get("loop") or {}
    cr = _read(d / "claude_run.json")
    design = _read(d / "design" / "design.json")
    row = {"task": task, "system": m["system"], "tier": m["tier"], "condition": m["cond"],
           "rounds": int(m["rounds"]), "seed": int(m["seed"]),
           "round": int(m["round"]) if m["round"] else int(m["rounds"]), "is_final": not m["round"],
           "namespace": ns, "key": f.stem,
           "scores": {k: (rec.get("dims", {}).get(k) or {}).get("score") for k in DIMS},
           "status": {k: (rec.get("dims", {}).get(k) or {}).get("status") for k in DIMS},
           "buildable": (rec.get("headline") or {}).get("buildable"),
           "critical_fail": (rec.get("headline") or {}).get("critical_fail") or [],
           "n_parts": len(design.get("parts", []) or []), "n_joints": len(design.get("joints", []) or []),
           "submitted_final": bool(meta.get("submitted_final")), "rounds_completed": meta.get("rounds_completed"),
           "stop": loop.get("stop"), "tool_calls": cr.get("tool_calls") or (loop.get("n_calls") if loop else None),
           "cost_usd": cr.get("cost_usd"), "wall_s": meta.get("wall_s") or cr.get("wall_s"),
           "model_ids": cr.get("model_ids")}
    return row


def collect(tasks=None, out: Path = SUMMARY):
    rows = []
    for task in tasks or core_tasks():
        for f in sorted((RESULTS / task / "eval_core").glob("*.json")):
            r = row_for(task, f)
            if r:
                rows.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "dims": DIMS, "dim_names": DIM_NAMES}, indent=1))
    return rows


def load(summary: Path = SUMMARY):
    return _read(summary).get("rows", [])


def is_main(r):
    return r["tier"] == MAIN["tier"] and r["condition"] == MAIN["condition"] and r["rounds"] == MAIN["rounds"]


def mean_table(rows, key=lambda r: r["system"]):
    """{group: {dim: (mean, n)}} plus the harness columns, over the rows given."""
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    out = {}
    for g, rs in by.items():
        cell = {}
        for d in DIMS:
            v = [r["scores"][d] for r in rs if r["scores"].get(d) is not None]
            cell[d] = (sum(v) / len(v), len(v)) if v else (None, 0)
        num = lambda f: [f(r) for r in rs if f(r) is not None]
        cell["_n_runs"] = len(rs)
        cell["_tasks"] = len({r["task"] for r in rs})
        for name, f in (("parts", lambda r: r["n_parts"]), ("joints", lambda r: r["n_joints"]),
                        ("tool_calls", lambda r: r["tool_calls"]), ("cost_usd", lambda r: r["cost_usd"])):
            v = num(f)
            cell["_" + name] = sum(v) / len(v) if v else None
        cell["_buildable"] = sum(1 for r in rs if r["buildable"]) / len(rs)
        cell["_finished"] = sum(1 for r in rs if r["submitted_final"]) / len(rs)
        out[g] = cell
    return out


def fmt(table, title, groups=None):
    lines = [title, "system".ljust(26) + "".join(d.rjust(9) for d in DIMS) + "   n_runs tasks  parts joints"]
    for g in groups or sorted(table):
        c = table[g]
        cells = "".join((f"{c[d][0] * 100:6.1f}/{c[d][1]:<2d}" if c[d][0] is not None else "     -   ").rjust(9) for d in DIMS)
        lines.append(f"{g[:26].ljust(26)}{cells}   {c['_n_runs']:5d} {c['_tasks']:5d} {c['_parts'] or 0:6.1f} {c['_joints'] or 0:6.1f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "tables"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--summary", default=str(SUMMARY))
    ns = ap.parse_args(argv)
    tasks = None if ns.tasks == "all" else ns.tasks.split(",")
    if ns.cmd == "collect":
        rows = collect(tasks, Path(ns.summary))
        print(json.dumps({"rows": len(rows), "tasks": len({r['task'] for r in rows}),
                          "systems": sorted({r["system"] for r in rows})}, indent=1))
        return
    rows = load(Path(ns.summary))
    main_rows = [r for r in rows if is_main(r) and r["is_final"]]
    print(fmt(mean_table(main_rows), "MAIN SETTING (Tier B, name_only+image, 3-round trajectory), score x100 / n tasks"))
    print()
    for r_ in (1, 2, 3):
        rr = [r for r in rows if is_main(r) and r["round"] == r_]
        if rr:
            print(fmt(mean_table(rr), f"ROUND {r_}"))
            print()
    abl = [r for r in rows if not is_main(r) and r["is_final"]]
    if abl:
        print(fmt(mean_table(abl, key=lambda r: f"{r['system']} | {r['tier']} | {r['condition']}"), "ABLATIONS"))


if __name__ == "__main__":
    main()
