"""Measured wall time of the office-chair run, and what the full benchmark would cost.

    .venv_eval/bin/python -m ppbench.v2.timing --task swivel_office_chair

Measured per design: episode wall time (tool arms: from the environment's start to the last call;
open models also report model-server time), evaluation wall time, and generation time of external
systems (from their meta.json). Serving start-up per open model is read from the vLLM logs.

The extrapolation multiplies measured means by the size of a full sweep, stated explicitly:
tasks x conditions x tiers x rounds x seeds for tool arms, tasks x seeds for external generators.
Numbers are sequential GPU/API wall hours; parallel slots divide them, and the table says how many
slots the measured runs actually used.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from ppbench.v2.task import RESULTS


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def collect(task_id):
    base = RESULTS / task_id
    rows = []
    for p in sorted((base / "eval").glob("*.json")):
        if p.name.startswith(("imagesim", "judge", "vlm")):
            continue
        r = json.loads(p.read_text())
        it = r.get("item") or {}
        meta = r.get("meta") or {}
        wall = meta.get("wall_s")
        loop = meta.get("loop") or {}
        gen = None
        if it.get("kind") == "external":
            m = Path(it["path"]) / "meta.json"
            if m.exists():
                pm = json.loads(m.read_text())
                for k in ("wall_s", "wall_s_total", "wall_time_s", "seconds"):
                    if isinstance(pm.get(k), (int, float)):
                        gen = float(pm[k])
                        break
                if gen is None:
                    txt = json.dumps(pm)
                    hits = [float(x) for x in re.findall(r'"wall_s[a-z_]*":\s*([0-9.]+)', txt)]
                    gen = sum(hits) if hits else None
        rows.append({"key": it.get("key"), "kind": it.get("kind"), "system": it.get("system"), "tier": it.get("tier"),
                     "rounds": it.get("rounds"), "episode_s": wall if it.get("kind") == "tool" else None,
                     "generate_s": gen, "eval_s": r.get("eval_wall_s"),
                     "tokens": (loop.get("usage") or {}), "transport": meta.get("transport")})
    return rows


def summarize(rows, tasks=40, conditions=6, tiers=2, rounds=(1, 2, 3), seeds_open=3, seeds_api=1, ext_seeds=3):
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["system"], r["rounds"])].append(r)
    table = []
    for (kind, sysname, rnd), rs in sorted(by.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2] or 0)):
        table.append({"kind": kind, "system": sysname, "rounds": rnd, "n": len(rs),
                      "episode_min": None if _mean([x["episode_s"] for x in rs]) is None else _mean([x["episode_s"] for x in rs]) / 60,
                      "generate_min": None if _mean([x["generate_s"] for x in rs]) is None else _mean([x["generate_s"] for x in rs]) / 60,
                      "eval_min": None if _mean([x["eval_s"] for x in rs]) is None else _mean([x["eval_s"] for x in rs]) / 60,
                      "prompt_tokens": _mean([(x["tokens"] or {}).get("prompt") for x in rs]),
                      "completion_tokens": _mean([(x["tokens"] or {}).get("completion") for x in rs])})
    # full-sweep extrapolation per system
    per_sys = defaultdict(dict)
    for t in table:
        per_sys[(t["kind"], t["system"])][t["rounds"]] = t
    sweep = []
    for (kind, sysname), byr in per_sys.items():
        if kind == "tool":
            api = str(sysname).startswith("claude")
            seeds = seeds_api if api else seeds_open
            ep_h = 0.0
            for r in rounds:
                t = byr.get(r) or next(iter(byr.values()))
                ep_h += (t["episode_min"] or 0) / 60 * tasks * conditions * tiers * seeds
            ev_h = _mean([t["eval_min"] for t in byr.values()]) or 0
            ev_h = ev_h / 60 * tasks * conditions * tiers * seeds * len(rounds)
            sweep.append({"system": sysname, "kind": kind, "designs": tasks * conditions * tiers * seeds * len(rounds),
                          "episode_hours_sequential": ep_h, "eval_cpu_hours_sequential": ev_h})
        elif kind == "external":
            t = next(iter(byr.values()))
            sweep.append({"system": sysname, "kind": kind, "designs": tasks * ext_seeds,
                          "generate_hours_sequential": (t["generate_min"] or 0) / 60 * tasks * ext_seeds,
                          "eval_cpu_hours_sequential": (t["eval_min"] or 0) / 60 * tasks * ext_seeds})
    return table, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="swivel_office_chair")
    ap.add_argument("--tasks", type=int, default=40)
    ns = ap.parse_args()
    rows = collect(ns.task)
    table, sweep = summarize(rows, tasks=ns.tasks)
    out = RESULTS / ns.task / "timing.json"
    out.write_text(json.dumps({"rows": rows, "table": table, "sweep": sweep, "assumptions": {
        "tasks": ns.tasks, "conditions": 6, "tiers": 2, "rounds": [1, 2, 3], "seeds_open": 3, "seeds_api": 1, "external_seeds": 3}}, indent=1))
    f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
    print(f"{'system':34s} {'kind':9s} {'r':>2s} {'n':>3s} {'episode':>8s} {'generate':>9s} {'eval':>6s}  (minutes, mean per design)")
    for t in table:
        print(f"{str(t['system'])[:34]:34s} {str(t['kind']):9s} {str(t['rounds'] or '-'):>2s} {t['n']:>3d} {f(t['episode_min']):>8s} {f(t['generate_min']):>9s} {f(t['eval_min']):>6s}")
    print(f"\nFull sweep at {ns.tasks} tasks (sequential hours):")
    tot_gpu = tot_cpu = 0.0
    for s in sorted(sweep, key=lambda s: s["system"] or ""):
        g = s.get("episode_hours_sequential", s.get("generate_hours_sequential", 0.0))
        tot_gpu += g
        tot_cpu += s["eval_cpu_hours_sequential"]
        print(f"  {str(s['system'])[:34]:34s} {s['designs']:>6d} designs  run {g:8.1f} h  eval {s['eval_cpu_hours_sequential']:7.1f} h")
    print(f"  TOTAL run {tot_gpu:.0f} h sequential, eval {tot_cpu:.0f} CPU-process hours -> {out}")


if __name__ == "__main__":
    main()
