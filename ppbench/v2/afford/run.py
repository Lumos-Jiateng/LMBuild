"""Level A pipeline.

    python -m ppbench.v2.afford.run prep   [--tasks all] [--workers 64]      segment + render every item
    python -m ppbench.v2.afford.run judge  --model qwen2.5-vl-32b [--workers 48]   blind part grounding
    python -m ppbench.v2.afford.run calib  [--tasks all]                     task sheets from real instances
    python -m ppbench.v2.afford.run score  [--tasks all] [--workers 64]      A.1 / A.2 / A.3 records
    python -m ppbench.v2.afford.run table                                    report tables

Items: the main setting (Tier B, name_only+image, 3 rounds, seed 0) for every agent, the domain-specific
generators' main-setting outputs, and the population of real instances. Every stage is resumable and
keyed on the design's stamp, so new runs are picked up by re-running the chain.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from ppbench.v2.afford import core, ground
from ppbench.v2.lexicon import Lexicon
from ppbench.v2.task import RESULTS

os.environ.setdefault("TMPDIR", "/tmp")


def adir(tid):
    return RESULTS / tid / "afford"


def _stamp(item):
    if item["kind"] == "population":
        return "pop-v1"
    from ppbench.v2 import report as R
    try:
        return f"{R._stamp(item):.0f}"
    except Exception:
        return "0"


def all_items(tid):
    t = core.task(tid)
    return core.items_tool(tid) + core.items_external(tid) + core.population_items(t)


# ------------------------------------------------------------------ prep
def _prep_one(args):
    tid, item, force = args
    out = adir(tid) / "items" / f"{item['key']}.json"
    st = _stamp(item)
    if not force and out.exists():
        try:
            m = json.loads(out.read_text())
            if m.get("stamp") == st and m.get("spec") == core.SPEC and all(Path(s["png"]).exists() for s in m["segments"]):
                return "cached"
        except (OSError, ValueError, KeyError):
            pass
    t = core.task(tid)
    try:
        d = core.load(t, item)
        segs = core.segments(d, item) if d.parts else []
        rec = {"spec": core.SPEC, "stamp": st, "task": tid, "item": {k: v for k, v in item.items() if k != "ref"},
               "n_parts": len(d.parts), "metric": core.metric(d), "segments": []}
        if item["kind"] == "population":
            rec["item"]["object_id"] = item["ref"].get("object_id")
            rec["item"]["dataset"] = item["ref"].get("dataset")
        lego_pop = item["kind"] == "population" and core.medium_of(item["ref"]) == "lego"
        if segs and not lego_pop:
            sheets = ground.render_design(d, segs, ground.SHEETS / tid / item["key"])
        else:
            sheets = [(i, "", 0.0) for i in range(len(segs))]
        for (si, png, vis), s in zip(sheets, segs):
            rec["segments"].append({"i": si, "parts": [d.parts[k].id for k in s], "png": png, "visible": vis,
                                    "labels": [d.parts[k].role for k in s] if item["kind"] == "population" else None})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, default=float))
        return "ok"
    except Exception:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".error.txt").write_text(traceback.format_exc())
        return "error"


def prep(tids, workers=64, force=False):
    jobs = []
    for tid in tids:
        try:
            jobs += [(tid, it, force) for it in all_items(tid)]
        except Exception as e:
            print(f"{tid}: items failed: {e}", flush=True)
    print(f"prep: {len(jobs)} items over {len(tids)} tasks", flush=True)
    tot, t0 = defaultdict(int), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_prep_one, jobs, chunksize=2), 1):
            tot[r] += 1
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)} {dict(tot)} {time.time() - t0:.0f}s", flush=True)
    print(f"PREP TOTAL {dict(tot)} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ judge
def _answers_path(tid, key, model):
    return adir(tid) / "ground" / f"{key}.{model}.json"


def judge(tids, model, workers=48, base_url=None):
    from openai import OpenAI
    client = OpenAI(base_url=base_url or ground.JUDGES[model], api_key="x", timeout=120)
    todo = []
    for tid in tids:
        t = core.task(tid)
        comps = core.components(t, Lexicon(tid, t))
        text = ground.prompt(t.raw["object"]["name"], comps)
        for mf in sorted((adir(tid) / "items").glob("*.json")):
            try:
                m = json.loads(mf.read_text())
            except (OSError, ValueError):
                continue
            ap = _answers_path(tid, m["item"]["key"], model)
            have = json.loads(ap.read_text()) if ap.exists() else {}
            if have.get("prompt") != ground.PROMPT_VERSION or have.get("stamp") != m["stamp"]:
                have = {"prompt": ground.PROMPT_VERSION, "stamp": m["stamp"], "n_opts": len(ground.options(comps)),
                        "answers": {}}
            need = [s for s in m["segments"] if s["png"] and str(s["i"]) not in have["answers"]]
            if need:
                todo.append((tid, m["item"]["key"], ap, have, need, text))
    n = sum(len(x[4]) for x in todo)
    print(f"judge {model}: {n} sheets in {len(todo)} items", flush=True)
    t0, done = time.time(), 0

    def one(job):
        tid, key, ap, have, need, text = job
        for s in need:
            a = ground.ask(client, model, s["png"], text)
            have["answers"][str(s["i"])] = a
        ap.parent.mkdir(parents=True, exist_ok=True)
        tmp = ap.with_suffix(".tmp")
        tmp.write_text(json.dumps(have))
        os.replace(tmp, ap)
        return len(need)
    with ThreadPoolExecutor(workers) as ex:
        for k in ex.map(one, todo):
            done += k
            if done and (done // 500) != ((done - k) // 500):
                el = time.time() - t0
                print(f"  {done}/{n} {el:.0f}s ({done / max(el, 1):.1f}/s)", flush=True)
    print(f"JUDGE TOTAL {model} {done} {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "judge", "calib", "score", "table"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--model", default="qwen2.5-vl-32b")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args(argv)
    tids = core.task_ids() if ns.tasks == "all" else ns.tasks.split(",")
    if ns.cmd == "prep":
        prep(tids, ns.workers, ns.force)
    elif ns.cmd == "judge":
        judge(tids, ns.model, ns.workers, ns.base_url)
    elif ns.cmd == "calib":
        from ppbench.v2.afford import calib
        calib.build(tids)
    elif ns.cmd == "score":
        from ppbench.v2.afford import score
        score.build(tids, ns.workers, ns.force)
    elif ns.cmd == "table":
        from ppbench.v2.afford import score
        score.tables(tids)


if __name__ == "__main__":
    main()
