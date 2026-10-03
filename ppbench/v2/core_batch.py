"""Run one served open model over the whole core set (tool arms, harness v2.4).

Configurations per model: 40 tasks x 6 conditions x tiers A, B, C x one 3-round trajectory x one seed, each a
`ppbench.v2.loop` process against the model's OpenAI-compatible endpoint, `concurrency` at a time. Outputs go to
results/v2/<task>/runs/core_v2.4/<system>__<tier>__<condition>__r<rounds>__s<seed>/, with the task frozen in
results/v2/<task>/task_snapshot_core.json. Each episode holds one cumulative trajectory and saves an immutable checkpoint
after each submit. A valid final submission is skipped; an unfinished attempt is preserved under _core/discarded/ before retry.

Jobs are interleaved by condition, tier, and task. Rounds are sequential revisions inside each job.

    .venv_eval/bin/python -m ppbench.v2.core_batch --model qwen3-vl-8b --system-name qwen3-vl-8b-instruct \
        --base-url http://127.0.0.1:8101/v1 --concurrency 40 [--no-vision] [--extra-body JSON] [--tasks a,b] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ppbench.v2.task import RESULTS

ROOT = Path(__file__).resolve().parents[2]
from ppbench.v2.task import CORE
RUNS = "core_v2.4"  # cumulative trajectories with an immutable checkpoint after every submit
CONDITIONS = ["name_only", "name_only+image", "attributes", "attributes+image", "functional", "functional+image"]
# Main experiment first (the user's decision, 2026-09-15): one Tier-B text+image trajectory of 3 cumulative rounds per task.
# The other conditions and Tier A are ablations, run later by passing --conditions / --tiers.
# Tier C (creation only, no catalog: the user's third track, 2026-09-16) runs the same main cell with --tiers C.
MAIN_CONDITIONS = ["name_only+image"]
MAIN_TIERS = ("B",)
LEGACY_RUNS = "core_v2.3"   # finished 3-round episodes from before the checkpoint change count as done (never rerun)
PROGRESS = RESULTS / "_core" / "progress"
CAP_FILE = RESULTS / "_core" / "concurrency_cap.txt"   # optional "<open-model cap> <claude cap>"; read when a batch starts


def cap(requested, which=0):
    try:
        return max(1, min(requested, int(CAP_FILE.read_text().split()[which])))
    except (OSError, ValueError, IndexError):
        return requested


def core_tasks():
    return [t["task_id"] for t in json.loads(CORE.read_text())]


def run_dir(task, system, tier, cond, rounds, seed):
    return RESULTS / task / "runs" / RUNS / f"{system}__{tier}__{cond}__r{rounds}__s{seed}"


def _ldraw_tasks():
    return {t["task_id"] for t in json.loads(CORE.read_text()) if any(p.get("catalog") == "LDraw" for p in t["subpart_pool"]["parts"])}


def valid_run(d, ldraw_tasks):
    """Count only a final submission that completed the requested trajectory."""
    f = d / "design" / "design.json"
    if not f.exists():
        return False
    try:
        meta = json.loads(f.read_text()).get("meta") or {}
    except (OSError, json.JSONDecodeError):
        return False
    match = re.search(r"__r(\d+)__s\d+$", d.name)
    expected = int(match.group(1)) if match else meta.get("rounds")
    if not meta.get("submitted_final") or meta.get("rounds_completed") != expected:
        return False
    task = d.parents[2].name
    if task in ldraw_tasks:
        return meta.get("pool_mesh_version") == 2
    return True


def legacy_done(task, system, tier, cond, rounds, seed, ldraw_tasks):
    """A core_v2.3 episode of the same configuration that ran the whole trajectory to a final submit.

    Until 2026-09-16 this accepted any exported design, so an episode that stopped after round 0 still counted as
    done and was never re-queued: qwen3-vl-30b-a3b had 28 of its 40 Tier B "finished" episodes at
    rounds_completed 0, and qwen3.5-35b-a3b 6. The bar is now the same as valid_run's, so those are rerun under
    the v2.4 harness (which also writes a rounds/round_n checkpoint per submit). The v2.3 directories are kept.
    """
    d = RESULTS / task / "runs" / LEGACY_RUNS / f"{system}__{tier}__{cond}__r{rounds}__s{seed}"
    f = d / "design" / "design.json"
    if not f.exists():
        return False
    try:
        meta = json.loads(f.read_text()).get("meta") or {}
    except (OSError, json.JSONDecodeError):
        return False
    if task in ldraw_tasks and meta.get("pool_mesh_version") != 2:
        return False
    return bool(meta.get("submitted_final")) and meta.get("rounds_completed") == rounds


# Stops that say something about the serving stack, not the model: an episode that ended this way is rerun.
INFRA_STOPS = {"repeated API errors"}
# PPBENCH_ALSO_INFRA="context overflow": also rerun episodes that stopped for this reason. Used on 2026-09-18 for models
# that had been served below their native context length (32768 / 49152), where the overflow was ours, not the model's.
INFRA_STOPS.update(x.strip() for x in os.environ.get("PPBENCH_ALSO_INFRA", "").split(",") if x.strip())


def model_ended(d, ldraw_tasks):
    """The model ended its own episode (context overflow, step limit, no tool calls after nudges).

    That is a result, not a broken job: with --keep-ended it is kept instead of rerun (the user's rule of
    2026-09-17, "do not re-run anything we have"). An episode stopped by the serving stack is not a result.
    """
    f = d / "design" / "design.json"
    if not f.exists():
        return False
    try:
        meta = json.loads(f.read_text()).get("meta") or {}
    except (OSError, json.JSONDecodeError):
        return False
    if d.parents[2].name in ldraw_tasks and meta.get("pool_mesh_version") != 2:
        return False
    stop = (meta.get("loop") or {}).get("stop")
    return bool(stop) and stop not in INFRA_STOPS


def read_rerun_list(path):
    """Exactly the episodes to rerun, one `task|tier|condition|rounds|seed` per line.

    Used when a fix changes what an episode would have been, but only for the episodes the fix touches: on
    2026-09-20 the ERNIE runs (the <name>/<arguments> tool-call shape, which the parser now reads) and the
    context-overflow episodes of the four models served below their native context. A listed episode is rerun
    whether or not it looks valid or ended by itself, and nothing outside the list is queued.
    """
    out = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        task, tier, cond, rounds, seed = (x.strip() for x in line.split("|"))
        out.add((task, tier, cond, int(rounds), int(seed)))
    return out


def jobs(system, tasks=None, conditions=None, tiers=MAIN_TIERS, rounds=3, seed=0, keep_ended=False, rerun=None):
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds is the total number in one cumulative episode; pass an integer such as 3")
    tasks = tasks or core_tasks()
    ld = _ldraw_tasks()
    out = []
    if rerun:
        return [(t, ti, c, r, sd, run_dir(t, system, ti, c, r, sd)) for t, ti, c, r, sd in sorted(rerun)]
    for cond in conditions or MAIN_CONDITIONS:
        for tier in tiers:
            for task in tasks:
                d = run_dir(task, system, tier, cond, rounds, seed)
                if valid_run(d, ld) or legacy_done(task, system, tier, cond, rounds, seed, ld):
                    continue
                if keep_ended and model_ended(d, ld):
                    continue
                out.append((task, tier, cond, rounds, seed, d))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system-name", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--tasks")
    ap.add_argument("--conditions")
    ap.add_argument("--tiers", default=",".join(MAIN_TIERS))
    ap.add_argument("--rounds", type=int, default=3,
                    help="total cumulative rounds in one episode (default: 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--keep-ended", action="store_true",
                    help="keep episodes the model ended itself (a result); rerun only missing or infra-stopped ones")
    ap.add_argument("--rerun-list",
                    help="file of `task|tier|condition|rounds|seed` lines: run exactly these episodes, forced, and "
                         "nothing else (the earlier directory is preserved under _core/discarded/ as usual)")
    ap.add_argument("--also-infra", default="",
                    help="comma list of extra stop reasons to treat as infra stops for this run (e.g. 'context overflow' "
                         "when the model was served below its native context length); such episodes are rerun")
    ns = ap.parse_args(argv)
    INFRA_STOPS.update(x.strip().replace("_", " ") for x in ns.also_infra.split(",") if x.strip())
    ns.concurrency = cap(ns.concurrency, 0)
    tiers = tuple(x.strip() for x in ns.tiers.split(",") if x.strip())
    if not tiers or any(x not in {"A", "B", "C"} for x in tiers):
        ap.error("--tiers must be a comma-separated subset of A, B, C")
    ldraw_tasks = _ldraw_tasks()
    todo = jobs(ns.system_name, ns.tasks.split(",") if ns.tasks else None,
                ns.conditions.split(",") if ns.conditions else None, tiers=tiers, rounds=ns.rounds, seed=ns.seed,
                keep_ended=ns.keep_ended, rerun=read_rerun_list(ns.rerun_list) if ns.rerun_list else None)
    if ns.limit:
        todo = todo[:ns.limit]
    PROGRESS.mkdir(parents=True, exist_ok=True)
    prog = PROGRESS / f"{ns.system_name}.json"
    state = {"system": ns.system_name, "model": ns.model, "total_todo": len(todo), "done": 0, "failed": 0, "started": time.time()}
    lock = threading.Lock()
    env = dict(os.environ, TMPDIR=os.environ.get("TMPDIR", "/tmp"), RENDER_THREADS=os.environ.get("RENDER_THREADS", "4"))

    def one(job):
        task, tier, cond, r, seed, d = job
        if d.exists():   # an unfinished earlier attempt: kept under _core/discarded/, never deleted
            dst = RESULTS / "_core" / "discarded" / f"{task}__{d.name}__{time.strftime('%Y%m%d-%H%M%S')}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(d / "occ_cache", ignore_errors=True)
            shutil.move(str(d), str(dst))
        d.mkdir(parents=True, exist_ok=True)
        logs = RESULTS / task / "logs_core"
        logs.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "ppbench.v2.loop", "--model", ns.model, "--system-name", ns.system_name,
               "--base-url", ns.base_url, "--task", task, "--tier", tier, "--condition", cond, "--rounds", str(r),
               "--seed", str(seed), "--out", str(d), "--extra-body", ns.extra_body,
               "--snapshot", str(RESULTS / task / "task_snapshot_core.json")] + (["--no-vision"] if ns.no_vision else [])
        with open(logs / f"ep_{d.name}.log", "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
        ok = rc == 0 and valid_run(d, ldraw_tasks)
        if not ok and ns.keep_ended and model_ended(d, ldraw_tasks):
            ok = True    # the model's own stop is a result: counted, and not retried below
        with lock:
            state["done" if ok else "failed"] += 1
            state["updated"] = time.time()
            prog.write_text(json.dumps(state))
        return ok

    prog.write_text(json.dumps(state))
    with ThreadPoolExecutor(ns.concurrency) as ex:
        results = list(ex.map(one, todo))
    retry = [j for j, ok in zip(todo, results) if not ok]
    if retry:   # one more pass for episodes that crashed (e.g. a transient server error)
        state["retried"] = len(retry)
        with ThreadPoolExecutor(ns.concurrency) as ex:
            list(ex.map(one, retry))
    state["finished"] = time.time()
    prog.write_text(json.dumps(state))
    print(json.dumps(state))


if __name__ == "__main__":
    main()
