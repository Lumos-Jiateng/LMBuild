"""Run Claude models over the core set through headless Claude Code (`claude -p`), one shell session per episode.

Tool arms: each episode is a `ppbench.v2.session` state (same environment, prompt and tools as the chair demo's
Claude runs) driven by `claude -p --model <id>`, whose only permitted tools are that session command and reading
files (the condition image, catalog sheets and renders). Outputs go to
results/v2/<task>/runs/core_v2.3/<system>__<tier>__<condition>__r<rounds>__s0/ with the task frozen in
task_snapshot_core.json. Per episode the CLI's JSON result (cost, token usage, turns, duration) is kept in
claude_run.json. When the CLI stops before the last submit, the session is finished anyway so whatever was built is
exported and marked `submitted_final: false`.

Nothing paid for is thrown away (2026-09-15). Every tool call is saved in session_state.json as it happens, and
claude_run.json (with the conversation's session id) is rewritten after every CLI attempt, marked `in_progress` until
the episode ends. A batch that is killed or hits the account limit can be started again: an interrupted episode is
resumed, not rerun. The same conversation continues (`claude -p --resume`) when its transcript can be found.
Otherwise a new conversation picks up the saved state. Earlier attempts stay in the record. A run directory is never
deleted: one that must start over (an outdated LDraw pool, a broken state) is moved to results/v2/_core/claude_discarded/.

Annotation arm (external generators): the same runner with `--mode annotate` drives `session init-annotate` states
prepared by ppbench.v2.core_externals.

    .venv_eval/bin/python -m ppbench.v2.claude_batch tool --models claude-haiku-4-5-20251001,claude-sonnet-5,claude-opus-5 \
        --concurrency 30 [--tasks a,b] [--conditions ...] [--rounds 3] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ppbench.v2.core_batch import CONDITIONS, RUNS, _ldraw_tasks, core_tasks, run_dir, valid_run
from ppbench.v2.task import RESULTS

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv_eval" / "bin" / "python")
class _Systems(dict):
    def __missing__(self, model):      # any other Claude model id is its own system name
        return model


SYSTEM = _Systems({"claude-haiku-4-5-20251001": "claude-haiku-4.5", "claude-sonnet-5": "claude-sonnet-5", "claude-opus-5": "claude-opus-5",
                   "claude-fable-5-1": "claude-fable-5.1"})
PROGRESS = RESULTS / "_core" / "progress"
PRESERVED = RESULTS / "_core" / "claude_preserved"    # copies of interrupted episodes taken on 2026-09-15 (see LIMITATIONS)
DISCARDED = RESULTS / "_core" / "claude_discarded"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
EPISODE_TIMEOUT_S = 3 * 3600
# `claude -p` ends when the model ends its turn with text, which often happens mid-episode. As in the open-model loop
# (up to 3 nudges without a tool call), the same conversation is resumed with a continuation message; it stops after 3
# consecutive continuations that made no new tool call, or when the episode is done.
NUDGE_LIMIT = 3
MAX_CONTINUATIONS = 12
CONTINUE_PROMPT = ("The episode is not finished: no submit has returned \"episode_done\": true yet. Continue working with the "
                   "session command until a submit result says \"episode_done\": true.")
CLI_ENV = {"TMPDIR": os.environ.get("TMPDIR", "/tmp"), "RENDER_THREADS": "4", "PYTHONPATH": str(ROOT),
           "BASH_DEFAULT_TIMEOUT_MS": "1800000", "BASH_MAX_TIMEOUT_MS": "3600000"}   # tool calls can take minutes on a busy machine
# Isolation (2026-09-15): runs until then used the PP_Bench checkout as working directory, so every evaluated session had the
# user's project auto-memory index (and the repo's git status) in context. Each episode now runs from its own directory
# outside the repository with auto-memory disabled; commands and state files use absolute paths.
ISO_ROOT = Path(os.environ.get("PPBENCH_ISO_ROOT", Path.home() / ".cache" / "ppbench_claude_iso"))
CLI_SETTINGS = '{"autoMemoryEnabled": false}'
SESSION_CMD = f"{PY} -m ppbench.v2.session"
LIMIT_MARKERS = ("hit your session limit", "usage limit", "rate limit", "you've reached your")   # the last: "You've reached your Fable limit" (2026-09-18)
# conservative cost of one 3-round trajectory, reserved while it runs (measured round-1 episodes: Haiku $0.37, Sonnet $0.94)
EST_TRAJECTORY_USD = {"claude-haiku-4-5-20251001": 1.2, "claude-sonnet-5": 3.0, "claude-opus-5": 6.0, "claude-fable-5-1": 10.0}


def _limit_hit(res):
    txt = (str(res.get("result") or "") + str(res.get("error") or "") + str(res.get("raw") or "")).lower()
    return any(m in txt for m in LIMIT_MARKERS)

TOOL_PROMPT = """You are the model being evaluated in a 3D assembly benchmark. Work only through the session command below; do not edit,
create or delete any files, and do not run any other programs except reading image files.

Your session state file: {state}

1. Run: {cmd} prompt --state {state}
   Read the whole output: it is your task, your tools and how to call them. Open every image file it names with your
   file-reading tool.
2. Build the object with the tools. Tool calls look like:
   {cmd} call --state {state} <tool_name> '<JSON arguments>'
   Each call prints one JSON result; image results are PNG paths you open to look at.
3. The episode may have several rounds: each submit ends a round and returns a review. Keep working until a submit
   result says "episode_done": true. If a call returns an error, read it and correct the call.
Reply at the end with one line: the number of parts and joints in your final design."""
RESUMED_NOTE = """

This episode was interrupted by an infrastructure failure and is being resumed. The session state already holds the work done so far
(parts, joints, round and tool budgets used). After reading the prompt output, call get_scene to see the current design and
continue from there; do not start over."""


def _rec(d):
    try:
        return json.loads((d / "claude_run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _attempt_cost(a):
    """An attempt refused by the account limit uses no tokens but the CLI still reports the resumed session's earlier
    total_cost_usd, so counting it would bill the same work again (2026-09-22: ~$190 of phantom Fable cost)."""
    u = a.get("usage") or {}
    if not any(u.get(k) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")):
        return 0.0
    return a.get("total_cost_usd") or 0.0


def _cost(rec):
    """Real spend of one episode, recomputed from its attempts (a stored cost_usd may hold phantom limit attempts)."""
    return sum(_attempt_cost(a) for a in rec.get("attempts", []))


def _write_rec(d, rec):
    tmp = d / "claude_run.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(d / "claude_run.json")


def _finished(d, ld, rerun_unisolated=False):
    rr = _rec(d)
    if not rr or rr.get("in_progress") or not valid_run(d, ld):
        return False
    if rerun_unisolated and not rr.get("isolated"):
        return False
    if rr.get("episode_done"):
        return True
    limited = any(_limit_hit(a) for a in rr.get("attempts", []))
    return rr.get("harness") == "continuations" and not limited


def _live_workdirs():
    """Working directories of this user's running `claude` processes."""
    out = set()
    for p in Path("/proc").iterdir():
        try:
            if p.name.isdigit() and (p / "comm").read_text().strip() == "claude":
                out.add(os.readlink(p / "cwd"))
        except OSError:
            pass
    return out


def _driven_elsewhere(d, live):
    """True while another batch process is driving this episode: its claude runs in the episode's workdir, or its record
    was rewritten within the last 20 min (a batch waiting out an account limit rewrites it every 15 min with no claude
    running). Two batches resuming one conversation would pay twice and corrupt the session state (2026-09-22)."""
    rr = _rec(d)
    if not rr.get("in_progress"):
        return False
    if rr.get("workdir") in live:
        return True
    try:
        return time.time() - (d / "claude_run.json").stat().st_mtime < 1200
    except OSError:
        return False


def tool_jobs(models, tasks=None, conditions=None, tiers=("A", "B"), rounds=3, seed=0, rerun_unisolated=False):
    out = []
    ld = _ldraw_tasks()
    live = _live_workdirs()
    for cond in conditions or CONDITIONS:
        for tier in tiers:
            for task in tasks or core_tasks():
                for m in models:
                    d = run_dir(task, SYSTEM[m], tier, cond, rounds, seed)
                    if not _finished(d, ld, rerun_unisolated) and not _driven_elsewhere(d, live):
                        out.append({"model": m, "task": task, "tier": tier, "cond": cond, "rounds": rounds, "seed": seed, "dir": d})
    return out


ANNOTATE_PROMPT = """You are the annotator in a 3D-object benchmark. Work only through the session command below; do not edit, create
or delete any files, and do not run any other programs except reading image files.

Your session state file: {state}

1. Run: {cmd} prompt --state {state}
   Read the whole output: it is your task, your tools and how to call them.
2. Open the condition image it names (if any) with your file-reading tool, then call get_scene and render and open the PNG
   files they return, so you know what each part is.
3. Do the task: a role and material for every part, the joints the real object needs (use measure for origins), the
   assembly sequence. Use check once or twice to see what is missing. Tool calls look like:
   {cmd} call --state {state} set_part_info '{{"instance_id": "part_00", "role": "...", "material": "..."}}'
4. Finish by calling submit ("episode_done": true). If a call returns an error, read it and correct the call.
Reply at the end with one line: the number of parts, roles and joints you declared."""
ANNOT_SUFFIX = "claude-haiku-4.5"


def annotate_jobs(models, tasks=None):
    from ppbench.v2.core_externals import annotation_items
    out = []
    for task in tasks or core_tasks():
        for cond, item in annotation_items(task):
            for m in models:
                folder, name = item["key"].split("/")
                d = RESULTS / task / "annotated_core" / folder / f"{name}__{SYSTEM[m]}"
                if not _finished(d, set()):
                    out.append({"model": m, "task": task, "cond": cond, "item": item, "dir": d})
    return out


# ---------------------------------------------------------------- resuming interrupted episodes

def _discard(d, why):
    """Move a run directory that has to start over out of the way (never delete it)."""
    if not d.exists():
        return
    dst = DISCARDED / f"{d.relative_to(RESULTS).as_posix().replace('/', '__')}__{time.strftime('%Y%m%d-%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(d / "occ_cache", ignore_errors=True)       # cache only
    shutil.move(str(d), str(dst))
    (dst / "discarded.json").write_text(json.dumps({"why": why, "source": str(d), "at": time.time()}))


def _state_calls(d):
    try:
        st = json.loads((d / "session_state.json").read_text())
        return st["env"]["counters"]["tool_calls"], st
    except (OSError, json.JSONDecodeError, KeyError):
        return None, None


def _find_transcript_session(d, workdir):
    """Session id of the latest `claude -p` conversation that ran this episode from `workdir` (its transcript names the state file)."""
    proj = CLAUDE_PROJECTS / re.sub(r"[^A-Za-z0-9]", "-", str(workdir))
    if not proj.is_dir():
        return None
    needle = (d / "session_state.json").relative_to(ROOT).as_posix()      # old prompts used relative, new ones absolute paths
    r = subprocess.run(["grep", "-l", "-F", "--include=*.jsonl", "-r", needle, str(proj)], capture_output=True, text=True)
    cands = []
    for f in r.stdout.split():
        f = Path(f)
        if f.parent != proj:
            continue
        with open(f, errors="ignore") as fh:
            head = "".join(next(fh, "") for _ in range(12))
        if needle in head and ("You are the model being evaluated" in head or "You are the annotator" in head):
            cands.append(f)
    return max(cands, key=lambda f: f.stat().st_mtime).stem if cands else None


def _prepare_resume(job, ld):
    """Decide how an existing run directory continues. Returns None (start fresh) or the resume dict for _drive."""
    d = job["dir"]
    key = f"{d.parents[2].name}__{d.name}" if job.get("item") is None else None
    pres = PRESERVED / key if key else None
    calls, st = _state_calls(d)
    if pres is not None and (pres / "session_state.json").exists():
        pcalls, _ = _state_calls(pres)
        if pcalls and (calls is None or pcalls > calls):       # the preserved copy got further than what is here now
            _discard(d, "superseded by the preserved copy of the interrupted episode")
            shutil.copytree(pres, d, ignore=shutil.ignore_patterns("preserved.json"))
            calls, st = _state_calls(d)
    if st is None:
        _discard(d, "no usable session state")
        return None
    # The preserved copies were taken from the core_v2.3 run directories, and a session state carries the absolute path
    # it exports to. Copied into a core_v2.4 directory unchanged, the episode ran here but wrote its design, rounds and
    # renders back into core_v2.3, so the batch saw no design.json and counted a finished episode as failed -- and ran
    # (and paid for) it again on the next pass. Re-point any state whose `out` is not this directory before it is driven.
    if st.get("out") and Path(st["out"]).resolve() != d.resolve():
        st["out"] = str(d)
        tmp = d / "session_state.json.tmp"
        tmp.write_text(json.dumps(st))
        tmp.replace(d / "session_state.json")
    if d.parents[2].name in ld and st["env"].get("pool_mesh_version", 1) != 2:
        _discard(d, "LDraw pool meshes before pool_mesh_version 2")
        return None
    prev = _rec(d)
    iso = ISO_ROOT / d.relative_to(RESULTS).as_posix().replace("/", "__")
    if prev.get("workdir"):
        workdir, isolated = Path(prev["workdir"]), bool(prev.get("isolated"))
    elif prev:                                       # records before the isolation fix
        workdir, isolated = (iso, True) if prev.get("isolated") else (ROOT, False)
    else:                                            # killed before its first record was written
        workdir, isolated = (iso, True) if iso.exists() else (ROOT, False)
    if calls == 0:                                   # nothing was built: a clean, isolated start on the same state
        return {"session": None, "workdir": iso, "isolated": True, "rec": prev, "how": "restart (no tool calls yet)"}
    sid = next((a.get("session_id") for a in reversed(prev.get("attempts", [])) if a.get("session_id")), None)
    if sid and not (CLAUDE_PROJECTS / re.sub(r"[^A-Za-z0-9]", "-", str(workdir)) / f"{sid}.jsonl").exists():
        sid = None
    sid = sid or _find_transcript_session(d, workdir)
    if sid:
        return {"session": sid, "workdir": workdir, "isolated": isolated, "rec": prev, "how": "resume conversation"}
    return {"session": None, "workdir": iso, "isolated": False, "rec": prev, "how": "new conversation on saved state"}


def run_annotate_episode(job):
    from ppbench.v2.annotate import create
    from ppbench.v2.task import Task
    d = job["dir"]
    state = d / "session_state.json"
    prompt = ANNOTATE_PROMPT.format(cmd=SESSION_CMD, state=state)
    resume = _prepare_resume(job, set()) if d.exists() else None
    if resume:
        return _drive(job["model"], prompt + (RESUMED_NOTE if resume["how"].startswith("new") else ""), state, d, resume)
    d.mkdir(parents=True, exist_ok=True)
    snap = RESULTS / job["task"] / "task_snapshot_core.json"
    task = Task(job["task"], snapshot=snap)
    env, info = create(task, job["item"], d, job["cond"])
    st = {"task_id": job["task"], "out": str(d), "system": SYSTEM[job["model"]], "transport": "claude -p (annotate)",
          "created": time.time(), "snapshot": str(snap), "annotation": info, "env": env.to_state()}
    state.write_text(json.dumps(st, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    return _drive(job["model"], prompt, state, d)


def _session(*args, timeout=600):
    return subprocess.run([PY, "-m", "ppbench.v2.session", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


EPISODE_LOCKS = Path("/dev/shm/ppb_claude_locks")     # host-local; every Claude batch runs on the run host


def run_tool_episode(job, ld=frozenset()):
    """One episode under an exclusive per-episode lock, so two batches (e.g. a second chain started beside
    claude_of_all50.sh) never drive the same episode; the loser skips it. The kernel drops the lock if we die."""
    import fcntl
    EPISODE_LOCKS.mkdir(parents=True, exist_ok=True)
    with open(EPISODE_LOCKS / job["dir"].relative_to(RESULTS).as_posix().replace("/", "__"), "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"ok": False, "error": "driven by another batch", "skipped": True}
        if _finished(job["dir"], ld):      # finished by another batch after our job list was built
            return {"ok": True, "cost": 0, "skipped": True}
        return _run_tool_episode(job, ld)


def _run_tool_episode(job, ld=frozenset()):
    d = job["dir"]
    state = d / "session_state.json"
    prompt = TOOL_PROMPT.format(cmd=SESSION_CMD, state=state)
    resume = _prepare_resume(job, ld) if d.exists() or (PRESERVED / f"{d.parents[2].name}__{d.name}").exists() else None
    if resume:
        return _drive(job["model"], prompt + (RESUMED_NOTE if resume["how"].startswith("new") else ""), state, d, resume)
    d.mkdir(parents=True, exist_ok=True)
    snap = RESULTS / job["task"] / "task_snapshot_core.json"
    r = _session("init", "--state", str(state), "--task", job["task"], "--tier", job["tier"], "--condition", job["cond"],
                 "--rounds", str(job["rounds"]), "--out", str(d), "--system", SYSTEM[job["model"]], "--snapshot", str(snap))
    if r.returncode != 0:
        return {"ok": False, "error": "init: " + r.stderr[-500:]}
    return _drive(job["model"], prompt, state, d)


def _cli(model, prompt, resume=None, workdir=None):
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--max-turns", "600",
           "--settings", CLI_SETTINGS, "--allowedTools", f"Bash({SESSION_CMD}:*)", "Read"]
    if resume:
        cmd += ["--resume", resume]
    workdir = Path(workdir or ISO_ROOT / "misc")
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=EPISODE_TIMEOUT_S, env=dict(os.environ, **CLI_ENV))
        try:
            return json.loads(p.stdout.strip())
        except json.JSONDecodeError:
            return {"is_error": True, "raw": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"is_error": True, "error": f"timeout after {EPISODE_TIMEOUT_S} s"}


def _drive(model, prompt, state, d, resume=None):
    """Run (or resume) one episode's conversation. claude_run.json is rewritten after every attempt, so an interruption
    at any point leaves the session id and the cost record on disk."""
    t0 = time.time()
    prev = (resume or {}).get("rec") or {}
    iso = ISO_ROOT / d.relative_to(RESULTS).as_posix().replace("/", "__")
    workdir = Path(resume["workdir"]) if resume else iso
    calls = lambda: json.loads(state.read_text())["env"]["counters"]["tool_calls"]
    done = lambda: json.loads(state.read_text())["env"]["done"]
    rec = {"model": model, "started": prev.get("started", t0), "harness": "continuations",
           "isolated": resume["isolated"] if resume else True, "workdir": str(workdir), "in_progress": True,
           "attempts": list(prev.get("attempts", [])), "model_ids": list(prev.get("model_ids", [])),
           "resumed": list(prev.get("resumed", [])), "wall_s_before": prev.get("wall_s", 0) or 0}
    if resume:
        rec["resumed"].append({"at": t0, "how": resume["how"], "session_id": resume["session"], "tool_calls": calls(),
                               "attempts_before": len(prev.get("attempts", []))})
    n_prev = len(rec["attempts"])
    _write_rec(d, rec)
    session, idle, n = (resume or {}).get("session"), 0, 0
    while n <= MAX_CONTINUATIONS:
        before = calls()
        res = _cli(model, prompt if session is None else CONTINUE_PROMPT, resume=session, workdir=workdir)
        rec["attempts"].append({k: res.get(k) for k in ("is_error", "subtype", "num_turns", "duration_ms", "total_cost_usd", "usage",
                                                         "session_id", "error", "raw", "stderr")} | {"result": str(res.get("result"))[-600:],
                                                                                                    "tool_calls_before": before, "tool_calls_after": calls(),
                                                                                                    "at": time.time()})
        for mid in (res.get("modelUsage") or {}):
            if mid not in rec["model_ids"]:
                rec["model_ids"].append(mid)
        _write_rec(d, rec)
        if _limit_hit(res):      # account limit: wait and repeat the same step (not a continuation, not the model's doing)
            if calls() > 0:
                session = res.get("session_id") or session
            time.sleep(900)
            continue
        if res.get("is_error") and session is None and calls() == 0:
            time.sleep(60 + random.random() * 60)      # CLI failed before the model did anything: start over
            n += 1
            continue
        session = res.get("session_id") or session
        if done() or session is None:
            break
        idle = idle + 1 if calls() == before else 0
        if idle >= NUDGE_LIMIT:
            break
        n += 1
    st = json.loads(state.read_text())
    if not st["env"]["done"]:
        _session("finish", "--state", str(state))
    new_cost = sum(_attempt_cost(a) for a in rec["attempts"][n_prev:])
    rec.update({"in_progress": False, "wall_s": round(rec["wall_s_before"] + time.time() - t0, 1),
                "tool_calls": st["env"]["counters"]["tool_calls"], "episode_done": st["env"]["done"],
                "continuations": max(0, len(rec["attempts"]) - 1),
                "cost_usd": _cost(rec)})
    _write_rec(d, rec)
    return {"ok": (d / "design" / "design.json").exists(), "cost": new_cost}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tool", "annotate"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--tasks")
    ap.add_argument("--conditions")
    ap.add_argument("--tiers", default="A,B")
    ap.add_argument("--rounds", type=int, default=3,
                    help="total cumulative rounds in one episode (default: 3)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rerun-unisolated", action="store_true", help="also rerun finished episodes from before the isolation fix")
    ap.add_argument("--budget-usd", type=float, help="stop starting episodes once spent + running estimate would pass this")
    ns = ap.parse_args(argv)
    from ppbench.v2.core_batch import cap
    ns.concurrency = cap(ns.concurrency, 1)
    models = ns.models.split(",")
    tiers = tuple(x.strip() for x in ns.tiers.split(",") if x.strip())
    if not tiers or any(x not in {"A", "B", "C"} for x in tiers):
        ap.error("--tiers must be a comma-separated subset of A, B, C")
    ld = _ldraw_tasks()
    if ns.mode == "annotate":
        todo = annotate_jobs(models, ns.tasks.split(",") if ns.tasks else None)
    else:
        todo = tool_jobs(models, ns.tasks.split(",") if ns.tasks else None, ns.conditions.split(",") if ns.conditions else None,
                         tiers=tiers, rounds=ns.rounds, rerun_unisolated=ns.rerun_unisolated)
        # interrupted episodes with work in them first, so their conversations are resumed while the transcripts are fresh
        todo.sort(key=lambda j: -(_state_calls(j["dir"])[0] or 0) if (j["dir"] / "session_state.json").exists() else 0)
    if ns.limit:
        todo = todo[:ns.limit]
    PROGRESS.mkdir(parents=True, exist_ok=True)
    prog = PROGRESS / f"claude_{ns.mode + '_' if ns.mode != 'tool' else ''}{'+'.join(SYSTEM[m] for m in models)}.json"
    # Budget (2026-09-15, the user's cap): money already spent on episodes of this run namespace counts, and every running
    # episode reserves a conservative per-trajectory estimate until its real cost is known.
    spent_before = sum(_cost(_rec(Path(f).parent))
                       for m in models for f in RESULTS.glob(f"*/runs/{RUNS}/{SYSTEM[m]}__*/claude_run.json"))
    state = {"models": models, "total_todo": len(todo), "done": 0, "failed": 0, "cost_usd": 0.0, "started": time.time(),
             "budget_usd": ns.budget_usd, "spent_before_usd": round(spent_before, 2), "skipped_budget": 0, "reserved_usd": 0.0}
    lock = threading.Lock()

    def one(job):
        est = EST_TRAJECTORY_USD.get(job["model"], 5.0) * max(1, ns.rounds) / 3
        with lock:
            if ns.budget_usd is not None and spent_before + state["cost_usd"] + state["reserved_usd"] + est > ns.budget_usd:
                state["skipped_budget"] += 1
                prog.write_text(json.dumps(state))
                return
            state["reserved_usd"] += est
        try:
            res = run_annotate_episode(job) if ns.mode == "annotate" else run_tool_episode(job, ld)
        except Exception as e:  # one broken episode must not stop the batch
            res = {"ok": False, "error": repr(e)}
        with lock:
            state["reserved_usd"] -= est
            state["done" if res.get("ok") else "failed"] += 1
            state["cost_usd"] += res.get("cost") or 0
            state["updated"] = time.time()
            prog.write_text(json.dumps(state))

    prog.write_text(json.dumps(state))
    with ThreadPoolExecutor(ns.concurrency) as ex:
        list(ex.map(one, todo))
    state["finished"] = time.time()
    prog.write_text(json.dumps(state))
    print(json.dumps(state))


if __name__ == "__main__":
    main()
