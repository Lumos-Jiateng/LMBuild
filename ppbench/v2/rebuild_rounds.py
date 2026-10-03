"""Rebuild the per-round checkpoints of legacy core_v2.3 episodes by replaying their stored trace.

The v2.3 harness kept only the final design, so rounds 1 and 2 of those episodes have no `rounds/round_n/`
directory and the board can show no parts, joints or materials for them — although `trace.json` holds every
tool call the model made, each tagged with its round. This replays that trace through a fresh AssemblyEnv and
calls the same `session.save_round_checkpoint` the v2.4 loop uses, so a rebuilt checkpoint is byte-compatible
with a natively written one.

Faithfulness gate: the replay is accepted only when its final design has the same part, joint and material
counts as the stored `design/design.json`. A replay that diverges is reported and nothing is written, because a
checkpoint that does not match what the model actually built is worse than a missing one. Rebuilt checkpoints
are marked `"rebuilt_from_trace": true` in their design meta.

    .venv_eval/bin/python -m ppbench.v2.rebuild_rounds [--systems a,b] [--tasks a,b] [--jobs 16] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ppbench.v2.core_batch import LEGACY_RUNS, core_tasks
from ppbench.v2.env import AssemblyEnv
from ppbench.v2.session import save_round_checkpoint
from ppbench.v2.task import RESULTS, Task

LEGACY_SYSTEMS = ["gpt-oss-120b", "qwen3-vl-30b-a3b-instruct", "qwen3.5-35b-a3b"]


def _counts(design_json):
    d = json.loads(Path(design_json).read_text())
    parts = d.get("parts") or []
    return (len(parts), len(d.get("joints") or []),
            len({p.get("material") for p in parts if p.get("material")}))


def candidates(systems, tasks, match=None):
    out = []
    for task in tasks:
        for d in sorted((RESULTS / task / "runs" / LEGACY_RUNS).glob("*__*__*__r*__s*")
                        if (RESULTS / task / "runs" / LEGACY_RUNS).exists() else []):
            system = d.name.split("__")[0]
            if systems and system not in systems:
                continue
            if not (d / "trace.json").exists() or not (d / "design" / "design.json").exists():
                continue
            if (d / "rounds").exists():
                continue
            if match and match not in d.name:
                continue
            out.append(d)
    return out


def rebuild(d: Path):
    """Replay one episode. Returns (dir, n_checkpoints, note)."""
    try:
        task_name, name = d.parents[2].name, d.name
        system, tier, cond, rr, _ = name.split("__")
        rounds = int(rr[1:])
        snap = RESULTS / task_name / "task_snapshot_core.json"
        task = Task(task_name, snapshot=snap if snap.exists() else RESULTS / task_name / "task_snapshot.json")
        trace = json.loads((d / "trace.json").read_text())
        work = Path(tempfile.mkdtemp(prefix="rebuild_", dir="/dev/shm"))
        try:
            env = AssemblyEnv(task, tier, cond, rounds, workdir=work)
            made = 0
            for call in trace:
                tool, args = call.get("tool"), call.get("args") or {}
                if not tool:
                    continue
                try:
                    res = env.call(tool, args)
                except Exception:
                    res = {"ok": False}
                if tool == "submit" and res.get("ok"):
                    save_round_checkpoint(env, system, "replayed from trace.json", res)
                    made += 1
                if env.s["done"]:
                    break
            final = env.design(system, {"transport": "replayed from trace.json", "rebuilt_from_trace": True})
            final.save(work / "design_replay")
            want, got = _counts(d / "design" / "design.json"), _counts(work / "design_replay" / "design.json")
            if want != got:
                return (str(d), 0, f"diverged: stored parts/joints/materials {want} vs replay {got}")
            src = work / "rounds"
            if not src.exists():
                return (str(d), 0, "replay produced no submit")
            for cp in sorted(src.iterdir()):
                dst = d / "rounds" / cp.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    continue
                shutil.copytree(cp, dst)
                dj = dst / "design" / "design.json"
                if dj.exists():
                    obj = json.loads(dj.read_text())
                    obj.setdefault("meta", {})["rebuilt_from_trace"] = True
                    dj.write_text(json.dumps(obj))
            return (str(d), made, "ok")
        finally:
            shutil.rmtree(work, ignore_errors=True)
    except Exception as e:
        return (str(d), 0, f"error: {e.__class__.__name__}: {e}"[:200])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(LEGACY_SYSTEMS))
    ap.add_argument("--tasks")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--match", help="only run directories whose name contains this, e.g. __name_only+image__r3__")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)
    systems = {s for s in ns.systems.split(",") if s} if ns.systems != "all" else None
    tasks = ns.tasks.split(",") if ns.tasks else core_tasks()
    todo = candidates(systems, tasks, ns.match)
    if ns.limit:
        todo = todo[:ns.limit]
    print(f"{len(todo)} legacy episodes without rounds/")
    if ns.dry_run:
        for d in todo[:10]:
            print("  ", d)
        return
    ok = div = err = 0
    with ProcessPoolExecutor(ns.jobs) as ex:
        for path, made, note in ex.map(rebuild, todo):
            if note == "ok":
                ok += 1
            elif note.startswith("diverged"):
                div += 1
            else:
                err += 1
            if note != "ok":
                print(f"  {note}  {path}")
    print(json.dumps({"rebuilt": ok, "diverged": div, "failed": err, "total": len(todo)}))


if __name__ == "__main__":
    main()
