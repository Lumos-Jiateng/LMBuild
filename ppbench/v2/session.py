"""Shell transport for terminal agents (e.g. Claude Code subagents).

The state lives in one JSON file; each call reloads it, dispatches through the
same AssemblyEnv the in-process loop uses, and saves. Images (condition image,
pool sheets, renders) are files the agent opens itself.

    python -m ppbench.v2.session init   --state S.json --task swivel_office_chair --tier A \
            --condition name_only+image --rounds 2 --out results/v2/swivel_office_chair/runs/<run> --system <model>
    python -m ppbench.v2.session prompt --state S.json
    python -m ppbench.v2.session call   --state S.json place_part '{"part_id": "P035", ...}'
    python -m ppbench.v2.session finish --state S.json
    python -m ppbench.v2.session init-annotate --state S.json --task swivel_office_chair --from partcrafter_np15_s0 \
            --system <annotator> --out results/v2/swivel_office_chair/annotated/<run>      # see ppbench/v2/annotate.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ppbench.v2.env import AssemblyEnv, system_prompt, tool_docs
from ppbench.v2.task import RESULTS, Task


def _load(state_path):
    st = json.loads(Path(state_path).read_text())
    task = Task(st["task_id"], snapshot=Path(st.get("snapshot") or RESULTS / st["task_id"] / "task_snapshot.json"))
    if st["env"].get("mode") == "annotate":
        from ppbench.v2.annotate import AnnotationEnv
        env = AnnotationEnv.from_state(st["env"], task, workdir=st["out"])
    else:
        env = AssemblyEnv.from_state(st["env"], task, workdir=st["out"])
    env.occ_dir = Path(st["out"]) / "occ_cache"
    return env, st


def _save(state_path, env, st):
    try:
        env.save_occ_cache()
    except Exception:   # the cache only saves time; never fail a call over it
        pass
    st["env"] = env.to_state()
    tmp = Path(str(state_path) + ".tmp")
    tmp.write_text(json.dumps(st))
    tmp.replace(state_path)


def save_round_checkpoint(env, system: str, transport: str, submit_result=None):
    """Persist the cumulative design immediately after each successful submit."""
    n = len(env.s["round_counters"])
    if n < 1:
        return None
    out = Path(env.workdir) / "rounds" / f"round_{n}"
    d = env.design(system, {"transport": transport, "checkpoint_round": n})
    d.save(out / "design")
    (out / "state.json").write_text(json.dumps(env.to_state()))
    (out / "trace.json").write_text(json.dumps(env.s["trace"], indent=0))
    if submit_result is not None:
        (out / "submit_result.json").write_text(json.dumps(submit_result, indent=1))
    return out


def user_prompt(task: Task, cond: dict, tier: str, sheets, vision=True, image_paths=True) -> str:
    lines = [f"TASK: {cond['text']}"]
    if cond.get("image"):
        if vision:
            ref = f" ({cond['image']['path']})" if image_paths else ""
            lines.append(f"The condition image {cond['image']['id']} is attached{ref}.")
        else:
            lines.append(f"(The task refers to image {cond['image']['id']}, which you cannot see.)")
    lines.append(f"Tier {tier}: " + {
        "A": "use catalog parts only.",
        "B": "use catalog parts, and create new parts where the catalog has none that fit.",
        "C": "there is no catalog: create every part you place.",
    }[tier])
    if vision and sheets and tier != "C":
        ref = (": " + ", ".join(sheets)) if image_paths else ""
        lines.append(f"A picture of every catalog part, labelled with id, name and size, is attached{ref}. "
                     "view_catalog_sheet shows it again.")
    lines.append("Start with list_materials, then create_part." if tier == "C" else
                 "Start with list_parts and list_materials.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("--state", required=True)
    a.add_argument("--task", required=True)
    a.add_argument("--tier", required=True, choices=["A", "B", "C"])
    a.add_argument("--condition", required=True)
    a.add_argument("--rounds", type=int, default=2)
    a.add_argument("--out", required=True)
    a.add_argument("--system", required=True)
    a.add_argument("--snapshot", help="task snapshot file (default results/v2/<task>/task_snapshot.json)")
    an = sub.add_parser("init-annotate")
    an.add_argument("--state", required=True)
    an.add_argument("--task", required=True)
    an.add_argument("--from", dest="source", required=True, help="external design key, e.g. partcrafter_np15_s0")
    an.add_argument("--condition", default="name_only+image")
    an.add_argument("--out", required=True)
    an.add_argument("--system", required=True, help="annotator name")
    for name in ("prompt", "finish", "status"):
        p = sub.add_parser(name)
        p.add_argument("--state", required=True)
    c = sub.add_parser("call")
    c.add_argument("--state", required=True)
    c.add_argument("tool")
    c.add_argument("args", nargs="?", default="{}")
    ns = ap.parse_args(argv)

    if ns.cmd == "init":
        snap = Path(ns.snapshot) if ns.snapshot else RESULTS / ns.task / "task_snapshot.json"
        task = Task(ns.task, snapshot=snap)
        out = Path(ns.out)
        out.mkdir(parents=True, exist_ok=True)
        env = AssemblyEnv(task, ns.tier, ns.condition, ns.rounds, workdir=out)
        st = {"task_id": ns.task, "out": str(out), "system": ns.system, "transport": "shell session",
              "created": time.time(), "snapshot": str(snap), "env": env.to_state()}
        _save(ns.state, env, st)
        print(json.dumps({"ok": True, "state": ns.state}))
        return

    if ns.cmd == "init-annotate":
        from ppbench.v2.annotate import create
        task = Task(ns.task, snapshot=RESULTS / ns.task / "task_snapshot.json")
        env, info = create(task, ns.source, Path(ns.out), ns.condition)
        st = {"task_id": ns.task, "out": str(ns.out), "system": ns.system, "transport": "shell session",
              "created": time.time(), "annotation": info, "env": env.to_state()}
        _save(ns.state, env, st)
        print(json.dumps({"ok": True, "state": ns.state, "parts": len(env.s["instances"]),
                          "generator_declared": info["generator_declared"]}))
        return

    env, st = _load(ns.state)
    if ns.cmd == "prompt" and env.s.get("mode") == "annotate":
        from ppbench.v2.annotate import prompt_text
        print(prompt_text(env, st))
        print("\nHOW TO CALL A TOOL FROM THE SHELL\n"
              f"  .venv_eval/bin/python -m ppbench.v2.session call --state {ns.state} <tool_name> '<JSON arguments>'\n"
              "Each call prints one JSON result. Image results are PNG file paths: open them to look at them.")
    elif ns.cmd == "prompt":
        cond = env.task.condition(env.s["condition_id"])
        from ppbench.v2.render import sheet_all
        sheets = [] if env.tier == "C" else [s for s in [sheet_all(env.task.id)] if s]
        print(system_prompt(env.tier))
        print("\nTOOLS\n" + tool_docs(env.tier))
        print("\n" + user_prompt(env.task, cond, env.tier, sheets))
        print("\nHOW TO CALL A TOOL FROM THE SHELL\n"
              f"  .venv_eval/bin/python -m ppbench.v2.session call --state {ns.state} <tool_name> '<JSON arguments>'\n"
              "Each call prints one JSON result. Image results are PNG file paths: open them to look at them.")
    elif ns.cmd == "call":
        try:
            args = json.loads(ns.args)
        except json.JSONDecodeError as e:
            res = {"ok": False, "error": f"arguments are not valid JSON: {e}"}
            env.s["counters"]["tool_calls"] += 1
            env.s["counters"]["errors"] += 1
        else:
            res = env.call(ns.tool, args)
        if ns.tool == "submit" and res.get("ok"):
            save_round_checkpoint(env, st["system"], st["transport"], res)
        _save(ns.state, env, st)
        print(json.dumps(res))
        if env.s["done"]:
            finish(env, st)
    elif ns.cmd == "finish":
        finish(env, st)
    elif ns.cmd == "status":
        print(json.dumps(env.t_get_scene()))


def finish(env, st):
    out = Path(st["out"])
    if env.s.get("mode") == "annotate":
        from ppbench.v2.annotate import export
        d = export(env, st)
    else:
        d = env.design(st["system"], {"transport": st["transport"]})
    d.save(out / "design")
    (out / "trace.json").write_text(json.dumps(env.s["trace"], indent=0))
    (out / "state.json").write_text(json.dumps(env.to_state()))
    print(json.dumps({"finished": True, "design": str(out / "design"), "parts": len(d.parts), "joints": len(d.joints)}),
          file=sys.stderr)


if __name__ == "__main__":
    main()
