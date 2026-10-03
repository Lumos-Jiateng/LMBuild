"""One episode driven by the Codex CLI (`codex exec`), as the paper's GPT-6 Astra / GPT-5.6 Sol arms were.

The agent works only through the shell session (`python -m ppbench.v2.session`), from its own working directory
outside the repository; the prompt below is the one those runs used, byte for byte.

    python scripts/run_codex_episode.py --model gpt-6-astra --task office_desk [--tier B] [--condition name_only+image] [--rounds 3]

Output: results/v2/<task>/runs/core_v2.4/<model>__<tier>__<condition>__r<rounds>__s0/ (the session exports the design and
a checkpoint per round there); the Codex JSON event log goes next to it as codex_attempt<N>.jsonl. An interrupted
episode is continued from its saved session state (up to 3 attempts), never restarted.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v2"
PY = ROOT / ".venv_eval" / "bin" / "python"


def progress(state: Path):
    try:
        env = json.loads(state.read_text())["env"]
        return env.get("done"), env.get("counters", {}).get("tool_calls", 0)
    except (OSError, KeyError, ValueError):
        return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--tier", default="B", choices=["A", "B", "C"])
    ap.add_argument("--condition", default="name_only+image")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iso-root", default=str(Path.home() / ".cache" / "ppbench_codex_iso"))
    ns = ap.parse_args()

    d = RESULTS / ns.task / "runs" / "core_v2.4" / f"{ns.model}__{ns.tier}__{ns.condition}__r{ns.rounds}__s0"
    d.mkdir(parents=True, exist_ok=True)
    state = d / "session_state.json"
    if not state.exists():
        subprocess.run([str(PY), "-m", "ppbench.v2.session", "init", "--state", str(state), "--task", ns.task,
                        "--tier", ns.tier, "--condition", ns.condition, "--rounds", str(ns.rounds), "--out", str(d),
                        "--system", ns.model, "--snapshot", str(RESULTS / ns.task / "task_snapshot_core.json")],
                       cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    iso = Path(ns.iso_root) / d.relative_to(RESULTS).as_posix().replace("/", "__")
    iso.mkdir(parents=True, exist_ok=True)
    command = f"{PY} -m ppbench.v2.session"
    prompt = (
        "You are the model being evaluated in a 3D assembly benchmark. Work only through the "
        "session command below. Do not edit, create, or delete files directly. Do not inspect the "
        "reference assembly, evaluator data, other results, or repository code. You may read only "
        "the condition image, catalog sheets, and renders named by your session.\n\n"
        f"Your session state file is {state}.\n"
        f"Run `{command} prompt --state {state}` first and read its full output. "
        "Open the image files it names. Build the object using only session tool calls such as "
        f"`{command} call --state {state} <tool_name> '<JSON arguments>'`. "
        "Each successful submit saves a round and returns feedback. Continue revising the same "
        "design until a submit returns `episode_done: true`. If the saved session is already "
        "partly complete, call get_scene and continue from that state; never start over. "
        "Inspect your renders and correct errors before submitting. "
        "At the end, report the final part and joint counts."
    )
    for attempt in range(1, 4):
        done, before = progress(state)
        if done:
            break
        with open(d / f"codex_attempt{attempt}.jsonl", "w") as log:
            rc = subprocess.run(["codex", "exec", "-m", ns.model, "-C", str(iso), "--add-dir", str(ROOT),
                                 "--skip-git-repo-check", "-s", "workspace-write", "--json", prompt],
                                cwd=iso, stdout=log, stderr=subprocess.STDOUT, timeout=3600,
                                env={**os.environ, "PYTHONPATH": str(ROOT), "RENDER_THREADS": "4"}).returncode
        done, after = progress(state)
        if done or (rc and after == before):
            break
    if not progress(state)[0]:     # export whatever was built, as the Claude driver does
        subprocess.run([str(PY), "-m", "ppbench.v2.session", "finish", "--state", str(state)], cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    print(json.dumps({"run": str(d), "done": progress(state)[0], "tool_calls": progress(state)[1]}))


if __name__ == "__main__":
    sys.exit(main())
