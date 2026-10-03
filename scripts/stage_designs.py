"""Stage the shipped designs (designs/, from LMBuild-designs.tar) where the evaluator looks for them.

designs/<task>/ mirrors results/v2/<task>/: runs/<namespace>/<run>/design (+ rounds/round_N/design) for the agents and
external_core/<condition>/<system>_s<n> (a directory, or <system>_s<n>.ldr + .meta.json) for the generators. This script
links each shipped run / generator output into results/v2/<task>/ with relative symlinks, so that

    python scripts/stage_designs.py [--tasks all|t1,t2] [--systems s1,s2] [--unstage]
    bash scripts/evaluate.sh <tasks>

re-scores exactly the designs behind the paper's tables (report collect-core discovers them as it discovers new runs).
Nothing in designs/ is modified; an existing real file or directory at a target is never replaced. --unstage removes
only the symlinks this script created (those that point into designs/).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = ROOT / "designs"
RESULTS = Path(os.environ.get("PPBENCH_RESULTS", ROOT / "results" / "v2"))


def entries(index, tasks, systems):
    seen = set()
    for e in index["designs"]:
        if tasks and e["task"] not in tasks or systems and e["system"] not in systems:
            continue
        rel = Path(e["path"]).relative_to(Path("designs") / e["task"])     # runs/<ns>/<run>/... or external_core/<c>/<x>
        parts = rel.parts
        if parts[0] == "runs":
            unit = Path(*parts[:3])                                          # runs/<ns>/<run>
            units = [unit]
        else:
            unit = Path(*parts[:3])                                          # external_core/<cond>/<system>_s<n>[.ldr]
            units = [unit]
            if unit.suffix == ".ldr":
                units.append(unit.with_name(unit.stem + ".meta.json"))
        for u in units:
            if (e["task"], u) not in seen:
                seen.add((e["task"], u))
                yield e["task"], u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--systems", default="")
    ap.add_argument("--unstage", action="store_true")
    ns = ap.parse_args()
    index = json.loads((DESIGNS / "index.json").read_text())
    tasks = None if ns.tasks == "all" else set(ns.tasks.split(","))
    systems = set(x for x in ns.systems.split(",") if x)
    made = kept = removed = absent = 0
    for task, unit in entries(index, tasks, systems):
        src = DESIGNS / task / unit
        dst = RESULTS / task / unit
        if ns.unstage:
            if dst.is_symlink() and str(DESIGNS.resolve()) in str(Path(os.path.realpath(dst))):
                dst.unlink()
                removed += 1
            continue
        if not src.exists():
            absent += 1
            continue
        if dst.exists() or dst.is_symlink():
            kept += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(os.path.relpath(src, dst.parent))
        made += 1
    if ns.unstage:
        print(f"removed {removed} symlinks")
    else:
        print(f"staged {made} links into {RESULTS} ({kept} already present, {absent} not shipped)")


if __name__ == "__main__":
    main()
