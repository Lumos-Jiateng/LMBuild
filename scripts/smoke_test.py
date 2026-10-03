"""Check an install: every task loads, and one scripted episode runs through the shell session and renders.

    PYTHONPATH=. .venv_eval/bin/python scripts/smoke_test.py [--no-render] [--tasks a,b]

1. For each of the 50 tasks: the frozen task snapshot, the condition image, three pool meshes and the reference
   (GLB for mesh references, the LDraw file through the LDraw inset meshes for brick references).
2. A 6-part desk on office_desk, Tier B, one round, driven through `python -m ppbench.v2.session` exactly as an agent
   would (create_part, place_part, add_joint, set_assembly_sequence, check, render, submit), written to a temp dir.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppbench.v2.task import CORE, RESULTS, Task  # noqa: E402


def check_tasks(ids):
    from ppbench.v2 import spec_v33
    bad = []
    for tid in ids:
        t0 = time.time()
        try:
            t = Task(tid, snapshot=RESULTS / tid / "task_snapshot_core.json")
            img = t.condition("name_only+image")["image"]
            assert img and Path(img["path"]).exists(), f"condition image missing: {img}"
            pool = t.pool
            for p in pool[:: max(1, len(pool) // 3)][:3]:
                v, f = t.pool_mesh(p["pool_part_id"])
                assert len(v) and len(f)
            if t.raw["reference"].get("reference_glb"):
                n = len(t.reference()["parts"])
            else:
                n = len((spec_v33._ldraw_reference(t) or {}).get("parts", []))
            assert n > 0, "reference has no parts"
            print(f"  ok  {tid:26s} pool {len(pool):4d}  reference parts {n:4d}  {time.time() - t0:5.1f}s", flush=True)
        except Exception as e:   # report every task, then fail
            bad.append(tid)
            print(f"  FAIL {tid}: {type(e).__name__}: {e}", flush=True)
    return bad


def episode(render=True):
    work = Path(tempfile.mkdtemp(prefix="ppbench_smoke_"))
    state, out = work / "S.json", work / "run"
    py = [sys.executable, "-m", "ppbench.v2.session"]

    def run(*args):
        r = subprocess.run(py + list(args), cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT),
                           **{k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}})
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-2000:])
        return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""

    def call(tool, args):
        res = json.loads(run("call", "--state", str(state), tool, json.dumps(args)))
        if not res.get("ok"):
            raise RuntimeError(f"{tool}: {res}")
        return res

    run("init", "--state", str(state), "--task", "office_desk", "--tier", "B", "--condition", "name_only+image",
        "--rounds", "1", "--out", str(out), "--system", "smoke-test",
        "--snapshot", str(RESULTS / "office_desk" / "task_snapshot_core.json"))
    wood = "solid_wood_hardwood"
    call("create_part", {"name": "top", "program": {"nodes": [{"id": "a", "op": "box", "size": [1.2, 0.6, 0.03]}], "output": "a"}})
    call("create_part", {"name": "leg", "program": {"nodes": [{"id": "a", "op": "box", "size": [0.05, 0.05, 0.72]}], "output": "a"}})
    call("place_part", {"part_id": "top", "instance_id": "top", "position": [0, 0, 0.735], "role": "desktop", "material": wood})
    for i, (x, y) in enumerate([(-0.55, -0.25), (0.55, -0.25), (-0.55, 0.25), (0.55, 0.25)], 1):
        call("place_part", {"part_id": "leg", "instance_id": f"leg{i}", "position": [x, y, 0.36], "role": "leg", "material": wood})
    call("place_part", {"part_id": "P001", "instance_id": "drawer", "position": [0.2, 0, 0.674], "role": "drawer", "material": wood})
    call("add_joint", {"joint_id": "drawer_slide", "type": "prismatic", "parent": "top", "child": "drawer",
                       "axis": [0, -1, 0], "origin": [0.2, 0.0, 0.72], "limits": [0, 0.3]})
    call("set_assembly_sequence", {"steps": [{"part": p} for p in ("leg1", "leg2", "leg3", "leg4", "top", "drawer")]})
    chk = call("check", {})
    assert chk["connected_groups"] == 1 and not chk["collisions"], chk
    if render:
        imgs = call("render", {"views": ["iso"]})["images"]
        assert all(Path(p).exists() for p in imgs), imgs
        print(f"  render -> {imgs[0]}")
    call("submit", {})
    d = json.loads((out / "design" / "design.json").read_text())
    print(f"  episode ok: {len(d.get('parts', []))} parts, {len(d.get('joints', []))} joint(s) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--tasks", default="all")
    ns = ap.parse_args()
    ids = [t["task_id"] for t in json.loads(CORE.read_text())] if ns.tasks == "all" else ns.tasks.split(",")
    print(f"1. {len(ids)} tasks")
    bad = check_tasks(ids)
    print("2. scripted episode (office_desk, Tier B, 1 round)")
    episode(render=not ns.no_render)
    if bad:
        sys.exit(f"FAILED tasks: {bad}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
