"""External generators over the core set (2026-09-14), one seed, outputs in results/v2/<task>/external_core/.

What each generator can consume decides which of the six conditions it runs under:
  image generators (PartCrafter np15 and np8, PartPacker): the three "+image" conditions of a task share one wild image,
      so they run once per task -> external_core/image/<system>_s0/ (used for all three image conditions);
  text generators (Cube3D + CubePart, BrickGPT, LegoACE): once per text condition (name_only, attributes, functional)
      with that condition's text -> external_core/<condition>/<system>_s0/; CubePart's part schema is the task's
      required part names (claims P), one entry each;
  Particulate: on every generated whole mesh (PartCrafter/PartPacker merged.glb, Cube3D shape.glb), front "auto",
      then the CPU bundle converter -> next to its input as particulate-<source>_s0/.
Every run records its inputs in meta.json (written by the runners); a run whose meta.json already exists is skipped.

    .venv_eval/bin/python -m ppbench.v2.core_externals image|text|particulate      (CUDA_VISIBLE_DEVICES set by the caller)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ppbench.v2.task import RESULTS

ROOT = Path(__file__).resolve().parents[2]
from ppbench.v2.task import CORE
TP = ROOT / "third_party"
BASE_PY = sys.executable
TEXT_CONDITIONS = ["name_only", "attributes", "functional"]
# main experiment first (2026-09-15): text-only generators get the name_only text (the text part of the main text+image
# condition); the attributes / functional texts are ablations for later (CORE_TEXT_CONDITIONS=name_only,attributes,functional)
RUN_TEXT_CONDITIONS = os.environ.get("CORE_TEXT_CONDITIONS", "name_only").split(",")
LOG = RESULTS / "_core" / "logs" / "externals.log"
CLAIMS = RESULTS / "_core" / "claims"


def tasks():
    """The core set, or the subset named in CORE_EXT_TASKS (comma-separated), so a long generator can be split
    across cards without two cards writing the same task."""
    ts = json.loads(CORE.read_text())
    only = [x for x in os.environ.get("CORE_EXT_TASKS", "").split(",") if x]
    return [t for t in ts if t["task_id"] in only] if only else ts


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(f"[{time.strftime('%F %T')}] {msg}\n")


def env():
    gpu = os.environ.get("GPU_ID", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    return dict(os.environ, HF_HOME=os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")), HF_HUB_OFFLINE="1",
                TMPDIR=os.environ.get("TMPDIR", "/tmp"), U2NET_HOME=os.path.expanduser(os.environ.get("U2NET_HOME", "~/.u2net")), PYTHONUNBUFFERED="1",
                CUDA_VISIBLE_DEVICES=gpu), gpu


def run(name, cmd, logfile):
    e, _ = env()
    Path(e["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(logfile, "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=e).returncode
    log(f"{name} exit={rc} {time.time() - t0:.0f}s")
    return rc


def image_jobs():
    for t in tasks():
        ims = {w["id"]: w["path"] for w in t["images"]["in_the_wild"]}
        img = ims[next(c["image"] for c in t["conditions"] if c.get("image"))]
        out = RESULTS / t["task_id"] / "external_core" / "image"
        logs = out / "logs"
        for np_ in (15, 8):
            if not (out / f"partcrafter_np{np_}_s0" / "meta.json").exists():
                run(f"{t['task_id']} partcrafter_np{np_}",
                    [str(TP / "partcrafter/.venv/bin/python"), "ppbench/baselines/run_partcrafter.py", "--image", str(ROOT / img),
                     "--out", str(out), "--num_parts", str(np_), "--seeds", "0", "--num_parts_source", f"fixed count ({np_}), same as the chair demo"],
                    logs / f"partcrafter_np{np_}.log")
        if not (out / "partpacker_s0" / "meta.json").exists():
            run(f"{t['task_id']} partpacker", [str(TP / "partpacker/.venv/bin/python"), "ppbench/baselines/run_partpacker.py",
                                               "--image", str(ROOT / img), "--out", str(out), "--seeds", "0"], logs / "partpacker.log")


def text_jobs():
    _, gpu = env()
    # CORE_EXT_LOCK (2026-09-21): a stream's own lock, so several text streams can share one card (the lock only keeps
    # two CubePart GPU stages of the SAME stream from overlapping); CORE_TEXT_SYSTEMS picks the generators of a stream
    lock = os.environ.get("CORE_EXT_LOCK", f"/tmp/ppbench_core_ext_gpu{gpu}.lock")
    systems = os.environ.get("CORE_TEXT_SYSTEMS", "cubepart,brickgpt,legoace").split(",")
    for t in tasks():
        schema = ",".join(p["part"] for p in t["required_parts"])
        for cond in RUN_TEXT_CONDITIONS:
            text = next(c["text"] for c in t["conditions"] if c["condition_id"] == cond)
            out = RESULTS / t["task_id"] / "external_core" / cond
            logs = out / "logs"
            cp = out / "cubepart_core_s0"
            # only a stream that runs CubePart may judge its folder: another stream sees an in-progress run (meta.json
            # written by the shape stage, parts/ not yet) and would move it away mid-run (2026-09-22, two runs lost)
            if "cubepart" in systems and (cp / "meta.json").exists() and not (cp / "parts").is_dir():
                # the part stage failed after the shape stage wrote meta.json (3 runs on 2026-09-17/18 ran out of memory
                # beside another process); kept aside, never deleted
                aside = out / f"cubepart_core_s0.failed-{time.strftime('%Y%m%d-%H%M%S')}"
                cp.rename(aside)
                log(f"{t['task_id']} {cond} cubepart: failed run moved to {aside.name}")
            if "cubepart" in systems and not (cp / "meta.json").exists():
                run(f"{t['task_id']} {cond} cubepart", [str(TP / "cube/.venv/bin/python"), "ppbench/baselines/run_cubepart.py",
                                                         "--prompt", text, "--parts", schema, "--out", str(out), "--seeds", "0",
                                                         "--setting", "core", "--gpu", gpu, "--lock", lock], logs / "cubepart.log")
            if "bricknet" in systems and not (out / "bricknet-14b_s0.meta.json").exists():
                # BrickNet-14B (Qwen3-14B + PT + SFT LoRA), one sample at seed 0 like every other generator
                run(f"{t['task_id']} {cond} bricknet", [BASE_PY, "ppbench/baselines/run_bricknet.py", "--caption", text,
                                                         "--out", str(out), "--size", "14b", "--n", "1", "--seed", "0", "--gpu", gpu],
                    logs / "bricknet.log")
            for sysname, script in (("brickgpt", "run_brickgpt.py"), ("legoace", "run_legoace.py")):
                if sysname not in systems:
                    continue
                # the brick runners write flat files (<out>/<system>_s0.meta.json); until 2026-09-18 only the directory
                # form was checked, so every ext_text pass re-generated (and overwrote) all 50 BrickGPT/LegoACE outputs
                if not (out / f"{sysname}_s0.meta.json").exists() and not (out / f"{sysname}_s0" / "meta.json").exists():
                    run(f"{t['task_id']} {cond} {sysname}", [BASE_PY, f"ppbench/baselines/{script}", "--caption", text,
                                                              "--out", str(out), "--seeds", "0", "--gpu", gpu], logs / f"{sysname}.log")


def particulate_jobs():
    # wait for the generators that produce the meshes (they may be running on other cards)
    while not all((CLAIMS / c / "done").exists() for c in ("ext_image", "ext_text")):
        time.sleep(300)
    py = str(TP / "particulate/.venv/bin/python")
    for t in tasks():
        base = RESULTS / t["task_id"] / "external_core"
        inputs = [(base / "image", f"particulate-partcrafter", base / "image" / "partcrafter_np15_s0" / "merged.glb"),
                  (base / "image", f"particulate-partcrafter-np8", base / "image" / "partcrafter_np8_s0" / "merged.glb"),
                  (base / "image", f"particulate-partpacker", base / "image" / "partpacker_s0" / "merged.glb")]
        inputs += [(base / c, "particulate-cube3d", base / c / "cubepart_core_s0" / "shape.glb") for c in RUN_TEXT_CONDITIONS]
        for out, sysname, mesh in inputs:
            run_dir = out / f"{sysname}_s0"
            if (run_dir / "bundle" / "bundle.json").exists() or not mesh.exists():
                continue
            rc = run(f"{t['task_id']} {out.name} {sysname}", [py, "ppbench/baselines/run_particulate.py", "--input", str(mesh), "--system", sysname,
                                                              "--seed", "0", "--out", str(out), "--up_dir", "Y", "--front=auto"],
                     out / "logs" / f"{sysname}.log")
            if rc == 0:
                run(f"{t['task_id']} {out.name} {sysname} bundle", [py, "ppbench/baselines/to_bundle_particulate.py", str(run_dir)],
                    out / "logs" / f"{sysname}_bundle.log")


def physx_jobs():
    """PhysX-Anything last in the queue (the user's decision, 2026-09-14): bf16 VLM on a whole card instead of the chair
    demo's int8 on a shared one, once per task (the three image conditions share one image), then the CPU converter."""
    _, gpu = env()
    for t in tasks():
        ims = {w["id"]: w["path"] for w in t["images"]["in_the_wild"]}
        img = ims[next(c["image"] for c in t["conditions"] if c.get("image"))]
        out = RESULTS / t["task_id"] / "external_core" / "image"
        run_dir = out / "physx-anything_s0"
        if (run_dir / "bundle" / "bundle.json").exists():
            continue
        if (run_dir / "meta.json").exists() and not (run_dir / "basic_info.json").exists():
            # A generation that ended without basic_info.json failed (36 of the 2026-09-16 runs on the run host GPU 0 hit
            # "Cuda error: 209" in nvdiffrast's fineRasterKernel, 4 ran out of memory). Kept aside, never deleted.
            aside = out / f"physx-anything_s0.failed-{time.strftime('%Y%m%d-%H%M%S')}"
            run_dir.rename(aside)
            log(f"{t['task_id']} physx-anything: failed run moved to {aside.name}")
        if not (run_dir / "meta.json").exists():
            run(f"{t['task_id']} physx-anything", [sys.executable, "ppbench/baselines/run_physx_anything.py", "--image", str(ROOT / img),
                                                   "--out", str(out), "--seeds", "0", "--gpu", gpu, "--vlm-precision", "bf16",
                                                   "--work", str(TP / "physx_anything" / "work" / f"core_{t['task_id']}")],
                out / "logs" / "physx-anything.log")
        if (run_dir / "meta.json").exists():
            run(f"{t['task_id']} physx-anything bundle", [sys.executable, "ppbench/baselines/to_bundle_physx_anything.py", str(run_dir)],
                out / "logs" / "physx-anything_bundle.log")


# systems the annotation arm completes (brick generators are left out: their parts are standard bricks, not functional parts)
ANNOTATABLE = [  # (system, report.EXTERNAL prefix, adapter check)
    ("partcrafter_np15", "partcrafter", "parts"), ("partcrafter_np8", "partcrafter", "parts"), ("partpacker", "partpacker", "parts"),
    ("cubepart_core", "cubepart", "parts"), ("particulate-partcrafter", "particulate-partcrafter", "bundle"),
    ("particulate-partcrafter-np8", "particulate-partcrafter", "bundle"), ("particulate-partpacker", "particulate-partpacker", "bundle"),
    ("particulate-cube3d", "particulate-cube3d", "bundle"), ("physx-anything", "physx-anything", "bundle"),
]


def annotation_items(task_id):
    """[(condition_id, item)] for every generated core-set external design of a task that the annotator can complete.
    Image-generator designs carry the image condition name_only+image (all three image conditions share the image);
    text-generator designs carry their own text condition."""
    base = RESULTS / task_id / "external_core"
    out = []
    for folder in ["image"] + TEXT_CONDITIONS:
        cond = "name_only+image" if folder == "image" else folder
        for sysname, prefix, kind in ANNOTATABLE:
            d = base / folder / f"{sysname}_s0"
            ok = (d / "parts").is_dir() and any((d / "parts").glob("*.glb")) if kind == "parts" else (d / "bundle" / "bundle.json").exists()
            if ok:
                out.append((cond, {"key": f"{folder}/{sysname}_s0", "kind": "external", "system": sysname, "prefix": prefix,
                                   "tier": "external", "seed": 0, "path": str(d), "condition": cond}))
    return out


if __name__ == "__main__":
    {"image": image_jobs, "text": text_jobs, "particulate": particulate_jobs, "physx": physx_jobs}[sys.argv[1]]()
