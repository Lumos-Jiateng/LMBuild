"""Collect every design for a task, evaluate, render, and write the site data.

    .venv_eval/bin/python -m ppbench.v2.report collect  --task swivel_office_chair      # evaluate + render (cached)
    .venv_eval/bin/python -m ppbench.v2.report collect-core --tasks all --rounds           # core-set runs -> eval_core/ (cached)
    .venv_eval/bin/python -m ppbench.v2.report imagesim --task swivel_office_chair      # DINOv2 (crosses into .venv_imagegen)
    .venv_eval/bin/python -m ppbench.v2.report items    --task swivel_office_chair      # judge inputs per tier
    .venv_eval/bin/python -m ppbench.v2.report site     --task swivel_office_chair      # site/data.json

Where designs come from:
  results/v2/<task>/runs/v2.1/<system>__<tier>__<condition>__r<rounds>__s<seed>/design   tool arms
  results/v2/<task>/external/<system>_s<seed>/                                       external generators
  the task's reference object                                                         ceiling arm
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ppbench.v2 import evaluate as E
from ppbench.v2 import external, render
from ppbench.v2.design import Design
from ppbench.v2.task import CORE, RESULTS, ROOT, Task

PROTOCOL = "v2.2"   # override with --protocol for core or later harnesses

# the core set: many tasks, the three core run namespaces, one record per design under <task>/eval_core
CORE_NAMESPACES = ["core_v2.4_clean", "core_v2.4", "core_v2.3"]   # first namespace holding a run name wins
CORE_FILTER = "*__B__name_only+image__r3__s0"                      # the main setting
CORE_SNAPSHOT = "task_snapshot_core.json"
TOP20_SOURCE = RESULTS / "_core" / "experiments" / "sol_astra_tier_b_image_r3.json"

EXTERNAL = {   # system prefix -> (adapter, kwargs, family, input note)
    "brickgpt": ("ldr", {}, "brick LLM (text -> LDraw)", "caption 'An office chair.' (text only)"),
    "bricknet": ("ldr", {}, "brick LLM (text -> LDraw)", "caption 'An office chair.' (text only)"),
    "legoace": ("ldr", {}, "brick generator (text -> LDraw)", "caption 'An office chair.' (text only)"),
    "partcrafter": ("glb", {"up": "y"}, "part-decomposed 3D generation (image -> part meshes)", "condition image W2"),
    "partpacker": ("glb", {"up": "y"}, "part-decomposed 3D generation (image -> part meshes)", "condition image W2"),
    "cubepart": ("glb", {"up": "y", "roles_from_names": True}, "text -> shape -> schema-controlled parts", "prompt 'An office chair' + part schema"),
    "physx-anything": ("bundle", {}, "image -> articulated, physical asset (VLM + TRELLIS)", "condition image W2"),
    "infinite-mobility": ("bundle", {}, "procedural articulated generator (OfficeChairFactory)", "category only, random seed"),
    "particulate-cube3d": ("bundle", {}, "text -> shape (Cube3D) -> parts + joints (Particulate)", "prompt 'An office chair'"),
    "particulate-partcrafter": ("bundle", {}, "image -> mesh (PartCrafter) -> parts + joints (Particulate)", "condition image W2"),
    "particulate-partpacker": ("bundle", {}, "image -> mesh (PartPacker) -> parts + joints (Particulate)", "condition image W2"),
}


def _json(o):
    return json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else (float(x) if isinstance(x, np.floating) else str(x)))


def discover(task_id, protocol=PROTOCOL, snapshot="task_snapshot.json"):
    base = RESULTS / task_id
    out = [{"key": "reference", "kind": "reference", "system": "reference", "tier": "ref", "seed": 0}]
    for d in sorted((base / "runs" / protocol).glob("*")):
        if (d / "design" / "design.json").exists():
            m = re.match(r"(.+?)__([ABC])__(.+?)__r(\d+)__s(\d+)$", d.name)
            if not m:
                continue
            common = {"kind": "tool", "system": m.group(1), "tier": m.group(2), "condition": m.group(3),
                      "trajectory_rounds": int(m.group(4)), "seed": int(m.group(5)), "snapshot": snapshot}
            checkpoints = sorted((d / "rounds").glob("round_*"))
            if checkpoints:
                for checkpoint in checkpoints:
                    cm = re.fullmatch(r"round_(\d+)", checkpoint.name)
                    if cm and (checkpoint / "design" / "design.json").exists():
                        rnd = int(cm.group(1))
                        out.append({**common, "key": f"{d.name}__round{rnd}", "rounds": rnd,
                                    "checkpoint_round": rnd, "path": str(checkpoint / "design")})
            else:
                out.append({**common, "key": d.name, "rounds": int(m.group(4)), "path": str(d / "design")})
    for d in sorted((base / "external").glob("*_s[0-9]*")):
        m = re.match(r"(.+)_s(\d+)$", d.name)
        sysname = m.group(1)
        prefix = next((p for p in sorted(EXTERNAL, key=len, reverse=True) if sysname.startswith(p)), None)
        if prefix is None:
            continue
        adapter = EXTERNAL[prefix][0]
        if adapter == "ldr" and not (d / "model.ldr").exists():
            continue
        if adapter == "glb" and not (d / "parts").is_dir():
            continue
        if adapter == "bundle" and not (d / "bundle" / "bundle.json").exists():
            continue
        out.append({"key": d.name, "kind": "external", "system": sysname, "prefix": prefix, "tier": "external",
                    "seed": int(m.group(2)), "path": str(d)})
    # annotation arm: a generator's fixed geometry + declarations by an annotator agent (ppbench/v2/annotate.py)
    for d in sorted((base / "annotated").glob("*__*")):
        if not (d / "design" / "design.json").exists():
            continue
        m = re.match(r"(.+)_s(\d+)__(.+)$", d.name)
        if not m:
            continue
        out.append({"key": f"{m.group(1)}_s{m.group(2)}+{m.group(3)}", "kind": "annotated", "system": f"{m.group(1)}+{m.group(3)}",
                    "generator": m.group(1), "annotator": m.group(3), "tier": "external", "seed": int(m.group(2)),
                    "path": str(d / "design")})
    return out


def load_design(task, item):
    if item["kind"] == "reference":
        return E.reference_design(task)
    if item["kind"] in ("tool", "annotated"):
        return Design.load(item["path"])
    adapter, kw, family, inp = EXTERNAL[item["prefix"]]
    meta = {"family": family, "input": item.get("input", inp)}
    p = Path(item["path"])
    mf = p.with_name(p.stem + ".meta.json") if p.suffix == ".ldr" else p / "meta.json"   # core brick runs are flat files
    if mf.exists():
        meta["producer_meta"] = json.loads(mf.read_text())
    if adapter == "ldr":
        return external.ldr_design(p if p.suffix == ".ldr" else p / "model.ldr", item["system"], meta)
    if adapter == "glb":
        return external.part_glb_design(p / "parts", item["system"], meta=meta, **kw)
    return external.bundle_design(p / "bundle", item["system"], meta)


def _stamp(item):
    if item["kind"] == "reference":
        return 0.0
    p = Path(item["path"])
    if p.is_file():
        return p.stat().st_mtime
    return max(f.stat().st_mtime for f in p.rglob("*") if f.is_file())


def evaluate_one(args):
    task_id, item, snapshot = args
    os.environ.setdefault("TMPDIR", "/tmp")
    base = RESULTS / task_id
    (base / "eval").mkdir(parents=True, exist_ok=True)
    out = base / "eval" / f"{item['key']}.json"
    if out.exists() and json.loads(out.read_text()).get("stamp") == _stamp(item):
        return item["key"], "cached"
    task = Task(task_id, snapshot=base / snapshot)
    anchors = json.loads((base / "reference_eval.json").read_text())["anchors"]
    pop = E.load_population(task_id)
    t0 = time.time()
    try:
        d = load_design(task, item)
        rec = E.evaluate(d, task, anchors, pop)
        render_extras(base, base / "eval", item["key"], d)
        rec["item"] = item
        rec["stamp"] = _stamp(item)
        rec["eval_wall_s"] = round(time.time() - t0, 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json(rec))
        return item["key"], f"ok {rec['eval_wall_s']}s"
    except Exception as e:
        import traceback
        (base / "eval").mkdir(parents=True, exist_ok=True)
        (base / "eval" / f"{item['key']}.error.txt").write_text(traceback.format_exc())
        return item["key"], f"error {e!r}"


def render_extras(base, eval_dir, key, design):
    """Clay views for the judge and DINOv2, and a point cloud for the diversity metric (the page colours its own views)."""
    if not design.parts:
        return
    rdir = base / "renders"
    render.render_parts(design.parts, rdir / f"{key}_clay", views=("cond", "iso", "front", "side"), size=384,
                        samples=16, color_key=lambda p: "all")
    render.grid([str(rdir / f"{key}_clay_{v}.png") for v in ("cond", "iso", "front", "side")],
                rdir / f"{key}_claygrid.png", cols=2, cell=384)
    np.save(eval_dir / f"{key}_points.npy", E.sample_points(design.parts, 6000))


def ensure_reference(task_id, snapshot="task_snapshot.json"):
    """Build and cache the task-specific reference anchors required by every design."""
    base = RESULTS / task_id
    out = base / "reference_eval.json"
    if out.exists():
        return out
    task = Task(task_id, snapshot=base / snapshot)
    rec = E.evaluate(E.reference_design(task), task, {}, E.load_population(task_id))
    rec["anchors"] = E.anchors_from(rec)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(_json(rec))
    tmp.replace(out)
    return out


def collect(task_id, workers=8, only=None, protocol=PROTOCOL, snapshot="task_snapshot.json"):
    # evaluation is CPU-only and embarrassingly parallel; a floor of 32 processes (4 Blender threads each)
    # keeps it well inside the machine without starving the model episodes
    workers = max(workers, int(os.environ.get("PPBENCH_MIN_EVAL_WORKERS", "32")))
    os.environ.setdefault("RENDER_THREADS", "4")
    ensure_reference(task_id, snapshot)
    items = discover(task_id, protocol, snapshot)
    if only:
        items = [i for i in items if re.search(only, i["key"])]
    with ProcessPoolExecutor(workers) as ex:
        for key, status in ex.map(evaluate_one, [(task_id, i, snapshot) for i in items]):
            print(f"{key:70s} {status}", flush=True)


# ================================================================ core set (many tasks, core namespaces)

def core_task_ids(spec="all"):
    """--tasks: every core task, the paired top-20 of the sol/astra experiment, or a comma list."""
    ids = [t["task_id"] for t in json.loads(CORE.read_text())]
    if spec == "top20":
        exp = json.loads(TOP20_SOURCE.read_text())
        return [t for t in exp["top_20_tasks"] if t in ids]
    if spec != "all":
        return [t.strip() for t in spec.split(",") if t.strip()]
    return ids


def core_snapshot(task_id, name=CORE_SNAPSHOT):
    """The core snapshot when the task has one, else the original."""
    return name if (RESULTS / task_id / name).exists() else "task_snapshot.json"


def _matches(name, pattern):
    """--filter is a glob when it carries glob characters, else a plain substring."""
    return fnmatch.fnmatch(name, pattern) if any(c in pattern for c in "*?[") else pattern in name


def discover_core_externals(task_id, pattern=CORE_FILTER, snapshot=CORE_SNAPSHOT):
    """Domain-specific generators' core-set outputs (ppbench/v2/core_externals.py): external_core/<condition>/<system>_s<n>
    (a directory, or <system>_s<n>.ldr for the brick generators). Key: <system>__ext__<condition>__s<n>."""
    out = []
    for cdir in sorted((RESULTS / task_id / "external_core").glob("*")):
        if not cdir.is_dir():
            continue
        for d in sorted(cdir.glob("*_s[0-9]*")):
            m = re.fullmatch(r"(.+)_s(\d+)(\.ldr)?", d.name)
            if not m:
                continue
            sysname = m.group(1)
            prefix = next((p for p in sorted(EXTERNAL, key=len, reverse=True) if sysname.startswith(p)), None)
            if prefix is None:
                continue
            adapter = EXTERNAL[prefix][0]
            if adapter == "ldr" and not (m.group(3) or (d / "model.ldr").exists()):
                continue
            if adapter == "glb" and not (d / "parts").is_dir():
                continue
            if adapter == "bundle" and not (d / "bundle" / "bundle.json").exists():
                continue
            key = f"{sysname}__ext__{cdir.name}__s{m.group(2)}"
            if pattern != CORE_FILTER and not _matches(key, pattern):   # the default filter names the tool-run main setting
                continue
            out.append({"key": key, "kind": "external", "system": sysname, "prefix": prefix, "tier": "external",
                        "condition": cdir.name, "input": f"core task, condition {cdir.name}", "seed": int(m.group(2)),
                        "task": task_id, "snapshot": snapshot, "path": str(d)})
    return out


def discover_core(task_id, namespaces=CORE_NAMESPACES, pattern=CORE_FILTER, rounds=False, snapshot=CORE_SNAPSHOT,
                  externals=False):
    """Tool runs for one task across the core namespaces; a run name found earlier wins (clean arm first).
    externals=True: the domain-specific generators' outputs instead (discover_core_externals)."""
    if externals:
        return discover_core_externals(task_id, pattern, snapshot)
    base = RESULTS / task_id
    out, seen = [], set()
    for ns in namespaces:
        for d in sorted((base / "runs" / ns).glob("*")):
            m = re.match(r"(.+?)__([ABC])__(.+?)__r(\d+)__s(\d+)$", d.name)
            if not m or d.name in seen or not _matches(d.name, pattern) or not (d / "design" / "design.json").exists():
                continue
            seen.add(d.name)
            common = {"kind": "tool", "system": m.group(1), "tier": m.group(2), "condition": m.group(3),
                      "trajectory_rounds": int(m.group(4)), "seed": int(m.group(5)),
                      "namespace": ns, "task": task_id, "snapshot": snapshot}
            out.append({**common, "key": d.name, "rounds": int(m.group(4)), "path": str(d / "design")})
            if not rounds:
                continue
            for cp in sorted((d / "rounds").glob("round_*")):   # per-round checkpoints feed the R1/R2/R3 ablation
                cm = re.fullmatch(r"round_(\d+)", cp.name)
                if cm and (cp / "design" / "design.json").exists():
                    out.append({**common, "key": f"{d.name}__round{cm.group(1)}", "rounds": int(cm.group(1)),
                                "checkpoint_round": int(cm.group(1)), "path": str(cp / "design")})
    return out


def _cached_stamp(out: Path):
    """The stamp of an already written record, or None when it is missing or unreadable."""
    try:
        return json.loads(out.read_text()).get("stamp")
    except (OSError, ValueError):
        return None


def core_anchors(args):
    """Reference anchors for one task, cached beside the results; {} when the task has no reference object."""
    task_id, snapshot, spec = (args + ("v2",))[:3] if isinstance(args, tuple) else args
    out = RESULTS / task_id / ("reference_eval_v3.json" if spec == "v3" else "reference_eval_core.json")
    if out.exists():
        try:
            return task_id, json.loads(out.read_text()).get("anchors") or {}, ""
        except (OSError, ValueError):
            pass
    try:
        task = Task(task_id, snapshot=RESULTS / task_id / snapshot)
        if spec == "v3":
            from ppbench.v2 import metrics_v3 as M
            anchors = M.anchors_v3(task)
            # the reference measured by the same instrument: the ceiling check, kept next to the anchors
            rec = M.evaluate_v3(M.reference_design(task), task, anchors, is_reference=True)
            rec["anchors"] = anchors
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(_json(rec))
            tmp.replace(out)
            return task_id, anchors, ""
        rec = E.evaluate(E.reference_design(task), task, {}, E.load_population(task_id))
        rec["anchors"] = E.anchors_from(rec)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(_json(rec))
        tmp.replace(out)
        return task_id, rec["anchors"], ""
    except Exception as e:   # the BrickNet tasks have no reference GLB: evaluate without anchors rather than lose the task
        return task_id, {}, f"{type(e).__name__}: {e}"


def evaluate_core_one(args):
    """One record under <task>/eval_core; never raises, so one bad design cannot abort the batch."""
    task_id, item, snapshot, anchors, renders, force, spec = (args + ("v2",))[:7]
    os.environ.setdefault("TMPDIR", "/tmp")
    base = RESULTS / task_id
    edir = base / ("eval_v3" if spec == "v3" else "eval_core")
    out = edir / f"{item['key']}.json"
    if item.get("kind", "tool") == "tool" and not Path(item["path"], "design.json").exists():
        return task_id, item["key"], "skip no design.json", ""
    stamp = _stamp(item)
    if spec == "v3":
        # the metric version belongs in the cache key: a design that has not changed still needs a
        # new record when the spec that scored it has. Without this, a metric fix only lands where
        # someone remembers to pass --force.
        from ppbench.v2.metrics_v3 import SPEC_VERSION
        stamp = f"{stamp}|{SPEC_VERSION}"
    if not force and out.exists() and _cached_stamp(out) == stamp:
        return task_id, item["key"], "cached", ""
    t0 = time.time()
    try:
        task = Task(task_id, snapshot=base / snapshot)
        d = load_design(task, item)
        if spec == "v3":
            from ppbench.v2 import metrics_v3 as M
            rec = M.evaluate_v3(d, task, anchors)
        else:
            rec = E.evaluate(d, task, anchors, E.load_population(task_id))
        edir.mkdir(parents=True, exist_ok=True)
        if renders:
            render_extras(base, edir, item["key"], d)
        rec["item"] = item
        rec["stamp"] = stamp
        rec["eval_wall_s"] = round(time.time() - t0, 1)
        out.write_text(_json(rec))
        return task_id, item["key"], f"ok {rec['eval_wall_s']}s", ""
    except Exception as e:
        import traceback
        edir.mkdir(parents=True, exist_ok=True)
        (edir / f"{item['key']}.error.txt").write_text(traceback.format_exc())
        return task_id, item["key"], f"error {e!r}", f"{type(e).__name__}: {e}"


def _core_summary(tally, errors, no_runs):
    """Per task: written / cached / skipped / failed, then the most common error messages."""
    print("\n== collect-core summary")
    for task_id in sorted(tally):
        c = tally[task_id]
        print(f"{task_id:28s} written {c['ok']:4d}  cached {c['cached']:4d}  skipped {c['skip']:3d}  failed {c['error']:4d}")
    total = {k: sum(c[k] for c in tally.values()) for k in ("ok", "cached", "skip", "error")}
    print(f"{'TOTAL':28s} written {total['ok']:4d}  cached {total['cached']:4d}  skipped {total['skip']:3d}  failed {total['error']:4d}")
    if no_runs:
        print(f"no matching runs: {', '.join(sorted(no_runs))}")
    groups = {}
    for task_id, key, msg in errors:
        groups.setdefault(msg[:160], []).append(f"{task_id}/{key}")
    for msg, keys in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"  {len(keys):4d} x {msg}   e.g. {keys[0]}")


def collect_core(tasks="all", namespaces=CORE_NAMESPACES, pattern=CORE_FILTER, rounds=False, workers=24,
                 limit=None, only=None, force=False, renders=False, spec="v2", externals=False):
    """Evaluate every core-set design of every requested task into <task>/eval_core, one process pool for the lot."""
    os.environ.setdefault("RENDER_THREADS", "4")
    task_ids = core_task_ids(tasks)
    jobs, snaps, no_runs = [], {}, []
    for task_id in task_ids:
        snaps[task_id] = core_snapshot(task_id)
        items = discover_core(task_id, namespaces, pattern, rounds, snaps[task_id], externals)
        if only:
            items = [i for i in items if only in i["key"]]
        if not items:
            no_runs.append(task_id)
        jobs += [(task_id, i) for i in items]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} designs over {len({t for t, _ in jobs})} tasks (namespaces {','.join(namespaces)}, filter {pattern})", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        # anchors first: one reference evaluation per task, shared by every design of that task
        anchors = {}
        for task_id, a, err in ex.map(core_anchors, [(t, snaps[t], spec) for t in sorted({t for t, _ in jobs})]):
            anchors[task_id] = a
            if err:
                print(f"{task_id:28s} no anchors ({err})", flush=True)
        tally = {t: {"ok": 0, "cached": 0, "skip": 0, "error": 0} for t in sorted({t for t, _ in jobs})}
        errors = []
        args = [(t, i, snaps[t], anchors[t], renders, force, spec) for t, i in jobs]
        for task_id, key, status, err in ex.map(evaluate_core_one, args):
            tally[task_id][status.split()[0]] += 1
            if err:
                errors.append((task_id, key, err))
            print(f"{task_id:24s} {key:70s} {status}", flush=True)
    _core_summary(tally, errors, no_runs)


def imagesim(task_id, snapshot="task_snapshot.json"):
    base = RESULTS / task_id
    task = Task(task_id, snapshot=base / snapshot)
    cond = task.condition("name_only+image")["image"]["path"]
    items = {p.stem.replace("_clay_cond", ""): str(p) for p in (base / "renders").glob("*_clay_cond.png")}
    job = base / "eval" / "imagesim_job.json"
    job.write_text(json.dumps({"condition": cond, "items": items}))
    py = ROOT / ".venv_imagegen" / "bin" / "python"
    env = dict(os.environ, HF_HOME=os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")), CUDA_VISIBLE_DEVICES="")
    subprocess.run([str(py), "-m", "ppbench.v2.imagesim", str(job), str(base / "eval" / "imagesim.json")], check=True, cwd=ROOT, env=env)


def judge_items(task_id):
    base = RESULTS / task_id
    recs = [json.loads(p.read_text()) for p in (base / "eval").glob("*.json") if not p.name.startswith(("imagesim", "judge", "vlm"))]
    tiers = {}
    for r in recs:
        it = r.get("item") or {}
        grid = base / "renders" / f"{it.get('key')}_claygrid.png"
        if not grid.exists():
            continue
        tiers.setdefault(it["tier"], {})[it["key"]] = str(grid)
    ref = tiers.pop("ref", {})
    out = {}
    for t, items in tiers.items():
        items = dict(items, **ref)
        p = base / "eval" / f"judge_items_{t}.json"
        p.write_text(json.dumps(items, indent=1))
        out[t] = str(p)
    print(json.dumps(out, indent=1))


def site(task_id, snapshot="task_snapshot.json"):
    """One JSON with everything the page shows; images stay as paths, the page builder embeds them."""
    base = RESULTS / task_id
    task = Task(task_id, snapshot=base / snapshot)
    recs = {}
    for p in sorted((base / "eval").glob("*.json")):
        if p.name.startswith(("imagesim", "judge", "vlm")):
            continue
        r = json.loads(p.read_text())
        recs[r["item"]["key"]] = r
    sim = json.loads((base / "eval" / "imagesim.json").read_text())["cosine"] if (base / "eval" / "imagesim.json").exists() else {}
    judges = {}
    for p in (base / "eval").glob("judge_*.json"):
        if p.name.startswith("judge_items"):
            continue
        judges[p.stem] = json.loads(p.read_text())
    # diversity across seeds (3.3)
    from ppbench.v2.evaluate import fscore
    groups = {}
    for k, r in recs.items():
        it = r["item"]
        groups.setdefault((it["system"], it["tier"]), []).append(k)
    diversity = {}
    for (s, t), keys in groups.items():
        pts = {k: np.load(base / "eval" / f"{k}_points.npy") for k in keys if (base / "eval" / f"{k}_points.npy").exists()}
        if len(pts) < 2:
            continue
        ds = []
        ks = sorted(pts)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = pts[ks[i]], pts[ks[j]]
                a = a - np.r_[(a.min(0)[:2] + a.max(0)[:2]) / 2, a.min(0)[2]]
                b = b - np.r_[(b.min(0)[:2] + b.max(0)[:2]) / 2, b.min(0)[2]]
                ds.append(1 - fscore(a, b)["fscore@0.05"])
        diversity[f"{s}|{t}"] = {"n_seeds": len(ks), "mean_pairwise_1_minus_fscore@0.05": float(np.mean(ds))}
    # evidence-grounded VLM checks (ppbench.v2.vlm_checks): summary per design and judge, and the evidence each judge saw
    vlm = json.loads((base / "eval" / "vlm_summary.json").read_text()) if (base / "eval" / "vlm_summary.json").exists() else {}
    evidence = {}
    for ej in (base / "evidence").glob("*/evidence.json"):
        e = json.loads(ej.read_text())
        if e["key"] not in recs:
            continue
        evidence[e["key"]] = {"labels": [[r["label"], r["part_id"], r["role"], r["material"]] for r in e["labels"]],
                              "sheets": {n: e[n] for n in ("overview", "materials", "clay")},
                              "motion": [{k: m[k] for k in ("joint", "claims", "image", "type", "axis_words", "parent_label",
                                                            "moving_labels", "limits", "zoomed") if k in m} for m in e["motion"]]}
    data = {"task": {"id": task.id, "name": task.name, "condition": task.condition("name_only+image"),
                     "required_parts": task.required_parts, "attributes": task.attributes, "subsystems": task.subsystems,
                     "kinematics": task.kinematics, "source": task.source},
            "records": recs, "imagesim": sim, "judges": judges, "vlm": vlm, "evidence": evidence, "diversity": diversity,
            "population": (E.load_population(task_id) or {}).get("aggregate"),
            "reference_anchors": json.loads((base / "reference_eval.json").read_text())["anchors"]}
    (base / "site").mkdir(exist_ok=True)
    (base / "site" / "data.json").write_text(_json(data))
    print(f"{len(recs)} records -> {base / 'site' / 'data.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "collect-core", "imagesim", "items", "site"])
    ap.add_argument("--task", default="swivel_office_chair")
    ap.add_argument("--workers", type=int, help="processes (default 8 for collect, 24 for collect-core)")
    ap.add_argument("--only", help="collect: regex on keys; collect-core: substring on keys")
    ap.add_argument("--protocol", default=PROTOCOL, help="run namespace under results/v2/<task>/runs")
    ap.add_argument("--snapshot", default="task_snapshot.json", help="task snapshot filename under results/v2/<task>")
    # collect-core
    ap.add_argument("--tasks", default="all", help="all | top20 | comma-separated task ids")
    ap.add_argument("--namespaces", default=",".join(CORE_NAMESPACES), help="run namespaces, earliest wins")
    ap.add_argument("--filter", default=CORE_FILTER, help="glob or substring on run directory names")
    ap.add_argument("--rounds", action="store_true", help="also evaluate the per-round checkpoints (<run>__round<n>)")
    ap.add_argument("--limit", type=int, help="evaluate at most this many designs (smoke test)")
    ap.add_argument("--force", action="store_true", help="re-evaluate even when the stamp is unchanged")
    ap.add_argument("--renders", action="store_true", help="also write clay views and point clouds (slow)")
    ap.add_argument("--externals", action="store_true", help="collect-core: evaluate the domain-specific generators' "
                    "external_core/ outputs instead of the tool runs")
    ap.add_argument("--spec", default="v2", choices=["v2", "v3"], help="v2: evaluate.py into eval_core; v3: metrics_v3.py into eval_v3")
    ns = ap.parse_args()
    {"collect": lambda: collect(ns.task, ns.workers or 8, ns.only, ns.protocol, ns.snapshot),
     "collect-core": lambda: collect_core(ns.tasks, [n for n in ns.namespaces.split(",") if n], ns.filter, ns.rounds,
                                          ns.workers or 24, ns.limit, ns.only, ns.force, ns.renders, ns.spec,
                                          ns.externals),
     "imagesim": lambda: imagesim(ns.task, ns.snapshot),
     "items": lambda: judge_items(ns.task), "site": lambda: site(ns.task, ns.snapshot)}[ns.cmd]()


if __name__ == "__main__":
    main()
