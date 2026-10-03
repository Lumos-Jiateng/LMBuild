"""Results board: every trajectory, checkpoint and evaluation record under results/v2, gathered for one page.

The board never computes a score. It reads what the pipeline already wrote -- run directories under
results/v2/<task>/runs/core_*/, their round checkpoints, claude_run.json, and evaluator outputs in
results/v2/<task>/eval/ (dims, attribute proxies, DINOv2 image similarity, VLM checks) -- so a record
appears on the page as soon as its file lands.

    .venv_eval/bin/python -m ppbench.v2.dashboard serve [--port 8770] [--tasks top20|all|a,b]   # live, rescans on reload
    .venv_eval/bin/python -m ppbench.v2.dashboard collect                                      # -> results/v2/_dashboard/data.json
    .venv_eval/bin/python -m ppbench.v2.dashboard snapshot                                     # -> results/v2/_dashboard/board.html
    .venv_eval/bin/python -m ppbench.v2.dashboard thumbs [--limit N]                           # renders checkpoints with no review render

Over ssh: `ssh -L 8770:localhost:8770 <host>`, then open http://localhost:8770.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ppbench.v2.claude_batch import _attempt_cost
from ppbench.v2.core_batch import CORE, _ldraw_tasks
from ppbench.v2.task import RESULTS

ROOT = Path(__file__).resolve().parents[2]
BOARD = RESULTS / "_dashboard"
TEMPLATE = Path(__file__).with_name("dashboard_template.html")
TOP20_SOURCE = RESULTS / "_core" / "experiments" / "sol_astra_tier_b_image_r3.json"
SPEC = ROOT / "docs" / "evaluation" / "v2_evaluator_spec.json"
QUEUE = ROOT / "scripts" / "core_models.txt"
VENDOR = ROOT / "reference_data" / "site" / "vendor"      # three.js + OrbitControls, served locally (no CDN)
PROGRESS = RESULTS / "_core" / "progress"
MAIN_SETTING = "B|name_only+image|3"
# one (task, system, setting) can exist in several namespaces: the clean Codex arm wins, then v2.4 checkpoints, then legacy v2.3
NS_ORDER = ["core_v2.4_clean", "core_v2.4", "core_v2.4_ablation_r1", "core_v2.3"]
RUN_RE = re.compile(r"(.+?)__([ABC])__(.+?)__r(\d+)__s(\d+)$")
ACTIVE_S = 30 * 60                     # a run touched within this window counts as running
STATUS_RANK = {"done": 0, "ended": 1, "running": 2, "partial": 3, "interrupted": 4, "stalled": 5, "invalid": 6}
FAMILY_ORDER = ["Claude", "GPT (Codex)", "Open model", "External generator"]
THUMB_W_SNAPSHOT = 240

_cache: dict[str, tuple] = {}
_json_cache: dict[str, tuple] = {}


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _read_cached(p: Path):
    """JSON files that rarely change (eval records, VLM summaries), keyed on mtime."""
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    hit = _json_cache.get(str(p))
    if hit and hit[0] == mt:
        return hit[1]
    val = _read(p)
    _json_cache[str(p)] = (mt, val)
    return val


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _listdir(p: Path) -> set[str]:
    try:
        return set(os.listdir(p))
    except OSError:
        return set()


def _rel(p) -> str | None:
    if not p:
        return None
    p = os.path.normpath(str(p))
    root = str(ROOT)
    return os.path.relpath(p, root) if p.startswith(root + os.sep) else p


def family(system: str) -> str:
    if system.startswith("claude-"):
        return "Claude"
    if system.startswith("gpt-") and not system.startswith("gpt-oss"):
        return "GPT (Codex)"
    return "Open model"


def ns_rank(ns: str) -> int:
    return NS_ORDER.index(ns) if ns in NS_ORDER else len(NS_ORDER)


# ------------------------------------------------------------------ catalog

def metric_catalog():
    """Columns the page can show. level 'round' reads a checkpoint, 'run' reads the whole trajectory."""
    m = [
        ("parts", "Parts", "Harness", "round", "count", "part instances in the checkpoint design"),
        ("joints", "Joints", "Harness", "round", "count", "declared joints in the checkpoint design"),
        ("created", "Created parts", "Harness", "round", "count", "instances whose geometry the model created"),
        ("materials", "Materials", "Harness", "round", "count", "distinct materials assigned"),
        ("tool_calls", "Tool calls", "Harness", "round", "count", "cumulative tool calls up to this checkpoint"),
        ("errors", "Tool errors", "Harness", "round", "count", "cumulative tool errors up to this checkpoint"),
        ("checks", "Check calls", "Harness", "round", "count", "cumulative check calls up to this checkpoint"),
        ("renders", "Render calls", "Harness", "round", "count", "cumulative render calls up to this checkpoint"),
    ]
    spec = _read(SPEC) or []
    for d in spec:
        k = d.get("key", "")
        if re.fullmatch(r"\d\.\d", k) and k != "4.3":          # 4.3 keeps presence/placement/connection apart: no score
            label = "Novelty" if k == "3.3" else d.get("name", k)
            desc = "1 - F-score@5cm to the reference (3.3 reports novelty, not a merged score)" if k == "3.3" else d.get("question", "")
            m.append((k, f"{k} {label}", "Evaluator", "round", "score", desc))
    m += [
        ("overall", "Overall", "Evaluator", "round", "score", "v3 headline: mean of the dimension scores that exist for this design"),
        ("attr", "A1-A4 attributes", "Evaluator", "round", "score", "mean of the attribute proxy scores (v2 only; v3 drops the chair-specific proxies)"),
        ("buildable", "Buildable", "Evaluator", "round", "score", "headline: 1 when no critical failure fired"),
        ("img", "Image sim.", "Evaluator", "round", "score", "DINOv2 cosine between the design render and the condition image"),
        ("vlm_parts", "VLM parts", "VLM", "round", "score", "required parts confirmed by the VLM checks, mean over judges"),
        ("vlm_attr", "VLM attributes", "VLM", "round", "score", "required attributes confirmed, mean over judges"),
        ("vlm_kin", "VLM kinematics", "VLM", "round", "score", "kinematics claims confirmed on pose triptychs, mean over judges"),
        ("vlm_aes", "VLM aesthetics", "VLM", "round", "score", "overall aesthetics, mean over judges"),
        ("wall_min", "Wall (min)", "Cost", "run", "num", "wall-clock time of the whole trajectory"),
        ("cost_usd", "Cost (USD)", "Cost", "run", "usd", "Claude CLI total_cost_usd, summed over attempts"),
        ("tokens_in", "Input tokens (k)", "Cost", "run", "num", "prompt tokens, including cache reads for Claude"),
        ("tokens_out", "Output tokens (k)", "Cost", "run", "num", "completion tokens"),
    ]
    return [dict(zip(("key", "label", "group", "level", "kind", "desc"), x)) for x in m]


def task_sets(core):
    ids = [t["task_id"] for t in core]
    exp = _read(TOP20_SOURCE) or {}
    top = [t for t in exp.get("top_20_tasks") or ids[:20] if t in ids]
    return {"top20": top, "all": ids}


def _condition_image(t, cond):
    conds = t.get("conditions") or []
    for c in (conds.values() if isinstance(conds, dict) else conds):
        if not isinstance(c, dict) or c.get("condition_id") != cond:
            continue
        im = c.get("image")
        if isinstance(im, dict) and im.get("path"):
            return im["path"]
        if isinstance(im, str):
            for x in t.get("images") or []:
                if isinstance(x, dict) and x.get("id") == im:
                    return x.get("path")
    return None


def task_meta(t):
    obj = t.get("object") or {}
    img = _condition_image(t, "name_only+image")
    return {"name": obj.get("name") or t["task_id"], "family": obj.get("family"),
            "pool": len((t.get("subpart_pool") or {}).get("parts") or []),
            "required_parts": len(t.get("required_parts") or []),
            "required_kinematics": len(t.get("required_kinematics") or []),
            "image": _rel(img if not img or os.path.isabs(img) else ROOT / img)}


def queued_systems():
    out = []
    for line in (QUEUE.read_text().splitlines() if QUEUE.exists() else []):
        if line.lstrip().startswith("#") or "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) > 3 and cols[1] != "EXTERNAL" and cols[3] not in out:
            out.append(cols[3])
    return out


def batches():
    out = []
    for f in sorted(PROGRESS.glob("*.json")):
        d = _read(f) or {}
        out.append({"name": f.stem, "done": d.get("done"), "total": d.get("total_todo"), "failed": d.get("failed"),
                    "cost_usd": d.get("cost_usd"), "budget_usd": d.get("budget_usd"), "updated": d.get("updated")})
    return out


# ------------------------------------------------------------------ one run

def _state_summary(s):
    inst = s.get("instances") or {}
    vals = list(inst.values()) if isinstance(inst, dict) else list(inst)
    return {"parts": len(vals),
            "created": sum(1 for v in vals if ((v.get("source") or {}).get("kind") == "created")),
            "joints": len(s.get("joints") or []),
            "materials": len({v.get("material") for v in vals if v.get("material")})}


def _eval_fields(rec):
    m, st = {}, {}
    dims = rec.get("dims") or {}
    for k, d in dims.items():
        st[k] = d.get("status")
        if isinstance(d.get("score"), (int, float)):
            m[k] = float(d["score"])
    nov = ((dims.get("3.3") or {}).get("metrics") or {}).get("novelty_1_minus_fscore@0.05")
    if isinstance(nov, (int, float)):
        m["3.3"] = float(nov)
    attrs = [a["score"] for a in rec.get("attributes") or [] if isinstance(a.get("score"), (int, float))]
    if attrs:
        m["attr"] = sum(attrs) / len(attrs)
    head = rec.get("headline") or {}
    if "buildable" in head:
        m["buildable"] = 1.0 if head["buildable"] else 0.0
    if isinstance(head.get("overall"), (int, float)):
        m["overall"] = float(head["overall"])
    return m, st, list(head.get("critical_fail") or [])


def _vlm_fields(entry):
    acc = {"vlm_parts": [], "vlm_attr": [], "vlm_kin": [], "vlm_aes": []}
    for j in (entry or {}).values():
        if not isinstance(j, dict):
            continue
        for src, dst in (("parts", "vlm_parts"), ("attributes", "vlm_attr"), ("kinematics", "vlm_kin")):
            if isinstance(j.get(src), (int, float)):
                acc[dst].append(j[src])
        aes = (j.get("aesthetics") or {}).get("overall")
        if isinstance(aes, (int, float)):
            acc["vlm_aes"].append(aes)
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


class TaskCtx:
    """Per-task files shared by every run: the eval listing, image similarity, VLM summary, board renders."""

    def __init__(self, task):
        base = RESULTS / task
        self.task = task
        # v3 first: the generic metrics supersede eval_core (v2) and the chair pilot's eval/
        self.eval_dirs = [(base / "eval_v3", "v3"), (base / "eval_core", "v2"), (base / "eval", "v2")]
        self.eval_names_by_dir = [(d, spec, _listdir(d)) for d, spec in self.eval_dirs]
        self.eval_dir = base / "eval"
        self.eval_names = _listdir(self.eval_dir)
        self.imagesim = ((_read_cached(self.eval_dir / "imagesim.json") or {}).get("cosine") or {}) if "imagesim.json" in self.eval_names else {}
        self.vlm = ((_read_cached(self.eval_dir / "vlm_summary.json") or {}).get("designs") or {}) if "vlm_summary.json" in self.eval_names else {}
        self.render_dir = BOARD / "renders" / task
        self.renders = _listdir(self.render_dir)
        self.sig = (tuple(_mtime(d) for d, _ in self.eval_dirs), _mtime(self.eval_dir / "imagesim.json"),
                    _mtime(self.eval_dir / "vlm_summary.json"), _mtime(self.render_dir))

    def eval_for(self, key, ns):
        """The newest-spec record for this design, or None. The same directory name can exist
        in another namespace, so the record's own item path has to agree."""
        for d, spec, names in self.eval_names_by_dir:
            if f"{key}.json" not in names:
                continue
            rec = _read_cached(d / f"{key}.json")
            path = str(((rec or {}).get("item") or {}).get("path") or "")
            if f"/runs/{ns}/" in path:
                rec = dict(rec, eval_spec=rec.get("spec") or spec)
                return rec
        return None


def scan_run(task, ns, d: Path, ctx: TaskCtx, ldraw):
    m = RUN_RE.match(d.name)
    system, tier, cond, R = m.group(1), m.group(2), m.group(3), int(m.group(4))
    names = _listdir(d)
    rounds_names = _listdir(d / "rounds") if "rounds" in names else set()
    eval_sig = tuple(_mtime(ctx.eval_dir / n) for n in sorted(ctx.eval_names) if n.startswith(d.name + "__round") or n == d.name + ".json")
    sig = (_mtime(d), _mtime(d / "rounds"), tuple(_mtime(d / "rounds" / n) for n in sorted(rounds_names)),
           _mtime(d / "design" / "design.json"), _mtime(d / "claude_run.json"), _mtime(d / "state.json"), ctx.sig, eval_sig)
    hit = _cache.get(str(d))
    if hit and hit[0] == sig and hit[1]["status"] in ("done", "invalid"):
        return hit[1]

    design = _read(d / "design" / "design.json") if "design" in names else None
    meta = (design or {}).get("meta") or {}
    state = _read(d / "state.json") if "state.json" in names else None
    crun = _read(d / "claude_run.json") if "claude_run.json" in names else None
    last = max([_mtime(d), _mtime(d / "rounds"), _mtime(d / "state.json"), _mtime(d / "loop_log.json"),
                _mtime(d / "claude_run.json")] + [_mtime(d / "rounds" / n) for n in rounds_names])
    age = time.time() - last

    valid = bool(design) and bool(meta.get("submitted_final")) and meta.get("rounds_completed") == R
    in_progress = bool(crun and crun.get("in_progress"))
    stop = (meta.get("loop") or {}).get("stop")              # the model's episode ended on its own: a result, not a broken job
    if design and task in ldraw and meta.get("pool_mesh_version") != 2:
        status = "invalid"
    elif valid and not in_progress:
        status = "done"
    elif design and stop:
        status = "ended"
    elif age < ACTIVE_S:
        status = "running"
    elif in_progress:
        status = "interrupted"
    elif design:
        status = "partial"
    else:
        status = "stalled"

    renders = _listdir(d / "renders") if "renders" in names else set()
    counters = (state or {}).get("round_counters") or meta.get("round_counters") or []
    rows = []
    for r in range(1, R + 1):
        cp = d / "rounds" / f"round_{r}"
        row = {"r": r, "checkpoint": False, "evaluated": False, "m": {}, "st": {}, "fail": [], "img": None}
        src = None
        if f"round_{r}" in rounds_names and (cp / "design" / "design.json").exists():
            src, key = cp, f"{d.name}__round{r}"
        elif r == R and valid and not rounds_names:          # legacy v2.3 keeps only the final design
            src, key = d, d.name
        if src is not None:
            row["checkpoint"] = True
            s = _read(src / "state.json") if src is cp else state
            if s:
                row["m"].update(_state_summary(s))
                row["seq"] = len(s.get("sequence") or [])
            c = counters[r - 1] if len(counters) >= r else (s or {}).get("counters")
            if c:
                row["m"].update({"tool_calls": c.get("tool_calls"), "errors": c.get("errors"),
                                 "checks": c.get("check"), "renders": c.get("render")})
            rec = ctx.eval_for(key, ns) or (ctx.eval_for(d.name, ns) if src is d else None)
            if rec:
                em, st, fail = _eval_fields(rec)
                row.update(evaluated=True, st=st, fail=fail, spec=rec.get("eval_spec"))
                row["m"].update(em)
                if isinstance(ctx.imagesim.get(rec["item"]["key"]), (int, float)):
                    row["m"]["img"] = float(ctx.imagesim[rec["item"]["key"]])
                row["m"].update(_vlm_fields(ctx.vlm.get(rec["item"]["key"])))
            row["design_dir"] = _rel(src / "design")
        elif len(counters) >= r and counters[r - 1]:
            c = counters[r - 1]
            row["m"].update({"tool_calls": c.get("tool_calls"), "errors": c.get("errors"),
                             "checks": c.get("check"), "renders": c.get("render")})
        # The harness renders a review right after every submit, so a round has a picture even when it has no
        # checkpoint design to read metrics from. Legacy core_v2.3 runs keep only the final design, and until
        # 2026-09-16 this lookup sat inside the branch above, so their r1/r2 review renders existed on disk but
        # never reached the page (all 80 gpt-oss-120b runs have r1_review_*.png and r2_review_*.png).
        # Preference order: the page render (768px JPEG, role colours, `thumbs --hq`), then an older 448px board
        # render, then the harness review render. The harness one is 448px because that is what the model is
        # shown, not what a reader should see.
        stem = f"{ns}__{d.name}__round{r}"
        board_jpg = f"{stem}_iso.jpg"
        board_png = f"{stem}_iso.png"
        if board_jpg in ctx.renders:
            row["img"] = _rel(ctx.render_dir / board_jpg)
            row["hq"] = True
            # render_hq writes the same four cameras the model was shown, so the object page can offer them all
            row["views"] = {v: _rel(ctx.render_dir / f"{stem}_{v}.jpg") for v in ("iso", "cond", "front", "side")
                            if f"{stem}_{v}.jpg" in ctx.renders}
        elif board_png in ctx.renders:
            row["img"] = _rel(ctx.render_dir / board_png)
        elif f"r{r}_review_iso.png" in renders:
            row["img"] = _rel(d / "renders" / f"r{r}_review_iso.png")
            row["views"] = {v: _rel(d / "renders" / f"r{r}_review_{v}.png") for v in ("iso", "cond", "front", "side")
                            if f"r{r}_review_{v}.png" in renders}
        if row.get("design_dir"):
            row["viewer"] = stem                          # /viewer.js builds the interactive geometry on demand
        row["m"] = {k: v for k, v in row["m"].items() if v is not None}
        rows.append(row)

    run_m = {}
    wall = (crun or {}).get("wall_s") or meta.get("wall_s")
    if wall:
        run_m["wall_min"] = wall / 60.0
    if crun:
        atts = crun.get("attempts") or []
        run_m["cost_usd"] = sum(_attempt_cost(a) for a in atts)
        u = [a.get("usage") or {} for a in atts]
        run_m["tokens_in"] = sum((x.get("input_tokens") or 0) + (x.get("cache_creation_input_tokens") or 0)
                                 + (x.get("cache_read_input_tokens") or 0) for x in u) / 1000
        run_m["tokens_out"] = sum(x.get("output_tokens") or 0 for x in u) / 1000
    usage = ((meta.get("loop") or {}).get("usage")) or {}
    if usage.get("prompt"):
        run_m["tokens_in"] = usage["prompt"] / 1000
        run_m["tokens_out"] = (usage.get("completion") or 0) / 1000

    rec = {"task": task, "system": system, "family": family(system), "ns": ns, "dir": _rel(d),
           "setting": f"{tier}|{cond}|{R}", "status": status, "stop": stop, "age_s": round(age),
           "rounds_completed": meta.get("rounds_completed") or sum(1 for x in rows if x["checkpoint"]),
           "legacy": not rounds_names and bool(design), "isolated": (crun or {}).get("isolated"),
           "transport": meta.get("transport"), "m": run_m, "rounds": rows}
    _cache[str(d)] = (sig, rec)
    return rec


# ------------------------------------------------------------------ the whole board

def resolve_tasks(spec, sets):
    if spec in sets:
        return sets[spec]
    return [t for t in spec.split(",") if t]


def collect(tasks_spec="all", workers=16):
    t0 = time.time()
    core = json.loads(CORE.read_text())
    sets = task_sets(core)
    tasks = resolve_tasks(tasks_spec, sets)
    ldraw = _ldraw_tasks()
    jobs = []
    ctxs = {}
    for task in tasks:
        runs = RESULTS / task / "runs"
        ctxs[task] = TaskCtx(task)
        for ns in sorted(_listdir(runs), key=ns_rank):
            if not ns.startswith("core_"):
                continue                                     # v2.1/v2.2 are chair pilots, not the core protocol
            for name in sorted(_listdir(runs / ns)):
                if RUN_RE.match(name):
                    jobs.append((task, ns, runs / ns / name))
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(lambda j: scan_run(*j, ctxs[j[0]], ldraw), jobs))

    chosen: dict[tuple, dict] = {}
    for r in sorted(recs, key=lambda x: (STATUS_RANK.get(x["status"], 9), ns_rank(x["ns"]))):
        k = (r["task"], r["system"], r["setting"])
        if k in chosen:
            chosen[k].setdefault("alternates", []).append({"ns": r["ns"], "status": r["status"], "dir": r["dir"]})
        else:
            chosen[k] = r
    runs = list(chosen.values())

    settings = {}
    for r in runs:
        tier, cond, R = r["setting"].split("|")
        s = settings.setdefault(r["setting"], {"id": r["setting"], "tier": tier, "cond": cond, "rounds": int(R), "n_runs": 0})
        s["n_runs"] += 1
    settings.setdefault(MAIN_SETTING, {"id": MAIN_SETTING, "tier": "B", "cond": "name_only+image", "rounds": 3, "n_runs": 0})
    order = sorted(settings.values(), key=lambda s: (s["id"] != MAIN_SETTING, s["tier"], s["cond"], -s["rounds"]))

    queued = queued_systems()
    systems = {r["system"] for r in runs} | set(queued)
    sys_order = sorted(systems, key=lambda s: (FAMILY_ORDER.index(family(s)), s))
    by_id = {t["task_id"]: t for t in core}
    return {
        "generated": time.time(), "scan_s": round(time.time() - t0, 2), "root": str(ROOT),
        "task_sets": {k: [t for t in v if t in tasks] for k, v in sets.items()},
        "tasks": {t: task_meta(by_id[t]) for t in tasks if t in by_id},
        "settings": order, "main_setting": MAIN_SETTING,
        "systems": {s: {"family": family(s)} for s in systems}, "system_order": sys_order,
        "queued": queued, "metrics": metric_catalog(), "runs": runs, "batches": batches(),
    }


def page(mode, data=None, thumbs=None):
    boot = f"window.BOARD_MODE = {json.dumps(mode)};"
    if data is not None:
        boot += "\nwindow.BOARD_DATA = " + json.dumps(data, separators=(",", ":")).replace("</", "<\\/") + ";"
    if thumbs is not None:
        boot += "\nwindow.BOARD_THUMBS = " + json.dumps(thumbs, separators=(",", ":")) + ";"
    return TEMPLATE.read_text().replace("/*BOOT*/", boot)


def viewer_js(task, key, design_rel):
    """Browser geometry for one checkpoint: decimated parts, joints with their moving sets, the declared
    build order and the evaluator's verdict. Built on demand by viewer_export and cached, because a full
    pass over every checkpoint would be tens of thousands of exports for designs nobody opens."""
    out = BOARD / "viewer" / task / f"{key}.js"
    src = ROOT / design_rel / "design.json"
    # the exporter's own code is part of the cache key: the moving-set fix of 2026-09-17 changed what a
    # joint moves, and a design.json mtime alone would have served the old answer forever -- it did, from
    # a second server sharing this directory, until this check was added.
    code = max((Path(__file__).parent / f).stat().st_mtime for f in ("viewer_export.py", "metrics_v3.py", "evaluate.py"))
    if out.exists() and out.stat().st_mtime >= max(src.stat().st_mtime, code):
        return out
    from ppbench.v2 import viewer_export
    from ppbench.v2.design import Design
    ekey = key.split("__", 1)[1] if "__" in key else key   # the eval record is keyed without the namespace
    rec = None
    for sub in ("eval_v3", "eval_core", "eval"):
        cand = RESULTS / task / sub / f"{ekey}.json"
        if cand.exists():
            rec = _read(cand)
            if rec and "1.1" in (rec.get("dims") or {}):
                break
            rec = None
    viewer_export.export(Design.load(ROOT / design_rel), task, key, out.parent, record=rec)
    return out


def thumb_bytes(rel, width):
    p = os.path.normpath(str(ROOT / rel))
    if not p.startswith(str(ROOT) + os.sep) or not p.lower().endswith((".png", ".jpg", ".jpeg")) or not os.path.exists(p):
        return None
    key = hashlib.sha1(f"{p}|{os.path.getmtime(p)}|{width}".encode()).hexdigest()
    cached = BOARD / "thumb_cache" / f"{key}.jpg"
    if not cached.exists():
        from PIL import Image
        im = Image.open(p)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        if width:
            im.thumbnail((width, width))
        cached.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
        tmp = cached.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, cached)
    return cached.read_bytes()


def snapshot(tasks_spec, out):
    data = collect(tasks_spec)
    thumbs = {}
    wanted = [r2["img"] for r in data["runs"] for r2 in r["rounds"] if r2.get("img")]
    wanted += [t["image"] for t in data["tasks"].values() if t.get("image")]
    for rel in dict.fromkeys(wanted):
        b = thumb_bytes(rel, THUMB_W_SNAPSHOT)
        if b:
            thumbs[rel] = "data:image/jpeg;base64," + base64.b64encode(b).decode()
    html = page("snapshot", data, thumbs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.1f} MB, {len(data['runs'])} runs, {len(thumbs)} images)")


def _render_one(job):
    task, key, ddir, hq, size, samples = job
    from ppbench.v2 import render, render_hq
    from ppbench.v2.design import Design
    out = BOARD / "renders" / task / key
    try:
        design = Design.load(ROOT / ddir)
        if hq:
            render_hq.render_design(design, task, out, size=size, samples=samples)
        else:
            render.render_parts(design.parts, out, views=("iso",), size=size, samples=samples)
        return task, key, None
    except Exception as e:                                # noqa: BLE001 -- one broken design must not stop the rest
        return task, key, str(e)[:160]


def render_missing(tasks_spec, limit=None, setting=MAIN_SETTING, hq=False, redo=False, jobs=1,
                   size=None, samples=None):
    """Board renders for round checkpoints.

    By default only checkpoints with no picture at all, at the harness's own 448px/16-sample quality. With --hq
    the page render is used instead (768px, 32 samples, coloured by role via render_hq), and --redo also replaces
    rounds that so far show only the harness review render, which is produced at 448px because it is what the
    model is shown, not what a reader should see. One iso view is ~5 s, so a full pass is minutes, not hours.
    """
    os.environ.setdefault("RENDER_THREADS", "4")
    size = size or (768 if hq else 448)
    samples = samples or (32 if hq else 16)
    data = collect(tasks_spec)
    todo = [(r["task"], f"{r['ns']}__{Path(r['dir']).name}__round{row['r']}", row["design_dir"], hq, size, samples)
            for r in data["runs"] if setting in ("all", r["setting"])
            for row in r["rounds"] if row["checkpoint"] and row.get("design_dir")
            and not (hq and row.get("hq")) and (redo or not row["img"])]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} checkpoints to render (hq={hq}, redo={redo}, {size}px/{samples}spp, jobs={jobs})", flush=True)
    done = fail = 0
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            for task, key, err in ex.map(_render_one, todo):
                done += err is None
                fail += err is not None
                if err:
                    print(f"FAILED {task} {key}: {err}", flush=True)
    else:
        for job in todo:
            task, key, err = _render_one(job)
            done += err is None
            fail += err is not None
            if err:
                print(f"FAILED {task} {key}: {err}", flush=True)
    print(json.dumps({"rendered": done, "failed": fail, "total": len(todo)}))


def serve(tasks_spec, host, port, ttl):
    state = {"data": None, "t": 0.0}
    lock = threading.Lock()

    def data(force=False):
        with lock:
            if force or state["data"] is None or time.time() - state["t"] > ttl:
                state["data"] = collect(tasks_spec)
                state["t"] = time.time()
            return state["data"]

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if "json" in ctype or "html" in ctype else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    body = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                            '<meta name="viewport" content="width=device-width, initial-scale=1">' + page("live") + "</html>")
                    self._send(200, body.encode(), "text/html; charset=utf-8")
                elif u.path == "/data.json":
                    self._send(200, json.dumps(data("force" in q)).encode(), "application/json")
                elif u.path == "/img":
                    b = thumb_bytes(q.get("p", [""])[0], int(q.get("w", ["360"])[0]))
                    self._send(200, b, "image/jpeg") if b else self._send(404, b"not found", "text/plain")
                elif u.path == "/viewer.js":
                    try:
                        p = viewer_js(q.get("task", [""])[0], q.get("key", [""])[0], q.get("p", [""])[0])
                        self._send(200, p.read_bytes(), "application/javascript")
                    except Exception as e:                # noqa: BLE001 -- report it in the page, never 500 the board
                        self._send(200, f"window.__VIEWER_ERROR={json.dumps(str(e)[:200])};".encode(), "application/javascript")
                elif u.path.startswith("/vendor/"):
                    f = VENDOR / os.path.basename(u.path)
                    self._send(200, f.read_bytes(), "application/javascript") if f.exists() else self._send(404, b"not found", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *a):
            pass

    print(f"results board on http://{host}:{port}  (tasks: {tasks_spec}; rescans when data is older than {ttl}s)")
    ThreadingHTTPServer((host, port), H).serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["serve", "collect", "snapshot", "thumbs"])
    ap.add_argument("--tasks", default=None, help="top20, all, or a comma list (serve/collect default all; snapshot/thumbs default top20)")
    ap.add_argument("--hq", action="store_true", help="thumbs: page-quality render (768px, 32 samples, role colours)")
    ap.add_argument("--redo", action="store_true", help="thumbs: also replace rounds that only have the 448px harness review render")
    ap.add_argument("--jobs", type=int, default=1, help="thumbs: parallel render processes")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--ttl", type=int, default=45, help="serve: seconds before a page reload triggers a rescan")
    ap.add_argument("--out", default=str(BOARD / "board.html"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--setting", default=MAIN_SETTING, help="thumbs: '<tier>|<condition>|<rounds>' or 'all'")
    ns = ap.parse_args(argv)
    if ns.cmd == "serve":
        serve(ns.tasks or "all", ns.host, ns.port, ns.ttl)
    elif ns.cmd == "collect":
        d = collect(ns.tasks or "all")
        BOARD.mkdir(parents=True, exist_ok=True)
        (BOARD / "data.json").write_text(json.dumps(d))
        print(f"{len(d['runs'])} runs over {len(d['tasks'])} tasks in {d['scan_s']}s -> {BOARD / 'data.json'}")
    elif ns.cmd == "snapshot":
        snapshot(ns.tasks or "top20", Path(ns.out))
    else:
        render_missing(ns.tasks or "top20", ns.limit, ns.setting, hq=ns.hq, redo=ns.redo, jobs=ns.jobs)


if __name__ == "__main__":
    main()
