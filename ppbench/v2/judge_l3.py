"""Level 3 VLM judge: decomposition, aesthetics, structure alignment.

Policy P8-P13 in docs/evaluation/metrics_feedback.md (2026-09-20 Level 3 review). The judge is the
primary evidence for all three dimensions; `spec_v34.py` blends it with the computed half.

Three calls per design, because the three questions need different pictures and the vLLM servers
cap a request at two images:

  3.1 decomposition        reference photo + <key>_decomp.png   (part-coloured and exploded views)
  3.2 aesthetics           <key>_claygrid.png ALONE             -- no photo: judged as a design, not
                                                                   as a copy (P9)
  3.3 structure alignment  reference photo + <key>_claygrid.png (fine-grained detail fidelity)

Every design is scored on its own on a 1-10 rubric with written anchors, normalised (s-1)/9,
temperature 0. Nothing tells the judge which system produced a design. Two non-contestant judges
score everything; their rank agreement is reported next to the scores (P12). The reference design is
scored like a submission and is the calibration anchor.

    .venv_eval/bin/python -m ppbench.v2.judge_l3 run --model qwen2.5-vl-32b \
        --base-url http://127.0.0.1:8111/v1 [--tasks all] [--workers 24]
    .venv_eval/bin/python -m ppbench.v2.judge_l3 agreement
"""
from __future__ import annotations

import argparse
import os
import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from ppbench.v2.judge_views import main_setting, reference_item
from ppbench.v2.task import RESULTS, Task

JUDGE_VERSION = "l3-v2"
MAX_IMG = 1024

# ---------------------------------------------------------------- rubrics (generic; no task wording)
# 3.1 prompt history. v1 (abstract questions): gemma-3-27b answered 7 on ~80 % of designs. v2 (owner's
# review, 2026-09-20) named the two errors -- one component chopped into pieces (a car door built from a
# stack of bricks), and parts that must be separate fused into one (a wheel and its axle as one solid) --
# but told the judge "several colours on a component = over-split" and "an error caps the criterion at
# 5". Qwen then listed the whole car "body" as one component, flagged every design's body as over-split
# and answered 4 on all 42 designs of a task. v3 keeps v2's two errors and its roadmap, but defines a
# component as a separately manufactured piece, says that several colours are correct when each colour is
# a real piece, grades by how many components are affected instead of capping, and asks for the counts
# the grade rests on. Examples are illustrations over several object kinds; the rule names no task.
# v4 (same morning): the v3 pilot showed the judge reciting the prompt's examples rather than reading the
# image -- two stacked boxes with no wheels scored 0.93 with "fused: wheels + axles". v4 makes it count
# only what it can see (a component the real product has that is not visible as its own coloured part
# counts as FUSED), anchors the unrecognisable case, and drops the worked answer from the reply format.
DECOMP = {
    "over_split": "How many real components are chopped into arbitrary pieces -- slabs, bricks, strips or slices that do not correspond to separately made parts (e.g. a door or a tabletop assembled from a stack of blocks)? 10 = none; 8 = one minor component; 6 = two or three components, or one major one; 3 = most components are built from arbitrary pieces; 1 = the whole object is a pile of bricks or fragments with no component-level parts.",
    "fused": "How many pairs of components that must be separate -- because one moves relative to the other, or because they are made separately -- are merged into a single part (e.g. a wheel and its axle, a door or lid and its frame, a drawer and its cabinet, an arm and its base)? 10 = none; 8 = one minor static pair; 6 = one moving pair, or two or three static pairs; 3 = several moving pairs; 1 = the object, or most of it, is one lump.",
    "boundaries": "Where one part meets another, does the seam fall on a real component edge rather than cutting through the middle of a component? 10 = every seam is on a real edge; 8 = nearly all; 6 = several seams cut through components; 3 = most seams are arbitrary; 1 = no seam corresponds to a real edge.",
    "overall": "Overall quality of the part breakdown, as a mechanical designer planning manufacture and assembly would rate it. 1 = unusable; 4 = poor; 7 = good; 10 = excellent.",
}
DECOMP_PROMPT = """You are a mechanical design reviewer. You are judging how a 3D object has been broken into parts.
The first image is a photograph, only so you know what kind of object this is.
The second image shows the design: in the top row the object is assembled and every part has its own
colour; in the bottom row the parts are pulled apart (exploded). One colour = one part. The design has
{n_parts} parts.

A COMPONENT means a piece a manufacturer makes separately and then assembles: for a car, the body shell,
each door, the bonnet, each wheel, each tyre, each seat; for a chair, the seat, the backrest, each leg,
each armrest; for a lamp, the base, each arm segment, the shade. Several colours on one area are CORRECT
when each colour is such a component. Two kinds of error matter:
- OVER-SPLIT: a single component is chopped into arbitrary pieces with no manufacturing reason -- a door or
  a tabletop built from a stack of slabs, bricks or strips.
- FUSED: two components that must be separate are one part -- because one moves relative to the other
  (a wheel and its axle, a door or lid and its frame, a drawer and its cabinet, an arm and its base), or
  because they are clearly made separately.

Judge ONLY what you can see in the second image. Do not assume a component exists because the product
normally has it.

Work through these steps before you score:
1. List the components a real product of this kind is assembled from (at most 10).
2. For each, find it in the design and decide from the colours and the exploded view:
   - its own part, or a few real sub-parts: correct;
   - chopped into arbitrary pieces: OVER-SPLIT;
   - merged into a neighbour it must be separate from: FUSED;
   - not visible as its own coloured part at all (absent, or buried inside a larger block): FUSED too --
     the design did not make it a separate part.
3. Check every moving component -- wheels, doors, lids, drawers, hinges, arms, knobs -- in particular.
4. If the design is not recognisable as this kind of object (for example a few plain blocks), most
   components are absent: that is the "one lump" case, and fused and overall must be 1-3.
5. Count what you found, then score each criterion with an integer from 1 to 10 from those counts:
{questions}

Designs in this set range from excellent to unusable. Score from your counts using the anchors, over the
whole 1-10 range; do not default to the same middle number for every design or every criterion.

Reply with JSON only, with exactly these keys (lists hold what YOU found in this design, and may be empty):
{{"components": [...], "over_split_found": [...], "fused_found": [...], "n_components": <int>, "n_over_split": <int>, "n_fused": <int>, "over_split": <1-10>, "fused": <1-10>, "boundaries": <1-10>, "overall": <1-10>, "reason": "<one sentence>"}}"""

# 3.1 prompt v1 (2026-09-20), kept selectable: see DECOMP_ACTIVE below.
DECOMP_V1 = {
    "boundaries": "Do the part boundaries fall where a real manufacturer would separate pieces -- at material changes, at moving interfaces, at the edges of components that are made separately? 1 = the cuts are arbitrary slices through what should be one piece, or two clearly different components share one part; 4 = some boundaries are sensible, several are arbitrary; 7 = mostly sensible with a few odd cuts; 10 = every boundary is where a real product's boundary is.",
    "granularity": "Is the number of parts right for this object -- neither one fused lump that hides real components, nor needlessly shattered into slivers? 1 = grossly wrong (one blob, or hundreds of fragments); 4 = clearly too coarse or too fine; 7 = close to right; 10 = exactly the breakdown a product drawing would use.",
    "separability": "Could each coloured part be made and handled as one physical piece, and could the assembly be put together from them? 1 = parts are disconnected clouds or pass through one another so they could never be assembled; 4 = several parts are implausible as single pieces; 7 = mostly manufacturable with minor issues; 10 = every part is a single, makeable, assemblable piece.",
    "overall": "Overall quality of the part breakdown, as a mechanical designer would rate it. 1 = unusable; 4 = poor; 7 = good; 10 = excellent.",
}
DECOMP_PROMPT_V1 = """You are judging how a 3D object has been broken into parts.
The first image is a photograph of the kind of object being designed, for context only.
The second image shows the design: the top row is the assembled object with each part in a different
colour, the bottom row separates the parts. The design has {n_parts} parts.
Judge the part breakdown only. Do not judge how closely it resembles the photograph, and ignore
rendering quality and the particular colours chosen. Rate each criterion with an integer from 1 to 10
using the anchors given:
{questions}

Reply with JSON only, e.g. {{"boundaries": 6, "granularity": 7, "separability": 5, "overall": 6, "reason": "one sentence"}}"""

AESTH = {
    "proportion": "Are the proportions and stance well judged for a real product of this kind -- masses balanced, nothing gratuitously fat, thin, tall or stunted? 1 = grossly distorted; 4 = noticeably awkward; 7 = plausible with small issues; 10 = convincing, deliberate proportions.",
    "coherence": "Do the parts look like they belong to one product -- a consistent formal language, matching thicknesses and radii, edges that line up and meet? 1 = an unrelated heap; 4 = several parts clash in style or scale; 7 = mostly consistent; 10 = fully resolved as one object.",
    "refinement": "How finished is the form -- deliberate surfaces and edges, or crude stand-in primitives and unresolved stumps? 1 = raw boxes standing in for everything; 4 = mostly primitives with a little shaping; 7 = considered shapes with rough spots; 10 = the level of resolution of a finished product.",
    "overall": "Overall design quality as an experienced industrial designer would rate it, judging the object on its own merits. 1 = unusable; 4 = poor; 7 = good; 10 = excellent.",
}
AESTH_PROMPT = """You are rating the design quality of a 3D object, shown rendered in plain grey clay
from four views. Judge the object on its own merits as a piece of product design. There is no
reference to compare against and you are NOT being asked whether it resembles any particular product
-- only whether it is well designed. Ignore rendering quality, lighting and colour.
Rate each criterion with an integer from 1 to 10 using the anchors given:
{questions}

Reply with JSON only, e.g. {{"proportion": 6, "coherence": 7, "refinement": 5, "overall": 6, "reason": "one sentence"}}"""

ALIGN = {
    "inventory": "Is every distinct part visible in the photograph also present in the design, and has nothing been invented that the photograph does not have? 1 = most parts missing or a different object; 4 = the main masses are there, several components missing or invented; 7 = all major parts present, a minor one missing; 10 = the full inventory, nothing missing, nothing extra.",
    "placement": "Is each part in the right position and orientation relative to the others, as in the photograph? 1 = the arrangement is unrelated; 4 = major parts in the wrong place or facing the wrong way; 7 = correct arrangement with small offsets; 10 = every part placed as in the photograph.",
    "relative_scale": "Are the relative sizes of the parts right -- this part against that part, not the overall size of the object? 1 = relative sizes unrelated; 4 = several parts badly out of scale with the rest; 7 = close with one or two off; 10 = every part in the right proportion to its neighbours.",
    "detail_fidelity": "Are the fine features matched: the COUNT of repeated elements (legs, wheels, slats, buttons, rungs, panels), the cross-section and profile of members (round vs square, tapered vs straight), cut-outs, and the curvature of surfaces? This is the fine-grain question, not the overall silhouette. 1 = no fine feature matches; 4 = counts or profiles broadly wrong; 7 = most counts and profiles right, some details simplified away; 10 = the fine features match the photograph.",
    "overall": "Overall, how faithfully does this design reproduce the structure of the object in the photograph? 1 = a different object; 4 = the right kind of object, structurally wrong; 7 = clearly this object with minor structural differences; 10 = structurally faithful in every visible respect.",
}
ALIGN_PROMPT = """You are judging how faithfully a 3D design reproduces the structure of a real object.
The first image is a photograph of the target object. The second image shows the design rendered in
plain grey clay from four views (front-right, rear-left, front, side).
Judge structure and detail, not beauty and not rendering quality, and ignore colour, material and
surface finish -- the design is deliberately shown in featureless clay. Rate each criterion with an
integer from 1 to 10 using the anchors given:
{questions}

Reply with JSON only, e.g. {{"inventory": 6, "placement": 7, "relative_scale": 5, "detail_fidelity": 4, "overall": 6, "reason": "one sentence"}}"""

# Which 3.1 prompt a run asks. v4 is the owner-requested over-split / fused prompt; at full scale (2026-09-21) it
# erased the closed/open gap for Qwen (+16.7 -> +1.8) and inverted it for gemma (+1.5 -> -5.6), so v1 is kept
# selectable until the owner picks one. Answers are tagged with prompt_version, so the two never mix.
DECOMP_ACTIVE = os.environ.get("PPB_DECOMP_PROMPT", "v1")   # v1 is the adopted 3.1 prompt (release default)

DIMS = {
    "3.1": ({"name": "decomposition", "rubric": DECOMP, "prompt": DECOMP_PROMPT, "sheet": "decomp", "photo": True,
             "version": "v4", "max_tokens": 800,
             "evidence": ("components", "over_split_found", "fused_found", "n_components", "n_over_split", "n_fused")}
            if DECOMP_ACTIVE == "v4" else
            {"name": "decomposition", "rubric": DECOMP_V1, "prompt": DECOMP_PROMPT_V1, "sheet": "decomp", "photo": True,
             "version": "v1", "max_tokens": 320, "evidence": ()}),
    "3.2": {"name": "aesthetics", "rubric": AESTH, "prompt": AESTH_PROMPT, "sheet": "clay", "photo": False,
            "version": "v1", "max_tokens": 320, "evidence": ()},
    "3.3": {"name": "structure_alignment", "rubric": ALIGN, "prompt": ALIGN_PROMPT, "sheet": "clay", "photo": True,
            "version": "v1", "max_tokens": 320, "evidence": ()},
}


# ---------------------------------------------------------------- plumbing
def _b64(im):
    buf = BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _img(path, cap=MAX_IMG):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((cap, cap))
    return _b64(im)


def _ask(client, model, content, max_tokens=320, retries=3):
    txt = ""
    for _ in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}],
                                               max_tokens=max_tokens, temperature=0.0)
            txt = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                return json.loads(m.group(0)), txt
        except Exception as e:
            txt = f"error: {e}"
            time.sleep(4)
    return None, txt


def _score(ans, rubric):
    """1-10 -> [0,1]. The dimension's judge score is the mean of the specific criteria; `overall` is
    kept as an independent holistic cross-check and is deliberately not folded in twice."""
    out, raw = {}, {}
    for c in rubric:
        raw[c] = (ans or {}).get(c)
        try:
            out[c] = max(0.0, min(1.0, (float(raw[c]) - 1) / 9))
        except (TypeError, ValueError):
            out[c] = None
    spec = [v for c, v in out.items() if c != "overall" and v is not None]
    return {"raw": raw, "scores": out,
            "judge_score": (sum(spec) / len(spec)) if spec else None,
            "overall": out.get("overall"),
            "reason": (ans or {}).get("reason")}


def _n_parts(task_id, key):
    for src in ("eval_v33", "eval_v32"):
        p = RESULTS / task_id / src / f"{key}.json"
        if p.exists():
            try:
                m = (json.loads(p.read_text()).get("dims") or {}).get("3.1", {}).get("metrics") or {}
                if m.get("n_parts"):
                    return int(m["n_parts"])
            except (OSError, ValueError):
                pass
    return None


# ---------------------------------------------------------------- the run
def run_task(task_id, model, base_url, api_key="local", workers=24, force=False, dims=tuple(DIMS), exclude=()):
    from openai import OpenAI
    out_p = RESULTS / task_id / "judge_l3" / f"{model.replace('/', '_')}.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if out_p.exists() and not force:
        try:
            have = json.loads(out_p.read_text()).get("items", {})
        except (OSError, ValueError):
            have = {}

    task = Task(task_id, snapshot=RESULTS / task_id / "task_snapshot_core.json")
    photo_path = task.condition("name_only+image")["image"]["path"]
    photo = _img(photo_path, 768)
    vdir = RESULTS / task_id / "judge_views"

    jobs = []
    for it in main_setting(task_id) + [reference_item(task_id)]:
        key = it["key"]
        if it.get("system") in exclude:
            continue
        for d in dims:
            old = have.get(key, {}).get(d, {})
            if old.get("judge_score") is not None and old.get("prompt_version", "v1") == DIMS[d]["version"]:
                continue
            sheet = vdir / f"{key}_{'decomp' if DIMS[d]['sheet'] == 'decomp' else 'claygrid'}.png"
            if not sheet.exists():
                continue
            jobs.append((key, d, str(sheet), it))

    if not jobs:
        return {"task": task_id, "model": model, "calls": 0, "status": "complete"}

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)

    def one(job):
        key, d, sheet, it = job
        cfg = DIMS[d]
        qs = "\n".join(f"- {k}: {v}" for k, v in cfg["rubric"].items())
        text = cfg["prompt"].format(questions=qs, n_parts=_n_parts(task_id, key) or "several")
        content = [{"type": "text", "text": text}]
        if cfg["photo"]:
            content.append({"type": "image_url", "image_url": {"url": photo}})
        content.append({"type": "image_url", "image_url": {"url": _img(sheet)}})
        ans, raw = _ask(client, model, content, max_tokens=cfg["max_tokens"])
        rec = _score(ans, cfg["rubric"])
        rec["prompt_version"] = cfg["version"]
        if cfg["evidence"]:
            rec["evidence"] = {e: (ans or {}).get(e) for e in cfg["evidence"]}
        rec["raw_text"] = raw[:300] if rec["judge_score"] is None else None
        return key, d, rec

    with ThreadPoolExecutor(workers) as ex:
        for key, d, rec in ex.map(one, jobs):
            have.setdefault(key, {})[d] = rec

    res = {"task": task_id, "model": model, "judge_version": JUDGE_VERSION,
           "photo": photo_path, "rubrics": {d: DIMS[d]["rubric"] for d in DIMS},
           "n_items": len(have), "items": have,
           "n_unparsed": sum(1 for v in have.values() for d in v if v[d]["judge_score"] is None),
           "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = out_p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(out_p)                       # atomic: a killed run never leaves a truncated judge file
    return {"task": task_id, "model": model, "calls": len(jobs), "unparsed": res["n_unparsed"]}


def run(tasks, model, base_url, api_key="local", workers=24, force=False, exclude=()):
    ids = tasks or [t["task_id"] for t in json.loads(
        (__import__("ppbench.v2.task", fromlist=["CORE"]).CORE).read_text())]
    t0 = time.time()
    for n, t in enumerate(ids, 1):
        try:
            r = run_task(t, model, base_url, api_key, workers, force, exclude=exclude)
            print(f"[{n}/{len(ids)}] {t:28s} calls {r.get('calls', 0):4d}  unparsed {r.get('unparsed', 0):3d}"
                  f"  {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[{n}/{len(ids)}] {t:28s} FAILED {type(e).__name__}: {e}", flush=True)


def agreement(tasks=None):
    """Spearman rank correlation between the judges, per dimension, over every shared design."""
    import numpy as np
    from itertools import combinations
    ids = tasks or [t["task_id"] for t in json.loads(
        (__import__("ppbench.v2.task", fromlist=["CORE"]).CORE).read_text())]
    per = {}
    for t in ids:
        for p in sorted((RESULTS / t / "judge_l3").glob("*.json")):
            if p.name.endswith(".tmp"):
                continue
            j = json.loads(p.read_text())
            for k, v in j["items"].items():
                for d, rec in v.items():
                    if rec.get("judge_score") is not None:
                        per.setdefault(p.stem, {}).setdefault(d, {})[f"{t}|{k}"] = rec["judge_score"]
    out = {}
    for a, b in combinations(sorted(per), 2):
        for d in sorted(DIMS):
            ka, kb = per[a].get(d, {}), per[b].get(d, {})
            keys = sorted(set(ka) & set(kb))
            if len(keys) < 20:
                continue
            x = np.array([ka[k] for k in keys]); y = np.array([kb[k] for k in keys])
            rx, ry = x.argsort().argsort(), y.argsort().argsort()
            out[f"{a}|{b}|{d}"] = {"n": len(keys), "spearman": round(float(np.corrcoef(rx, ry)[0, 1]), 3),
                                   "mean_a": round(float(x.mean()), 3), "mean_b": round(float(y.mean()), 3)}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "agreement"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--exclude-systems", default="", help="comma list of systems to skip")
    ns = ap.parse_args()
    ts = None if ns.tasks == "all" else [x.strip() for x in ns.tasks.split(",") if x.strip()]
    if ns.cmd == "run":
        run(ts, ns.model, ns.base_url, ns.api_key, ns.workers, ns.force,
            exclude=tuple(x for x in ns.exclude_systems.split(",") if x))
    else:
        agreement(ts)
