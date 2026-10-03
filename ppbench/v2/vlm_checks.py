"""Evidence-grounded VLM checks: kinematics claims, attribute claims, required parts, and design quality.

The rule-based dimensions (integrity, stability, geometry, matching of joints and roles) read only the 3D
structure. What a rule cannot settle is whether the structure *means* what it claims: whether the part labelled
"lever" looks and sits like a height lever, whether the declared swivel really turns the seat about the column,
whether a chair looks like a finished product. Those questions go to a VLM, but only with evidence rendered
from the design's own geometry, never from the producer's text alone, and every answer must cite that evidence:

  overview   role-coloured renders with a numbered label on every part (L1, L2, ...) and a legend
  materials  renders coloured by declared material class, with a legend
  motion     for the best candidate joint of each kinematics claim (from the rule-based matcher): three poses
             of that joint alone, moving parts highlighted, its axis drawn, plus the joint record
  clay       four neutral views next to the condition photo, with computed symmetry and proportion facts

Grounding rule: an answer's cited labels must exist, and at least one must belong to the parts the claim is
about (the moving set of the joint, or the candidate parts of the role). An answer that fails this is recorded
as `ungrounded` and scores 0, whatever the model said. Designs that do not declare roles or joints are skipped
for the checks that need them, exactly as in the rule-based dimensions.

    .venv_eval/bin/python -m ppbench.v2.vlm_checks evidence --workers 24            # CPU, cached per design
    .venv_eval/bin/python -m ppbench.v2.vlm_checks judge --model <served> --base-url <url> --tag <judge>
    .venv_eval/bin/python -m ppbench.v2.vlm_checks summarize
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np

from ppbench.v2.task import RESULTS, Task

TASK = "swivel_office_chair"
EVID = RESULTS / TASK / "evidence"
SNAPSHOT = "task_snapshot.json"
SIZE = 512
MAT_COLORS = {"plastic": (0.88, 0.89, 0.90), "heavy_metal": (0.49, 0.53, 0.57), "light_metal": (0.73, 0.76, 0.80),
              "wood": (0.66, 0.47, 0.29), "foam": (0.91, 0.66, 0.49), "textile": (0.43, 0.35, 0.56),
              "rubber": (0.17, 0.17, 0.17), "glass": (0.62, 0.83, 0.88), "ceramic": (0.95, 0.94, 0.90)}


def _font(sz):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except OSError:
        return ImageFont.load_default()


# id pass palette: 8-bit targets on a 5-level grid (124 codes); Blender gets their sRGB-decoded linear values and the
# Standard view transform encodes them back, so a pixel is decoded as the nearest target (edge blends are rejected)
_LEVELS = (0, 64, 128, 192, 255)
ID_TARGETS = np.array([(r, g, b) for r in _LEVELS for g in _LEVELS for b in _LEVELS if (r, g, b) != (0, 0, 0)], float)


def _srgb_to_linear(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _decode_ids(id_png):
    """Per-pixel code (index into ID_TARGETS + 1), 0 for background or ambiguous edge pixels."""
    from PIL import Image
    px = np.asarray(Image.open(id_png).convert("RGB")).astype(float)
    flat = px.reshape(-1, 3)
    d = ((flat[:, None, :] - ID_TARGETS[None, :, :]) ** 2).sum(-1)
    k = d.argmin(1)
    ok = d[np.arange(len(k)), k] < 18.0 ** 2
    return np.where(ok, k + 1, 0).reshape(px.shape[:2])


def _label_image(png, id_png, labels):
    """Draw each label at the most central pixel of the part's visible region in the id pass; parts with
    fewer than 25 visible pixels in this view get no label (a label on an occluder would be false evidence)."""
    from PIL import Image, ImageDraw
    im = Image.open(png).convert("RGB")
    ids = _decode_ids(id_png)
    d = ImageDraw.Draw(im)
    f = _font(15)
    spots = []
    for lab, code in labels:
        ys, xs = np.nonzero(ids == code)
        if len(xs) < 25:
            continue
        mx, my = np.median(xs), np.median(ys)
        k = int(np.argmin((xs - mx) ** 2 + (ys - my) ** 2))
        spots.append((lab, float(xs[k]), float(ys[k])))
    placed = []
    for lab, u, v in sorted(spots, key=lambda t: t[2]):
        for _ in range(6):                      # nudge overlapping labels apart
            if all(abs(u - pu) > 26 or abs(v - pv) > 17 for pu, pv in placed):
                break
            v += 17
        placed.append((u, v))
        box = d.textbbox((u, v), lab, font=f, anchor="mm")
        d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(255, 255, 255), outline=(20, 20, 20))
        d.text((u, v), lab, fill=(10, 10, 10), font=f, anchor="mm")
    im.save(png)
    os.unlink(id_png)


def _tile(paths, out, titles, cols=2, cell=SIZE, legend=None):
    from PIL import Image, ImageDraw
    rows = (len(paths) + cols - 1) // cols
    leg_h = 34 if legend else 0
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 26) + leg_h), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = _font(16)
    if legend:
        x = 8
        for name, rgb in legend:
            d.rectangle([x, rows * (cell + 26) + 8, x + 18, rows * (cell + 26) + 26], fill=tuple(int(255 * c) for c in rgb), outline=(0, 0, 0))
            d.text((x + 24, rows * (cell + 26) + 8), name, fill=(0, 0, 0), font=_font(14))
            x += 34 + int(d.textlength(name, font=_font(14)))
    for i, (p, t) in enumerate(zip(paths, titles)):
        x, y = (i % cols) * cell, (i // cols) * (cell + 26)
        sheet.paste(Image.open(p).convert("RGB").resize((cell, cell)), (x, y + 26))
        d.text((x + 6, y + 4), t, fill=(0, 0, 0), font=f)
    sheet.save(out)
    return str(out)


# ---------------------------------------------------------------- evidence

def _rgb_hex(c):
    return "#%02x%02x%02x" % tuple(int(255 * x) for x in c)


def evidence_for(key, rec, design, task):
    """Render the evidence images and write the evidence record for one design."""
    import trimesh
    from ppbench.v2 import analysis, materials, render
    from ppbench.v2.design import axis_angle
    from ppbench.v2.metrics_v3 import kinematic_moving_set
    from ppbench.v2.lexicon import Lexicon
    from ppbench.v2.render_hq import ROLE_COLORS, UNKNOWN
    out = EVID / key
    out.mkdir(parents=True, exist_ok=True)
    lex = Lexicon(task.id)
    parts = design.parts
    labels = {p.id: f"L{i + 1}" for i, p in enumerate(parts)}
    rng = np.random.default_rng(0)
    sidx = {p.id: rng.choice(len(p.vertices), min(400, len(p.vertices)), replace=False) for p in parts}
    table = []
    for p in parts:
        m = materials.resolve(p.material)
        table.append({"label": labels[p.id], "part_id": p.id, "role": p.role, "canonical_role": lex.canon(p.role),
                      "material": m[0] if m else p.material, "material_class": m[1] if m else None,
                      "bbox_min_m": np.round(p.vertices.min(0), 3).tolist(), "bbox_max_m": np.round(p.vertices.max(0), 3).tolist()})

    def render_colored(prefix, meshes, views, which_labels=None, frame=None):
        """meshes: [(id, vertices, faces, srgb)]; labels for parts in which_labels (None = all, False = none)."""
        scene = trimesh.Scene()
        cmap, idmap, codes = {}, {}, []
        for i, (pid, v, f, c) in enumerate(meshes):
            name = f"n{i:03d}"
            scene.add_geometry(trimesh.Trimesh(np.asarray(v) @ render.ZUP_TO_YUP.T, f, process=False), node_name=name, geom_name=name)
            cmap[name] = [float(x) ** 2.2 for x in c]
            code = i + 1 if i < len(ID_TARGETS) else 0
            idmap[name] = _srgb_to_linear(ID_TARGETS[code - 1]).tolist() if code else [0.0, 0.0, 0.0]
            if code and pid in labels and which_labels is not False and (which_labels is None or pid in which_labels):
                codes.append((labels[pid], code))
        glb = str(out / f"{prefix}_scene.glb")
        scene.export(glb)
        job = {"mode": "design", "glb": glb, "out": str(out / prefix), "size": SIZE, "samples": 16,
               "views": {v: render.VIEWS[v] for v in views}, "colors": cmap, "ground": True,
               "view_transform": "Standard", "world_strength": 0.35, "light_scale": 0.55, "ground_rgb": [0.3, 0.31, 0.31]}
        if codes:
            job["id_colors"] = idmap
        if frame is not None:
            job["frame"] = [list(map(float, frame[0])), list(map(float, frame[1]))]
        render._run(job, timeout=1800)
        os.unlink(glb)
        paths = {v: str(out / f"{prefix}_{v}.png") for v in views}
        if codes:
            for v in views:
                _label_image(paths[v], str(out / f"{prefix}_{v}_id.png"), codes)
        return paths

    base = [(p.id, p.vertices, p.faces) for p in parts]
    views = ("iso", "front", "side", "back_iso")
    ov = render_colored("overview", [(i, v, f, ROLE_COLORS.get(lex.canon(p.role), UNKNOWN)) for (i, v, f), p in zip(base, parts)], views)
    overview = _tile([ov[v] for v in views], out / "overview.png", [f"{v} view (L# = part labels, see table)" for v in views])
    mat_col = lambda p: MAT_COLORS.get((materials.resolve(p.material) or (None, None))[1], (0.85, 0.2, 0.75))
    mv = render_colored("materials", [(i, v, f, mat_col(p)) for (i, v, f), p in zip(base, parts)], ("iso", "back_iso"))
    material_img = _tile([mv["iso"], mv["back_iso"]], out / "materials.png",
                         ["materials, front-right", "materials, back-left"],
                         legend=[(k.replace("_", " "), c) for k, c in MAT_COLORS.items()] + [("undeclared", (0.85, 0.2, 0.75))])
    clay = render_colored("clay", [(i, v, f, (0.62, 0.64, 0.66)) for i, v, f in base], ("cond", "iso", "front", "side"), which_labels=False)
    clay_img = _tile([clay[v] for v in ("cond", "iso", "front", "side")], out / "clay.png",
                     ["condition camera", "front-right", "front", "right side"])

    # motion evidence: the best candidate joint of each kinematics claim, as chosen by the rule-based matcher
    motion = []
    joints = {j.id: j for j in design.joints}
    chosen = {}
    for c in (rec["dims"].get("2.2", {}).get("metrics", {}) or {}).get("claims", []):
        best = max((b for b in c.get("best", []) if b.get("joint")), key=lambda b: b["q"], default=None)
        if best and best["q"] > 0 and best["joint"] in joints:
            chosen.setdefault(best["joint"], []).append(c["id"])
    rows = analysis.pair_table(parts, analysis.occupancies(parts), frozenset(frozenset((j.parent, j.child)) for j in design.joints)) if chosen else []
    V = np.vstack([p.vertices for p in parts])
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    for jid, claim_ids in chosen.items():
        j = joints[jid]
        if j.axis is None or j.origin is None or not np.linalg.norm(j.axis):
            continue
        mset, _ = kinematic_moving_set(parts, rows, design.joints, j)
        ax = np.asarray(j.axis, float) / np.linalg.norm(j.axis)
        org = np.asarray(j.origin, float)
        if j.type in ("continuous", "cylindrical", "ball") or (j.type == "revolute" and j.limits is None):
            # cylindrical joints are shown turning about their axis; ball joints turning about the declared axis
            poses = [("0°", "rot", 0.0), ("90°", "rot", math.pi / 2), ("180°", "rot", math.pi)]
        elif j.type == "revolute":
            a0, a1 = j.limits
            poses = [(f"{math.degrees(x):.0f}°", "rot", x) for x in (a0, (a0 + a1) / 2, a1)]
        elif j.type == "prismatic":
            a0, a1 = j.limits if j.limits is not None else (0.0, 0.0)
            poses = [(f"{x * 100:+.0f} cm", "lin", x) for x in (a0, (a0 + a1) / 2, a1)]
        else:
            continue
        rod = trimesh.creation.cylinder(radius=max(0.004, 0.006 * diag), height=0.45 * diag, sections=16)
        z = np.array([0.0, 0.0, 1.0])
        cz = np.cross(z, ax)
        R = np.eye(3) if np.linalg.norm(cz) < 1e-6 else axis_angle(cz / np.linalg.norm(cz), math.acos(np.clip(ax @ z, -1, 1)))
        rod_v = np.asarray(rod.vertices) @ R.T + org
        # rotations read best from about 50° off the axis (the rod stays visible), translations looking across it
        best_v, best_s = "iso", -1.0
        for v, (az, el) in render.VIEWS.items():
            a, e = math.radians(az), math.radians(el)
            vd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            c_ = abs(vd @ ax)
            sc = (1 - abs(c_ - 0.65) if poses[0][1] == "rot" else 1 - c_) + (0.1 if v in ("iso", "back_iso", "cond") else 0)
            if sc > best_s:
                best_v, best_s = v, sc
        # small moving sets (a caster wheel) get a zoomed frame around the moving parts and the axis
        mv_pts = np.vstack([p.vertices for p in parts if p.id in mset] + [org[None]])
        mlo, mhi = mv_pts.min(0), mv_pts.max(0)
        frame = None
        if np.linalg.norm(mhi - mlo) < 0.3 * diag:
            cen = (mlo + mhi) / 2
            h = max(np.linalg.norm(mhi - mlo) * 0.9, 0.1 * diag)
            frame = (cen - h, cen + h)
            rod = trimesh.creation.cylinder(radius=max(0.002, 0.004 * h), height=1.4 * h, sections=16)
            rod_v = np.asarray(rod.vertices) @ R.T + org
        frames = []
        for k, (title, kind, val) in enumerate(poses):
            meshes = []
            for p in parts:
                vv = p.vertices
                if p.id in mset:
                    vv = (vv - org) @ axis_angle(ax, val).T + org if kind == "rot" else vv + ax * val
                col = (0.95, 0.55, 0.12) if p.id in mset else ((0.25, 0.45, 0.8) if p.id == j.parent else (0.72, 0.73, 0.75))
                meshes.append((p.id, vv, p.faces, col))
            meshes.append(("__axis__", rod_v, np.asarray(rod.faces), (0.85, 0.1, 0.1)))
            fr = render_colored(f"joint_{jid}_{k}", meshes, (best_v,), which_labels=set(mset) | {j.parent}, frame=frame)
            frames.append(fr[best_v])
        img = _tile(frames, out / f"joint_{jid}.png", [f"{jid}: {t}" for t, _, _ in poses], cols=3)
        motion.append({"joint": jid, "claims": claim_ids, "image": img, "view": best_v, "zoomed": frame is not None, "type": j.type,
                       "parent": j.parent, "parent_label": labels.get(j.parent), "child": j.child, "child_label": labels.get(j.child),
                       "moving_labels": sorted((labels[x] for x in mset), key=lambda s: int(s[1:])), "axis": np.round(ax, 3).tolist(),
                       "axis_words": "vertical" if abs(ax[2]) > 0.9 else ("horizontal" if abs(ax[2]) < 0.2 else "tilted"),
                       "origin_m": np.round(org, 3).tolist(), "limits": j.limits})
    m21 = rec["dims"].get("2.1", {}).get("metrics", {})
    m32 = rec["dims"].get("3.2", {}).get("metrics", {})
    seat = (m21.get("role_predicates") or {}).get("seat_height") or {}
    evid = {"key": key, "labels": table, "overview": overview, "materials": material_img, "clay": clay_img, "motion": motion,
            "facts": {"extent_m": m21.get("extent_m"), "seat_top_m": seat.get("seat_top_m"),
                      "reference_seat_top_range_m": seat.get("reference_range_m"),
                      "mirror_symmetry_iou": m32.get("symmetry_iou"), "reference_mirror_symmetry_iou": m32.get("reference_symmetry_iou")},
            "declares": (design.meta or {}).get("declares", {}), "stamp": rec.get("stamp")}
    (out / "evidence.json").write_text(json.dumps(evid, indent=1))
    keep = {"overview.png", "materials.png", "clay.png"} | {Path(m["image"]).name for m in motion}
    for f in out.glob("*.png"):     # keep only the tiled sheets
        if f.name not in keep:
            f.unlink()
    return evid


def _evidence_one(key):
    os.environ.setdefault("TMPDIR", "/tmp")
    os.environ.setdefault("RENDER_THREADS", "4")
    from ppbench.v2.report import load_design
    from ppbench.v2.evaluate import prepare
    base = RESULTS / TASK
    rec = json.loads((base / "eval" / f"{key}.json").read_text())
    ej = EVID / key / "evidence.json"
    if ej.exists() and json.loads(ej.read_text()).get("stamp") == rec.get("stamp"):
        return key, "cached"
    try:
        task = Task(TASK, snapshot=base / SNAPSHOT)
        d = load_design(task, rec["item"])
        if rec["item"]["kind"] != "reference":
            prepare(d, task.reference())
        if not d.parts:
            return key, "no parts"
        evidence_for(key, rec, d, task)
        return key, "ok"
    except Exception as e:
        import traceback
        (EVID / key).mkdir(parents=True, exist_ok=True)
        (EVID / key / "error.txt").write_text(traceback.format_exc())
        return key, f"error {e!r}"


def build_evidence(workers=24, only=None):
    from concurrent.futures import ProcessPoolExecutor
    keys = sorted(p.stem for p in (RESULTS / TASK / "eval").glob("*.json") if not p.name.startswith(("imagesim", "judge", "vlm")))
    if only:
        keys = [k for k in keys if re.search(only, k)]
    with ProcessPoolExecutor(workers) as ex:
        for k, st in ex.map(_evidence_one, keys):
            if st != "cached":
                print(f"{k:60s} {st}", flush=True)


# ---------------------------------------------------------------- judging

def _b64(path, side=1400):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = BytesIO()
    im.save(buf, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def _ask(client, model, text, images, max_tokens=900):
    content = [{"type": "text", "text": text}] + images
    last = ""
    for _ in range(3):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}],
                                               max_tokens=max_tokens, temperature=0.0)
            last = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", last, re.S)
            if m:
                return json.loads(m.group(0)), last
        except Exception as e:
            last = f"error: {e}"
            time.sleep(5)
    return None, last


def _num(x):
    try:
        return max(0.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _cited(ans, key="evidence"):
    """L# labels cited anywhere in an answer's evidence entries (free text such as "L1 and L2" counts)."""
    out = set()
    for e in (ans or {}).get(key, []) or []:
        txt = json.dumps(e) if isinstance(e, dict) else str(e)
        out |= set(re.findall(r"\bL\d+\b", txt))
    return out


def _table_text(evid, only=None):
    rows = [r for r in evid["labels"] if only is None or r["label"] in only]
    return "\n".join(f"{r['label']}: role={r['role']!r} material={r['material'] or 'undeclared'} "
                     f"box z {r['bbox_min_m'][2]:.2f}-{r['bbox_max_m'][2]:.2f} m" for r in rows)


EVIDENCE_RULE = """Base every judgement ONLY on what the images show and on the facts listed. Cite evidence as
{{"image": "<which image>", "label": "<L#>", "observation": "<what you see>"}}; use only labels that appear in the
images or tables. If the evidence does not show something, say so and score it low."""

KIN_PROMPT = """You are verifying a kinematics claim about a 3D design of {object_name}.
CLAIM {cid}: {claim}
The claim requires: joint type {req_type}, axis {req_axis}, range {req_range}.
The design declares joint {jid}: type={jtype}, axis {axis_words} {axis}, parent {plabel} ({prole}), child {clabel} ({crole}),
origin {origin} m, limits {limits}. Moving with the child: {moving}.
(The rule-based matcher picked this joint as the closest candidate; it may still not realise the claim.)
IMAGE 1 shows that joint alone at three positions from the same camera: orange = moving parts, blue = parent, red rod = axis.
IMAGE 2 is the labelled overview of the whole design. Part table:
{table}
Answer four questions: (a) are the moving parts the part the claim names, and is the parent the part it moves relative to?
(b) do the declared joint type and axis match what the claim requires (a rotation does not realise a slide, a horizontal axis
does not realise a vertical one)? (c) does the motion shown in the three frames realise the claim? (d) is the motion physically
sensible: no part passes through another, the pivot sits where the parts actually meet?
""" + EVIDENCE_RULE + """
Reply with JSON only: {{"parts_match": true/false, "type_axis_match": true/false, "motion_realises_claim": true/false, "physically_sensible": true/false,
"score": <0-10>, "evidence": [...], "reason": "<one sentence>"}}"""

PART_PROMPT = """You are verifying that a 3D design of {object_name} really contains required parts.
For each required part below, the candidate labels are the parts the design itself declared with that role.
Look at IMAGE 1 (labelled role-coloured views) and decide whether the candidates LOOK like that part and sit where that part
belongs on this object. Part table:
{table}
Required parts:
{req}
""" + EVIDENCE_RULE + """
Reply with JSON only: {{"parts": [{{"id": "P1", "present": true/false, "labels": ["L#"], "score": <0-10>, "evidence": [...]}}, ...],
"reason": "<one sentence>"}}"""

ATTR_PROMPT = """You are verifying an attribute claim about a 3D design of {object_name}.
CLAIM {cid}: {claim}
IMAGE 1: {img1}. IMAGE 2: {img2}.
Part table:
{table}
Computed facts: {facts}
Decide whether the design, as shown, satisfies the claim for a real {object_name}.
""" + EVIDENCE_RULE + """
Reply with JSON only: {{"satisfied": true/false, "score": <0-10>, "evidence": [...], "reason": "<one sentence>"}}"""

AESTH_PROMPT = """You are rating a 3D design of {object_name}. IMAGE 1 is a photograph of the target object. IMAGE 2 shows the
design in plain grey clay from four views (condition camera, front-right, front, right side). Ignore rendering quality and colour.
Computed facts about the design: {facts}
Rate each criterion 1-10 using the anchors, and give at least one observation per criterion naming the view it comes from:
- recognisable: is it recognisably the requested object? 1 = no; 5 = the broad category but not the requested object; 10 = unmistakably one.
- resemblance: shape, parts and arrangement compared with the photograph. 1 = unrelated; 4 = major parts missing or misplaced; 7 = close with minor differences; 10 = matches.
- proportion: realistic, deliberate proportions and symmetry. 1 = grossly distorted; 4 = clearly off; 7 = plausible; 10 = convincing.
- coherence: parts meet, consistent style, nothing floating, intersecting or broken. 1 = scattered; 4 = several visible flaws; 7 = mostly clean; 10 = fully resolved.
- overall: overall design quality as a product designer would rate it. 1 = unusable; 4 = poor; 7 = good; 10 = excellent.
Reply with JSON only: {{"recognisable": n, "resemblance": n, "proportion": n, "coherence": n, "overall": n,
"observations": [{{"criterion": "...", "view": "...", "observation": "..."}}], "reason": "<one sentence>"}}"""


def judge_design(client, model, key, task, rec):
    from ppbench.v2.lexicon import Lexicon
    base = RESULTS / TASK
    evid = json.loads((EVID / key / "evidence.json").read_text())
    lex = Lexicon(task.id)
    all_labels = {r["label"] for r in evid["labels"]}
    declares = evid.get("declares") or {}
    has_roles = any(r["role"] for r in evid["labels"])
    out = {"key": key, "model": model, "kinematics": [], "parts": [], "attributes": [], "aesthetics": None}

    # kinematics claims: one call per chosen joint (covers every claim that joint is the best candidate for)
    motion_by_claim = {}
    for m in evid["motion"]:
        for cid in m["claims"]:
            motion_by_claim[cid] = m
    for k in task.kinematics:
        m = motion_by_claim.get(k["id"])
        if m is None:
            if rec["dims"].get("2.2", {}).get("status") == "skipped":
                out["kinematics"].append({"id": k["id"], "score": None, "status": "skipped (no joints declared)"})
            else:
                out["kinematics"].append({"id": k["id"], "score": 0.0, "status": "no candidate joint"})
            continue
        claim = f"{k['moving_part']} {k['motion']} relative to {k['relative_to']} ({k['joint_type']}, {k['axis']}, range: {k.get('range')})"
        roles = {r["label"]: r["role"] for r in evid["labels"]}
        text = KIN_PROMPT.format(object_name=task.name, cid=k["id"], claim=claim, req_type=k["joint_type"], req_axis=k["axis"], req_range=k.get("range"),
                                 jid=m["joint"], jtype=m["type"], plabel=m["parent_label"],
                                 prole=roles.get(m["parent_label"]), clabel=m["child_label"], crole=roles.get(m["child_label"]),
                                 axis_words=m["axis_words"], axis=m["axis"], origin=m["origin_m"], limits=m["limits"],
                                 moving=", ".join(m["moving_labels"]), table=_table_text(evid))
        ans, raw = _ask(client, model, text, [_b64(m["image"], 1500), _b64(evid["overview"], 1100)])
        allowed = set(m["moving_labels"]) | {m["parent_label"]}
        cited = _cited(ans)
        grounded = bool(cited & allowed) and cited <= all_labels
        score = _num((ans or {}).get("score")) / 10 if grounded else 0.0
        # the judge's own verdicts bound its score: a joint it says does not realise the claim cannot score high
        if grounded and not all((ans or {}).get(f) is True for f in ("parts_match", "type_axis_match", "motion_realises_claim")):
            score = min(score, 0.3)
        out["kinematics"].append({"id": k["id"], "joint": m["joint"], "score": max(0.0, min(1.0, score)), "grounded": grounded,
                                  "answer": ans, "raw": None if ans else raw[:300], "status": "judged" if grounded else "ungrounded"})

    # required parts: one call for all P claims, candidates are the declared parts of each role
    if has_roles:
        req, cand = [], {}
        for pc in task.required_parts:
            labs = [r["label"] for r in evid["labels"] if lex.satisfies(pc["part"], r["canonical_role"])]
            cand[pc["id"]] = labs
            req.append(f"{pc['id']} {pc['part']} ({pc['why']}): candidates {', '.join(labs) if labs else 'none declared'}")
        ans, raw = _ask(client, model, PART_PROMPT.format(object_name=task.name, table=_table_text(evid), req="\n".join(req)), [_b64(evid["overview"], 1400)],
                        max_tokens=2000)
        got = {str(p.get("id")).split()[0]: p for p in (ans or {}).get("parts", []) if isinstance(p, dict) and p.get("id")}
        out["parts_raw"] = None if got else raw[:1500]
        for pc in task.required_parts:
            g = got.get(pc["id"]) or {}
            labs = set(re.findall(r"\bL\d+\b", json.dumps(g.get("labels", [])))) | _cited(g)
            grounded = bool(cand[pc["id"]]) and bool(labs & set(cand[pc["id"]])) and labs <= all_labels
            score = _num(g.get("score")) / 10 if grounded and g.get("present") is True else 0.0
            out["parts"].append({"id": pc["id"], "score": max(0.0, min(1.0, score)), "grounded": grounded, "answer": g,
                                 "status": "no candidate" if not cand[pc["id"]] else ("judged" if grounded else "ungrounded")})
    else:
        out["parts"] = [{"id": pc["id"], "score": None, "status": "skipped (no roles)"} for pc in task.required_parts]

    # attribute claims
    for a in task.attributes:
        if not has_roles:
            out["attributes"].append({"id": a["id"], "score": None, "status": "skipped (no roles)"})
            continue
        if a.get("kind") == "material":
            imgs = [_b64(evid["materials"], 1300), _b64(evid["overview"], 1100)]
            i1, i2 = "renders coloured by declared material class (legend: plastic light grey, heavy metal dark grey, light metal silver, wood brown, foam peach, textile purple, rubber black, glass cyan, magenta undeclared)", "labelled role-coloured overview"
        else:
            imgs = [_b64(evid["overview"], 1300), _b64(evid["clay"], 1100)]
            i1, i2 = "labelled role-coloured overview", "neutral clay views"
        motion_txt = "; ".join(f"{m['joint']} {m['type']} {m['axis_words']} moving {','.join(m['moving_labels'])}" for m in evid["motion"]) or "none matched"
        facts = dict(evid["facts"], declared_moving_joints=motion_txt)
        ans, raw = _ask(client, model, ATTR_PROMPT.format(object_name=task.name, cid=a["id"], claim=a["statement"], img1=i1, img2=i2,
                                                         table=_table_text(evid), facts=json.dumps(facts)), imgs)
        cited = _cited(ans)
        grounded = bool(cited) and cited <= all_labels
        score = _num((ans or {}).get("score")) / 10 if grounded else 0.0
        if grounded and (ans or {}).get("satisfied") is not True:
            score = min(score, 0.3)
        out["attributes"].append({"id": a["id"], "score": max(0.0, min(1.0, score)), "grounded": grounded, "answer": ans,
                                  "raw": None if ans else raw[:300], "status": "judged" if grounded else "ungrounded"})

    # design quality
    cond = task.condition("name_only+image")["image"]["path"]
    ans, raw = _ask(client, model, AESTH_PROMPT.format(object_name=task.name, facts=json.dumps(evid["facts"])), [_b64(cond, 700), _b64(evid["clay"], 1100)])
    obs = [o for o in (ans or {}).get("observations", []) if isinstance(o, dict) and o.get("view")]
    crit = ("recognisable", "resemblance", "proportion", "coherence", "overall")
    sc = {}
    for c in crit:
        try:
            sc[c] = max(0.0, min(1.0, (float((ans or {}).get(c)) - 1) / 9))
        except (TypeError, ValueError):
            sc[c] = None
    grounded = len({o.get("criterion") for o in obs}) >= 3
    out["aesthetics"] = {"scores": sc if grounded else {c: 0.0 for c in crit}, "grounded": grounded, "answer": ans,
                         "raw": None if ans else raw[:300]}
    return out


def run_judge(model, base_url, tag, workers=12, only=None):
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    base = RESULTS / TASK
    task = Task(TASK, snapshot=base / SNAPSHOT)
    client = OpenAI(base_url=base_url, api_key="local", timeout=900)
    outdir = base / "eval" / f"vlm_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    keys = sorted(p.parent.name for p in EVID.glob("*/evidence.json"))
    if only:
        keys = [k for k in keys if re.search(only, k)]

    def one(key):
        o = outdir / f"{key}.json"
        rec = json.loads((base / "eval" / f"{key}.json").read_text())
        if o.exists() and json.loads(o.read_text()).get("stamp") == rec.get("stamp"):
            return key, "cached"
        try:
            res = judge_design(client, model, key, task, rec)
            res["stamp"] = rec.get("stamp")
            o.write_text(json.dumps(res, indent=1))
            return key, "ok"
        except Exception as e:
            return key, f"error {e!r}"
    with ThreadPoolExecutor(workers) as ex:
        for k, st in ex.map(one, keys):
            print(f"{tag} {k:58s} {st}", flush=True)


def summarize(only_judges=None):
    """Per design and judge: mean score per check family and per-claim verdicts with the evidence cited;
    rank agreement (Spearman) between the first two judges per family."""
    base = RESULTS / TASK
    out = {}
    judges = sorted(p.name[4:] for p in (base / "eval").glob("vlm_*") if p.is_dir())
    if only_judges:
        judges = [j for j in judges if j in only_judges]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def claim(x):
        a = x.get("answer") if isinstance(x.get("answer"), dict) else {}
        ev = [e for e in a.get("evidence", []) or [] if isinstance(e, dict)][:4]
        return {"score": x.get("score"), "status": x.get("status"), "joint": x.get("joint"), "reason": a.get("reason"),
                "verdict": {f: a.get(f) for f in ("parts_match", "type_axis_match", "motion_realises_claim", "physically_sensible",
                                                  "present", "satisfied") if f in a},
                "evidence": [{"image": str(e.get("image", ""))[:40], "label": str(e.get("label", ""))[:40],
                              "observation": str(e.get("observation", ""))[:200]} for e in ev]}
    for tag in judges:
        for f in (base / "eval" / f"vlm_{tag}").glob("*.json"):
            r = json.loads(f.read_text())
            A = r.get("aesthetics") or {}
            claims = r["kinematics"] + r["parts"] + r["attributes"]
            out.setdefault(r["key"], {})[tag] = {
                "model": r["model"],
                "kinematics": mean([k["score"] for k in r["kinematics"]]),
                "parts": mean([p["score"] for p in r["parts"]]),
                "attributes": mean([a["score"] for a in r["attributes"]]),
                "aesthetics": A.get("scores", {}),
                "aesthetics_reason": (A.get("answer") or {}).get("reason"),
                "aesthetics_observations": [o for o in (A.get("answer") or {}).get("observations", []) if isinstance(o, dict)][:6],
                "claims": {x["id"]: claim(x) for x in claims},
                "ungrounded": sum(1 for x in claims if x.get("status") == "ungrounded") + (0 if A.get("grounded", True) else 1)}
    agree = {}
    if len(judges) >= 2:
        a, b = judges[:2]
        for fam in ("kinematics", "parts", "attributes", "overall"):
            get = lambda v, t: v[t]["aesthetics"].get("overall") if fam == "overall" else v[t][fam]
            pairs = [(get(v, a), get(v, b)) for v in out.values() if a in v and b in v]
            pairs = [p for p in pairs if None not in p]
            if len(pairs) >= 3:
                from scipy.stats import spearmanr
                agree[fam] = {"n": len(pairs), "spearman": float(spearmanr(*zip(*pairs)).statistic)}
    (base / "eval" / "vlm_summary.json").write_text(json.dumps({"judges": judges, "designs": out, "agreement": agree}, indent=1))
    print(json.dumps({"judges": judges, "n_designs": len(out), "agreement": agree}, indent=1))


def main():
    global TASK, EVID, SNAPSHOT
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evidence", "judge", "summarize"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--tag")
    ap.add_argument("--judges", help="summarize: comma-separated judge tags to include (default: all)")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--snapshot", default=SNAPSHOT, help="task snapshot filename under results/v2/<task>")
    ns = ap.parse_args()
    TASK, SNAPSHOT = ns.task, ns.snapshot
    EVID = RESULTS / TASK / "evidence"
    if ns.cmd == "evidence":
        build_evidence(ns.workers, ns.only)
    elif ns.cmd == "judge":
        run_judge(ns.model, ns.base_url, ns.tag, min(ns.workers, 16), ns.only)
    else:
        summarize(ns.judges.split(",") if ns.judges else None)


if __name__ == "__main__":
    main()
