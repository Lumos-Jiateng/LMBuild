"""Level A (affordance): the objects a score is computed on.

A.1 Geometry, A.2 Parts and A.3 Kinematics each ask one question per functional component of the object,
e.g. for an off-road vehicle:

    A.1  are the wheels appropriately shaped, sized, oriented and positioned to contact the ground and roll?
    A.2  does the vehicle contain the components driving needs: wheels, axles, steering, supports?
    A.3  can the wheels rotate about their axles and, where required, steer along the intended axes?

Where each ingredient comes from (nothing below is written per object):

* which components and which motions: the task's own Wikipedia-cited claims (required parts P,
  subsystem parts F, kinematics K), canonicalised with the task lexicon;
* what "appropriate" means: the population of real instances of the category that the benchmark
  already holds (the task's reference + the extended set's references of the same Wikidata class,
  same medium), plus typical dimensions quoted from Wikipedia where the article gives them;
* which part of a design is which component: a visual judge that sees one highlighted segment at a
  time and never sees the design's declared names (afford/ground.py), so a box called "wheel" is not
  a wheel and a well-built wheel called "round thing" is one.

This module: task components and claims, the items to score, loading, segmentation, descriptors.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from ppbench.v2.design import Design, Joint, Part
from ppbench.v2.lexicon import Lexicon, tokens
from ppbench.v2.task import RESULTS, ROOT, Task, to_zup

SPEC = "A-v1"
CACHE = RESULTS / "_afford"
from ppbench.v2.task import CORE as CORE_JSON, BENCH
EXT_JSON = BENCH / "extended_v2.json"   # not shipped; only needed to rebuild the population sheets
LEGO_DATASETS = ("BrickNet", "BrickComposer")
MAX_SEGMENTS = {"tool": 60, "external": 40, "brick": 24, "population": 60}
POP_CAP = 12                       # real instances per task (the reference first)
# a K claim's partner phrase that is the object itself or its surroundings names a frame, not a part
FRAME_WORDS = {"ground", "floor", "wall", "each", "other", "water", "air", "road", "surface", "user", "person",
               "sofa", "bed", "table", "desk", "ceiling"}


# ------------------------------------------------------------------ tasks
_TASKS: dict = {}


def task(tid) -> Task:
    if tid not in _TASKS:
        _TASKS[tid] = Task(tid, snapshot=RESULTS / tid / "task_snapshot_core.json")
    return _TASKS[tid]


def task_ids():
    return [t["task_id"] for t in json.loads(CORE_JSON.read_text())]


def medium(t: Task) -> str:
    ds = (t.raw.get("reference") or {}).get("dataset") or ""
    return "lego" if ds.startswith(LEGO_DATASETS) else "real"


def _self_phrases(t: Task):
    """The object's own names as token tuples ("off-road vehicle", "jeep", "four-wheel drive", ...)."""
    o = t.raw.get("object") or {}
    out = set()
    for s in [o.get("name", ""), *(o.get("aliases") or []), t.id.replace("_", " ")]:
        tk = tokens(s)
        if tk:
            out.add(tuple(tk))
    return out


def components(t: Task, lex: Lexicon):
    """The functional components the claims name, one entry per canonical role.

    P parts, F subsystem parts and K moving parts are components. A K claim's `relative_to` is kept as a
    frame (it tells A.3 what must stay put) but is not a component on its own. Phrases that are the object
    itself ("dumpster rotates relative to ground") or its surroundings are frames, not parts."""
    selfp = _self_phrases(t)
    out, by = [], {}

    def add(phrase, claim, why=None, kind="P"):
        tk = tokens(phrase)
        if not tk or tuple(tk) in selfp or set(tk) <= FRAME_WORDS:
            return None
        c = lex.canon(phrase) or " ".join(tk)
        if c not in by:
            by[c] = {"id": f"C{len(out) + 1}", "canon": c, "name": _display(phrase), "phrases": [], "claims": [],
                     "why": why, "kinds": []}
            out.append(by[c])
        e = by[c]
        if phrase not in e["phrases"]:
            e["phrases"].append(phrase)
        if claim not in e["claims"]:
            e["claims"].append(claim)
        if kind not in e["kinds"]:
            e["kinds"].append(kind)
        if why and not e["why"]:
            e["why"] = why
        return e
    for p in t.required_parts:
        add(p["part"], p["id"], p.get("why"), "P")
    for f in t.subsystems:
        for n in f.get("parts", []):
            add(n, f["id"], None, "F")
    for k in t.kinematics:
        add(k["moving_part"], k["id"], None, "K")
    return _merge_alternatives(out)


def _alts(c):
    """Token tuples a component answers to: its phrases, each 'X or Y' alternative, a parenthetical acronym."""
    al = set()
    for ph in c["phrases"]:
        for a in re.split(r"\s+or\s+|/", re.sub(r"\(([^)]*)\)", r" or \1", ph)):
            tk = tokens(a)
            if tk:
                al.add(tuple(tk))
    return al


def _merge_alternatives(comps):
    """"turntable or stirrer" and "turntable" are one component, as are "central processing unit (CPU)" and
    "CPU": a judge offered both would split its answer between two names for one thing."""
    merged = []
    for c in comps:
        host = next((m for m in merged if (tuple(tokens(c["name"])) in _alts(m)) or (tuple(tokens(m["name"])) in _alts(c))), None)
        if host is None:
            merged.append(dict(c))
            continue
        for k in ("phrases", "claims", "kinds"):
            host[k] = host[k] + [x for x in c[k] if x not in host[k]]
        host["why"] = host["why"] or c["why"]
        host.setdefault("merged", []).append(c["canon"])
    for k, c in enumerate(merged):
        c["id"] = f"C{k + 1}"
    return merged


def _display(phrase):
    s = re.sub(r"\([^)]*\)", "", phrase).strip()
    return " ".join(w if (w.isupper() and len(w) > 1) else w.lower() for w in s.split())


def kin_claims(t: Task, lex: Lexicon, comps):
    by = {}
    for c in comps:
        for cn in [c["canon"], *c.get("merged", [])]:
            by[cn] = c
    out = []
    for k in t.kinematics:
        mc = lex.canon(k["moving_part"]) or " ".join(tokens(k["moving_part"]))
        rc = lex.canon(k.get("relative_to") or "") or " ".join(tokens(k.get("relative_to") or ""))
        out.append({"id": k["id"], "moving": by.get(mc, {}).get("id"), "moving_phrase": k["moving_part"],
                    "relative": by.get(rc, {}).get("id"), "relative_phrase": k.get("relative_to"),
                    "joint_type": k.get("joint_type"), "motion": k.get("motion"), "axis_text": k.get("axis"),
                    "range_text": k.get("range")})
    return out


# ------------------------------------------------------------------ items
def items_tool(tid):
    from ppbench.v2 import report as R
    return [dict(it, family="llm") for it in R.discover_core(tid) if it.get("seed") == 0]


def items_external(tid):
    from ppbench.v2 import report as R
    out = []
    for pat in ("*__ext__name_only__s0", "*__ext__image__s0"):
        out += [dict(it, family="domain") for it in R.discover_core_externals(tid, pat)]
    return out


def load(t: Task, item) -> Design:
    """The design as scored: tool runs are metric with a declared front; unitless outputs get the oracle
    scale of Level 1 (their absolute size is then not a result and is not scored)."""
    kind = item["kind"]
    if kind == "population":
        return pop_design(item)
    from ppbench.v2 import report as R
    d = R.load_design(t, item)
    if kind == "external":
        from ppbench.v2.spec_v33 import prepare_external
        prepare_external(d, t)
    return d


def metric(d: Design) -> bool:
    return (d.meta or {}).get("scale_mode") != "unitless" and "oracle_scale" not in (d.meta or {})


# ------------------------------------------------------------------ population (real instances)
def _ext_entries(t: Task):
    q = (t.raw.get("object") or {}).get("wikidata")
    ents = [e for e in _ext() if e["object"].get("wikidata") == q]
    return ents


_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        # The release ships the population sheets (results/v2/<task>/afford/sheet_v37.json) instead of the extended set;
        # without extended_v2.json only the reference itself is a population item, which is all scoring reads.
        _EXT = json.loads(EXT_JSON.read_text()) if EXT_JSON.exists() else []
    return _EXT


def population_items(t: Task, cap=POP_CAP):
    """The task's reference first, then real instances of the same category and medium."""
    med = medium(t)
    out = [{"kind": "population", "key": "pop__reference", "system": "reference", "family": "real",
            "task": t.id, "ref": t.raw["reference"], "role": "reference"}]
    seen = {t.raw["reference"].get("object_id")}
    ents = [e for e in _ext_entries(t) if (medium_of(e["reference"]) == med)]
    ents.sort(key=lambda e: (0 if e["reference"].get("dataset") in ("Artiverse", "Fusion 360 Gallery Assembly") else 1,
                             e["reference"].get("object_id")))
    for e in ents:
        if len(out) >= cap:
            break
        oid = e["reference"].get("object_id")
        if oid in seen or not _pop_loadable(e["reference"]):
            continue
        seen.add(oid)
        out.append({"kind": "population", "key": f"pop__{oid}", "system": "real", "family": "real", "task": t.id,
                    "ref": e["reference"], "role": "extended"})
    if len(out) < 4:   # few extended instances: the task's own alternates (geometry only; Artiverse and LEGO)
        for alt in (t.raw["reference"].get("alternates") or []):
            if len(out) >= cap:
                break
            r = _alternate_ref(alt)
            if r and alt not in seen:
                seen.add(alt)
                out.append({"kind": "population", "key": f"pop__{alt}", "system": "real", "family": "real",
                            "task": t.id, "ref": r, "role": "alternate"})
    return out


def medium_of(ref) -> str:
    return "lego" if (ref.get("dataset") or "").startswith(LEGO_DATASETS) else "real"


def _pop_loadable(ref):
    if medium_of(ref) == "lego":
        return any(c.get("format") == "LDR" and (ROOT / c["path"]).exists() for c in (ref.get("cad_files") or []))
    return bool(ref.get("reference_glb")) and (ROOT / ref["reference_glb"]).exists()


def _alternate_ref(oid: str):
    m = re.match(r"lego-sft-(\d+)$", oid)
    if m:
        p = next(iter(sorted(BENCH.glob(f"tasks/*/reference/{oid}.ldr"))), BENCH / "missing" / f"{oid}.ldr")
        if p.exists():
            return {"object_id": oid, "dataset": "BrickNet corpus (alternate)", "cad_files": [{"format": "LDR", "path": str(p.relative_to(ROOT))}],
                    "articulation": {"joints": []}}
    m = re.match(r"artiverse-(.+?)-(fpModel|3dfModel|texverse|wss|[A-Za-z0-9]+)-([0-9a-f\-]+)$", oid)
    if m:
        cat, src, short = m.groups()
        for d in (BENCH / "assets" / "artiverse" / cat / src).glob(f"{short}*"):
            if (d / f"{d.name}.segmented.glb").exists():
                return {"object_id": oid, "dataset": "Artiverse (alternate)", "artiverse_dir": str(d)}
    return None


def pop_design(item) -> Design:
    """A real instance as a Design: parts with their dataset labels as roles, and its annotated joints.

    The joint parent is taken from the annotation when it names one (Artiverse `base`); otherwise it is
    the part that holds the child at the joint origin -- the nearest part to the origin that does not move
    with the child. Before 2026-09-22 reference_design dropped every joint without an annotated parent,
    which silently removed all Fusion and PartNeXt joints from the reference arm."""
    ref = item["ref"]
    if medium_of(ref) == "lego":
        return _lego_pop(ref)
    if ref.get("artiverse_dir"):
        return _artiverse_dir_design(Path(ref["artiverse_dir"]))

    art = dict(ref.get("articulation") or {})
    art["joints"] = [j for j in art.get("joints") or [] if j.get("axis") is not None and j.get("origin") is not None]
    ref = {**ref, "articulation": art}

    class _T:
        raw = {"reference": ref}
    from ppbench.v2.task import load_reference
    r = load_reference(_T)
    parts = [Part(f"n{p['node']:03d}", p["vertices"], p["faces"], p["role_raw"], p.get("material"),
                  {"kind": "reference", "node": p["node"]}) for p in r["parts"]]
    joints = []
    for j in r["joints"]:
        kids = list(j.get("child_nodes") or [])
        moving = set(j.get("moves_nodes") or kids)
        if not kids:
            continue
        org = np.asarray(j["origin"], float)

        def dist(n):
            return float(np.min(np.linalg.norm(r["parts"][n]["vertices"] - org, axis=1)))
        child = min(kids, key=dist)
        cands = [n for n in (j.get("parent_nodes") or []) if n not in moving] or \
                [p["node"] for p in r["parts"] if p["node"] not in moving]
        if not cands:
            continue
        parent = min(cands, key=dist)
        t = j["type"]
        lim = None
        if t in ("revolute",) and j.get("range") and None not in j["range"]:
            lim = [float(j["range"][0]), float(j["range"][1])]
        if t == "prismatic":
            pr = j.get("prismatic_range") or j.get("range")
            if pr and None not in pr:
                lim = [float(pr[0]), float(pr[1])]
        if t == "cylindrical" and j.get("prismatic_range"):
            lim = list(j["prismatic_range"])
        joints.append(Joint(j["id"], {"screw": "cylindrical"}.get(t, t), f"n{parent:03d}", f"n{child:03d}",
                            j["axis"], org.tolist(), lim))
        # children that move together with the first one are joined to it rigidly
        for n in kids:
            if n != child:
                joints.append(Joint(f"{j['id']}_rig{n}", "fixed", f"n{child:03d}", f"n{n:03d}", None, org.tolist(), None))
    return Design("reference", "ref", parts, joints, None,
                  {"scale_mode": "metric" if ref.get("dataset") != "PartNeXt" else "prior_scaled",
                   "dataset": ref.get("dataset"), "object_id": ref.get("object_id")})


def _artiverse_dir_design(d: Path) -> Design:
    from ppbench.v2 import glb
    nodes = glb.read_nodes(d / f"{d.name}.segmented.glb")
    allv = np.vstack([to_zup(n["vertices"]) for n in nodes if len(n["vertices"])])
    lift = -allv[:, 2].min()
    parts, pid_of = [], {}
    for i, n in enumerate(nodes):
        if not len(n["faces"]):
            continue
        v = to_zup(n["vertices"])
        v[:, 2] += lift
        pid = (n.get("extras") or {}).get("id")
        label = (n.get("extras") or {}).get("label", n["name"])
        parts.append(Part(f"n{i:03d}", v, n["faces"], label, None, {"kind": "reference", "node": i}))
        pid_of.setdefault(pid, []).append(f"n{i:03d}")
    art = json.loads((d / f"{d.name}.articulations.json").read_text())
    joints = []
    for a in art.get("articulations", []):
        kids = pid_of.get(a.get("pid")) or []
        base = [x for b in (a.get("base") or []) for x in (pid_of.get(b) or [])]
        if not kids or not base or a.get("type") in ("free",) or a.get("axis") is None:
            continue
        ax = to_zup(a["axis"])
        org = to_zup(a["origin"]) + np.array([0, 0, lift]) if a.get("origin") is not None else \
            np.vstack([p.vertices for p in parts if p.id in kids]).mean(0)
        t = {"screw": "cylindrical"}.get(a["type"], a["type"])
        lim = None
        if t == "revolute" and "rangeMin" in a:
            lim = [float(a["rangeMin"]), float(a["rangeMax"])]
        if t == "prismatic" and "rangeMin" in a:
            lim = [float(a["rangeMin"]), float(a["rangeMax"])]
        joints.append(Joint(f"a{a['pid']}", t, base[0], kids[0], ax.tolist(), np.asarray(org).tolist(), lim))
        for k in kids[1:]:
            joints.append(Joint(f"a{a['pid']}_rig{k}", "fixed", kids[0], k, None, np.asarray(org).tolist(), None))
    return Design("reference", "ref", parts, joints, None, {"scale_mode": "metric", "dataset": "Artiverse"})


def _lego_pop(ref) -> Design:
    """An LDraw model with its connector-derived joints, in the frame ldr_design uses (LDU -> m, Z up)."""
    from ppbench.v2 import external
    from ppbench.core.partlib import PartLib
    cad = [c for c in ref.get("cad_files") or [] if c.get("format") == "LDR"][0]
    d = external.ldr_design(ROOT / cad["path"], "reference")
    lib = PartLib()
    V0 = None
    for p in d.parts:
        p.role = lib.name(p.source.get("stem", "")) or None
    # ldr_design dropped the ground offset: recover it from the raw transform of the first part
    from ppbench.core.canon import resolve
    from ppbench.core.ir import Design as LDesign
    ld = resolve(LDesign.from_ldr((ROOT / cad["path"]).read_text(), source="ref", design_id="r"), lib)
    raw_z = []
    from ppbench import config as C
    from ppbench.core.plyio import read_ply
    for stem, T in zip(ld.stems, ld.poses):
        ply = C.INSET / f"{stem}.ply"
        if ply.exists():
            v, _ = read_ply(ply)
            vw = ((v @ T[:3, :3].T + T[:3, 3]) * external.LDU_M) @ external.LDRAW_TO_WORLD.T
            raw_z.append(vw[:, 2].min())
    z0 = min(raw_z) if raw_z else 0.0
    ids = {p.id for p in d.parts}
    joints = []
    for j in (ref.get("articulation") or {}).get("joints", []):
        kids = [f"brick_{n:03d}" for n in (j.get("child_nodes") or []) if f"brick_{n:03d}" in ids]
        moving = {f"brick_{n:03d}" for n in (j.get("moves_nodes") or [])} | set(kids)
        if not kids:
            continue
        org = (np.asarray(j["origin"], float) * external.LDU_M) @ external.LDRAW_TO_WORLD.T - np.array([0, 0, z0])
        ax = np.asarray(j["axis"], float) @ external.LDRAW_TO_WORLD.T
        byid = {p.id: p for p in d.parts}

        def dist(pid):
            return float(np.min(np.linalg.norm(byid[pid].vertices - org, axis=1)))
        cands = [p.id for p in d.parts if p.id not in moving]
        if not cands:
            continue
        parent = min(cands, key=dist)
        child = min(kids, key=dist)
        t = j["type"]
        lim = [float(j["range"][0]), float(j["range"][1])] if t == "revolute" and j.get("range") else None
        if t == "prismatic" and j.get("range"):
            lim = [float(x) * external.LDU_M for x in j["range"]]
        joints.append(Joint(j["id"], t, parent, child, ax.tolist(), org.tolist(), lim))
        for k in sorted(moving - {child}):
            joints.append(Joint(f"{j['id']}_rig{k}", "fixed", child, k, None, org.tolist(), None))
    d.joints = joints
    d.meta.update({"dataset": ref.get("dataset"), "object_id": ref.get("object_id")})
    return d


# ------------------------------------------------------------------ segmentation
def segments(d: Design, item) -> list[list[int]]:
    """Groups of part indices a judge is shown one at a time.

    One part is one segment, so four wheels are four segments and can be counted. Only a design with
    more parts than a picture can separate is grouped, by Ward clustering of part centres: brick models
    (LDraw outputs and LEGO references), where one brick is rarely a component on its own."""
    n = len(d.parts)
    if n == 0:
        return []
    brick = all((p.source or {}).get("kind") == "ldraw" for p in d.parts)
    cap = MAX_SEGMENTS["brick"] if brick and item["kind"] != "tool" else MAX_SEGMENTS.get(item["kind"], 40)
    if n <= cap:
        return [[i] for i in range(n)]
    from scipy.cluster.hierarchy import fcluster, linkage
    C = np.array([(p.vertices.min(0) + p.vertices.max(0)) / 2 for p in d.parts])
    lab = fcluster(linkage(C, "ward"), t=cap, criterion="maxclust")
    return [list(np.nonzero(lab == k)[0]) for k in sorted(set(lab))]


# ------------------------------------------------------------------ descriptors
def sample(parts, n=3000, seed=0):
    import trimesh
    Vs, Fs, off = [], [], 0
    for p in parts:
        Vs.append(np.asarray(p.vertices, float))
        Fs.append(np.asarray(p.faces, np.int64) + off)
        off += len(p.vertices)
    if not Vs:
        return np.zeros((0, 3))
    m = trimesh.Trimesh(np.vstack(Vs), np.vstack(Fs), process=False)
    if m.area <= 0:
        return np.vstack(Vs)
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(pts)


def pca_frame(P):
    c = (P.min(0) + P.max(0)) / 2
    X = P - P.mean(0)
    w, V = np.linalg.eigh(np.cov(X.T) + 1e-18 * np.eye(3))
    proj = X @ V
    ext = proj.max(0) - proj.min(0)
    o = np.argsort(-ext)
    return V[:, o].T, ext[o], c     # axes (rows) sorted by extent, extents, bbox centre


def roundness(P, axes):
    """Best surface-of-revolution score about one of the three frame axes, and that axis index."""
    X = P - P.mean(0)
    best = (0.0, 2)
    for k in range(3):
        a, u, v = axes[k], axes[(k + 1) % 3], axes[(k + 2) % 3]
        x, y = X @ u, X @ v
        ang = np.arctan2(y, x)
        r = np.hypot(x, y)
        bins = np.floor((ang + np.pi) / (2 * np.pi) * 36).astype(int).clip(0, 35)
        mx = np.zeros(36)
        np.maximum.at(mx, bins, r)
        if mx.mean() <= 0 or (mx == 0).sum() > 3:
            continue
        s = float(max(0.0, 1 - (mx.std() / mx.mean()) / 0.2))
        if s > best[0]:
            best = (s, k)
    return best


def object_stats(d: Design):
    V = np.vstack([p.vertices for p in d.parts])
    lo, hi = V.min(0), V.max(0)
    ext = hi - lo
    horiz = sorted([float(ext[0]), float(ext[1])], reverse=True)
    return {"lo": lo, "hi": hi, "diag": float(np.linalg.norm(ext)), "height": float(ext[2]),
            "long": horiz[0], "wide": horiz[1], "fp_centre": (lo[:2] + hi[:2]) / 2,
            "fp_half_diag": float(np.linalg.norm(ext[:2]) / 2) + 1e-12}


def instance_desc(parts, obj, is_metric=True):
    """Scale-free descriptors of one component instance (+ absolute size when the design has units)."""
    P = sample(parts, 2500)
    if len(P) < 20:
        return None
    axes, ext, c = pca_frame(P)
    e1 = max(float(ext[0]), 1e-9)
    rnd, rk = roundness(P, axes)
    lo, hi = P.min(0), P.max(0)
    H = max(obj["height"], 1e-9)
    d = {"rel_size": e1 / max(obj["diag"], 1e-9),
         "ratio_mid": float(ext[1]) / e1, "ratio_min": float(ext[2]) / e1,
         "roundness": rnd,
         "sym_axis_z": float(abs(axes[rk][2])),
         "long_axis_z": float(abs(axes[0][2])), "thin_axis_z": float(abs(axes[2][2])),
         "height_pos": float((c[2] - obj["lo"][2]) / H),
         "ground": float(lo[2] - obj["lo"][2] <= 0.02 * H + 1e-9),
         "radial": float(np.linalg.norm(c[:2] - obj["fp_centre"]) / obj["fp_half_diag"])}
    if is_metric:
        d["abs_size"] = e1
    d["_axes"] = axes.tolist()
    d["_sym_axis"] = axes[rk].tolist()
    d["_centre"] = c.tolist()
    d["_ext"] = ext.tolist()
    return d


ASPECTS = {"size": ["rel_size", "abs_size"], "shape": ["ratio_mid", "ratio_min", "roundness"],
           "orientation": ["sym_axis_z", "long_axis_z", "thin_axis_z"],
           "position": ["height_pos", "ground", "radial"]}
LOG_DESC = {"rel_size", "abs_size", "ratio_mid", "ratio_min"}
