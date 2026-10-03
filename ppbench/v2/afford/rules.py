"""Level 2 (Affordance), spec v3.7: 2.1 Geometry, 2.2 Parts, 2.3 Kinematics -- rule-based, no VLM anywhere.

History. v3.6 (2026-09-22 afternoon) replaced the v3.2 layout and the VLM-grounded Level A draft. The owner's
review of v3.6 the same evening asked for four changes, which are v3.7:

* 2.1 is geometry. A design that retrieves catalogue parts must not lose points for not creating, and a design
  that creates good parts must not lose points either -- so creation is reported, never scored, and Tier C
  (creation only) is scored with the same rule;
* 2.3 is bound to concrete joints: the reference object's own annotated joints where it has them (Artiverse,
  and the LEGO references' connector joints) plus the claimed motions of *visible* parts, and every joint the
  design declares is checked for being a reasonable one;
* 2.2 is easier on the design: a part that is there but not named is found by geometry, and an internal module
  that is not named is found by looking inside -- a part no ray from outside can reach.

Where each ingredient comes from (nothing below is written per object):

* components and motions: the task's current verified claims in core_v2.json (P, F, K) -- the agents never saw
  them -- plus, as motion targets, the reference's annotated joints;
* which part of a design is which component: its declared role through the task lexicon (Grounder); a part
  whose role matches nothing, or a design with no roles, by geometry against the reference's labelled parts;
* what "right" means: the reference object and the population of real instances of the category, grounded by
  their dataset labels only.

    2.1 = 0.6 shape + 0.4 part geometry                       (role-less designs: shape only)
    2.2 = 2/3 visible components + 1/3 internal components    (count-aware; floating = 1/2; unnamed interior = 1/2)
    2.3 = F1(target recall, joint precision)

    python -m ppbench.v2.afford.rules sheets [--tasks all]
    python -m ppbench.v2.afford.rules score  [--tasks all] [--workers 96] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

from ppbench.v2.afford import calib, core, kin
from ppbench.v2.lexicon import Lexicon
from ppbench.v2.lexicon import tokens as _tokens
from ppbench.v2.task import RESULTS, Task

SPEC = "v3.7"
OUT = "eval_v37"
SHEET = "sheet_v37.json"
N_POINTS = 20000
TAU = 0.025                  # of the bounding diagonal, as v3.2 2.1
TAU_PART = 0.05              # a component's own geometry: parts are small, place errors count at twice the tolerance
TAU_GROUND = 0.05            # geometric grounding
W21 = {"shape": 0.6, "parts": 0.4}
W22 = {"visible": 2 / 3, "internal": 1 / 3}
UNNAMED_INTERIOR = 0.5       # an interior module whose identity the design does not state
UNCLAIMED_JOINT = 0.5        # a physically sound joint on no target motion (a door's lever handle when unclaimed)
INTERIOR_EXPOSED_MAX = 0.05  # a part is interior when at most 5% of its boundary cells touch the outside air
MAX_SLOTS = 8
VISIBILITY_MIN_GROUNDED = 0.3
# a claimed phrase headed by one of these names a function carried by parts, not a part: "steering system" is
# present when a part names steering (the steering wheel, the steering column)
ABSTRACT_HEADS = {"system", "mechanism", "control", "interface", "electronic", "hardware", "source", "drive"}
SPELLING = {"tyre": "tire", "castor": "caster", "aluminium": "aluminum", "colour": "color", "centre": "center"}
ROUND_AXLE_WORDS = {"axle", "shaft", "spindle", "centerline", "centreline", "hub", "center", "centre"}
MAIN_EVAL = "eval_v34"
# a joint relation read off one *extended* instance is that asset's annotation, not the category's (the drill
# press reference marks its spindle as a horizontal joint, against the claim's "vertical axis along the column")
MIN_EVIDENCE = 2
# references whose annotated joints are motion targets. Artiverse joints are curated articulations; the LEGO
# references' joints come from their connectors (a wheel on an axle pin). Fusion 360 assembly joints are CAD
# mates -- the steam engine's 18 are mostly screws -- and are not motions of the object.
GT_ARTIVERSE = ("Artiverse",)
GT_LEGO = ("BrickNet", "BrickComposer")


def tokens(raw):
    """lexicon.tokens plus British/American spelling ('tyre' is a tire, 'castor' a caster)."""
    return [SPELLING.get(t, t) for t in _tokens(raw)]


# ------------------------------------------------------------------ task context
@lru_cache(maxsize=64)
def task(tid) -> Task:
    return Task(tid)          # current core_v2.json, not the frozen run snapshot (see module doc)


class Grounder:
    """Free-text part name -> component id. The component's own phrases (and their 'X or Y' alternatives,
    spaced and joined: 'back-rest' / 'backrest') are matched as contiguous token runs and the match that ends
    last wins (the head noun of an English compound is its last word: 'wheel hub' is a hub). Then a label that
    is the tail of a phrase ('spout' / 'outlet spout'), the task lexicon's synonyms, '<component> <WordNet part
    of it>' ('wheel rim'), and an abstract head ('steering system' <- 'steering column'). Unmatched names stay
    unmatched; nothing is guessed."""

    def __init__(self, t, comps, lex):
        self.lex = lex
        self.comps = comps
        self.runs = []          # (token tuple, comp id)
        self.abstract = []      # (modifier token set, comp id)
        self.canon = {}
        from ppbench.v2.spec_v35 import meronyms
        mer = meronyms()
        for c in comps:
            for cn in [c["canon"], *c.get("merged", [])]:
                self.canon.setdefault(cn, c["id"])
            for a in core._alts(c):
                a = tuple(SPELLING.get(x, x) for x in a)
                self.runs.append((a, c["id"]))
                if len(a) > 1:
                    self.runs.append((("".join(a),), c["id"]))
                if a[-1] in ABSTRACT_HEADS and len(a) > 1:
                    self.abstract.append((set(a[:-1]) - {"power"}, c["id"]))
            for ph in c["phrases"]:
                for m in mer.get(ph.lower().strip(), []):
                    mt = tuple(tokens(m))
                    if mt and len(mt) == 1:
                        base = tuple(tokens(ph))
                        if base:
                            self.runs.append((base + mt, c["id"]))

    def match(self, label):
        """Comma-separated conjuncts in order ('door panel, handle' is the door panel, as a dataset writes a merged
        segment); the name outside brackets first, the bracketed text only when that fails ('door leaf
        (15-panel hinged door slab)' is the panel)."""
        if not label:
            return None
        main = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", str(label))
        for part in [x for x in main.split(",") if x.strip()] or [main]:
            r = self._match(part)
            if r is not None:
                return r
        for inner in re.findall(r"\(([^)]*)\)|\[([^\]]*)\]", str(label)):
            r = self._match(" ".join(inner))
            if r is not None:
                return r
        return None

    def _match(self, s):
        tk = tokens(s)
        if not tk:
            return None
        variants = [tk] + [tk[:i] + [tk[i] + tk[i + 1]] + tk[i + 2:] for i in range(len(tk) - 1)]
        best = None
        for v in variants:
            for run, cid in self.runs:
                n = len(run)
                for i in range(len(v) - n + 1):
                    if tuple(v[i:i + n]) == run:
                        key = (i + n - len(v), n)        # ends last first, then longest
                        if best is None or key > best[0]:
                            best = (key, cid)
        if best:
            return best[1]
        for run, cid in self.runs:
            if len(run) > 1 and len(tk) < len(run) and tuple(tk) == run[-len(tk):]:
                if best is None or len(tk) > best[0]:
                    best = (len(tk), cid)
        if best:
            return best[1]
        c = self.lex.canon(s)
        if c is not None and c in self.canon:
            return self.canon[c]
        st = set(tk)
        for mods, cid in self.abstract:
            if mods and mods & st:
                return cid
        return None


def _gt_kind(ds):
    ds = ds or ""
    if ds.startswith(GT_ARTIVERSE):
        return "artiverse"
    if ds.startswith(GT_LEGO):
        return "lego"
    return None


@lru_cache(maxsize=16)
def reference_design(tid):
    t = task(tid)
    item = next(i for i in core.population_items(t) if i["key"] == "pop__reference")
    try:
        return core.load(t, item)
    except Exception:
        return None


@lru_cache(maxsize=64)
def context(tid):
    """Claimed components, plus -- on an Artiverse reference -- one motion-only component per annotated moving part
    the claims do not name (a knob, a lever handle, a caster connector): the reference's joints are motion targets,
    and a design's part must be able to answer to them."""
    t = task(tid)
    lex = Lexicon(tid, t)
    comps = core.components(t, lex)
    g0 = Grounder(t, comps, lex)
    extra = []
    ref = reference_design(tid)
    if ref is not None and _gt_kind(t.raw["reference"].get("dataset")) == "artiverse":
        byid = {p.id: p for p in ref.parts}
        in_ref = {g0.match(p.role) for p in ref.parts} - {None}
        byc = {c["id"]: c for c in comps}
        # claimed motions whose moving part the reference never names: the reference's unnamed moving part is it
        # ('door leaf' is the claim's 'panel' -- both swing relative to the frame)
        open_claims = [(g0.match(k["moving_part"]), motion_of(k)) for k in t.kinematics]
        open_claims = [(c, m) for c, m in open_claims if c and m and c not in in_ref]
        seen = {}
        for j in ref.joints:
            if j.type == "fixed" or j.child not in byid:
                continue
            label = (byid[j.child].role or "").split(",")[0].strip()
            if not label or g0.match(label) is not None:
                continue
            key = " ".join(tokens(label))
            if not key or key in seen:
                continue
            seen[key] = True
            host = next((c for c, m in open_claims if m == joint_motion(j.type)), None)
            if host is not None:
                byc[host]["phrases"] = byc[host]["phrases"] + [label]
                byc[host].setdefault("aliases", []).append(label)
                open_claims = [(c, m) for c, m in open_claims if c != host]
                continue
            extra.append({"id": f"M{len(extra) + 1}", "canon": key, "name": core._display(label), "phrases": [label],
                          "claims": [], "kinds": ["GT"], "motion_only": True})
    comps = comps + extra
    return t, lex, comps, Grounder(t, comps, lex)


def motion_of(claim):
    jt = (claim.get("joint_type") or "").lower()
    if jt == "screw":
        return "rotate"
    return calib.motion_of(claim)


def joint_motion(jtype):
    return {"revolute": "rotate", "continuous": "rotate", "cylindrical": "rotate", "screw": "rotate",
            "prismatic": "slide", "ball": "ball"}.get(jtype)


def claims(tid):
    t, lex, comps, g = context(tid)
    ids = {c["id"] for c in comps}
    out = []
    for k in t.kinematics:
        mv = g.match(k["moving_part"])
        rtk = tokens(k.get("relative_to") or "")
        rel = None if (not rtk or set(rtk) <= core.FRAME_WORDS or tuple(rtk) in core._self_phrases(t)) \
            else g.match(k.get("relative_to"))
        mtk = tokens(re.sub(r"\([^)]*\)", " ", k["moving_part"]))
        mods = []
        for i in range(len(mtk) - 1):
            for j in range(i + 1, len(mtk)):
                cid = g.match(" ".join(mtk[i:j]))
                if cid and cid != mv and cid not in mods and any(r == tuple(mtk[i:j]) for r, c2 in g.runs if c2 == cid):
                    mods.append(cid)
        out.append({"id": k["id"], "moving": mv if mv in ids else None, "moving_phrase": k["moving_part"],
                    "modifier_comps": mods, "relative": rel if rel != mv else None,
                    "relative_phrase": k.get("relative_to"), "joint_type": k.get("joint_type"),
                    "motion_kind": motion_of(k), "axis_text": k.get("axis"), "range_text": k.get("range")})
    return out


# ------------------------------------------------------------------ instances
def instances(d, part_comp, cid, singular):
    """Instances of component cid: its parts, where parts of the *same* component joined by a fixed joint or with
    overlapping boxes (> 20% of the smaller) are one instance -- a tire on its rim is one wheel, a drawer's front
    and box one drawer, while four legs fixed to one seat stay four legs."""
    def comp(pid):
        v = part_comp.get(pid)
        return v[0] if isinstance(v, tuple) else v
    pids = [p.id for p in d.parts if comp(p.id) == cid]
    if not pids:
        return []
    if singular:
        return [pids]
    return _union_groups(d, pids)


def _union_groups(d, pids):
    par = {x: x for x in pids}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    S = set(pids)
    for j in d.joints:
        if j.type == "fixed" and j.parent in S and j.child in S:
            par[find(j.parent)] = find(j.child)
    byid = {p.id: p for p in d.parts}
    box = {x: (byid[x].vertices.min(0), byid[x].vertices.max(0)) for x in pids}
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            lo = np.maximum(box[a][0], box[b][0])
            hi = np.minimum(box[a][1], box[b][1])
            if np.all(hi > lo):
                inter = float(np.prod(hi - lo))
                va = float(np.prod(np.maximum(box[a][1] - box[a][0], 1e-9)))
                vb = float(np.prod(np.maximum(box[b][1] - box[b][0], 1e-9)))
                if inter > 0.2 * min(va, vb):
                    par[find(a)] = find(b)
    groups = defaultdict(list)
    for x in pids:
        groups[find(x)].append(x)
    return list(groups.values())


def plural(c):
    """A claim names several of a component only in the plural ('wheels', 'brakes'); a singular name
    ('keyboard', 'kingpin') asks for one, whatever the real instances' label granularity."""
    for ph in c["phrases"]:
        w = re.sub(r"\([^)]*\)", " ", ph).strip().split()
        if w and ((w[-1].lower().endswith("s") and not w[-1].lower().endswith("ss")) or w[-1].lower() in ("feet", "teeth")):
            return True
    return False


# ------------------------------------------------------------------ sheets (population, labels only)
def relations_in(d, pc, insts, claim, mot):
    """Per instance of the moving component, the joint of a real instance that realises the claim: the most
    specific joint that moves it (fewest parts), never one that also carries the claim's relative_to part.
    calib.relations_in took the first joint that moved the instance, so a caster wheel's claim read the
    caster's vertical swivel instead of the wheel's own axle."""
    out = []
    ok_types = calib.MOTION_TYPES[mot]
    scene = kin.Scene(d)
    rel_parts = {pid for pid, (cid, _) in pc.items() if claim.get("relative") and cid == claim["relative"]}
    for inst in insts:
        best = None
        for j in d.joints:
            if j.type == "fixed" or j.type not in ok_types:
                continue
            ms = kin.moving_set(d, j)
            if not (set(inst) & ms) or len(ms) >= len(d.parts) or (ms & rel_parts):
                continue
            if best is None or len(ms) < len(best[1]):
                best = (j, ms)
        if best:
            r = kin.relation(d, best[0], inst, best[1], scene)
            if r:
                out.append(r)
    return out


def _finish_expect(exp, rels, k=None):
    # a door leaf is most of a door: the cap is what real instances move, with margin, not a flat 60%
    mf = [r["moving_frac"] for r in rels]
    exp["moving_frac_max"] = float(min(0.98, max(0.75, 1.5 * np.median(mf), 1.1 * max(mf))))
    if k is not None:   # the claim's own words win over extended instances on the axis direction
        words = set(tokens(k.get("axis_text") or ""))
        if "vertical" in words or "horizontal" in words:
            exp["axis_dir"] = "vertical" if "vertical" in words else "horizontal"
    return exp


def generic_expectation(k, mot):
    """No real instance realises the claim: what any joint of that kind must satisfy, plus what the claim's own
    axis words say ('vertical', 'horizontal', 'axle')."""
    words = set(tokens((k or {}).get("axis_text") or ""))
    jt = ((k or {}).get("joint_type") or "").lower()
    exp = {"types": sorted({"revolute", "continuous"} if mot == "rotate" else {"prismatic"} if mot == "slide" else {"ball"}),
           "axis": None, "axis_dir": "vertical" if "vertical" in words else ("horizontal" if "horizontal" in words else None),
           "offset": None, "moving_frac_max": 0.75, "generic": True,
           "round_axle": bool(mot == "rotate" and (words & ROUND_AXLE_WORDS or jt == "continuous"))}
    if mot == "rotate":
        exp["span"] = 2 * math.pi if jt == "continuous" else math.radians(30)
    elif mot == "slide":
        exp["travel_ratio"] = 0.2
    return exp


def build_sheet(tid):
    t, lex, comps, g = context(tid)
    pops = []
    for item in core.population_items(t):
        try:
            d = reference_design(tid) if item["key"] == "pop__reference" else core.load(t, item)
        except Exception:
            continue
        if d is None or not d.parts:
            continue
        pc = {p.id: (g.match(p.role), "label") for p in d.parts}
        pops.append((item, d, pc))
    npop = len(pops)
    stats = {c["id"]: {"present": 0, "counts": []} for c in comps}
    for item, d, pc in pops:
        for c in comps:
            n = len(instances(d, pc, c["id"], False))
            if n:
                stats[c["id"]]["present"] += 1
                stats[c["id"]]["counts"].append(n)
    ref_item = next((x for x in pops if x[0]["key"] == "pop__reference"), None)
    ref_cnt = defaultdict(int)
    if ref_item:
        for cid, _ in ref_item[2].values():
            if cid:
                ref_cnt[cid] += 1
    sheet_comps = []
    for c in comps:
        s = stats[c["id"]]
        vis = s["present"] / max(npop, 1)
        # modelled as a separate part by the reference, or by enough of the real instances: dataset label
        # vocabularies differ (PartNeXt chairs say "cushion" and "support rod"), so the population alone
        # under-reports parts every chair has
        visible = bool(ref_cnt.get(c["id"])) or ((vis >= calib.VISIBLE_MIN) if npop >= 2 else (s["present"] >= 1))
        counts = s["counts"]
        cmin = min(int(max(1, math.floor(np.quantile(counts, 0.25)))) if counts else 1, MAX_SLOTS) if plural(c) else 1
        singular = bool(np.median(counts) <= 1.0) if counts else True
        sheet_comps.append({**c, "visible_frac": vis, "n_present": s["present"], "visible": bool(visible),
                            "count_min": cmin if visible else 1, "singular": singular and not c.get("motion_only"),
                            "counts": counts, "in_reference": int(ref_cnt.get(c["id"], 0))})
    byc = {c["id"]: c for c in sheet_comps}
    # whether the reference's labels can say which parts are visible at all: LEGO brick names ('plate 2x4') and some
    # curated CAD part names ground under 30% of the reference's parts, and "not seen in the reference" then means
    # nothing -- every declarable claimed motion is a target there
    ref_rate = (sum(1 for v in ref_item[2].values() if v[0]) / max(len(ref_item[2]), 1)) if ref_item else 0.0
    visibility_known = ref_rate >= VISIBILITY_MIN_GROUNDED

    # ---- motion targets: the reference's annotated joints first
    targets = {}
    gt = _gt_kind(t.raw["reference"].get("dataset"))
    if ref_item and gt:
        _, rd, rpc = ref_item
        scene = kin.Scene(rd)
        groups = defaultdict(list)
        for j in rd.joints:
            mot = joint_motion(j.type)
            if j.type == "fixed" or mot is None:
                continue
            cid = rpc.get(j.child, (None,))[0]
            if cid is None:
                continue
            ms = kin.moving_set(rd, j)
            if len(ms) >= len(rd.parts):
                continue
            inst = [x for x in ms if rpc.get(x, (None,))[0] == cid] or [j.child]
            r = kin.relation(rd, j, inst, ms, scene)
            if r is None:
                continue
            par = rpc.get(j.parent, (None,))[0]
            groups[(cid, mot)].append((r, par))
        for (cid, mot), rows in groups.items():
            rels = [r for r, _ in rows]
            pars = [p for _, p in rows if p and p != cid]
            exp = _finish_expect(calib.expectation(rels, mot, {}), rels)
            targets[(cid, mot)] = {"id": f"GT:{byc[cid]['name']}:{mot}", "moving": cid, "motion_kind": mot,
                                   "slots": min(MAX_SLOTS, len(rows)), "expect": exp, "expect_source": f"ground truth ({gt})",
                                   "relative": max(set(pars), key=pars.count) if pars else None,
                                   "claims": [], "modifier_comps": []}
    # ---- claimed motions: of visible parts they are targets; of internal parts they only vouch for a joint
    declared_only, kin_rows = [], []
    for k in claims(tid):
        mot = k["motion_kind"]
        row = dict(k)
        cm = byc.get(k["moving"]) if k["moving"] else None
        if mot is None or cm is None:
            row["role"] = "not scored"
            row["why_not"] = ("no declarable motion (" + str(k["joint_type"]) + ")") if mot is None \
                else "moving part is the object itself, its surroundings, or unnamed"
            kin_rows.append(row)
            continue
        key = (cm["id"], mot)
        if key in targets:
            targets[key]["claims"].append(k["id"])
            targets[key]["relative"] = targets[key]["relative"] or k["relative"]
            targets[key]["modifier_comps"] = sorted(set(targets[key]["modifier_comps"]) | set(k["modifier_comps"]))
            row["role"] = "target (merged with ground truth)"
            kin_rows.append(row)
            continue
        rels, evidence = [], 0
        for item, d, pc in pops:
            insts = instances(d, pc, cm["id"], False)
            if not insts:
                continue
            try:
                r = relations_in(d, pc, insts, k, mot)
            except Exception:
                r = []
            if r:
                evidence += 1
                rels += r
        if evidence >= MIN_EVIDENCE:
            exp, src = _finish_expect(calib.expectation(rels, mot, k), rels, k), "real instances"
        else:
            exp, src = generic_expectation(k, mot), "generic"
        entry = {"id": k["id"], "moving": cm["id"], "motion_kind": mot, "slots": cm["count_min"], "expect": exp,
                 "expect_source": src, "relative": k["relative"], "claims": [k["id"]],
                 "modifier_comps": k["modifier_comps"]}
        if cm["visible"] or not visibility_known:
            targets[key] = entry
            row["role"] = "target"
        else:
            declared_only.append(entry)
            row["role"] = "internal: vouches for a declared joint, not a target"
        kin_rows.append(row)
    sheet = {"spec": SPEC, "task": tid, "medium": core.medium(t), "n_population": npop, "ground_truth": gt,
             "reference_grounded_frac": ref_rate, "visibility_known": visibility_known,
             "population": [{"key": i["key"], "n_parts": len(d.parts),
                             "grounded": sum(1 for v in pc.values() if v[0])} for i, d, pc in pops],
             "components": sheet_comps, "claims": kin_rows, "targets": list(targets.values()),
             "declared_only": declared_only}
    import hashlib
    sheet["stamp"] = hashlib.sha1(json.dumps(sheet, sort_keys=True, default=float).encode()).hexdigest()[:12]
    out = RESULTS / tid / "afford" / SHEET
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sheet, indent=1, default=float))
    return sheet


def load_sheet(tid):
    p = RESULTS / tid / "afford" / SHEET
    return json.loads(p.read_text()) if p.exists() else build_sheet(tid)


# ------------------------------------------------------------------ geometry helpers
def sample_labelled(parts, n=N_POINTS, seed=0):
    """Area-weighted surface samples and the index of the part each came from."""
    import trimesh
    rng = np.random.default_rng(seed)
    meshes = [trimesh.Trimesh(p.vertices, p.faces, process=False) for p in parts]
    areas = np.array([m.area for m in meshes])
    if areas.sum() <= 0:
        return np.zeros((0, 3)), np.zeros(0, int)
    counts = rng.multinomial(n, areas / areas.sum())
    P, L = [], []
    for i, (m, k) in enumerate(zip(meshes, counts)):
        if k:
            pts, _ = trimesh.sample.sample_surface(m, int(k), seed=int(rng.integers(1 << 30)))
            P.append(pts)
            L.append(np.full(len(pts), i))
    return np.vstack(P), np.concatenate(L)


def normalise(P):
    lo, hi = P.min(0), P.max(0)
    c = np.r_[(lo[:2] + hi[:2]) / 2, lo[2]]
    diag = float(np.linalg.norm(hi - lo))
    return (P - c) / max(diag, 1e-9), diag


def rotz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def fscore(a, b, tau):
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0, 0.0
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    p, r = float((da < tau).mean()), float((db < tau).mean())
    return (2 * p * r / (p + r) if p + r > 0 else 0.0), p, r


@lru_cache(maxsize=16)
def reference_geometry(tid):
    """The reference's normalised samples and per-component masks (claimed components only)."""
    t, lex, comps, g = context(tid)
    d = reference_design(tid)
    if d is None or not d.parts:
        return None
    P, L = sample_labelled(d.parts, seed=1)
    Pn, diag = normalise(P)
    comp_of = np.array([g.match(p.role) or "" for p in d.parts], dtype=object)
    return {"P": Pn, "L": L, "diag": diag,
            "comp_pts": {c["id"]: Pn[comp_of[L] == c["id"]] for c in comps
                         if not c.get("motion_only") and (comp_of == c["id"]).any()}}


def interior_parts(scene):
    """Parts no path from outside reaches: fill every part's voxels, flood the outside air from the grid border,
    and call a part interior when at most 5% of its boundary cells touch that air. A motor inside a housing, an
    engine under a bonnet, a PCB inside a mouse shell."""
    from scipy import ndimage
    occ = {pid: scene.occ(pid) for pid in scene.byid}
    occ = {k: o for k, o in occ.items() if o is not None and o.count}
    if len(occ) < 2:
        return set(), {}
    lo = np.min([o.lo for o in occ.values()], 0) - 2
    hi = np.max([o.hi for o in occ.values()], 0) + 2
    shape = tuple(int(x) for x in hi - lo)
    if np.prod(shape) > 60_000_000:
        return set(), {}
    U = np.zeros(shape, bool)
    for o in occ.values():
        s = tuple(slice(int(a - l), int(a - l + n)) for a, l, n in zip(o.lo, lo, o.grid.shape))
        U[s] |= o.grid
    lab, _ = ndimage.label(~U)
    border = set(np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                            lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])))
    border.discard(0)
    outside = np.isin(lab, list(border))
    near_out = ndimage.binary_dilation(outside)
    exposed, out = {}, set()
    for pid, o in occ.items():
        s = tuple(slice(int(a - l), int(a - l + n)) for a, l, n in zip(o.lo, lo, o.grid.shape))
        bnd = o.grid & ~ndimage.binary_erosion(o.grid)
        nb = int(bnd.sum())
        e = float((bnd & near_out[s]).sum() / max(nb, 1))
        exposed[pid] = e
        if e <= INTERIOR_EXPOSED_MAX:
            out.add(pid)
    return out, exposed


# ------------------------------------------------------------------ scoring one design
def is_created(p, family):
    if family == "domain":
        return True               # a generator's output is geometry it produced itself
    return (p.source or {}).get("kind") == "created"


def roles_declared(d):
    named = [p for p in d.parts if (p.role or "").strip() and not re.fullmatch(r"(part|mesh|node|object|segment|brick)?[\s_\-]*\d*",
                                                                             (p.role or "").strip().lower())]
    return len(named) >= 0.5 * max(len(d.parts), 1)


def geometric_match(ref, Dr, DL, idx):
    """Component of design part idx by where it lies: at least half its surface within 5% of the diagonal of the
    reference's parts of one component."""
    from scipy.spatial import cKDTree
    Q = Dr[DL == idx]
    if len(Q) < 5 or ref is None:
        return None
    best = (0.0, None)
    for cid, pts in ref["comp_pts"].items():
        if len(pts) < 5:
            continue
        dist, _ = cKDTree(pts).query(Q)
        f = float((dist < TAU_GROUND).mean())
        if f > best[0]:
            best = (f, cid)
    return best[1] if best[0] >= 0.5 else None


def score_design(tid, item, family):
    t, lex, comps, g = context(tid)
    sheet = load_sheet(tid)
    scomps = {c["id"]: c for c in sheet["components"]}
    rec = {"spec": SPEC, "task": tid, "key": item["key"], "system": item["system"], "tier": item.get("tier"),
           "family": family, "sheet": sheet["stamp"]}
    try:
        d = core.load(t, item)
    except Exception as e:
        d = None
        rec["load_error"] = str(e)[:200]
    if d is None or not d.parts:
        z = {"score": 0.0, "status": "fail", "notes": ["empty or unloadable design"]}
        rec["dims"] = {"2.1": dict(z), "2.2": dict(z), "2.3": dict(z) if sheet["targets"] else
                       {"score": None, "status": "skipped", "notes": ["no motion target for this task"]}}
        rec["empty"] = True
        return rec
    ref = reference_geometry(tid)
    D, DL = sample_labelled(d.parts, seed=0)
    Dn, ddiag = normalise(D)
    R = np.eye(3)
    m21 = {}
    if ref is not None and len(D):
        best = None
        for yaw in (0, 90, 180, 270):
            Rz = rotz(yaw)
            f, p, r = fscore(Dn @ Rz.T, ref["P"], TAU)
            if best is None or f > best[0]:
                best = (f, p, r, yaw, Rz)
        m21.update({"shape_f": best[0], "precision": best[1], "recall": best[2], "best_yaw_deg": best[3],
                    "scale_ratio": ddiag / max(ref["diag"], 1e-9)})
        R = best[4]
    Dr = Dn @ R.T
    # ---------------- grounding
    named = roles_declared(d)
    by_role = {p.id: (g.match(p.role) if named else None) for p in d.parts}
    part_comp, how = {}, {}
    for i, p in enumerate(d.parts):
        c = by_role[p.id]
        if c is not None:
            part_comp[p.id], how[p.id] = c, "role"
            continue
        c = geometric_match(ref, Dr, DL, i) if ref is not None else None
        part_comp[p.id], how[p.id] = c, ("geometry" if c else None)
    # ---------------- 2.1 geometry
    if ref is not None and len(D):
        per = {}
        if named:
            for cid, rp in ref["comp_pts"].items():
                c = scomps.get(cid)
                if c is None or not c["visible"] or len(rp) < 20:
                    continue
                idx = [i for i, p in enumerate(d.parts) if by_role[p.id] == cid]
                Q = Dr[np.isin(DL, idx)] if idx else np.zeros((0, 3))
                per[c["name"]] = fscore(Q, rp, TAU_PART)[0] if len(Q) else 0.0
        m21["part_f"] = per
        # part geometry compares a *declared* part with the reference's part of that name; a part found by where it
        # lies would be compared with the very region that found it
        m21["parts"] = float(np.mean(list(per.values()))) if per else None
        created = np.array([is_created(p, family) for p in d.parts])
        cidx = np.nonzero(created)[0]
        Q = Dr[np.isin(DL, cidx)] if len(cidx) else np.zeros((0, 3))
        _, cp, cr = fscore(Q, ref["P"], TAU) if len(Q) else (0.0, None, 0.0)
        m21.update({"created_part_frac": float(created.mean()), "created_surface_frac": float(len(Q) / max(len(Dr), 1)),
                    "created_precision": cp, "created_recall": cr})
        terms = {"shape": m21["shape_f"], "parts": m21["parts"]}
        live = {k: v for k, v in terms.items() if v is not None}
        s21 = sum(W21[k] * v for k, v in live.items()) / sum(W21[k] for k in live)
        rec_21 = {"score": s21, "status": "pass", "metrics": m21, "terms": terms,
                  "notes": [f"weights {W21}, renormalised over the live terms; creation is reported, not scored"]}
    else:
        rec_21 = {"score": None, "status": "skipped", "metrics": m21, "notes": ["no reference geometry"]}
    # ---------------- 2.2 parts
    scene = kin.Scene(d)
    interior, exposed = interior_parts(scene)
    floating, fsrc = _unattached(tid, item["key"], d)
    per22, vis, intl = {}, [], []
    # a brick model's hidden bricks are fill, not modules: interior LDraw bricks never stand for an engine
    anon = [p.id for p in d.parts if part_comp[p.id] is None and p.id in interior
            and (p.source or {}).get("kind") != "ldraw"]
    anon_groups = _union_groups(d, anon) if anon else []
    anon_left = len(anon_groups)
    for c in sheet["components"]:
        if c.get("motion_only"):
            continue
        insts = instances(d, part_comp, c["id"], c["singular"])
        m = c["count_min"]
        vals = sorted([0.5 if all(x in floating for x in grp) else 1.0 for grp in insts], reverse=True)[:m]
        src = "named or placed" if vals else None
        if not c["visible"] and not vals and anon_left > 0:
            vals, anon_left, src = [UNNAMED_INTERIOR], anon_left - 1, "unnamed interior module"
        s = float(sum(vals) / m)
        per22[c["name"]] = {"score": s, "n": len(insts), "need": m, "visible": c["visible"], "evidence": src}
        (vis if c["visible"] else intl).append(s)
    groups = {"visible": float(np.mean(vis)) if vis else None, "internal": float(np.mean(intl)) if intl else None}
    live = {k: v for k, v in groups.items() if v is not None}
    s22 = sum(W22[k] * v for k, v in live.items()) / sum(W22[k] for k in live) if live else None
    rec_22 = {"score": s22, "status": "pass" if s22 is not None else "skipped",
              "metrics": {"groups": groups, "components": per22, "roles_declared": named,
                          "grounded_by_role": sum(1 for v in how.values() if v == "role"),
                          "grounded_by_geometry": sum(1 for v in how.values() if v == "geometry"),
                          "interior_parts": len(interior), "unnamed_interior_modules": len(anon_groups),
                          "floating_source": fsrc, "n_parts": len(d.parts)},
              "notes": [f"weights {W22}; an unnamed interior module stands for one unmet internal component at "
                        f"{UNNAMED_INTERIOR}"]}
    # ---------------- 2.3 kinematics
    rec["dims"] = {"2.1": rec_21, "2.2": rec_22, "2.3": _kinematics(d, scene, sheet, scomps, part_comp)}
    return rec


def _kinematics(d, scene, sheet, scomps, part_comp):
    targets = sheet["targets"]
    if not targets:
        return {"score": None, "status": "skipped", "notes": ["no motion target for this task (no annotated reference "
                                                              "joint, no claimed motion of a visible part)"]}
    contact = scene.contact() if d.joints else {}
    moving = [j for j in d.joints if j.type != "fixed" and j.axis is not None]
    motions = [(t, True) for t in targets] + [(t, False) for t in sheet["declared_only"]]
    # q for every (motion, instance, joint): one sweep each, shared by recall and precision
    best_by_joint = {j.id: (0.0, None) for j in moving}
    recall, per = [], {}
    for t, is_target in motions:
        c = scomps.get(t["moving"])
        insts = instances(d, part_comp, t["moving"], c["singular"] if c else False)
        if not insts:
            for cid in t.get("modifier_comps") or []:
                insts += instances(d, part_comp, cid, scomps[cid]["singular"] if cid in scomps else False)
        rel_parts = [p.id for p in d.parts if t.get("relative") and part_comp.get(p.id) == t["relative"]]
        exp = t["expect"]
        inst_best = []
        for inst in insts:
            expi = _instance_expect(scene, inst, exp)
            b = None
            for j in moving:
                try:
                    r = kin.joint_quality(scene, j, inst, expi, t["motion_kind"], rel_parts, contact)
                except Exception as e:
                    r = {"joint": j.id, "q": 0.0, "error": str(e)[:120]}
                if not r:
                    continue
                if r["q"] > best_by_joint[j.id][0]:
                    best_by_joint[j.id] = (r["q"], t["id"])
                if b is None or r["q"] > b["q"]:
                    b = r
            inst_best.append(b)
        if is_target:
            m = max(1, int(t["slots"]))
            qs = sorted([b["q"] if b else 0.0 for b in inst_best], reverse=True)
            s = float(sum(qs[:m]) / m)
            recall.append(s)
            per[t["id"]] = {"score": s, "moving": (scomps.get(t["moving"]) or {}).get("name"), "slots": m,
                            "n_instances": len(insts), "source": t["expect_source"], "claims": t.get("claims"),
                            "best": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items() if k != "sweep"}
                                     if b else None for b in inst_best][:6]}
    # precision: every declared moving joint, judged by the best target (or internal claim) it realises; a joint
    # on no motion is judged as a joint -- does it move a proper sub-assembly, freely, over a modest range -- at half
    joints = {}
    for j in moving:
        q, tid_ = best_by_joint[j.id]
        if q > 0:
            joints[j.id] = {"score": q, "realises": tid_}
            continue
        mot = joint_motion(j.type)
        gq = 0.0
        if mot in ("rotate", "slide"):
            exp = generic_expectation({"joint_type": j.type}, mot)
            exp["types"] = [j.type]
            try:
                r = kin.joint_quality(scene, j, [j.child], exp, mot, [], contact)
                gq = r["q"] if r else 0.0
            except Exception:
                gq = 0.0
        joints[j.id] = {"score": UNCLAIMED_JOINT * gq, "realises": None, "sound_as_a_joint": gq}
    R_ = float(np.mean(recall))
    P_ = float(np.mean([v["score"] for v in joints.values()])) if joints else 0.0
    # F1, not a weighted mean: a design whose joints realise no target motion scores 0 however sound they are
    # (Particulate's generic joints earned 14-25 under 0.5 R + 0.5 P), and one that realises targets but also
    # declares junk joints is pulled down by them
    s = 2 * R_ * P_ / (R_ + P_) if R_ + P_ > 0 else 0.0
    return {"score": s, "status": "pass", "metrics": {
        "recall": R_, "precision": P_, "targets": per, "n_moving_joints": len(moving),
        "n_joints_on_a_motion": sum(1 for v in joints.values() if v["realises"]),
        "joints": dict(list(joints.items())[:40])},
        "notes": [f"F1 of target recall and joint precision; unclaimed but sound joints count {UNCLAIMED_JOINT}"]}


def _instance_expect(scene, inst, exp):
    """A generic claim about a round part turning on an axle ('wheels rotate about the axle centreline') asks
    for the part's own symmetry axis through its centre -- the physics of a wheel, not a per-object rule."""
    if not exp.get("generic") or not exp.get("round_axle"):
        return exp
    parts = [scene.byid[x] for x in inst if x in scene.byid]
    P = core.sample(parts, 1500)
    if len(P) < 20:
        return exp
    axes, ext, c = core.pca_frame(P)
    rnd, _ = core.roundness(P, axes)
    if rnd >= 0.6:
        return {**exp, "axis": "symmetry", "offset": "centre"}
    return exp


def _unattached(tid, key, d):
    rec = RESULTS / tid / MAIN_EVAL / f"{key}.json"
    if rec.exists():
        try:
            m = json.loads(rec.read_text())["dims"]["1.1"].get("metrics") or {}
            if "unattached_parts" in m:
                return set(m["unattached_parts"]), "level1"
        except (OSError, ValueError, KeyError):
            pass
    return set(), "unknown"


# ------------------------------------------------------------------ items and driver
EXCLUDED = {"glm-4.6v-flash", "kimi-vl-a3b-instruct"}


def items(tid, tiers=("A", "B", "C")):
    """The reported main setting: every agent's name_only+image, 3 rounds, seed 0 final design on each tier, and
    the domain generators' core outputs."""
    out = []
    for f in sorted((RESULTS / tid / MAIN_EVAL).glob("*.json")):
        b = f.stem
        # PPB_AFFORD_EXTRA (2026-09-25, ablation analysis): "rounds" adds the round-k checkpoints of the 3-round
        # name_only+image trajectories, "prompts" the one-round attributes/functional+image runs; unset = main setting
        extra = set(x for x in os.environ.get("PPB_AFFORD_EXTRA", "").split(",") if x)
        if "__round" in b:
            p = b.split("__")
            if "rounds" in extra and len(p) == 6 and p[1] in tiers and p[2:5] == ["name_only+image", "r3", "s0"] \
                    and p[0] not in EXCLUDED:
                try:
                    out.append((json.loads(f.read_text())["item"], "llm"))
                except (OSError, ValueError, KeyError):
                    pass
            continue
        p = b.split("__")
        if "prompts" in extra and len(p) == 5 and p[0] not in EXCLUDED and p[1] in ("A", "B") and p[3:] == ["r1", "s0"] \
                and p[2] in ("attributes+image", "functional+image"):
            try:
                out.append((json.loads(f.read_text())["item"], "llm"))
            except (OSError, ValueError, KeyError):
                pass
            continue
        if p[0] in EXCLUDED:
            continue
        try:
            item = json.loads(f.read_text())["item"]
        except (OSError, ValueError, KeyError):
            continue
        # tier token "A1" = Tier A one-round ablation runs (GPT-6 / GPT-5.6 have no A r3)
        if len(p) == 5 and p[2] == "name_only+image" and p[4] == "s0" and (
                (p[1] in tiers and p[3] == "r3") or (p[1] + p[3][1:] in tiers and p[3] == "r1")):
            out.append((item, "llm"))
        elif len(p) >= 2 and p[1] == "ext":
            out.append((item, "domain"))
    return out


def _stamp(item):
    from ppbench.v2 import report as R
    try:
        return f"{R._stamp(item):.0f}"
    except Exception:
        return "?"


def _one(args):
    tid, item, family, force = args
    os.environ.setdefault("TMPDIR", "/tmp")
    out = RESULTS / tid / OUT / f"{item['key']}.json"
    try:
        sheet = load_sheet(tid)
        stamp = f"{SPEC}|{_stamp(item)}|{sheet['stamp']}"
        if not force and out.exists():
            try:
                if json.loads(out.read_text()).get("stamp_all") == stamp:
                    return "cached"
            except (OSError, ValueError):
                pass
        rec = score_design(tid, item, family)
        rec["stamp_all"] = stamp
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, default=float))
        return "ok"
    except Exception:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".error.txt").write_text(traceback.format_exc())
        return "error"


def build(tids, workers=96, force=False, tiers=("A", "B", "C")):
    jobs = []
    for tid in tids:
        jobs += [(tid, it, fam, force) for it, fam in items(tid, tiers)]
    jobs.sort(key=lambda j: hash(j[1]["key"]) % 97)
    print(f"score {SPEC}: {len(jobs)} designs over {len(tids)} tasks", flush=True)
    tot, t0 = defaultdict(int), time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=1), 1):
            tot[r] += 1
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} {dict(tot)} {time.time() - t0:.0f}s", flush=True)
    print(f"SCORE TOTAL {dict(tot)} {time.time() - t0:.0f}s", flush=True)


def _sheet_one(tid):
    try:
        s = build_sheet(tid)
        vis = sum(c["visible"] for c in s["components"] if not c.get("motion_only"))
        n = sum(1 for c in s["components"] if not c.get("motion_only"))
        src = defaultdict(int)
        for t in s["targets"]:
            src[t["expect_source"]] += 1
        return (f"{tid:26s} pop={s['n_population']:2d} comps {n} ({vis} visible) targets {len(s['targets'])} "
                f"{dict(src)} internal-claims {len(s['declared_only'])}")
    except Exception as e:
        return f"{tid:26s} FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheets", "score"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tiers", default="A,B,C")
    a = ap.parse_args(argv)
    tids = core.task_ids() if a.tasks == "all" else a.tasks.split(",")
    if a.cmd == "sheets":
        with ProcessPoolExecutor(min(a.workers, 25)) as ex:
            for line in ex.map(_sheet_one, tids):
                print(line, flush=True)
    else:
        build(tids, a.workers, a.force, tuple(a.tiers.split(",")))


if __name__ == "__main__":
    main()
