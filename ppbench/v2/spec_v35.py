"""Spec v3.5: Level 4 (Realization) re-specified as P.1 Sequence, P.2 Material, P.3 Operability.

Requested 2026-09-22 after the Level-4 audit (docs/evaluation/metrics_feedback.md, P18). Levels 1-3 are
carried through from the eval_v34 record unchanged; only the three Level-4 dimensions are recomputed, and
they get new ids:

    P.1 sequence      (was 4.2)  can a robot execute the declared order, state by state
    P.2 material      (was 4.1)  is each part made of what its role needs: reference instance + cited claims
    P.3 operability   (was 4.3)  are the parts and subsystems there, and does operating it reach the parts

A design that declares no sequence / no material / no roles scores 0.0 on that dimension (owner's rule of
2026-09-22), with the reason in the record, rather than being skipped.

    python -m ppbench.v2.spec_v35 build [--tasks a,b] [--workers 48] [--force]
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ppbench.v2 import analysis, voxel
from ppbench.v2 import spec_v33 as V33
from ppbench.v2.evaluate import _dim
from ppbench.v2.task import RESULTS, Task

SPEC_VERSION = "v3.5"
INSERT_EXTRA_M = 0.05             # an approach path runs the part's own extent plus this, as in v3.1 4.2
INSERT_TOL_CELLS = 2              # raw-cell overlap an approach may add before it counts as blocked. Raw, not the
                                  # eroded core v3.1 used: a 2 cm plate has no core at 1 cm and passed through anything
UP_Z = 0.3                        # a motion direction with z above this comes from below the floor
AXES = [np.array(v, float) for v in ([0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1])]
ORACLE_MAX_PARTS = 160            # the order search is O(n^2) state checks; bigger designs report None


# ================================================================ P.1 sequence

class BuildState:
    """The geometry every prefix check needs, computed once per design."""

    def __init__(self, parts, occs, rows, jedges, weights, free_standing):
        self.parts, self.occs, self.w = parts, occs, np.asarray(weights, float)
        self.n = len(parts)
        self.idx = {p.id: i for i, p in enumerate(parts)}
        self.adj = defaultdict(set)
        for r in rows:
            self.adj[r["i"]].add(r["k"])
            self.adj[r["k"]].add(r["i"])
        for i, k in jedges:
            self.adj[i].add(k)
            self.adj[k].add(i)
        z = np.vstack([p.vertices for p in parts])[:, 2]
        self.ground = float(z.min())
        self.band = max(0.005, 0.01 * float(np.ptp(z)))          # the 1.3 ground band
        self.zmin = np.array([float(p.vertices[:, 2].min()) for p in parts])
        self.free_standing = free_standing is not False
        self.res = occs[0].res
        self.ground_cell = int(np.floor(self.ground / self.res))
        self._stab = {}

    # -- one component: touches the floor, and stands on its own contacts
    def component_ok(self, comp):
        key = frozenset(comp)
        if key in self._stab:
            return self._stab[key]
        grounded = bool(self.zmin[list(comp)].min() <= self.ground + self.band)
        if not grounded:
            out = (False, "floating", 0.0)
        elif not self.free_standing:
            out = (True, None, None)
        else:
            st = analysis.stability([self.occs[i] for i in comp], masses=list(self.w[list(comp)]), band=self.band)
            t = float(st["critical_tilt_deg"])
            out = (True, None, t) if t > 0 else (False, "tips over", t)
        self._stab[key] = out
        return out

    def state_margin(self, placed, target):
        """How safely the state stands: its weakest body's critical tilt against the task's 1.3 target tilt,
        in [0, 1]; 0 when any body floats or tips."""
        s = 1.0
        for c in self.components(placed):
            ok, _, t = self.component_ok(c)
            if not ok:
                return 0.0
            if t is not None:
                s = min(s, min(1.0, t / target))
        return s

    def components(self, placed):
        placed = set(placed)
        seen, out = set(), []
        for s in placed:
            if s in seen:
                continue
            comp, stack = [], [s]
            seen.add(s)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in self.adj[u]:
                    if v in placed and v not in seen:
                        seen.add(v)
                        stack.append(v)
            out.append(comp)
        return out

    # -- can part i reach its place along a straight line, past the parts already placed and the floor
    def insertable(self, i, placed, declared=None):
        cands = []
        if declared is not None:
            d = np.asarray(declared, float)
            if np.linalg.norm(d) > 1e-9:
                cands.append(d / np.linalg.norm(d))
        cands += AXES
        o = self.occs[i]
        P = voxel.union([self.occs[k] for k in placed]) if placed else None
        rest = voxel.overlap_cells(o, P, deep=False) if P is not None else 0
        ext = float(np.linalg.norm(self.parts[i].vertices.max(0) - self.parts[i].vertices.min(0))) + INSERT_EXTRA_M
        nsteps = int(ext / self.res) + 1
        for d in cands:
            if d[2] > UP_Z and self.zmin[i] > self.ground + self.band:
                continue                      # it would have to come up through the floor
            blocked = False
            if P is not None:
                for t in range(1, nsteps + 1, 2):
                    off = np.round(-d * t).astype(np.int64)
                    sh = voxel.Occ(o.lo + off, o.grid, o.points[:0], self.res, 0)
                    sh._core = o.core
                    if voxel.overlap_cells(sh, P, deep=False) > rest + INSERT_TOL_CELLS:
                        blocked = True
                        break
            if not blocked:
                return True, d.tolist()
        return False, None

    def step(self, i, placed, declared=None, strict=False):
        """Add part i to `placed`: (ok, reason). strict = the whole prefix must be one body."""
        ok_ins, _ = self.insertable(i, placed, declared)
        if not ok_ins:
            return False, "blocked insertion"
        comps = self.components(list(placed) + [i])
        if strict and len(comps) > 1:
            return False, "not one body"
        mine = next(c for c in comps if i in c)
        for c in ([mine] if not strict else comps):
            ok, why, _ = self.component_ok(c)
            if not ok:
                return False, why
        return True, None


def oracle_order(bs: BuildState, strict=False):
    """A greedy bottom-up order search: at each step, the lowest part whose placement keeps the state valid.
    Returns the number of parts it could place. A lower bound on what the best order achieves."""
    if bs.n > ORACLE_MAX_PARTS:
        return None
    placed, left = [], set(range(bs.n))
    order_pref = sorted(left, key=lambda i: (bs.zmin[i], -bs.occs[i].volume))
    while left:
        for i in order_pref:
            if i in left and bs.step(i, placed, None, strict)[0]:
                placed.append(i)
                left.discard(i)
                break
        else:
            break
    return len(placed)


TILT_TARGET_DEFAULT = 30.0        # 1.3's target when the task has no reference tilt


def p1_sequence(design, parts, occs, rows, jedges, weights, free_standing, oracle=True, target_tilt=None):
    """P.1 = 1/2 E + 1/2 S.
    E = executable fraction: parts placed, in the declared order, before the first state that cannot be executed.
    S = state robustness: over every declared step k (checked even after a failure), s_k = 0 if the step cannot
        be executed, else the weakest body's critical tilt over the 1.3 target tilt, capped at 1; divided by the
        number of parts, so parts the order omits count 0."""
    seq = design.sequence or []
    if not parts:
        return _dim("computed", "fail", 0.0, {"n_parts": 0}, ["empty design"])
    idx = {p.id: i for i, p in enumerate(parts)}
    order, declared, seen = [], {}, set()
    for s_ in seq:
        pid = s_.get("part") if isinstance(s_, dict) else s_
        if pid in idx and pid not in seen:
            seen.add(pid)
            order.append(idx[pid])
            declared[idx[pid]] = s_.get("direction") if isinstance(s_, dict) else None
    if not order:
        return _dim("computed", "fail", 0.0, {"declared": bool(seq), "n_parts": len(parts)},
                    ["no usable assembly sequence: scored 0 (owner's rule, 2026-09-22)"])
    bs = BuildState(parts, occs, rows, jedges, weights, free_standing)
    n = len(parts)
    target = float(target_tilt) if target_tilt else TILT_TARGET_DEFAULT
    res = {}
    margins = []
    for strict in (False, True):
        placed, first, valid = [], None, []
        for k, i in enumerate(order):
            ok, why = bs.step(i, placed, declared.get(i), strict)
            valid.append(ok)
            if not ok and first is None:
                first = {"step": k, "part": parts[i].id, "reason": why}
            placed.append(i)                 # later steps are still checked
            if not strict:
                margins.append(bs.state_margin(placed, target) if ok else 0.0)
        executed = first["step"] if first else len(order)
        res[strict] = {"executable_frac": executed / n, "valid_state_frac": float(np.mean(valid)),
                       "first_failure": first}
    orc = {}
    if oracle:
        for strict in (False, True):
            m = oracle_order(bs, strict)
            orc[strict] = None if m is None else m / n
    main = res[False]
    E = main["executable_frac"]
    S = float(sum(margins)) / n
    score = 0.5 * E + 0.5 * S
    ceiling = orc.get(False)
    return _dim("computed", "pass" if score >= 0.8 else ("fail" if score == 0 else "degraded"), float(score), {
        "n_parts": n, "n_steps": len(order), "coverage": len(order) / n,
        "executable_frac": E, "state_robustness": S, "target_tilt_deg": target,
        "valid_state_frac": main["valid_state_frac"],
        "state_margins": [round(x, 3) for x in margins][:200],
        "first_failure": main["first_failure"],
        "strict_one_body": res[True],
        "oracle_executable_frac": ceiling, "oracle_strict_executable_frac": orc.get(True),
        "free_standing": bs.free_standing}, [
        "P.1 = 0.5 E + 0.5 S",
        "E = (parts placed before the first non-executable state) / n_parts",
        "S = sum over declared steps k of s_k / n_parts; s_k = 0 if the state is not executable, else "
        "min over its bodies of min(1, critical tilt / the task's 1.3 target tilt)",
        "a state is executable when the new part has a straight approach (declared direction or an axis, never up "
        "through the floor) past the placed parts, and every placed body (connected parts are one solid) touches "
        "the floor and stands on its own; no re-orientation of the build",
        "strict_one_body and oracle_* are reported, not scored"])


# ================================================================ record plumbing

_TASKS = {}


def _task(task_id):
    if task_id not in _TASKS:
        _TASKS[task_id] = Task(task_id, snapshot=RESULTS / task_id / "task_snapshot_core.json")
    return _TASKS[task_id]


def geometry(design, task):
    parts = design.parts
    occs = analysis.occupancies(parts)
    rows = V33._pairs(parts, occs, design.joints)
    jedges = V33.joint_edges(parts, occs, design.joints)
    weights, _ = V33._masses(parts, occs)
    return parts, occs, rows, jedges, weights


def load(task_id, item):
    from ppbench.v2 import report as R
    task = _task(task_id)
    design = R.load_design(task, item)
    if item.get("kind") == "external":
        V33.prepare_external(design, task)
    return task, design


# ================================================================ P.2 material

# Material words in the cited claims -> the library's classes (materials.py). A superset of
# materials.CLAIM_WORDS, kept here so the frozen v3.1 4.1 does not move. Words the library has no class for
# map to the nearest one and say so: a fibre composite is a polymer matrix (plastic), paper and cardboard
# are cellulose (wood), stone / brick / clay / concrete are mineral (ceramic).
MAT_WORDS = {
    "metal": {"heavy_metal", "light_metal"}, "metallic": {"heavy_metal", "light_metal"},
    "steel": {"heavy_metal"}, "iron": {"heavy_metal"}, "brass": {"heavy_metal"}, "copper": {"heavy_metal"},
    "bronze": {"heavy_metal"}, "wire": {"heavy_metal"}, "stainless": {"heavy_metal"},
    "aluminum": {"light_metal"}, "aluminium": {"light_metal"}, "titanium": {"light_metal"},
    "magnesium": {"light_metal"},
    "plastic": {"plastic"}, "plastics": {"plastic"}, "thermoplastic": {"plastic"},
    "polyurethane": {"plastic"}, "pvc": {"plastic"}, "acrylic": {"plastic"}, "polycarbonate": {"plastic"},
    "abs": {"plastic"}, "fiberglass": {"plastic"}, "fibreglass": {"plastic"}, "composite": {"plastic"},
    "wood": {"wood"}, "wooden": {"wood"}, "timber": {"wood"}, "plywood": {"wood"}, "hardwood": {"wood"},
    "hardwoods": {"wood"}, "maple": {"wood"}, "fiberboard": {"wood"}, "particle": {"wood"},
    "paper": {"wood"}, "cardboard": {"wood"}, "cellulose": {"wood"},
    "glass": {"glass"},
    "ceramic": {"ceramic"}, "porcelain": {"ceramic"}, "stone": {"ceramic"}, "brick": {"ceramic"},
    "clay": {"ceramic"}, "concrete": {"ceramic"},
    "rubber": {"rubber"}, "rubbery": {"rubber"}, "elastomer": {"rubber"}, "elastomers": {"rubber"},
    "silicone": {"rubber"},
    "foam": {"foam"},
    "fabric": {"textile"}, "fabrics": {"textile"}, "textile": {"textile"}, "cloth": {"textile"},
    "leather": {"textile"}, "cotton": {"textile"}, "nylon": {"textile"}, "polyester": {"textile"},
    "mesh": {"textile"}, "hair": {"textile"}, "straw": {"textile"},
}
FAMILY = {"heavy_metal": "metal", "light_metal": "metal", "plastic": "polymer", "rubber": "soft",
          "foam": "soft", "textile": "soft", "wood": "wood", "glass": "glass", "ceramic": "ceramic"}
SAME_FAMILY_CREDIT = 0.5          # steel where the reference has aluminium: the right kind, not the same
MAT_GRID = 0.02                   # correspondence grid, as a fraction of the object's diagonal
SUBJECT_STOP = r"\b(are|is|was|were|use|uses|used|can|may|might|typically|commonly|generally|usually|often|" \
               r"contain|contains|consist|consists|have|has|feature|features)\b"


def material_words(text):
    """Material words standing on their own. A word joined to another by a hyphen or dash is a different term
    ("metal-oxide-semiconductor", "lithium-ion"), except the fibre composites; "carbon" alone is carbon."""
    import re
    t = re.sub(r"carbon[\s\-–]*fib(?:er|re)", " fiberglass ", text.lower())
    t = re.sub(r"glass[\s\-–]*fib(?:er|re)", " fiberglass ", t)
    out = []
    for m in re.finditer(r"[a-z]+", t):
        before = t[m.start() - 1] if m.start() > 0 else " "
        after = t[m.end()] if m.end() < len(t) else " "
        if before in "-–—" or after in "-–—":
            continue
        if m.group(0) in MAT_WORDS:
            out.append(m.group(0))
    return out


def claim_material_groups(statement):
    out = []
    for w in material_words(statement):
        g = MAT_WORDS[w]
        if g not in out:
            out.append(set(g))
    return out


def claim_subject(statement):
    import re
    s = re.split(SUBJECT_STOP, statement, maxsplit=1, flags=re.I)[0]
    s = re.sub(r"^\s*(the|a|an|many|most|some|modern|home)\s+", "", s.strip(), flags=re.I)
    return [x.strip() for x in re.split(r"\s*,\s*|\s+and\s+|\s+or\s+", s) if x.strip()]


PART_MATS = Path(__file__).parent / "lexicons" / "part_materials_v35.json"
_PM = None


def extracted_part_materials(task_id):
    """Per-part typical materials read from the task's cited Wikipedia pages by an LLM, each with a verbatim
    quote (scripts/extract_part_materials_v35.py)."""
    global _PM
    if _PM is None:
        _PM = json.loads(PART_MATS.read_text()).get("tasks", {}) if PART_MATS.exists() else {}
    return (_PM.get(task_id) or {}).get("parts", {})


def wiki_material_requirements(task, lex):
    """The task's verified material claims as (subject roles, allowed classes, claim id, quote).
    A claim about the object as a whole ("Chairs are made of wood, metal or plastic") has roles=None
    and applies to the whole design, volume-weighted; a claim about a part applies to the parts
    that realise that role."""
    from ppbench.v2.lexicon import tokens
    obj = task.raw.get("object", {})
    obj_heads = {tokens(n)[-1] for n in [obj.get("name", "")] + list(obj.get("aliases", [])) if tokens(n)}
    obj_heads |= set(tokens(task.id.replace("_", " ")))
    reqs = []
    for a in task.raw.get("required_attributes", []):
        if a.get("kind") != "material":
            continue
        groups = claim_material_groups(a["statement"])
        if not groups:
            continue
        allowed = set().union(*groups)
        roles = []
        for subj in claim_subject(a["statement"]):
            tk = tokens(subj)
            if not tk or tk[-1] in obj_heads:
                continue
            c = lex.canon(subj)
            if c is not None and c not in roles:
                roles.append(c)
        reqs.append({"claim": a["id"], "statement": a["statement"], "roles": roles or None,
                     "allowed": sorted(allowed), "source": "verified claim"})
    covered = {r for q in reqs for r in (q["roles"] or [])}
    for part, rec in extracted_part_materials(task.id).items():
        c = lex.canon(part)
        if c is None or c in covered:
            continue
        covered.add(c)
        reqs.append({"claim": f"W:{part}", "statement": rec["quote"], "roles": [c], "allowed": rec["classes"],
                     "source": f"wikipedia '{rec.get('page')}' via LLM, quote verified"})
    return reqs


PRIOR = Path(__file__).parent / "lexicons" / "material_prior_v35.json"
PRIOR_MIN_SHARE = 0.10            # a class this common for the role in the category is an accepted choice
PRIOR_MIN_PARTS = 5               # fewer annotated parts of a role than this: fall back to the category
_PRIOR = None


def prior_accept(task_id, role):
    """Classes the role takes in >= 10 % of its annotated parts across the task's Artiverse category."""
    global _PRIOR
    if _PRIOR is None:
        _PRIOR = json.loads(PRIOR.read_text())["tasks"] if PRIOR.exists() else {}
    t = _PRIOR.get(task_id)
    if not t:
        return set()
    c = t["roles"].get(role) if role else None
    if not c or sum(c.values()) < PRIOR_MIN_PARTS:
        c = t["all"]
    n = max(sum(c.values()), 1)
    return {k for k, v in c.items() if v / n >= PRIOR_MIN_SHARE}


def prior_role(task_id, role):
    """prior_accept restricted to a role the category actually annotates (no fallback to the whole category)."""
    prior_accept(task_id, None)                       # loads the table
    t = (_PRIOR or {}).get(task_id)
    c = (t or {}).get("roles", {}).get(role) if role else None
    if not c or sum(c.values()) < PRIOR_MIN_PARTS:
        return set()
    n = sum(c.values())
    return {k for k, v in c.items() if v / n >= PRIOR_MIN_SHARE}


_EXT = {}


def ref_exterior(task, rp):
    """The reference part's exterior class (Artiverse annotates interior and exterior; load_reference reads
    only the interior)."""
    from ppbench.v2.task import _artiverse_dir
    if rp.get("pid") is None:
        return set()
    if task.id not in _EXT:
        d = _artiverse_dir(task.raw["reference"]["record"]) if task.raw["reference"].get("dataset") == "Artiverse" else None
        _EXT[task.id] = json.loads((d / "material.json").read_text()) if d and (d / "material.json").exists() else {}
    m = _EXT[task.id].get(str(rp["pid"])) or {}
    return {m["exterior_material"]} if m.get("exterior_material") else set()


def _norm_cells(vertices, faces, c, diag, res):
    o = voxel.voxelize((vertices - c) / diag, faces, res)
    return set(map(tuple, np.argwhere(o.grid) + o.lo))


def _frame(vs):
    V = np.vstack(vs)
    lo, hi = V.min(0), V.max(0)
    return np.r_[(lo[:2] + hi[:2]) / 2, lo[2]], max(float(np.linalg.norm(hi - lo)), 1e-9)


def _agree(cls, accept_exact):
    if cls is None:
        return 0.0
    if cls in accept_exact:
        return 1.0
    return SAME_FAMILY_CREDIT if FAMILY.get(cls) in {FAMILY.get(a) for a in accept_exact} else 0.0


FUNC_MATS = Path(__file__).parent / "lexicons" / "part_function_materials_v35.json"
REF_OVERLAP_MIN = 0.25            # a reference part speaks for a design part covering >= this share of the design part
_FM = None
_POSITIONAL = {"left", "right", "front", "rear", "back", "upper", "lower", "top", "bottom", "l", "r", "fl", "fr", "rl",
               "rr", "inner", "outer", "center", "centre", "middle", "side", "a", "b", "c", "d"}


def role_key(name):
    """A part name without numbers, positions or parenthetical notes: 'front-left wheel rim (12x11)' -> 'wheel rim'."""
    import re
    r = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", str(name or "").lower())
    r = re.sub(r"[^a-z]+", " ", r)
    return " ".join(w for w in r.split() if w not in _POSITIONAL)


def judged_classes(task_id, name):
    global _FM
    if _FM is None:
        _FM = json.loads(FUNC_MATS.read_text()).get("tasks", {}) if FUNC_MATS.exists() else {}
    rec = (_FM.get(task_id) or {}).get(role_key(name)) or {}
    return set(rec.get("classes") or []), rec.get("function")


def p2_material(design, task, lex, ref, vols):
    """P.2 = (1/N) sum_i a_i over the design's parts that have a basis.
    a_i = 1 if the part's declared class is in its accepted set A_i, 0.5 if in the same family, else 0 (0 also when
    the part declares no library material). A_i, by evidence, first non-empty of:
      1. evidence about the part's role: the task's wiki material claims and quote-verified Wikipedia materials for
         the role, the role's >= 10 % classes across the Artiverse category, and the reference parts occupying the
         same place (their interior/exterior class and their role's category classes);
      2. a functional judgement for the part's name (LLM, object + part name only, never the design's material)."""
    from ppbench.v2 import materials
    parts = design.parts
    res_m = [materials.resolve(p.material) for p in parts]
    cls = [r[1] if r else None for r in res_m]
    if not any(cls):
        return _dim("declared+computed", "fail", 0.0,
                    {"declared": any(p.material for p in parts), "n_parts_with_material": 0},
                    ["no part carries a library material: scored 0 (owner's rule, 2026-09-22)"])
    canon = [lex.canon(p.role) for p in parts]
    wreqs = wiki_material_requirements(task, lex)
    wiki_by_role = defaultdict(set)
    for w in wreqs:
        for r in (w["roles"] or []):
            wiki_by_role[r] |= set(w["allowed"])

    # reference parts occupying each design part's place (Artiverse tasks)
    ref_at = [set() for _ in parts]
    rparts = [p for p in (ref or {}).get("parts", []) if p.get("material_class")]
    if rparts:
        dc, dd = _frame([p.vertices for p in parts])
        rc, rd = _frame([p["vertices"] for p in ref["parts"]])
        dsets = [_norm_cells(p.vertices, p.faces, dc, dd, MAT_GRID) for p in parts]
        for rp in rparts:
            rs = _norm_cells(rp["vertices"], rp["faces"], rc, rd, MAT_GRID)
            acc = {rp["material_class"]} | ref_exterior(task, rp) | prior_role(task.id, lex.canon(rp.get("role_raw")))
            for i, ds in enumerate(dsets):
                if ds and len(rs & ds) >= REF_OVERLAP_MIN * len(ds):
                    ref_at[i] |= acc

    rows_, scores, src_count = [], [], defaultdict(int)
    for i, p in enumerate(parts):
        c = canon[i]
        A = set(wiki_by_role.get(c, set())) | prior_role(task.id, c) | ref_at[i]
        src, func = ("evidence", None) if A else (None, None)
        if not A:
            A, func = judged_classes(task.id, p.role or p.id)
            src = "judged" if A else None
        if not A:
            src_count["no basis"] += 1
            continue
        a = _agree(cls[i], A)
        scores.append(a)
        src_count[src] += 1
        rows_.append({"part": p.id, "role": p.role, "material": p.material, "class": cls[i], "accepted": sorted(A),
                      "basis": src, "function": func, "score": a})
    if not scores:
        return _dim("declared+computed", "skipped", None, {"n_parts": len(parts)}, ["no part has a basis"])
    score = float(np.mean(scores))
    return _dim("declared+computed", "pass" if score >= 0.8 else ("fail" if score == 0 else "degraded"), score, {
        "n_parts": len(parts), "n_scored": len(scores), "basis": dict(src_count),
        "n_without_material": sum(1 for c_ in cls if not c_),
        "parts": rows_[:120], "wiki_requirements": wreqs, "design_classes": sorted({c_ for c_ in cls if c_})}, [
        "P.2 = mean over parts of a_i; a_i = 1 if the declared class is accepted, 0.5 same family, 0 otherwise or "
        "when no library material is declared",
        "accepted set: evidence for the part's role (wiki claims + quote-verified Wikipedia materials, Artiverse "
        "category classes >= 10 %, reference parts in the same place); else a function judgement for the part name"])


# ================================================================ P.3 operability

from ppbench.v2 import operability as OP                     # noqa: E402  (predicates are reused as they are)
from ppbench.v2.lexicon import Lexicon, tokens                # noqa: E402

MERONYMS = Path(__file__).parent / "lexicons" / "meronyms_v35.json"
_MER = None
# A person works the object through these. Heads, matched on the claim's own part phrase.
CONTROL_HEADS = {"pedal", "lever", "handle", "knob", "button", "switch", "crank", "trigger", "joystick", "dial",
                 "handlebar", "throttle", "keyboard", "touchpad", "trackpad", "key", "cyclic", "collective",
                 "tiller", "grip", "handwheel"}
CONTROL_PHRASES = {("steering", "wheel")}
SEAT_HEADS = {"seat", "saddle"}
HUMAN_GEOM = {"clearance", "reachable", "grip", "surface", "colocated", "ground_contact"}   # calibratable
HUMAN_SCALE = HUMAN_GEOM - {"ground_contact"}                                  # need a person-sized object
GROUND_CELL_FRAC = 0.04           # ground-contact clusters: 4 % of the object's diagonal, capped at op-v1's 5 cm
CALIB_V35 = OP.SHEETS / "_calibration_v35.json"


def meronyms():
    global _MER
    if _MER is None:
        _MER = json.loads(MERONYMS.read_text())["table"] if MERONYMS.exists() else {}
    return _MER


class Ctx(OP.OpCtx):
    """OpCtx with one addition for presence: a claim part X is realised by a design part named '<X> <m>' where
    m is a WordNet part of X ("front left wheel rim" realises "wheels"). Both conditions are required: a bare
    part-of-X (a 'gear' for 'engine') does not."""

    def roles(self, phrase, broad=True):
        ids = super().roles(phrase, broad)
        if not ids:
            # two canonical names that differ only by spacing or a hyphen ("back rest" / "back-rest", left by a
            # bridge next to an automatic group) are one role
            c = self.lex.canon(phrase)
            key = "".join(ch for ch in (c or "") if ch.isalpha())
            if key:
                ids = [i for i, p in enumerate(self.parts)
                       if "".join(ch for ch in (self.canon[p.id] or "") if ch.isalpha()) == key]
        if ids or not broad:
            return ids
        mer = {tuple(tokens(m)) for m in meronyms().get(str(phrase).lower().strip(), [])}
        ph = tokens(phrase)
        if not mer or not ph:
            return ids
        head = ph[-1]
        out = []
        for i, p in enumerate(self.parts):
            tk = tokens(p.role or p.id)
            if head in tk[:-1] and any(tuple(tk[len(tk) - len(m):]) == m for m in mer if m):
                out.append(i)
        # a part of X comes with what is fixed to it: the rim and the tyre bolted to it are one wheel
        fixed = defaultdict(set)
        for j in self.joints:
            if j.type == "fixed" and j.parent in self.idx and j.child in self.idx:
                fixed[self.idx[j.parent]].add(self.idx[j.child])
                fixed[self.idx[j.child]].add(self.idx[j.parent])
        more = {k for i in out for k in fixed[i]}
        return sorted(set(out) | more)


def _is_control(phrase):
    tk = tokens(phrase)
    return bool(tk) and (tk[-1] in CONTROL_HEADS or tuple(tk[-2:]) in CONTROL_PHRASES)


def sheet_v35(task_id):
    """op-v1's claim-generated sheet (F, K, P + the authored human layer), for every task, plus one generic
    'operate' capability per subsystem that contains a control: a person can reach the control (from the seat
    when the claims name one) and it is connected, through moving joints, to what the subsystem moves."""
    human = json.loads((OP.SHEETS / "_human.json").read_text()) if (OP.SHEETS / "_human.json").exists() else {}
    s = OP.generate_sheet(task_id, human)
    task = _task(task_id)
    lex = Lexicon(task_id, task)
    raw = task.raw
    allp = [p["part"] for p in raw.get("required_parts", [])] + \
           [n for f in raw.get("functional_subsystems", []) for n in f.get("parts", [])]
    seat = next((p for p in allp if tokens(p) and tokens(p)[-1] in SEAT_HEADS), None)
    movers = {(k.get("moving_part") or "").lower() for k in raw.get("required_kinematics", [])}
    extra = []
    for f in raw.get("functional_subsystems", []):
        for c in [n for n in f.get("parts", []) if _is_control(n)]:
            if lex.canon(c) is None:
                continue
            reqs = [{"pred": "reachable", "target": c, "claim": f["id"], **({"from": seat} if seat and lex.canon(seat) else {})}]
            outs = [n for n in f.get("parts", []) if n.lower() in movers and n != c] or \
                   [cn["to"] for cn in f.get("connections", []) if cn.get("to") and cn["to"] != c]
            for o in outs[:3]:
                if lex.canon(o) is not None:
                    reqs.append({"pred": "chain", "from": c, "to": o, "carries": "motion", "max_hops": 6,
                                 "claim": f["id"]})
            extra.append({"id": f"O{f['id']}", "name": f"operate {f['subsystem']} by its {c}", "claims": [f["id"]],
                          "requirements": reqs})
    s["capabilities"] += extra
    s["spec"] = "op-v2"
    return s


def calibrate_v35(tasks, workers=8):
    """The reference object scores the human-posture checks of its own sheet; one it fails is mis-specified
    for this object (scale, pose) and is disabled for every design. Presence, chains and joints are never
    disabled. LDraw references carry no roles and cannot be calibrated."""
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for t, sc in ex.map(_calib_one, tasks):
            if sc is not None:
                out[t] = sc
    CALIB_V35.write_text(json.dumps(out, indent=1))
    return out


def _calib_one(task_id):
    try:
        from ppbench.v2.evaluate import reference_design
        task = _task(task_id)
        d = reference_design(task)
        parts, occs, rows, _, _ = geometry(d, task)
        ctx = Ctx(d, task, Lexicon(task_id, task), parts, occs, rows, d.joints, [], occs[0].res)
        sc = {}
        for cap in sheet_v35(task_id)["capabilities"]:
            for r in cap["requirements"]:
                if r["pred"] in HUMAN_GEOM:
                    try:
                        sc[OP.req_key(cap["id"], r)] = float(OP.PREDICATES[r["pred"]](ctx, r)[0])
                    except Exception:
                        pass
        return task_id, sc
    except Exception:
        return task_id, None


_SHEETS, _CALIB = {}, None


def p3_operability(design, task, parts, occs, rows, per_joint, simulate=True, free_standing=True, weights=None):
    """P.3 = (1/|C|) sum_c w_c,  w_c = min over the links of capability c (a chain is as strong as its weakest link).

    Capabilities C, all from the task's cited claims (sheet op-v2):
      F  one per functional subsystem: every part it names present, every power/motion path it names present
         (a path that carries motion must end on a part that can move), and -- when it has a control -- the control
         within reach and driving the moving part
      M  one per claimed motion K: it works in the MuJoCo simulation (rolls when pushed / moves through its range)
      H  one per authored human-use capability (can be stopped, seats a person, ...)
      P  required parts no subsystem covers, all present
    """
    global _CALIB
    if not any(p.role for p in parts):
        return _dim("computed", "fail", 0.0, {"declared_roles": False},
                    ["no part carries a role: nothing can be located; scored 0 (owner's rule, 2026-09-22)"])
    if task.id not in _SHEETS:
        _SHEETS[task.id] = sheet_v35(task.id)
    sheet = _SHEETS[task.id]
    if _CALIB is None:
        _CALIB = json.loads(CALIB_V35.read_text()) if CALIB_V35.exists() else {}
    calib = _CALIB.get(task.id, {})
    ctx = Ctx(design, task, Lexicon(task.id, task), parts, occs, rows, design.joints, per_joint, occs[0].res)

    # -- simulation of every claimed motion (MuJoCo); falls back to the kinematic check if the engine fails
    sim_score, sim_rows, sim_info = None, [], {}
    if simulate:
        from ppbench.v2 import sim_v35
        try:
            vols = weights if weights is not None else [o.volume for o in occs]   # the 1.3 mass model
            sim_score, sim_rows, sim_info = sim_v35.simulate(design, task, parts, rows, ctx,
                                                             [v * sim_v35.DEFAULT_DENSITY for v in vols],
                                                             free_standing)
        except Exception as e:  # noqa: BLE001
            sim_score, sim_info = None, {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    sim_by_claim = {r["claim"]: r for r in sim_rows}

    # -- parts that can move at all: the kinematic moving set of any declared moving joint
    from ppbench.v2.metrics_v3 import kinematic_moving_set
    movable = set()
    for j in design.joints:
        if j.type != "fixed" and j.parent in ctx.idx and j.child in ctx.idx:
            movable |= kinematic_moving_set(parts, rows, design.joints, j)[0]

    caps_by_id, present, n_off, n_unc = {}, {}, 0, 0
    for cap in sheet["capabilities"]:
        links = []
        for r in cap["requirements"]:
            if r["pred"] == "ground_contact":         # the contact-cluster cell follows the object's size
                r = {**r, "cell_m": min(float(r.get("cell_m", 0.05)), GROUND_CELL_FRAC * ctx.diag)}
            try:
                sc, det = OP.PREDICATES[r["pred"]](ctx, r)
            except Exception as e:  # noqa: BLE001
                sc, det = 0.0, {"error": f"{type(e).__name__}: {e}"}
            row = {**r, "score": float(sc), "detail": det}
            if r["pred"] == "chain" and r.get("carries") == "motion" and sc > 0:
                B = {parts[i].id for i in ctx.roles(r["to"], broad=False)}
                if not (B & movable):
                    row.update({"score": 0.0, "why": "the driven part has no joint: it cannot move"})
            if r["pred"] == "joint_enables" and sim_by_claim:
                continue                              # claimed motions are their own capabilities (below)
            ref = calib.get(OP.req_key(cap["id"], r))
            if r["pred"] in HUMAN_GEOM and ref is not None and ref < OP.CEILING_MIN:
                row["disabled_by_ceiling"] = True
                n_off += 1
            elif r["pred"] in HUMAN_SCALE and not calib:
                row["disabled_uncalibrated"] = True     # no reference with roles: posture cannot be put to scale
                n_unc += 1
            links.append(row)
            if r["pred"] == "part_present":
                present[r["role"].lower()] = max(present.get(r["role"].lower(), 0.0), float(sc))
        caps_by_id[cap["id"]] = {"id": cap["id"], "name": cap["name"], "links": links}
    # every claimed motion is its own capability, scored by the simulation: a doubtful claim ("a dining chair's
    # seat tilts") then costs one capability, not the subsystem it is attached to
    if sim_by_claim:
        caps_by_id.pop("K", None)
        for cid, sr in sim_by_claim.items():
            caps_by_id[f"M{cid}"] = {"id": f"M{cid}", "name": f"{sr.get('moving_part')} {sr.get('motion') or 'moves'}",
                                     "links": [{"pred": "simulated_motion", "role": sr.get("moving_part"),
                                                "score": float(sr["score"]), "claim": cid,
                                                "sim": {k: v for k, v in sr.items() if k != "claim"}}]}
    # a subsystem's control check (O<F>) is one more link of that subsystem
    for cid in [c for c in caps_by_id if c.startswith("O")]:
        f = caps_by_id.pop(cid)
        if cid[1:] in caps_by_id:
            caps_by_id[cid[1:]]["links"] += [{**x, "from_control_check": True} for x in f["links"]]
        else:
            caps_by_id[cid] = f
    caps = []
    for c in caps_by_id.values():
        live = [x for x in c["links"] if not x.get("disabled_by_ceiling") and not x.get("disabled_uncalibrated")]
        if not live:
            continue
        w = float(min(x["score"] for x in live))
        weakest = min(live, key=lambda x: x["score"])
        caps.append({"id": c["id"], "name": c["name"], "kind": c["id"][0], "w": w, "n_links": len(live),
                     "weakest": {k: weakest.get(k) for k in ("pred", "role", "from", "to", "target", "anchor", "why")
                                 if weakest.get(k)}, "links": c["links"]})
    if not caps:
        return _dim("computed", "skipped", None, {"sheet": sheet.get("spec")}, ["the sheet has no usable link"])
    score = float(np.mean([c["w"] for c in caps]))
    by_kind = defaultdict(list)
    for c in caps:
        by_kind[c["kind"]].append(c["w"])
    return _dim("computed", "pass" if score >= 0.8 else ("fail" if score == 0 else "degraded"), score, {
        "sheet_spec": sheet.get("spec"), "calibrated": bool(calib), "n_capabilities": len(caps),
        "n_capabilities_working": sum(1 for c in caps if c["w"] >= 0.5),
        "fully_operable": float(min(c["w"] for c in caps)),
        "by_kind": {k: float(np.mean(v)) for k, v in by_kind.items()},
        "inventory": float(np.mean(list(present.values()))) if present else None,
        "parts_missing": sorted(k for k, v in present.items() if v < 1)[:30],
        "simulation": sim_score, "simulation_tests": sim_rows, "simulation_info": sim_info,
        "n_requirements_disabled_by_ceiling": n_off, "n_requirements_disabled_uncalibrated": n_unc,
        "capability_scores": {c["id"]: round(c["w"], 3) for c in caps}, "capabilities": caps}, [
        "P.3 = mean over capabilities c of w_c; w_c = min over the links of c",
        "F (per functional subsystem): its named parts present; its named power/motion paths present, a motion path "
        "ending on a part that can move; its control reachable and driving it",
        "M (per claimed motion): works in MuJoCo; H (human use); P (required parts outside subsystems)",
        "fully_operable = min over all capabilities (reported)"])


# ================================================================ one record

MAIN_TIERS = {"A", "B"}


def in_scope(name, scope):
    p = name[:-5].split("__")
    if scope == "all":
        return True
    if scope == "B":                                   # the owner's first pass: Tier B final designs only
        return p[1] == "B" and p[2:] == ["name_only+image", "r3", "s0"]
    if scope == "C":                                   # Tier C (creation only) final designs, for the B vs C ablation
        return p[1] == "C" and p[2:] == ["name_only+image", "r3", "s0"]
    if scope == "rounds":                              # round-k checkpoints of the 3-round name_only+image trajectories
        return len(p) == 6 and p[1] in ("A", "B", "C") and p[2:5] == ["name_only+image", "r3", "s0"]
    if scope == "prompts":                             # one-round attributes / functional + image runs, Tiers A and B
        return len(p) == 5 and p[1] in ("A", "B") and p[2] in ("attributes+image", "functional+image") and p[3:] == ["r1", "s0"]
    if scope == "r1main":                              # 2026-10-01, the 200-task core at R1 (the user's choice): Tier B
        # name_only+image round-1 state -- the round-1 checkpoint of the 3-round trajectory (original 50) or the
        # one-round run (150 new tasks), plus the final 3-round design of a trajectory that ended inside round 1 --
        # and the domain generators' outputs
        if p[1] == "ext":
            return len(p) == 4 and p[2] in ("image", "name_only")
        if len(p) == 6:
            return p[1] == "B" and p[2:5] == ["name_only+image", "r3", "s0"] and p[5] == "round1"
        return p[1] == "B" and p[2] == "name_only+image" and p[3] in ("r1", "r3") and p[4] == "s0"
    if scope == "A1":                                  # Tier A one-round ablation runs (GPT-6 / GPT-5.6 have no A r3)
        return p[1] == "A" and p[2:] == ["name_only+image", "r1", "s0"]
    if p[1] in MAIN_TIERS:
        return p[2:] == ["name_only+image", "r3", "s0"]
    return p[1] == "ext" and len(p) == 4 and p[2] in ("image", "name_only")


def level4(task, design, rec, free_standing):
    parts = design.parts
    if not parts:
        z = _dim("computed", "fail", 0.0, {"n_parts": 0}, ["empty design"])
        return {"P.1": z, "P.2": dict(z), "P.3": dict(z)}
    parts, occs, rows, jedges, weights = geometry(design, task)
    try:
        ref = task.reference()
    except Exception:
        ref = None
    lex = Lexicon(task.id, task)
    per_joint = ((rec.get("dims") or {}).get("2.3") or {}).get("metrics", {}).get("per_joint") or []
    out = {}
    tgt = (((rec.get("dims") or {}).get("1.3") or {}).get("metrics") or {}).get("target_tilt_deg")
    for k, fn in (("P.1", lambda: p1_sequence(design, parts, occs, rows, jedges, weights, free_standing,
                                              target_tilt=tgt)),
                  ("P.2", lambda: p2_material(design, task, lex, ref, [o.volume for o in occs])),
                  ("P.3", lambda: p3_operability(design, task, parts, occs, rows, per_joint,
                                                 free_standing=free_standing, weights=weights))):
        try:
            out[k] = fn()
        except Exception as e:
            out[k] = _dim("computed", "error", None, {"error": f"{type(e).__name__}: {e}"},
                          [traceback.format_exc()[-800:]])
    return out


def _one(args):
    task_id, name, force = args
    base = RESULTS / task_id
    src, dst = base / "eval_v34" / name, base / "eval_v35" / name
    try:
        rec = json.loads(src.read_text())
    except (OSError, ValueError):
        return "bad"
    want = f"{rec.get('stamp')}|{SPEC_VERSION}"
    if not force and dst.exists():
        try:
            if json.loads(dst.read_text()).get("stamp") == want:
                return "cached"
        except (OSError, ValueError):
            pass
    item = rec.get("item") or {}
    if not item or item.get("kind") == "reference":
        return "skip"
    path = Path(item.get("path", ""))
    if item.get("kind", "tool") in ("tool", "annotated") and not (path / "design.json").exists():
        return "gone"
    try:
        task, design = load(task_id, item)
        fs = V33.get_anchors(task_id, task).get("free_standing")
        dims4 = level4(task, design, rec, fs)
        D = dict(rec.get("dims") or {})
        D.update(dims4)
        out = {k: rec[k] for k in ("system", "tier", "meta", "item") if k in rec}
        out.update({"notes": (rec.get("notes") or []) + ["Level 4 recomputed under spec v3.5 (P.1-P.3)"],
                    "spec": SPEC_VERSION, "derived_from": rec.get("spec"), "dims": D,
                    "headline": rec.get("headline"), "stamp": want})
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=1, default=float))
        return "ok"
    except Exception:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.with_suffix(".error.txt").write_text(traceback.format_exc())
        return "error"


def build(tasks=None, workers=48, force=False, scope="main", limit=None):
    ids = tasks or sorted(p.name for p in RESULTS.iterdir() if (p / "task_snapshot_core.json").exists())
    jobs = []
    for t in ids:
        d = RESULTS / t / "eval_v34"
        if d.is_dir():
            jobs += [(t, f.name, force) for f in sorted(d.glob("*.json")) if in_scope(f.name, scope)]
    if limit:
        jobs = jobs[:limit]
    jobs.sort(key=lambda j: hash(j) % 997)             # spread big and small tasks over the pool
    print(f"{len(jobs)} records over {len(ids)} tasks, scope={scope}", flush=True)
    tot = defaultdict(int)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=2), 1):
            tot[r] += 1
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)}  {dict(tot)}  {time.time() - t0:.0f}s", flush=True)
    print(f"TOTAL {dict(tot)}  {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "calibrate", "sheets"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--scope", default="main", choices=["main", "B", "C", "A1", "rounds", "prompts", "all", "r1main"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    ids = None if a.tasks == "all" else a.tasks.split(",")
    all_ids = ids or sorted(p.name for p in RESULTS.iterdir() if (p / "task_snapshot_core.json").exists())
    if a.cmd == "calibrate":
        c = calibrate_v35(all_ids, a.workers)
        print(f"calibrated {len(c)} of {len(all_ids)} tasks")
    elif a.cmd == "sheets":
        for t in all_ids:
            s = sheet_v35(t)
            print(f"{t:26s} caps={len(s['capabilities']):2d} "
                  f"ops={sum(1 for c in s['capabilities'] if c['id'].startswith('O'))} "
                  f"reqs={sum(len(c['requirements']) for c in s['capabilities'])}")
    else:
        build(ids, a.workers, a.force, a.scope, a.limit)


if __name__ == "__main__":
    main()
