"""Export a design for the in-browser viewer: decimated, quantised part meshes plus
everything the viewer can show on top of the geometry.

One `designs/<key>.js` per design, which registers itself on `window.__DESIGNS`
(the page loads it on demand with a script tag, so no fetch is needed):

    parts     id, role, canonical role, material, material class, quantised positions, indices
    joints    type, parent, child, axis, origin, limits, the parts each joint moves, depth
    sequence  the declared build order with insertion directions
    verdict   floating parts and colliding pairs, as the evaluator found them

Positions are uint16 over the design's bounding box (0.1 mm steps on a 1 m object),
so a whole chair is a few hundred kB. Moving sets come from the same declared
joint graph the evaluator sweeps, so what moves in the viewer is what was scored.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from ppbench.v2 import analysis, materials
from ppbench.v2.metrics_v3 import kinematic_moving_set
from ppbench.v2.lexicon import Lexicon

FACE_BUDGET = int(__import__("os").environ.get("PPBENCH_VIEWER_FACES", "30000"))   # per design; the published site uses less
MIN_FACES = 60


def _decimate(v, f, target):
    if len(f) <= target:
        return v, f
    try:
        import fast_simplification
        v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1.0 - target / len(f))
        if len(f2):
            return np.asarray(v2, float), np.asarray(f2, np.int64)
    except Exception:
        pass
    return v, f


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def export(design, task_id, key, out_dir, record=None, face_budget=None, min_faces=None):
    """Export browser geometry, with optional per-design detail overrides."""
    face_budget = FACE_BUDGET if face_budget is None else int(face_budget)
    min_faces = MIN_FACES if min_faces is None else int(min_faces)
    lex = Lexicon(task_id)
    parts = design.parts
    import trimesh
    areas = np.array([trimesh.Trimesh(p.vertices, p.faces, process=False).area for p in parts]) if parts else np.zeros(0)
    V = np.vstack([p.vertices for p in parts]) if parts else np.zeros((1, 3))
    lo, hi = V.min(0), V.max(0)
    span = np.maximum(hi - lo, 1e-6)
    out_parts = []
    for p, a in zip(parts, areas):
        target = max(min_faces, int(face_budget * a / max(areas.sum(), 1e-12)))
        v, f = _decimate(p.vertices, p.faces, target)
        q = np.round((v - lo) / span * 65535).clip(0, 65535).astype(np.uint16)
        big = len(v) > 65535
        m = materials.resolve(p.material)
        out_parts.append({"id": p.id, "role": p.role, "canon": lex.canon(p.role), "material": m[0] if m else None,
                          "mclass": m[1] if m else None, "density": m[2] if m else None,
                          "src": (p.source or {}).get("id"), "pos": _b64(q),
                          "idx": _b64(f.astype(np.uint32 if big else np.uint16)), "idx32": big})
    # joints with moving sets and depth (outer joints move more parts)
    joints = []
    if design.joints and parts:
        occs = analysis.occupancies(parts)
        jp = frozenset(frozenset((j.parent, j.child)) for j in design.joints)
        rows = analysis.pair_table(parts, occs, jp)
        ids = {p.id for p in parts}
        for j in design.joints:
            if j.parent not in ids or j.child not in ids:
                continue
            ms, looped = kinematic_moving_set(parts, rows, design.joints, j) if j.type != "fixed" else ({j.child}, False)
            joints.append({"id": j.id, "type": j.type, "parent": j.parent, "child": j.child, "axis": j.axis,
                           "origin": j.origin, "limits": j.limits, "moving": sorted(ms), "looped": looped})
        joints.sort(key=lambda j: -len(j["moving"]))
    verdict = {"floating": [], "colliding": []}
    if record:
        m11 = record["dims"]["1.1"]["metrics"]
        verdict = {"floating": m11.get("floating_parts", []),
                   "colliding": [[c["a"], c["b"], c["cm3"]] for c in m11.get("colliding_pairs", [])]}
    rec = {"key": key, "lo": lo.round(6).tolist(), "span": span.round(6).tolist(), "parts": out_parts, "joints": joints,
           "sequence": design.sequence or [], "verdict": verdict,
           "declares": (design.meta or {}).get("declares", {}), "n_faces": int(sum(len(base64.b64decode(p["idx"])) //
                                                                                  (12 if p["idx32"] else 6) for p in out_parts))}
    out = Path(out_dir) / f"{key}.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("(window.__DESIGNS=window.__DESIGNS||{})[" + json.dumps(key) + "]=" + json.dumps(rec, separators=(",", ":")) + ";")
    return out, rec["n_faces"]
