#!/usr/bin/env python3
"""Convert a Particulate run (written by run_particulate.py) into the benchmark's common bundle.

    <run>/native/prediction.npz + native/parts/part_XX.ply + meta.json  ->  <run>/bundle/{bundle.json, parts/*.glb}

Frame: parts, joint axes and origins are mapped back from Particulate's +Z-up normalized frame to the
INPUT mesh frame (glTF +Y up, the generator's own unitless scale), i.e. v_in = R^T (v_model * scale + center).
So bundle "up" is "y" and "front" is the input front recorded in meta.json.

Joints (one per predicted parent->child edge whose both ends survive the face-level refinement; if an edge
endpoint was dropped, the child is re-attached to its nearest surviving ancestor, as infer.py --eval does):
  * revolute: axis = plucker direction, origin = projection of the child part's centroid onto the axis line,
    limits = revolute_range (radians, unchanged by the similarity transform);
  * prismatic: axis = prismatic_axis, origin = child centroid, limits = prismatic_range * scale;
  * class 3 (both revolute and prismatic, i.e. a cylindrical joint): emitted as TWO joint entries with the
    same parent/child (ids <j>_rev, <j>_pri), flagged in bundle["warnings"];
  * neither: fixed, axis [0,1,0] placeholder, origin = child centroid, limits null.
Particulate predicts no semantic labels and never a "continuous" type (revolute limits are always finite).

Usage: python ppbench/baselines/to_bundle_particulate.py RUN_DIR [RUN_DIR ...]   (needs numpy + trimesh)
"""
import json, sys
from pathlib import Path

import numpy as np
import trimesh


def induced_edges(hierarchy, alive):
    children = {}
    for p, c in hierarchy:
        children.setdefault(int(p), []).append(int(c))

    def alive_desc(n):
        out = []
        for c in children.get(n, []):
            out.extend([c] if c in alive else alive_desc(c))
        return out

    return [(p, c) for p in sorted(alive) for c in alive_desc(p)]


def convert(run):
    run = Path(run)
    meta = json.loads((run / "meta.json").read_text())
    z = np.load(run / "native" / "prediction.npz")
    R, center, scale = z["rotation"], z["center"], float(z["scale"])
    to_in_pt = lambda v: (np.asarray(v, float) * scale + center) @ R  # row-vector form of R^T (v*s + c)
    to_in_dir = lambda d: np.asarray(d, float) @ R

    out = run / "bundle"
    (out / "parts").mkdir(parents=True, exist_ok=True)
    for f in (out / "parts").glob("*.glb"):
        f.unlink()
    pids = [int(p) for p in z["unique_part_ids"]]
    parts, centroid, warnings = [], {}, []
    for pid in pids:
        m = trimesh.load(run / "native" / "parts" / f"part_{pid:02d}.ply", process=False)
        m = trimesh.Trimesh(vertices=to_in_pt(m.vertices), faces=m.faces, process=False)
        centroid[pid] = m.vertices.mean(0) if len(m.faces) == 0 else m.triangles_center.mean(0)
        if len(m.faces):
            centroid[pid] = (m.triangles_center * m.area_faces[:, None]).sum(0) / max(m.area, 1e-12)
        pid_s = f"part_{pid:02d}"
        m.export(out / "parts" / f"{pid_s}.glb")
        parts.append({"id": pid_s, "file": f"parts/{pid_s}.glb", "name": None, "material": None, "density_kg_m3": None})

    hier = [tuple(e) for e in z["motion_hierarchy"].tolist()]
    edges = induced_edges(hier, set(pids))
    if len(edges) != len([e for e in hier if e[0] in pids and e[1] in pids]):
        warnings.append("some predicted hierarchy nodes had no faces after refinement; children re-attached to nearest surviving ancestor")
    joints = []
    rnd = lambda v: [round(float(x), 6) for x in v]
    for p, c in edges:
        jid, ps, cs = f"joint_{p:02d}_{c:02d}", f"part_{p:02d}", f"part_{c:02d}"
        rev, pri = bool(z["is_part_revolute"][c]), bool(z["is_part_prismatic"][c])
        if rev:
            pl = z["revolute_plucker"][c]
            a = pl[:3] / (np.linalg.norm(pl[:3]) + 1e-8)
            pt = np.cross(pl[3:], a)  # articulation_utils.plucker_to_axis_point (model frame)
            a_in = to_in_dir(a); a_in /= np.linalg.norm(a_in)
            pt_in = to_in_pt(pt)
            org = pt_in + np.dot(centroid[c] - pt_in, a_in) * a_in
            lo, hi = z["revolute_range"][c]
            joints.append({"id": jid + ("_rev" if pri else ""), "type": "revolute", "parent": ps, "child": cs,
                           "axis": rnd(a_in), "origin": rnd(org), "limits": rnd([lo, hi])})
        if pri:
            a = z["prismatic_axis"][c]
            a_in = to_in_dir(a / (np.linalg.norm(a) + 1e-8)); a_in /= np.linalg.norm(a_in)
            lo, hi = z["prismatic_range"][c]
            joints.append({"id": jid + ("_pri" if rev else ""), "type": "prismatic", "parent": ps, "child": cs,
                           "axis": rnd(a_in), "origin": rnd(centroid[c]), "limits": rnd([lo * scale, hi * scale])})
        if rev and pri:
            warnings.append(f"{jid}: predicted class 3 (revolute AND prismatic, cylindrical); emitted as two joints")
        if not rev and not pri:
            joints.append({"id": jid, "type": "fixed", "parent": ps, "child": cs,
                           "axis": [0.0, 1.0, 0.0], "origin": rnd(centroid[c]), "limits": None})

    roots = sorted(set(pids) - {c for _, c in edges})
    if len(roots) > 1:
        warnings.append(f"forest with {len(roots)} roots: {roots}")
    bundle = {
        "source": {"system": meta["system"], "model": meta["model"], "code_commit": meta["code_commit"],
                   "weights_revision": meta["weights"]["revision"], "input_mesh": meta["input_mesh"],
                   "run_dir": str(run.resolve())},
        "up": "y", "front": meta["input_front"], "units": "unitless",
        "parts": parts, "joints": joints,
        "root_parts": [f"part_{r:02d}" for r in roots],
        "notes": "Frame = input mesh frame (glTF +Y up, generator scale). Joint origin: revolute = child centroid "
                 "projected onto the axis; prismatic/fixed = child area-weighted centroid. Revolute limits in radians; "
                 "prismatic limits in input units. Fixed joints carry a placeholder axis.",
        "warnings": warnings,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1))
    types = {}
    for j in joints:
        types[j["type"]] = types.get(j["type"], 0) + 1
    print(f"{run.name}: {len(parts)} parts, {len(joints)} joints {types}, roots {len(roots)}")
    return bundle


if __name__ == "__main__":
    for r in sys.argv[1:]:
        convert(r)
