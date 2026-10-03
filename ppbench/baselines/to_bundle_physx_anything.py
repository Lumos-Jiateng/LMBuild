#!/usr/bin/env python3
"""Convert PhysX-Anything native outputs (one physx-anything_s<seed>/ dir) to the PP_Bench common bundle.

    .venv_eval/bin/python ppbench/baselines/to_bundle_physx_anything.py RUN_DIR [RUN_DIR ...]

Writes RUN_DIR/bundle/{bundle.json, parts/<part_id>.glb}.

Frame: MuJoCo basic.xml world frame at rest, without the demo lift of the base body (pos "0 0 1").
  * objs/<k>/<k>.obj are Z-up (3_split.py rotates the TRELLIS Y-up GLB +90 deg about X) and live in the
    unit cube [-0.5, 0.5]^3, the same frame as the VLM voxels (voxel/32 - 0.5).
  * basic.xml scales every mesh uniformly by s = max(Dimension cm)/100, so metres = normalized * s,
    and the hinge/ball positions are pos_normalized * s. All bodies have pos 0, so the rest pose is just
    the scaled meshes. Units are "m".
Joints come from basic_info.json group_info (what both the URDF and the MJCF were generated from):
  group '0' = base parts; group j = [child labels, parent group id, params, type] where type is
  C (hinge: axis, axis position voxel/32-0.5, range deg/180), B (slide: dir, -, range voxel/32),
  CB (hinge + slide), D (ball about a point), A (free body), E (fixed).
  * C with range [-1, 1] (i.e. +-180 deg) -> "continuous" (as in basic.urdf); otherwise "revolute",
    limits = range * pi rad.
  * B -> "prismatic", limits = range * s metres. basic.xml writes the slide range unscaled, in normalized
    units; basic.urdf does the same. Both are recorded in the joint's "note".
  * CB -> two joints on the same parent/child pair: "<id>_slide" prismatic and "<id>_rot" revolute/continuous.
  * D -> type "spherical", A -> type "floating". These fall outside revolute|continuous|prismatic|fixed
    and are kept with a note rather than dropped.
  * Parts in the same group are rigid: "fixed" joints from the group's anchor part to each other member.
Parent/child part ids for a group joint: child = the part of the child group whose sampled surface comes
closest to the union of the parent group's part surfaces; parent = the parent-group part nearest to that
child part (the minimum over 20k surface samples per part, via a cKDTree). Missing meshes fall back to the
voxel point clouds ind_<label>.npy.
Front: the signed horizontal axis the seat faces, estimated as the direction from the backrest centroid to
the seat centroid, snapped to the dominant +-x/+-y axis. Parts are matched by name ("back" vs "seat"/
"cushion"). The result is null when the names do not match; the method is recorded in bundle.json.
"""
import json, re, sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _num(s):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return float(m.group(0)) if m else None


def density_kg_m3(raw):
    """VLM density strings look like '1.2 g/cm^3' (4_simready_gen.py: float(split('g/cm')[0]) * 1000)."""
    v = _num(raw)
    if v is None:
        return None
    s = str(raw).lower().replace(" ", "")
    if "kg/m" in s:
        return v
    return v * 1000.0  # g/cm^3 (upstream convention; bare numbers are treated the same way)


def load_part_mesh(run, label):
    p = run / "objs" / str(label) / f"{label}.obj"
    if not p.exists():
        return None
    m = trimesh.load(p, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m


def surface_points(run, label, mesh, n=20000, seed=0):
    if mesh is not None and len(mesh.faces):
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
        return np.asarray(pts)
    v = run / f"ind_{label}.npy"
    if v.exists():
        return np.load(v) / 32.0 - 0.5
    return None


def main(run):
    run = Path(run).resolve()
    info = json.loads((run / "basic_info.json").read_text())
    dims = [int(x) for x in re.findall(r"\d+", info["dimension"])]
    s = max(dims) / 100.0  # identical to 4_simready_gen.py's MJCF mesh scale
    out = run / "bundle"
    (out / "parts").mkdir(parents=True, exist_ok=True)
    notes = []

    parts, pts, meshes = [], {}, {}
    for p in info["parts"]:
        lab = p["label"]
        pid = f"l_{lab}"
        mesh = load_part_mesh(run, lab)
        meshes[lab] = mesh
        pts[lab] = surface_points(run, lab, mesh)
        entry = {"id": pid, "file": None, "name": p["name"], "material": p["material"],
                 "density_kg_m3": density_kg_m3(p["density"]), "density_raw": p["density"],
                 "youngs_modulus_gpa": _num(p.get("Young's Modulus (GPa)")),
                 "poisson_ratio": _num(p.get("Poisson's Ratio")), "affordance_rank": p.get("priority_rank"),
                 "description": p.get("Basic_description")}
        if mesh is not None:
            m = mesh.copy()
            m.apply_scale(s)
            f = out / "parts" / f"{pid}.glb"
            m.export(f)
            entry["file"] = f"parts/{pid}.glb"
        else:
            notes.append(f"{pid}: no mesh in objs/ (3_split.py assigned it no faces); file=null")
        parts.append(entry)

    g = info["group_info"]
    groups = {"0": list(g["0"])}
    for k, v in g.items():
        if k != "0":
            groups[k] = list(v[0])

    def nearest_pair(child_labels, parent_labels):
        parent_labels = [l for l in parent_labels if l not in child_labels] or list(groups["0"])
        P = [(l, pts[l]) for l in parent_labels if pts.get(l) is not None]
        C = [(l, pts[l]) for l in child_labels if pts.get(l) is not None]
        if not P or not C:
            return (child_labels[0], parent_labels[0], None)
        allp = np.concatenate([q for _, q in P]); owner = np.concatenate([[l] * len(q) for l, q in P])
        tree = cKDTree(allp)
        best = None
        for l, q in C:
            d, i = tree.query(q, k=1)
            j = int(np.argmin(d))
            if best is None or d[j] < best[2]:
                best = (l, int(owner[i[j]]), float(d[j]))
        return best

    joints = []
    anchor = {}
    # group anchors: base -> part closest to everything else is irrelevant; use first listed member,
    # other groups -> the member that touches the parent group.
    order = ["0"] + sorted([k for k in groups if k != "0"], key=int)
    for k in order:
        members = groups[k]
        if k == "0":
            anchor[k] = members[0]
            continue
        _, par_gid, params, typ = g[k]
        par_labels = groups.get(str(par_gid), groups["0"])
        child_l, parent_l, dist = nearest_pair(members, par_labels)
        anchor[k] = child_l
        base = {"parent": f"l_{parent_l}", "child": f"l_{child_l}", "group": int(k), "parent_group": int(par_gid),
                "native_type": typ, "contact_gap_m": None if dist is None else round(dist * s, 5)}
        params = [float(x) for x in params]
        jid = f"g{k}"

        def rot(idx_axis=slice(0, 3), idx_pos=slice(3, 6), idx_rng=slice(6, 8), suffix=""):
            lo, hi = params[idx_rng]
            cont = lo == -1 and hi == 1
            return dict(base, id=jid + suffix, type="continuous" if cont else "revolute",
                        axis=params[idx_axis], origin=[x * s for x in params[idx_pos]],
                        limits=None if cont else [lo * np.pi, hi * np.pi])

        if typ == "C":
            joints.append(rot())
        elif typ == "B":
            joints.append(dict(base, id=jid, type="prismatic", axis=params[0:3], origin=[0.0, 0.0, 0.0],
                               limits=[params[6] * s, params[7] * s],
                               note="native slide range is voxel/32 (normalized); multiplied by the metric "
                                    "scale here; basic.xml/basic.urdf keep it unscaled"))
        elif typ == "CB":
            joints.append(rot(suffix="_rot"))
            joints.append(dict(base, id=jid + "_slide", type="prismatic", axis=params[8:11],
                               origin=[x * s for x in params[3:6]], limits=[params[14] * s, params[15] * s],
                               note="CB = revolute + prismatic on the same part pair; slide range scaled to metres"))
        elif typ == "D":
            joints.append(dict(base, id=jid, type="spherical", axis=None, origin=[x * s for x in params[3:6]],
                               limits=None, note="native type D (ball joint about a point); not in the common enum"))
        elif typ == "A":
            joints.append(dict(base, id=jid, type="floating", axis=None, origin=[0.0, 0.0, 0.0], limits=None,
                               note="native type A (free body, MJCF freejoint); not in the common enum"))
        else:  # E or unknown
            joints.append(dict(base, id=jid, type="fixed", axis=None, origin=[0.0, 0.0, 0.0], limits=None))

    for k in order:  # rigid members within a group
        for l in groups[k]:
            if l != anchor[k]:
                joints.append({"id": f"fix_l_{anchor[k]}_l_{l}", "type": "fixed", "parent": f"l_{anchor[k]}",
                               "child": f"l_{l}", "axis": None, "origin": [0.0, 0.0, 0.0], "limits": None,
                               "group": int(k), "note": "same PhysX-Anything group (rigid)"})

    # front: backrest centroid -> seat centroid, horizontal, snapped to an axis
    def centroid(pred):
        c = [pts[p["label"]] for p in info["parts"] if pred(p["name"].lower()) and pts.get(p["label"]) is not None]
        return np.concatenate(c).mean(0) if c else None
    back = centroid(lambda n: "back" in n)
    seat = centroid(lambda n: ("seat" in n or "cushion" in n) and "back" not in n)
    front, front_how = None, "backrest->seat centroid direction (horizontal), snapped to the dominant axis"
    if back is not None and seat is not None:
        d = (seat - back)[:2]
        i = int(np.argmax(np.abs(d)))
        front = ("+" if d[i] > 0 else "-") + "xy"[i]
        front_how += f"; raw horizontal vector (m) = {[round(float(x * s), 4) for x in d]}"
    else:
        front_how = "not determined: no part names containing 'back' and 'seat'/'cushion'"

    lo = np.min([np.min(q, 0) for q in pts.values() if q is not None], 0) * s
    hi = np.max([np.max(q, 0) for q in pts.values() if q is not None], 0) * s
    bundle = {
        "source": "physx-anything",
        "native_dir": str(run),
        "up": "z", "front": front, "front_method": front_how,
        "units": "m",
        "frame": ("MuJoCo basic.xml world frame at rest without the base body's demo lift (0 0 1): "
                  "objs/*.obj (Z-up, unit cube [-0.5,0.5]^3) uniformly scaled by max(Dimension cm)/100"),
        "metric_scale": s, "dimension_cm_vlm": info["dimension"],
        "bbox_m": [lo.round(4).tolist(), hi.round(4).tolist()],
        "object_name": info.get("object_name"), "category": info.get("category"),
        "parts": parts, "joints": joints,
        "parent_child_rule": ("child = the child-group part with the smallest surface distance to the parent group's "
                              "parts; parent = the parent-group part attaining that distance (20k surface samples "
                              "per part, cKDTree); intra-group parts are fixed to that anchor"),
        "notes": notes,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1))
    print(f"{run.name}: {len(parts)} parts, {len(joints)} joints, front={front}, scale={s}")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
