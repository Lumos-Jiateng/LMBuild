"""Scene analysis shared by the metered `check` tool and the evaluator.

One pass over a set of posed meshes: voxelise each part, find every pair whose
grids come within a cell, and for those pairs measure contact and
interpenetration. Connectivity, floaters and stability follow from the pair
table. The tool shows the model a subset of this; the evaluator reads all of it,
so the model is never checked against a different geometry than it is scored on.
"""
from __future__ import annotations

import numpy as np

from ppbench.v2 import voxel

# A pair "collides" when its deep overlap exceeds both an absolute floor and a
# fraction of the smaller part. The reference office chair itself has a seat /
# backrest overlap of 5.4% of the seat (artist meshes interpenetrate), so the
# fraction is set at about twice that; the sweep below reports other values.
COLLIDE_ABS_M3 = 1e-6          # 1 cm^3
COLLIDE_FRAC = 0.10
FRAC_SWEEP = (0.0, 0.01, 0.05, 0.10, 0.25)
# A declared joint joins two parts only where both parts are: its origin within this of each surface.
# The check tool, the system prompt and the evaluator all use this one rule (protocol v2.2).
JOINT_CONNECT_TOL_M = 0.02


def occupancies(parts, res=voxel.RES):
    return [voxel.voxelize(p.vertices, p.faces, res) for p in parts]


def pair_table(parts, occs, joint_pairs=frozenset()):
    """Every pair within a cell of each other: touching, deep and raw overlap volume."""
    n = len(parts)
    lo = np.array([o.lo for o in occs])
    hi = np.array([o.hi for o in occs])
    rows = []
    for i in range(n):
        near = np.where(np.all(lo[i] - 1 <= hi[i + 1:], 1) & np.all(lo[i + 1:] <= hi[i] + 1, 1))[0] + i + 1
        for k in near:
            a, b = occs[i], occs[k]
            touch = voxel.touching(a, b)
            if not touch:
                continue
            deep = voxel.overlap_cells(a, b, deep=True) * a.res ** 3
            raw = voxel.overlap_cells(a, b, deep=False) * a.res ** 3
            small = max(min(a.volume, b.volume), 1e-12)
            key = frozenset((parts[i].id, parts[k].id))
            rows.append({"a": parts[i].id, "b": parts[k].id, "i": i, "k": k, "deep_m3": deep, "raw_m3": raw,
                         "frac_of_smaller": deep / small, "joint_pair": key in joint_pairs,
                         "collides": deep > COLLIDE_ABS_M3 and deep / small > COLLIDE_FRAC})
    return rows


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, k in edges:
        ri, rk = find(i), find(k)
        if ri != rk:
            parent[ri] = rk
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def surface_distance(point, occ):
    p = np.asarray(point, float)
    return float(np.min(np.linalg.norm(occ.points - p, axis=1))) if len(occ.points) else float("inf")


def stability(occs, masses=None, extra_masses=(), band=0.005):
    """Tilt and margin of the whole assembly treated as one rigid body.

    masses: per-part mass (kg); None = uniform density (volume stands in for mass).
    extra_masses: [(point, mass)] added loads, e.g. an occupant."""
    vols = np.array([o.volume for o in occs])
    w = vols if masses is None else np.asarray(masses, float)
    cents = np.array([o.centroid() for o in occs])
    tot = w.sum()
    com = (cents * w[:, None]).sum(0)
    for pt, m in extra_masses:
        com = com + np.asarray(pt, float) * m
        tot += m
    com = com / max(tot, 1e-12)
    pts = np.vstack([o.points for o in occs])
    out = voxel.tilt_and_margin(pts, com, band=band)
    out["com"] = com.tolist()
    out["total_weight"] = float(tot)
    return out


def summarize(parts, joints, res=voxel.RES, occs=None):
    """The public check: what a model sees when it calls `check`."""
    occs = occs or occupancies(parts, res)
    jp = frozenset(frozenset((j.parent, j.child)) for j in joints)
    rows = pair_table(parts, occs, jp)
    idx = {p.id: i for i, p in enumerate(parts)}
    edges = [(r["i"], r["k"]) for r in rows]
    joint_notes = []
    comps = None  # computed below, after joints are checked against the same rule the evaluator uses
    zmin = min(float(p.vertices[:, 2].min()) for p in parts) if parts else 0.0
    grounded = {i for i in range(len(parts)) if parts[i].vertices[:, 2].min() <= zmin + 0.005}
    for j in joints:
        if j.parent not in idx or j.child not in idx:
            joint_notes.append(f"{j.id}: parent or child does not exist")
            continue
        if j.origin is not None:
            dp = surface_distance(j.origin, occs[idx[j.parent]])
            dc = surface_distance(j.origin, occs[idx[j.child]])
            if max(dp, dc) <= JOINT_CONNECT_TOL_M:
                edges.append((idx[j.parent], idx[j.child]))
            else:
                joint_notes.append(f"{j.id}: origin is {dp:.3f} m from {j.parent} and {dc:.3f} m from {j.child}; "
                                   f"a joint connects its parts only when its origin is within {JOINT_CONNECT_TOL_M} m of both")
        else:
            joint_notes.append(f"{j.id}: no origin, so it does not connect its parts")
    comps = components(len(parts), edges)
    floating = [[parts[i].id for i in c] for c in comps[1:]]
    st = stability(occs) if parts else {}
    below = [p.id for p in parts if p.vertices[:, 2].min() < -0.005]
    coll = [r for r in rows if r["collides"] and not r["joint_pair"]]
    return {
        "n_parts": len(parts),
        "collisions": [{"parts": [r["a"], r["b"]], "overlap_cm3": round(r["deep_m3"] * 1e6, 2),
                        "frac_of_smaller_part": round(r["frac_of_smaller"], 3)} for r in
                       sorted(coll, key=lambda r: -r["deep_m3"])][:20],
        "n_collisions": len(coll),
        "connected_groups": len(comps),
        "not_connected_to_main_group": floating[:20],
        "lowest_point_z_m": round(zmin, 4),
        "parts_below_ground": below,
        "resting_on_ground": [parts[i].id for i in sorted(grounded)][:20],
        "stability": {"critical_tilt_deg": round(st.get("critical_tilt_deg", 0.0), 2),
                      "support_margin_m": round(st.get("support_margin_m", 0.0), 4)} if st else {},
        "joint_issues": joint_notes,
        "voxel_res_m": res,
    }, occs, rows
