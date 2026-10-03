"""Tier B part creation: a JSON CSG program compiled by manifold3d.

A created part is an immutable program of named nodes, in metres, in the part's
own frame. manifold3d guarantees a closed, manifold, consistently wound solid,
so a created part is well formed by construction and its volume is exact; the
only thing left to check is whether it is one piece.

    {"nodes": [
       {"id": "disc", "op": "cylinder", "radius": 0.03, "height": 0.02},
       {"id": "hub",  "op": "cylinder", "radius": 0.008, "height": 0.03},
       {"id": "hole", "op": "cylinder", "radius": 0.003, "height": 0.04},
       {"id": "wheel0", "op": "union", "inputs": ["disc", "hub"]},
       {"id": "wheel", "op": "difference", "inputs": ["wheel0", "hole"]}],
     "output": "wheel"}

Primitives are centred on the origin: box (size [x,y,z]); cylinder (radius,
height, optional radius_top: a cone or frustum) along +Z; sphere (radius);
torus (major_radius, minor_radius) in the XY plane; extrude (polygon [[x,y],...],
height) along Z; revolve (profile [[r,z],...] with r >= 0, degrees) about Z.
Operations: union / difference / intersection / hull (inputs: earlier node ids;
difference subtracts inputs[1:] from inputs[0]); transform (input, position,
rotation [rx,ry,rz] degrees, applied X then Y then Z about fixed axes);
mirror (input, normal [x,y,z]).

After compiling, the part is recentred on its bounding-box centre, the same
convention every pool part follows, and the offset is reported.
"""
from __future__ import annotations

import numpy as np

MAX_NODES = 96
MAX_EXTENT_M = 3.0
MIN_DIM_M = 0.0005
SEGMENTS = 48


class CSGError(ValueError):
    pass


def _num(v, name, lo=None, hi=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise CSGError(f"{name} must be a number, got {v!r}")
    if not np.isfinite(x):
        raise CSGError(f"{name} must be finite")
    if lo is not None and x < lo:
        raise CSGError(f"{name}={x} is below the minimum {lo}")
    if hi is not None and x > hi:
        raise CSGError(f"{name}={x} is above the maximum {hi}")
    return x


def _vec(v, n, name, lo=None, hi=None):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise CSGError(f"{name} must be a list of {n} numbers, got {v!r}")
    return [_num(x, f"{name}[{i}]", lo, hi) for i, x in enumerate(v)]


def _poly(v, name):
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        raise CSGError(f"{name} must be a list of at least 3 [x, y] points")
    return [_vec(p, 2, f"{name}[{i}]", -MAX_EXTENT_M, MAX_EXTENT_M) for i, p in enumerate(v)]


def compile_program(prog: dict):
    """-> (trimesh.Trimesh recentred, info dict). Raises CSGError with a message meant for the model."""
    import manifold3d as mf
    import trimesh

    M, CS = mf.Manifold, mf.CrossSection
    if not isinstance(prog, dict) or not isinstance(prog.get("nodes"), list):
        raise CSGError("program must be an object with a 'nodes' list and an 'output' id")
    nodes = prog["nodes"]
    if not nodes:
        raise CSGError("program has no nodes")
    if len(nodes) > MAX_NODES:
        raise CSGError(f"at most {MAX_NODES} nodes")
    built: dict = {}
    D, E = MIN_DIM_M, MAX_EXTENT_M
    for k, nd in enumerate(nodes):
        if not isinstance(nd, dict):
            raise CSGError(f"node {k} is not an object")
        nid, op = nd.get("id"), nd.get("op")
        if not isinstance(nid, str) or not nid:
            raise CSGError(f"node {k} needs a string id")
        if nid in built:
            raise CSGError(f"duplicate node id {nid!r}")

        def inp(key="inputs", many=True):
            ids = nd.get(key)
            ids = ids if many else [ids]
            if many and (not isinstance(ids, list) or not ids):
                raise CSGError(f"node {nid!r} ({op}) needs '{key}': a list of earlier node ids")
            for i in ids:
                if i not in built:
                    raise CSGError(f"node {nid!r} refers to {i!r}, which is not an earlier node")
            return [built[i] for i in ids]

        if op == "box":
            s = _vec(nd.get("size"), 3, f"{nid}.size", D, E)
            g = M.cube(s, True)
        elif op == "cylinder":
            r = _num(nd.get("radius"), f"{nid}.radius", 0.0, E)
            h = _num(nd.get("height"), f"{nid}.height", D, E)
            rt = _num(nd.get("radius_top", r), f"{nid}.radius_top", 0.0, E)
            if max(r, rt) < D:
                raise CSGError(f"{nid}: radius too small")
            g = M.cylinder(h, r, rt, SEGMENTS, True)
        elif op == "sphere":
            g = M.sphere(_num(nd.get("radius"), f"{nid}.radius", D, E), SEGMENTS)
        elif op == "torus":
            R = _num(nd.get("major_radius"), f"{nid}.major_radius", D, E)
            r = _num(nd.get("minor_radius"), f"{nid}.minor_radius", D, E)
            if r >= R:
                raise CSGError(f"{nid}: minor_radius must be smaller than major_radius")
            g = M.revolve(CS.circle(r, SEGMENTS).translate([R, 0.0]), SEGMENTS)
        elif op == "extrude":
            pts = _poly(nd.get("polygon"), f"{nid}.polygon")
            h = _num(nd.get("height"), f"{nid}.height", D, E)
            cs = CS([pts])
            if cs.area() < D * D:
                raise CSGError(f"{nid}: polygon has (near) zero area; check the point order and values")
            g = M.extrude(cs, h).translate([0.0, 0.0, -h / 2])
        elif op == "revolve":
            pts = _poly(nd.get("profile"), f"{nid}.profile")
            if any(p[0] < 0 for p in pts):
                raise CSGError(f"{nid}: profile radii (first coordinate) must be >= 0")
            deg = _num(nd.get("degrees", 360.0), f"{nid}.degrees", 1.0, 360.0)
            cs = CS([pts])
            if cs.area() < D * D:
                raise CSGError(f"{nid}: profile has (near) zero area")
            g = M.revolve(cs, SEGMENTS, deg)
        elif op in ("union", "difference", "intersection", "hull"):
            gs = inp()
            if op == "union":
                g = M.batch_boolean(gs, mf.OpType.Add) if len(gs) > 1 else gs[0]
            elif op == "difference":
                g = gs[0] - M.batch_boolean(gs[1:], mf.OpType.Add) if len(gs) > 1 else gs[0]
            elif op == "intersection":
                g = M.batch_boolean(gs, mf.OpType.Intersect) if len(gs) > 1 else gs[0]
            else:
                g = M.batch_hull(gs)
        elif op == "transform":
            (g,) = inp("input", many=False)
            rot = _vec(nd.get("rotation", [0, 0, 0]), 3, f"{nid}.rotation", -3600, 3600)
            pos = _vec(nd.get("position", [0, 0, 0]), 3, f"{nid}.position", -E, E)
            g = g.rotate(rot).translate(pos)
        elif op == "mirror":
            (g,) = inp("input", many=False)
            nrm = _vec(nd.get("normal"), 3, f"{nid}.normal")
            if np.linalg.norm(nrm) < 1e-9:
                raise CSGError(f"{nid}: mirror normal must be non-zero")
            g = g.mirror(nrm)
        else:
            raise CSGError(f"node {nid!r}: unknown op {op!r}; ops are box, cylinder, sphere, torus, extrude, "
                           "revolve, union, difference, intersection, hull, transform, mirror")
        built[nid] = g

    out = prog.get("output")
    if out not in built:
        raise CSGError(f"output {out!r} is not a node id")
    g = built[out]
    if g.is_empty():
        raise CSGError("the output solid is empty (a difference or intersection removed everything)")
    mesh = g.to_mesh()
    v = np.asarray(mesh.vert_properties, float)[:, :3]
    f = np.asarray(mesh.tri_verts, np.int64)
    tm = trimesh.Trimesh(v, f, process=False)
    lo, hi = tm.bounds
    if np.any(hi - lo > E):
        raise CSGError(f"part extent {np.round(hi - lo, 3).tolist()} m exceeds {E} m on some axis")
    centre = (lo + hi) / 2
    tm.apply_translation(-centre)
    pieces = g.decompose()
    vols = sorted((p.volume() for p in pieces), reverse=True)
    info = {"size_m": np.round(hi - lo, 4).tolist(), "volume_m3": float(g.volume()),
            "n_pieces": len(pieces),
            "piece_volumes_m3": [round(x, 9) for x in vols[:8]],
            "recentred_by_m": np.round(-centre, 5).tolist(), "n_triangles": int(len(f))}
    return tm, info
