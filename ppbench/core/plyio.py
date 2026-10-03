"""Reader for the BrickNet inset PLYs. Carried from BrickForge/runs/blockfeat/plyio.py
so ppbench has no import-path dependency on BrickForge's run directories.

BrickNet's collision loader negates Z on load (`_Z_FLIP`), so we do the same: the
returned vertices are in the part's LDraw frame, the frame the connector labels
in `labels.json.xz` live in.
"""
from __future__ import annotations

import numpy as np


def read_ply(path):
    with open(path, "rb") as fh:
        buf = fh.read()
    end = buf.index(b"end_header\n") + len(b"end_header\n")
    hdr = buf[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in hdr:
        raise ValueError(f"unsupported ply format in {path}")

    n_v = n_f = None
    vprops, elem = [], None
    for ln in hdr:
        t = ln.split()
        if t[0] == "element":
            elem = t[1]
            if elem == "vertex":
                n_v = int(t[2])
            elif elem == "face":
                n_f = int(t[2])
        elif t[0] == "property" and elem == "vertex":
            vprops.append((t[2], t[1]))
    dt = {"double": "f8", "float": "f4"}
    names = [p[0] for p in vprops]
    if names[:3] != ["x", "y", "z"]:
        raise ValueError(f"unexpected vertex props {names} in {path}")
    vdt = np.dtype([(n, dt[k]) for n, k in vprops])

    v = np.frombuffer(buf, dtype=vdt, count=n_v, offset=end)
    verts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    verts[:, 2] *= -1.0

    off = end + n_v * vdt.itemsize
    rec = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    f = np.frombuffer(buf, dtype=rec, count=n_f, offset=off)
    if not np.all(f["n"] == 3):
        raise ValueError(f"non-triangular face in {path}")
    faces = f["i"].astype(np.int64)
    faces = faces[:, ::-1].copy()   # Z negation mirrors; restore outward winding
    return verts, faces
