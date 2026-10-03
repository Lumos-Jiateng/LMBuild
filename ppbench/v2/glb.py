"""Read a GLB into world-space triangle meshes, one per mesh node.

trimesh flattens a glTF scene well but drops node extras, and Artiverse keeps
its part ids and labels in exactly those extras. This reader keeps them. It
handles what the benchmark files contain: triangle primitives, float POSITION,
uint8/16/32 indices, node matrices or TRS, nested nodes.
"""
from __future__ import annotations

import json
import struct

import numpy as np

_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _chunks(raw: bytes):
    if raw[:4] != b"glTF":
        raise ValueError("not a GLB file")
    off, js, binary = 12, None, b""
    while off < len(raw):
        ln, kind = struct.unpack_from("<I4s", raw, off)
        body = raw[off + 8: off + 8 + ln]
        if kind == b"JSON":
            js = json.loads(body)
        elif kind == b"BIN\x00":
            binary = body
        off += 8 + ln
    return js, binary


def _accessor(js, binary, i):
    a = js["accessors"][i]
    bv = js["bufferViews"][a["bufferView"]]
    dt = np.dtype(_COMP[a["componentType"]])
    nc = _NCOMP[a["type"]]
    stride = bv.get("byteStride") or dt.itemsize * nc
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    n = a["count"]
    if stride == dt.itemsize * nc:
        arr = np.frombuffer(binary, dtype=dt, count=n * nc, offset=start)
    else:
        idx = start + np.arange(n)[:, None] * stride + np.arange(nc)[None, :] * dt.itemsize
        arr = np.frombuffer(binary, dtype=np.uint8)[(idx[..., None] + np.arange(dt.itemsize)).reshape(-1)]
        arr = arr.view(dt)
    return arr.reshape(n, nc) if nc > 1 else arr


def _local(node):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    T = np.eye(4)
    if "scale" in node:
        T = np.diag(list(node["scale"]) + [1.0]) @ T
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M = np.eye(4)
        M[:3, :3] = R
        T = M @ T
    if "translation" in node:
        M = np.eye(4)
        M[:3, 3] = node["translation"]
        T = M @ T
    return T


def read_nodes(path) -> list[dict]:
    """[{index, name, extras, matrix (world), vertices (world, float64), faces, material}]"""
    js, binary = _chunks(open(path, "rb").read())
    nodes = js.get("nodes", [])
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i

    def world(i):
        T = _local(nodes[i])
        while i in parent:
            i = parent[i]
            T = _local(nodes[i]) @ T
        return T

    out = []
    for i, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        W = world(i)
        vs, fs, mat, base = [], [], None, 0
        for pr in js["meshes"][n["mesh"]]["primitives"]:
            if pr.get("mode", 4) != 4:
                continue
            v = _accessor(js, binary, pr["attributes"]["POSITION"]).astype(np.float64)
            if "indices" in pr:
                f = _accessor(js, binary, pr["indices"]).astype(np.int64).reshape(-1, 3)
            else:
                f = np.arange(len(v)).reshape(-1, 3)
            vs.append(v)
            fs.append(f + base)
            base += len(v)
            mat = pr.get("material", mat)
        if not vs:
            continue
        v = np.vstack(vs)
        v = v @ W[:3, :3].T + W[:3, 3]
        out.append({"index": i, "name": n.get("name"), "extras": n.get("extras") or {},
                    "matrix": W, "vertices": v, "faces": np.vstack(fs), "material": mat,
                    "parent": parent.get(i)})
    return out


def node_table(path) -> list[dict]:
    """All nodes (mesh or not) with extras and children, for group lookups."""
    js, _ = _chunks(open(path, "rb").read())
    return [{"index": i, "name": n.get("name"), "extras": n.get("extras") or {},
             "children": n.get("children", []), "mesh": "mesh" in n} for i, n in enumerate(js.get("nodes", []))]
