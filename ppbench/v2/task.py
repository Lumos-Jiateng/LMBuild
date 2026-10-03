"""A benchmark_v2 task: its conditions, pool and reference, in one world frame.

World frame for every design and every metric: metres, +Z up, ground plane
z = 0, front of the object toward -Y. The v2 files are glTF (+Y up) and the
Artiverse front is +Z, so a point (x, y, z) of a source file maps to
(x, -z, y) here; the Artiverse front +Z becomes -Y.

Pool parts keep their source orientation and are centred on their bounding box
(the pool builder centred them; this recentres again and records any offset).
The reference is grounded: translated so its lowest point sits at z = 0.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from ppbench.v2 import glb

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
CORE = Path(os.environ.get("PPBENCH_CORE", BENCH / "core_v2.json"))
RESULTS = Path(os.environ.get("PPBENCH_RESULTS", ROOT / "results" / "v2"))
ARTIVERSE = [BENCH / "assets" / "artiverse"]


def task_dir(task_id: str) -> Path:
    """benchmark/tasks/<task_id>: images, reference, pool meshes, pool sheets, knowledge."""
    return BENCH / "tasks" / task_id

YUP_TO_ZUP = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def to_zup(v):
    return np.asarray(v, float) @ YUP_TO_ZUP.T


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=4)
def _core_tasks(path=str(CORE)):
    return {t["task_id"]: t for t in json.loads(Path(path).read_text())}


class Task:
    def __init__(self, task_id: str, snapshot: Path | None = None):
        """Load from a frozen snapshot when given, else from core_v2.json."""
        if snapshot is not None and Path(snapshot).exists():
            snap = json.loads(Path(snapshot).read_text())
            self.raw = snap["task"]
            self.source = snap["source"]
        else:
            self.raw = _core_tasks()[task_id]
            st = CORE.stat()
            self.source = {"file": str(CORE.relative_to(ROOT)), "sha256": file_sha(CORE), "mtime": st.st_mtime}
        self.id = task_id
        self.name = self.raw["object"]["name"]
        self._pool_cache: dict = {}

    def freeze(self, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"source": self.source, "task": self.raw}, indent=1))

    # ------------------------------------------------------------ conditions
    def condition(self, cid: str) -> dict:
        c = next(c for c in self.raw["conditions"] if c["condition_id"] == cid)
        img = None
        if c.get("image"):
            w = next(w for w in self.raw["images"]["in_the_wild"] if w["id"] == c["image"])
            img = {"id": w["id"], "path": str(ROOT / w["path"]), "passed_fidelity": w["fidelity"]["passed"]}
        return {"condition_id": cid, "text": c["text"], "image": img}

    # ------------------------------------------------------------ pool
    # Version 2 (2026-09-14): LDraw pool meshes (BrickNet GLBs) are stored in LDraw units (1 LDU = 0.4 mm) while the
    # catalog lists sizes in metres, so v1 loaded them 2500x too large (every placement of such a part failed). Meshes are
    # now scaled by their catalog's unit; any other mesh whose extent disagrees with its catalog size by more than 5x is
    # rescaled to the catalog size. Environments record the version they loaded (AssemblyEnv state "pool_mesh_version").
    POOL_MESH_VERSION = 2
    CATALOG_M_PER_UNIT = {"LDraw": 0.0004}

    @property
    def pool(self) -> list[dict]:
        out = []
        for p in self.raw["subpart_pool"]["parts"]:
            sx, sy, sz = p["size_m"]
            out.append({"pool_part_id": p["pool_part_id"], "name": p["name"],
                        "size_m": [round(sx, 4), round(sz, 4), round(sy, 4)],   # Z-up: (x, depth, height)
                        "mesh": str(ROOT / p["mesh"]), "catalog": p.get("catalog")})
        return out

    def pool_mesh(self, pid: str):
        """(vertices Z-up centred, faces) for a pool part."""
        if pid not in self._pool_cache:
            p = next((q for q in self.pool if q["pool_part_id"] == pid), None)
            if p is None:
                raise KeyError(pid)
            nodes = glb.read_nodes(p["mesh"])
            vs, fs, base = [], [], 0
            for n in nodes:
                vs.append(n["vertices"])
                fs.append(n["faces"] + base)
                base += len(n["vertices"])
            v = to_zup(np.vstack(vs))
            v = v * self.CATALOG_M_PER_UNIT.get(p.get("catalog"), 1.0)
            ext, size = float((v.max(0) - v.min(0)).max()), float(max(p["size_m"]))
            if size > 0 and ext > 0 and not (0.2 < ext / size < 5.0):
                v = v * (size / ext)
            lo, hi = v.min(0), v.max(0)
            v = v - (lo + hi) / 2
            self._pool_cache[pid] = (v, np.vstack(fs))
        return self._pool_cache[pid]

    # ------------------------------------------------------------ claims
    @property
    def required_parts(self):
        return self.raw.get("required_parts", [])

    @property
    def attributes(self):
        return self.raw.get("required_attributes", [])

    @property
    def subsystems(self):
        return self.raw.get("functional_subsystems", [])

    @property
    def kinematics(self):
        return self.raw.get("required_kinematics", [])

    # ------------------------------------------------------------ reference
    @lru_cache(maxsize=1)
    def reference(self) -> dict:
        return load_reference(self)


def _artiverse_dir(record: str):
    cat, src, mid = record.split("/")[0], record.split("/")[1].split("-")[0], record.split("-", 1)[1]
    for root in ARTIVERSE:
        d = root / cat / src / mid
        if (d / f"{mid}.segmented.glb").exists():
            return d
    return None


def load_reference(task: Task) -> dict:
    """Parts (role, mesh, material), joints and bounds of the reference, in the world frame.

    Materials come from Artiverse's `material.json`, keyed by part id. The
    reference GLB carries no ids, so each of its nodes is matched to the
    segmented source GLB by label and nearest bounding-box centre; the match
    distance is kept so a bad match is visible.
    """
    ref = task.raw["reference"]
    nodes = glb.read_nodes(ROOT / ref["reference_glb"])
    allv = np.vstack([to_zup(n["vertices"]) for n in nodes])
    lift = -allv[:, 2].min()
    parts = []
    for i, n in enumerate(nodes):
        v = to_zup(n["vertices"])
        v[:, 2] += lift
        parts.append({"node": i, "role_raw": n["name"], "vertices": v, "faces": n["faces"],
                      "material": None, "material_class": None, "density": None, "pid": None})

    src = {"dataset": ref.get("dataset"), "materials": "none"}
    d = _artiverse_dir(ref["record"]) if ref.get("dataset") == "Artiverse" else None
    if d is not None:
        seg = glb.read_nodes(d / f"{d.name}.segmented.glb")
        mats = json.loads((d / "material.json").read_text()) if (d / "material.json").exists() else {}
        centres = [(s["extras"].get("id"), s["extras"].get("label", s["name"]), s["vertices"].mean(0)) for s in seg]
        for p, n in zip(parts, nodes):
            c = n["vertices"].mean(0)
            best = min(centres, key=lambda t: np.linalg.norm(t[2] - c))
            p["pid"] = best[0]
            p["pid_label"] = best[1]
            p["pid_match_dist_m"] = float(np.linalg.norm(best[2] - c))
            m = mats.get(str(best[0])) or {}
            p["material"] = m.get("submaterial")
            p["material_class"] = m.get("material")
            p["density"] = m.get("density")
        src["materials"] = str((d / "material.json").relative_to(ROOT))
        art = json.loads((d / f"{d.name}.articulations.json").read_text())
    else:
        art = {"articulations": []}

    pid_node = {p["pid"]: p["node"] for p in parts if p["pid"] is not None}
    if d is not None:
        # joints may name an ArticulatedPartVirtual group; its geometry is its member meshes
        table = glb.node_table(d / f"{d.name}.segmented.glb")
        by_index = {n["index"]: n for n in table}
        for n in table:
            gid = n["extras"].get("id")
            if n["extras"].get("type") == "ArticulatedPartVirtual" and gid not in pid_node:
                members = [by_index[c]["extras"].get("id") for c in n["children"] if c in by_index]
                pid_node[gid] = [pid_node[m] for m in members if m in pid_node]
    raw_by_pid = {a["pid"]: a for a in art.get("articulations", [])}
    joints = []
    for j in ref.get("articulation", {}).get("joints", []):
        a = raw_by_pid.get(int(j["id"][1:])) if j["id"][1:].isdigit() else None
        axis = to_zup(j["axis"])
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        if j.get("origin") is not None:
            origin = to_zup(j["origin"])
            origin[2] += lift
        else:   # some Artiverse prismatic joints carry no origin (only the axis matters): use the moving parts' centroid
            moving = [p["vertices"] for p in parts if p.get("node") in set(j.get("child_nodes") or [])]
            origin = np.vstack(moving).mean(0) if moving else np.zeros(3)
        parent_nodes = []
        for b in (a or {}).get("base", []):
            nb = pid_node.get(b)
            parent_nodes += nb if isinstance(nb, list) else ([nb] if nb is not None else [])
        child_nodes = list(j.get("child_nodes") or [])
        limits_rot = j.get("range")
        pr = None
        if a and ("prismatic_rangeMax" in a or "prismatic_rangeMin" in a):
            pr = [float(a.get("prismatic_rangeMin") or 0.0), float(a.get("prismatic_rangeMax") or 0.0)]
        joints.append({"id": j["id"], "type": j["type"], "label": j.get("label"), "axis": axis.tolist(),
                       "origin": origin.tolist(), "range": limits_rot, "prismatic_range": pr,
                       "parent_nodes": parent_nodes, "child_nodes": child_nodes,
                       "moves_nodes": list(j.get("moves_nodes") or []), "claims": j.get("claims", [])})
    allz = np.vstack([p["vertices"] for p in parts])
    return {"object_id": ref["object_id"], "parts": parts, "joints": joints,
            "bounds": [allz.min(0).tolist(), allz.max(0).tolist()], "source": src,
            "realizes": ref.get("realizes", {})}
