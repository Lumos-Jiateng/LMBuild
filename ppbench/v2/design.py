"""The design record: what every producer emits and every metric reads.

A design is posed part meshes in the world frame (metres, +Z up, ground z = 0,
front -Y) with, when the producer declares them, a role and a material per
part, joints between parts, and an assembly sequence. Geometry and declarations
stay apart: metrics of Level 1 read only the meshes; roles, materials, joints
and the sequence are claims the design makes and are scored as such.

On disk: `design.json` (everything but triangles) and `design.glb` (one node
per part, named by part id, world coordinates baked).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINT_TYPES = ("fixed", "revolute", "continuous", "prismatic", "cylindrical", "ball")


@dataclass
class Part:
    id: str
    vertices: np.ndarray
    faces: np.ndarray
    role: str | None = None
    material: str | None = None
    source: dict = field(default_factory=dict)


@dataclass
class Joint:
    id: str
    type: str
    parent: str
    child: str
    axis: list | None = None
    origin: list | None = None
    limits: list | None = None     # radians for rotations, metres for translations; None = unbounded / undeclared


@dataclass
class Design:
    system: str
    tier: str
    parts: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    sequence: list | None = None   # [{"part": id, "direction": [x,y,z] | None}]
    meta: dict = field(default_factory=dict)

    def part(self, pid):
        return next((p for p in self.parts if p.id == pid), None)

    # ----------------------------------------------------------- io
    def save(self, out_dir):
        import trimesh
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rec = {"system": self.system, "tier": self.tier, "meta": self.meta,
               "parts": [{"id": p.id, "role": p.role, "material": p.material, "source": p.source,
                          "n_triangles": int(len(p.faces))} for p in self.parts],
               "joints": [j.__dict__ for j in self.joints], "sequence": self.sequence}
        (out / "design.json").write_text(json.dumps(rec, indent=1, default=_np))
        scene = trimesh.Scene()
        rng = np.random.default_rng(7)
        for p in self.parts:
            m = trimesh.Trimesh(p.vertices, p.faces, process=False)
            col = (np.r_[rng.uniform(0.25, 0.9, 3), 1.0] * 255).astype(np.uint8)
            m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(col, (len(p.faces), 1)))
            scene.add_geometry(m, node_name=p.id, geom_name=p.id)
        if self.parts:
            scene.export(str(out / "design.glb"))
        return out

    @classmethod
    def load(cls, out_dir):
        from ppbench.v2 import glb
        out = Path(out_dir)
        rec = json.loads((out / "design.json").read_text())
        geo = {}
        if (out / "design.glb").exists():
            for n in glb.read_nodes(out / "design.glb"):
                geo.setdefault(n["name"], []).append(n)
        parts = []
        for p in rec["parts"]:
            ns = geo.get(p["id"], [])
            if not ns:
                continue
            vs, fs, base = [], [], 0
            for n in ns:
                vs.append(n["vertices"])
                fs.append(n["faces"] + base)
                base += len(n["vertices"])
            parts.append(Part(p["id"], np.vstack(vs), np.vstack(fs), p.get("role"), p.get("material"), p.get("source") or {}))
        joints = [Joint(**j) for j in rec.get("joints") or []]
        return cls(rec["system"], rec["tier"], parts, joints, rec.get("sequence"), rec.get("meta") or {})


def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def rotation_matrix(rot_deg):
    """Extrinsic X, then Y, then Z, in degrees (the convention of every tool)."""
    rx, ry, rz = np.radians(rot_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle(axis, angle):
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
