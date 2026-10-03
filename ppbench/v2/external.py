"""Adapters: outputs of external generators -> Design.

These systems neither see the catalog nor call tools, so each design declares
only what its producer emits. Anything the producer does not emit (roles,
materials, joints, a sequence) is marked undeclared and the corresponding
dimensions are `skipped`, never scored as zero and never filled in by us.

    ldr_design       BrickGPT / BrickNet / LegoACE .ldr: LDraw inset meshes, LDU -> m, -Y up -> +Z up
    part_glb_design  a directory of one mesh per part (CubePart, PartCrafter, PartPacker), unitless
    physx_design     PhysX-Anything MuJoCo XML: per-part meshes, densities, joints (metric)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ppbench.v2.design import Design, Joint, Part
from ppbench.v2.task import to_zup

LDU_M = 0.0004
LDRAW_TO_WORLD = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])   # (x, y, z)_LDraw -> (x, -z, -y)


def ldr_design(path, system, meta=None) -> Design:
    """LDraw +Y is down and a model's front faces -Z; world +Z is up and the front faces -Y.
    (x, y, z) -> (x, z, -y) would put the front at +Y, so the map is (x, -z, -y) with det = -1
    compensated by flipping face winding."""
    from ppbench.core.canon import resolve
    from ppbench.core.ir import Design as LDesign
    from ppbench.core.partlib import PartLib
    from ppbench.core.plyio import read_ply
    from ppbench import config as C
    text = Path(path).read_text()
    lib = PartLib()
    d = resolve(LDesign.from_ldr(text, source=system, design_id=Path(path).stem), lib)
    parts, missing = [], 0
    for k, (stem, T) in enumerate(zip(d.stems, d.poses)):
        ply = C.INSET / f"{stem}.ply"
        if not ply.exists():
            missing += 1
            continue
        v, f = read_ply(ply)
        vw = (v @ T[:3, :3].T + T[:3, 3]) * LDU_M
        vw = vw @ LDRAW_TO_WORLD.T
        parts.append(Part(f"brick_{k:03d}", vw, f[:, ::-1].copy(), None, None, {"kind": "ldraw", "stem": stem}))
    if parts:
        z0 = min(p.vertices[:, 2].min() for p in parts)
        for p in parts:
            p.vertices[:, 2] -= z0
    m = {"scale_mode": "metric", "units_note": "LDraw units at 0.4 mm (real brick scale)", "front_declared": False,
         "declares": {"roles": False, "materials": False, "joints": False, "sequence": False},
         "ingest": {"n_lines_parts": len(d.stems), "n_missing_meshes": missing, **(d.ingest or {})}, **(meta or {})}
    return Design(system, "external", parts, [], None, m)


MATERIAL_WORDS = [   # free-text material -> library submaterial (first match wins)
    ("stainless", "stainless_steel"), ("steel", "steel"), ("iron", "steel"), ("chrome", "steel"), ("metal", "steel"),
    ("alumin", "aluminum"), ("polypropylene", "polypropylene_PP"), ("polyethylene", "polyethylene_PE"),
    ("pvc", "polyvinyl_chloride_PVC"), ("acrylic", "acrylic_PMMA"), ("abs", "ABS"), ("nylon", "ABS"), ("plastic", "ABS"),
    ("silicone", "silicone_rubber"), ("rubber", "synthetic_rubber"), ("polyurethane", "polyurethane_foam"),
    ("memory foam", "memory_foam"), ("foam", "polyurethane_foam"), ("sponge", "polyurethane_foam"),
    ("mesh", "polyester_mesh"), ("polyester", "polyester_mesh"), ("leather", "cotton_fabric"), ("fabric", "cotton_fabric"),
    ("cloth", "cotton_fabric"), ("textile", "cotton_fabric"), ("cotton", "cotton_fabric"),
    ("plywood", "plywood"), ("mdf", "particle_board_MDF"), ("pine", "solid_wood_softwood"), ("wood", "solid_wood_hardwood"),
    ("glass", "tempered_glass"), ("ceramic", "porcelain"), ("porcelain", "porcelain"),
]


def map_material(text):
    t = str(text or "").lower()
    for w, sub in MATERIAL_WORDS:
        if w in t:
            return sub
    return None


def _yaw_to_front(front):
    """Rotation about Z that turns the declared front (after Z-up conversion) into -Y."""
    import math
    from ppbench.v2.design import axis_angle
    vec = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}.get(front)
    if vec is None:
        return None
    ang = math.atan2(-1, 0) - math.atan2(vec[1], vec[0])
    return axis_angle([0, 0, 1], ang)


def bundle_design(bundle_dir, system, meta=None) -> Design:
    """The common bundle every articulated external baseline is converted to (see ppbench/baselines/to_bundle_*.py)."""
    b = Path(bundle_dir)
    spec = json.loads((b / "bundle.json").read_text())
    up = spec.get("up", "z")
    front = spec.get("front")
    if up == "y" and front:   # express the front in the Z-up frame: (x, y, z) -> (x, -z, y)
        front = {"+x": "+x", "-x": "-x", "+z": "-y", "-z": "+y"}.get(front, front)
    R = _yaw_to_front(front) if front else None
    conv = (lambda v: to_zup(v)) if up == "y" else (lambda v: np.asarray(v, float))
    parts, raw_mats = [], {}
    for p in spec["parts"]:
        if not p.get("file"):   # declared by the producer with no geometry (PhysX-Anything, computer_mouse): same as an empty mesh
            continue
        v, f = _read_mesh(b / p["file"])
        if not len(f):
            continue
        v = conv(v)
        if R is not None:
            v = v @ R.T
        mat = map_material(p.get("material"))
        raw_mats[p["id"]] = p.get("material")
        parts.append(Part(str(p["id"]), v, f, p.get("name"), mat, {"kind": "generated", "file": p["file"],
                                                                   "material_raw": p.get("material"), "density_raw": p.get("density_kg_m3")}))
    z0 = min(pp.vertices[:, 2].min() for pp in parts) if parts else 0.0
    for pp in parts:
        pp.vertices[:, 2] -= z0
    ids = {pp.id for pp in parts}
    joints = []
    for j in spec.get("joints", []):
        if str(j.get("parent")) not in ids or str(j.get("child")) not in ids:
            continue
        ax = conv(j["axis"]) if j.get("axis") is not None else None
        org = conv(j["origin"]) if j.get("origin") is not None else None
        if R is not None:
            ax = ax @ R.T if ax is not None else None
            org = org @ R.T if org is not None else None
        if org is not None:
            org = org - np.array([0, 0, z0])
        t = j.get("type")
        t = {"floating": "fixed", "planar": "fixed"}.get(t, t)
        joints.append(Joint(str(j["id"]), t, str(j["parent"]), str(j["child"]),
                            None if ax is None else ax.tolist(), None if org is None else org.tolist(), j.get("limits")))
    named = any(pp.role for pp in parts)
    m = {"scale_mode": "metric" if spec.get("units") == "m" else "unitless", "front_declared": bool(front),
         "declares": {"roles": named, "materials": any(pp.material for pp in parts), "joints": bool(joints), "sequence": False},
         "bundle_source": spec.get("source"), "raw_materials": raw_mats, **(meta or {})}
    return Design(system, "external", parts, joints, None, m)


def _read_mesh(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=False)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def part_glb_design(parts_dir, system, up="y", roles_from_names=False, meta=None) -> Design:
    """One file per part. up='y': glTF convention, converted to +Z up with front -Y.
    Part names become roles only when the producer was given a part schema (CubePart)."""
    files = sorted([p for p in Path(parts_dir).iterdir() if p.suffix.lower() in (".glb", ".obj", ".ply", ".stl")])
    parts = []
    for k, f in enumerate(files):
        v, fc = _read_mesh(f)
        if not len(fc):
            continue
        if up == "y":
            v = to_zup(v)
        role = None
        if roles_from_names:
            role = re.sub(r"^\d+_", "", f.stem).replace("_", " ")
        parts.append(Part(f"part_{k:02d}", v, fc, role, None, {"kind": "generated", "file": f.name}))
    if parts:
        z0 = min(p.vertices[:, 2].min() for p in parts)
        for p in parts:
            p.vertices[:, 2] -= z0
    m = {"scale_mode": "unitless", "declares": {"roles": roles_from_names, "materials": False, "joints": False, "sequence": False},
         **(meta or {})}
    return Design(system, "external", parts, [], None, m)
