#!/usr/bin/env python3
"""Convert Infinite-Mobility (procedural Infinigen + Blender) URDF outputs to the PP_Bench common bundle.

    .venv_eval/bin/python ppbench/baselines/to_bundle_infinite_mobility.py RUN_DIR [RUN_DIR ...]

RUN_DIR = results/v2/<task>/external/infinite-mobility_s<seed>/, containing native/<k>/scene.urdf (+ meshes,
data_infos_<k>.json). Writes RUN_DIR/bundle/{bundle.json, parts/<part_id>.glb}.

Frame: the URDF world frame at the rest configuration (every joint value 0). World poses come from forward
kinematics over the URDF tree: T(child link) = T(parent link) @ T(joint origin xyz/rpy). A visual mesh is
placed at T(link) @ T(visual origin) with the mesh scale applied. Units: Infinigen builds in Blender metres,
so "m".

Parts: one per URDF link that has at least one visual mesh (all of its visuals merged into one GLB). The
generator adds visual-less helper links ("l_world", "abstract_<p>_<c>", "link_abstract_*"). They are collapsed:
for every part link C, walk up the tree to the nearest ancestor part link P and collect the joints on that path.
  * each non-fixed joint on the path becomes one bundle joint P -> C (a path holding several, such as
    revolute_prismatic, produces several entries with the same parent/child pair);
  * a path with only fixed joints becomes one fixed joint (id = the joint directly above C);
  * a part link with no part ancestor (hangs off l_world) is a root part; its world joint is not emitted.
Joint "origin" = world position of the joint frame at rest (T(parent link) @ T(joint origin)); "axis" = the
URDF axis rotated into the world frame by that frame's rotation, normalised. Limits: revolute/prismatic ->
[lower, upper] from <limit>; continuous/fixed -> null.
Material: URDF <material> entries only carry a generated name and an optional texture, so the material is
taken from the part's .mtl files: every `newmtl` name except Blender's per-object default (named like the
part index). Density: not produced by the generator -> null.
Up: the URDF axis (y or z) with the largest bbox extent, which must also exceed both other extents (a chair is
taller than wide). Front: horizontal direction from the backrest centroid to the seat centroid, snapped to the
dominant signed horizontal axis. Backrest/seat parts are identified by data_infos part names ("back" vs
"seat"). If the backrest is merged into the seat mesh ("chair_seat_whole"), the fallback uses that one mesh:
from the centroid of its upper half to the centroid of its lower half. bundle["front_method"] records which rule was used.
"""
import json, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

AX = "xyz"


def _vec(s, n=3, default=0.0):
    if s is None:
        return np.full(n, default, float)
    return np.array([float(x) for x in s.split()], float)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # URDF fixed-axis roll-pitch-yaw


def _origin(el):
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, :3] = _rpy(*_vec(o.get("rpy")))
        T[:3, 3] = _vec(o.get("xyz"))
    return T


def parse_urdf(path):
    root = ET.parse(path).getroot()
    links = {}
    for l in root.findall("link"):
        vis = []
        for v in l.findall("visual"):
            m = v.find("geometry/mesh")
            if m is None:
                continue
            mat = v.find("material")
            tex = mat.find("texture") if mat is not None else None
            vis.append({"T": _origin(v), "file": m.get("filename"), "scale": _vec(m.get("scale"), default=1.0),
                        "urdf_material": None if mat is None else mat.get("name"),
                        "texture": None if tex is None else tex.get("filename")})
        links[l.get("name")] = vis
    joints = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        ax = j.find("axis")
        joints[j.get("name")] = {
            "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"), "T": _origin(j),
            "axis": _vec(ax.get("xyz")) if ax is not None else np.array([1.0, 0.0, 0.0]),
            "limit": None if lim is None else [float(lim.get("lower", 0)), float(lim.get("upper", 0))]}
    return links, joints


def resolve_mesh(urdf_dir, fname):
    p = Path(fname)
    for cand in (p if p.is_absolute() else urdf_dir / p, urdf_dir / "objs" / p.parent.name / p.name):
        if cand.exists():
            return cand
    hits = list(urdf_dir.rglob(p.name))
    return hits[0] if hits else None


def load_mesh(path):
    s = trimesh.load(path, process=False)
    geoms = list(s.geometry.values()) if isinstance(s, trimesh.Scene) else [s]
    geoms = [g for g in geoms if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not geoms:
        return None, False
    try:
        return trimesh.util.concatenate(geoms), True
    except Exception:
        return trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces, process=False) for g in geoms]), False


def mtl_materials(mesh_path, part_index):
    names, maps = [], []
    for mtl in sorted(Path(mesh_path).parent.glob("*.mtl")):
        for line in mtl.read_text(errors="ignore").splitlines():
            t = line.strip().split()
            if not t:
                continue
            if t[0] == "newmtl" and len(t) > 1:
                n = " ".join(t[1:])
                if n != str(part_index) and n not in names:
                    names.append(n)
            elif t[0].startswith("map_") and t[-1] not in maps:
                maps.append(t[-1])
    return names, maps


def convert(run):
    run = Path(run).resolve()
    urdfs = sorted((run / "native").rglob("scene.urdf"), key=lambda p: len(p.parts))
    if not urdfs:
        raise FileNotFoundError(f"no scene.urdf under {run / 'native'}")
    urdf = urdfs[0]
    udir = urdf.parent
    links, joints = parse_urdf(urdf)
    warnings = []

    parent_joint = {j["child"]: j for j in joints.values()}
    roots = [l for l in links if l not in parent_joint]
    if len(roots) != 1:
        warnings.append(f"URDF has {len(roots)} root links: {roots}")
    Tw, frame = {}, {}

    def fk(l):
        if l in Tw:
            return Tw[l]
        j = parent_joint.get(l)
        if j is None:
            Tw[l] = np.eye(4)
        else:
            frame[j["name"]] = fk(j["parent"]) @ j["T"]
            Tw[l] = frame[j["name"]]
        return Tw[l]

    for l in links:
        fk(l)

    # semantic names from data_infos_<k>.json (part index -> part_name)
    sem = {}
    for f in sorted(run.glob("native/**/data_infos_*.json")):
        try:
            for case in json.loads(f.read_text()):
                for p in case.get("part", []):
                    sem[int(Path(p["file_name"]).stem)] = p["part_name"]
        except Exception as e:  # noqa: BLE001
            warnings.append(f"could not parse {f.name}: {e}")

    out = run / "bundle"
    (out / "parts").mkdir(parents=True, exist_ok=True)
    for f in (out / "parts").glob("*.glb"):
        f.unlink()

    parts, meshes = [], {}
    for name, vis in links.items():
        if not vis:
            continue
        geo, mats, maps, tex_ok = [], [], [], True
        idx = int(m.group(1)) if (m := re.fullmatch(r"l_(\d+)", name)) else None
        for v in vis:
            mp = resolve_mesh(udir, v["file"])
            if mp is None:
                warnings.append(f"{name}: mesh {v['file']} not found")
                continue
            mesh, ok = load_mesh(mp)
            if mesh is None:
                warnings.append(f"{name}: mesh {mp.name} has no faces")
                continue
            tex_ok &= ok
            S = np.eye(4); S[:3, :3] = np.diag(v["scale"])
            mesh.apply_transform(Tw[name] @ v["T"] @ S)
            geo.append(mesh)
            n, mm = mtl_materials(mp, idx if idx is not None else Path(mp).stem)
            mats += [x for x in n if x not in mats]
            maps += [x for x in mm if x not in maps]
        if not geo:
            continue
        try:
            mesh = trimesh.util.concatenate(geo)
        except Exception:
            mesh = trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces, process=False) for g in geo])
            tex_ok = False
        if not tex_ok:
            warnings.append(f"{name}: textures could not be merged; GLB exported without texture")
        mesh.export(out / "parts" / f"{name}.glb")
        meshes[name] = mesh
        parts.append({"id": name, "file": f"parts/{name}.glb", "name": name,
                      "semantic_name": sem.get(idx), "material": "; ".join(mats) if mats else None,
                      "material_names": mats, "texture_maps": maps, "density_kg_m3": None,
                      "n_faces": int(len(mesh.faces))})

    part_ids = set(meshes)
    bj = []
    rnd = lambda v: [round(float(x), 6) for x in v]
    root_parts = []
    for c in [p["id"] for p in parts]:
        path, l = [], c
        while l in parent_joint and (l == c or l not in part_ids):
            j = parent_joint[l]
            path.append(j)
            l = j["parent"]
        if l not in part_ids:
            root_parts.append(c)
            continue
        moving = [j for j in reversed(path) if j["type"] != "fixed"]
        for j in moving or [path[0]]:
            F = frame[j["name"]]
            typ = j["type"]
            axis = F[:3, :3] @ j["axis"]
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            bj.append({"id": j["name"], "type": typ, "parent": l, "child": c,
                       "axis": None if typ == "fixed" else rnd(axis), "origin": rnd(F[:3, 3]),
                       "limits": j["limit"] if typ in ("revolute", "prismatic") else None,
                       "native_path": [x["name"] for x in reversed(path)]})
        if len(moving) > 1:
            warnings.append(f"{l}->{c}: {len(moving)} non-fixed joints in series ({[j['type'] for j in moving]}); "
                            "emitted as separate entries with the same parent/child")
    if len(root_parts) != 1:
        warnings.append(f"{len(root_parts)} root parts (no part ancestor): {root_parts}")

    allv = np.concatenate([m.vertices for m in meshes.values()])
    lo, hi = allv.min(0), allv.max(0)
    ext = hi - lo
    up_i = 1 if ext[1] >= ext[2] else 2
    up = AX[up_i]
    if not (ext[up_i] > ext[(up_i + 1) % 3] and ext[up_i] > ext[(up_i + 2) % 3]):
        warnings.append(f"up axis ambiguous: bbox extents {rnd(ext)}; picked {up} (larger of y/z)")
    hor = [i for i in range(3) if i != up_i]

    def centroid(pred):
        ms = [meshes[p["id"]] for p in parts if p["semantic_name"] and pred(p["semantic_name"].lower())]
        if not ms:
            return None
        m = trimesh.util.concatenate(ms)
        return (m.triangles_center * m.area_faces[:, None]).sum(0) / max(m.area, 1e-12)

    back = centroid(lambda n: "back" in n)
    seat = centroid(lambda n: "seat" in n and "back" not in n)
    front, how, d = None, None, None
    if back is not None and seat is not None:
        d = (seat - back)[hor]
        how = "backrest centroid -> seat centroid (area-weighted), horizontal, snapped to dominant axis"
    else:
        sm = [p["id"] for p in parts if p["semantic_name"] and "seat" in p["semantic_name"].lower()]
        if sm:
            v = meshes[sm[0]].vertices
            mid = (v[:, up_i].min() + v[:, up_i].max()) / 2
            upper, lower = v[v[:, up_i] > mid], v[v[:, up_i] <= mid]
            if len(upper) and len(lower):
                d = (lower.mean(0) - upper.mean(0))[hor]
                how = (f"no separate backrest part: in the seat mesh {sm[0]}, vertex centroid of the upper half "
                       "-> vertex centroid of the lower half, horizontal, snapped to dominant axis")
    if d is not None:
        k = int(np.argmax(np.abs(d)))
        front = ("+" if d[k] > 0 else "-") + AX[hor[k]]
        how += f"; raw horizontal vector ({AX[hor[0]]},{AX[hor[1]]}) = {rnd(d)}"
        if abs(d[k]) < 2 * abs(d[1 - k]) or abs(d[k]) < 0.02:
            warnings.append(f"front estimate weak: {rnd(d)}")
    else:
        how = "not determined: no parts named back/seat in data_infos"

    # consistency: are the vertical (swivel/lift) joint axes along the declared up axis?
    vert = [j for j in bj if j["type"] in ("continuous", "revolute", "prismatic") and j["axis"]
            and abs(j["axis"][up_i]) > 0.99]
    if not vert:
        warnings.append(f"no non-fixed joint axis is parallel to up={up}; joint axes may be in a different frame "
                        "than the meshes (upstream writes Blender Z-up axes while rotating meshes)")

    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    types = {}
    for j in bj:
        types[j["type"]] = types.get(j["type"], 0) + 1
    bundle = {
        "source": {"system": "infinite-mobility", "factory": meta.get("factory", "OfficeChairFactory"),
                   "seed": meta.get("seed"), "code_commit": meta.get("commit"), "urdf": str(urdf)},
        "up": up, "front": front, "front_method": how, "units": "m",
        "frame": "URDF world frame at rest (all joint values 0), forward kinematics over joint origins",
        "bbox_m": [rnd(lo), rnd(hi)],
        "parts": parts, "joints": bj, "root_parts": root_parts,
        "joint_type_counts": types,
        "notes": ("Visual-less helper links (l_world, abstract_*) are collapsed; each bundle joint keeps the native "
                  "joint name as id and lists the collapsed native chain in native_path. Material = .mtl newmtl "
                  "names (Infinigen shader names); density not provided by the generator."),
        "warnings": warnings,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1))
    print(f"{run.name}: {len(parts)} parts, {len(bj)} joints {types}, up={up}, front={front}, "
          f"bbox={rnd(ext)}, warnings={len(warnings)}")
    return bundle


if __name__ == "__main__":
    for r in sys.argv[1:]:
        convert(r)
