"""The v2 tool protocol: assemble an articulated object from pool parts (Tier A),
from pool parts plus parts the model creates (Tier B), or from parts the model
creates and nothing else (Tier C).

One environment, three tiers; the only difference is what a part may come from:
Tier A gives the catalog, Tier B the catalog and `create_part`, Tier C only
`create_part` -- in Tier C the catalog is not listed, not shown and not
placeable, so every part in the design is one the model designed. Every call
returns JSON. Errors are returned to the model as tool
results and counted, never repaired. The state is plain JSON so the same
episode can be driven in-process (local models) or from a shell (terminal
agents) and replayed.

What a model must produce, and why each piece exists:
  parts      posed meshes                  -> Level 1, 2.1, 3.2 (computed)
  role       what each part is             -> 2.1 predicates, 3.1 coverage, 4.3 subsystems
  material   one of list_materials         -> 4.1 (declared, then checked)
  joints     type / axis / origin / limits -> 2.2 (matched and swept)
  sequence   order and insertion direction -> 4.2 (precedence, hand count, prefix stability)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from ppbench.v2 import analysis, csg, materials, voxel
from ppbench.v2.design import JOINT_TYPES, Design, Joint, Part, rotation_matrix
from ppbench.v2.task import RESULTS, Task, task_dir

ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,48}$")
ANCHORS = {"center": (0, 0, 0), "top": (0, 0, 1), "bottom": (0, 0, -1), "left": (-1, 0, 0),
           "right": (1, 0, 0), "front": (0, -1, 0), "back": (0, 1, 0)}
PROTOCOL_VERSION = "v2.2"   # v2.1 added `measure` and view notes; v2.2 made check/prompt/evaluator share one joint-connection rule
BUDGETS = {"max_tool_calls": 160, "max_check_calls": 6, "max_render_calls": 4, "max_create_calls": 30,
           "max_parts": 80}
# Tier C has no catalog to pick from, so the create budget stands in for one: the core-set pools hold 39 parts
# on average (14 to 160). Tier B keeps 30, which only 2 of its 482 main-cell episodes ever reached.
TIER_C_BUDGETS = {"max_create_calls": 60}
MAX_COORD = 5.0

SYSTEM = """You assemble a physical object out of 3D parts by calling tools. The result is judged as a real,
buildable, working product: it must hold together, stand, look like what was asked for, move where the
real object moves, be made of sensible materials, and be buildable in the order you give.

WORLD FRAME
- Units are metres. +Z is up. The ground is the plane z = 0. The FRONT of the object faces -Y
  (a person sitting on a chair faces -Y). Standing in front of the object (at -Y) and looking at it,
  +X is to your right.
- Real-world size matters. Build the object at its real size.

PARTS
{parts_source}
- Every part's local origin is the centre of its bounding box. place_part puts that centre at
  `position` and applies `rotation` = [rx, ry, rz] degrees about the world X, then Y, then Z axes.
  So an unrotated part placed at z = size_z / 2 rests exactly on the ground.
- attach moves a part so that one face centre of its bounding box (top, bottom, left, right, front,
  back, center) lands on a face centre of another part's box, plus an optional offset. It is the
  easiest way to stack parts exactly: attach(child, "bottom", parent, "top").
- Parts count as connected when their real surfaces touch (gap under about 4 mm), or when a joint joins
  them at a place where both parts are: the joint's origin within 2 cm of each part's surface. A joint
  whose origin is far from either part connects nothing.
  Parts that interpenetrate a lot collide. Anything not connected to the rest is floating.

DECLARATIONS (all are scored)
- role: say what each part is, in plain words ("seat", "backrest", "caster wheel", ...).
- material: choose one per part from list_materials.
- joints: declare every place the real object moves. add_joint(joint_id, type, parent, child, axis,
  origin, limits). type is one of fixed, revolute, continuous, prismatic, cylindrical, ball. axis is a
  direction in world coordinates; origin is a world point on the axis where parent and child meet;
  limits = [low, high] in radians (revolute) or metres (prismatic, cylindrical = the sliding range);
  omit limits for continuous. The child moves relative to the parent, carrying whatever is attached
  to the child.
- assembly sequence: set_assembly_sequence with every part once, in the order a person would build
  the object, each step optionally with the direction the part travels as it is put in place
  (e.g. [0, 0, -1] for lowering it from above).

ROUNDS AND BUDGETS
- submit ends a round. If review rounds remain you are shown a check of the design and renders, and
  you may revise it and submit again. After the final round the episode ends.
- check and render are metered; use them to verify, not to search. Tool calls are budgeted.

Keep reasoning brief. Prefer making tool calls over explaining."""

PARTS_CATALOG = """- list_parts shows the catalog: every part has an id, a name and size_m = [x, y, z] extents in metres
  in its default orientation. Parts are reusable: place the same part id as many times as you need."""

PARTS_CREATED_ONLY = """- There is no catalog in this episode. Every part is one you design yourself with create_part (below);
  list_parts shows the parts you have created so far, with an id, a name and size_m = [x, y, z] extents
  in metres in the part's default orientation. Created parts are reusable: place the same part id as
  many times as you need, so design a part once and place it wherever the object repeats it."""

SYSTEM_TIER_B = """
CREATING PARTS (this episode allows it)
- If no catalog part fits, create one with create_part(name, program). A program is JSON:
  {"nodes": [...], "output": node_id}, in metres, in the part's own frame. Primitives are centred on
  the origin: {"id","op":"box","size":[x,y,z]}, {"op":"cylinder","radius","height","radius_top"?}
  (axis +Z), {"op":"sphere","radius"}, {"op":"torus","major_radius","minor_radius"} (in the XY plane),
  {"op":"extrude","polygon":[[x,y],...],"height"} (along Z), {"op":"revolve","profile":[[r,z],...],
  "degrees"?} (about Z). Operations on earlier node ids: {"op":"union"|"difference"|"intersection"|
  "hull","inputs":[ids]} (difference subtracts the rest from the first), {"op":"transform","input":id,
  "position":[x,y,z],"rotation":[rx,ry,rz]}, {"op":"mirror","input":id,"normal":[x,y,z]}.
- The created part is recentred on its bounding box and is then placed with place_part like any
  catalog part, using its name as part_id. Prefer catalog parts when they fit; create what the catalog
  cannot give you."""

SYSTEM_TIER_C = """
CREATING PARTS (this episode has no catalog: every part starts here)
- Create a part with create_part(name, program), then place copies of it with place_part(name, ...).
  A program is JSON: {"nodes": [...], "output": node_id}, in metres, in the part's own frame. Primitives
  are centred on the origin: {"id","op":"box","size":[x,y,z]}, {"op":"cylinder","radius","height",
  "radius_top"?} (axis +Z), {"op":"sphere","radius"}, {"op":"torus","major_radius","minor_radius"}
  (in the XY plane), {"op":"extrude","polygon":[[x,y],...],"height"} (along Z), {"op":"revolve",
  "profile":[[r,z],...],"degrees"?} (about Z). Operations on earlier node ids: {"op":"union"|
  "difference"|"intersection"|"hull","inputs":[ids]} (difference subtracts the rest from the first),
  {"op":"transform","input":id,"position":[x,y,z],"rotation":[rx,ry,rz]}, {"op":"mirror","input":id,
  "normal":[x,y,z]}.
- The created part is recentred on its bounding box, and is then placed with place_part using its name
  as part_id. Design each part at the real size it has in the real object."""

TOOLS = [
    ("list_parts", {"query": "optional substring filter on the part name"},
     "The catalog (and parts you created): id, name, size_m."),
    ("inspect_part", {"part_id": "catalog id or created part name"},
     "Size, volume and a thumbnail image path for one part."),
    ("view_catalog_sheet", {}, "A picture of every catalog part, labelled with id, name and size (not metered)."),
    ("list_materials", {}, "The material vocabulary with densities."),
    ("place_part", {"part_id": "str", "instance_id": "your name for this placed copy", "position": "[x,y,z] m",
                    "rotation": "[rx,ry,rz] deg, optional", "role": "str, optional", "material": "str, optional"},
     "Place a copy of a part. Returns its world bounding box, what it touches and any collision."),
    ("move_part", {"instance_id": "str", "position": "[x,y,z], optional", "rotation": "[rx,ry,rz], optional",
                   "translate_by": "[dx,dy,dz], optional"}, "Move or rotate a placed part."),
    ("attach", {"instance_id": "part to move", "anchor": "center|top|bottom|left|right|front|back",
                "target": "instance id, or 'ground'", "target_anchor": "center|top|bottom|left|right|front|back",
                "offset": "[dx,dy,dz], optional"},
     "Move a part so its anchor lands on the target's anchor plus offset (target 'ground': the point (x, y, 0) below it)."),
    ("measure", {"a": "instance id", "b": "instance id, or 'ground'"},
     "Closest distance between the two parts' real surfaces (not their boxes), the closest points on each, "
     "whether they count as touching, and any overlap. Use it to close gaps and to place joint origins."),
    ("remove_part", {"instance_id": "str"}, "Remove a placed part (and joints that use it)."),
    ("set_part_info", {"instance_id": "str", "role": "optional", "material": "optional"}, "Set role and/or material."),
    ("add_joint", {"joint_id": "str", "type": "fixed|revolute|continuous|prismatic|cylindrical|ball",
                   "parent": "instance id", "child": "instance id", "axis": "[x,y,z]", "origin": "[x,y,z]",
                   "limits": "[low, high], optional"}, "Declare a joint."),
    ("remove_joint", {"joint_id": "str"}, "Remove a joint."),
    ("set_assembly_sequence", {"steps": "[{\"part\": instance_id, \"direction\": [x,y,z] optional}, ...]"},
     "Declare the build order (every part exactly once)."),
    ("get_scene", {}, "Every placed part with pose, world box, role, material; joints; sequence; budgets left."),
    ("check", {}, "METERED. Collisions, connectivity, ground contact, stability, joint issues, missing declarations."),
    ("render", {"views": "optional list from cond, iso, front, side, back_iso, top"},
     "METERED. Pictures of the current design (paths to PNG files)."),
    ("submit", {}, "End the current round."),
]
TOOL_CREATE = ("create_part", {"name": "new part id", "program": "CSG program (see CREATING PARTS)"},
               "Create a new part from a CSG program (Tier B and Tier C).")


# Tier C has no catalog, so the tools that show or reach one are gone or say something else.
TIER_C_HIDDEN = {"view_catalog_sheet"}
TIER_C_DOCS = {
    "list_parts": ("The parts you have created: id, name, size_m. There is no catalog in this episode.", None),
    "inspect_part": ("Size, volume and a thumbnail image path for one part you created.", {"part_id": "created part name"}),
}


def tool_docs(tier: str) -> str:
    tools = TOOLS + ([TOOL_CREATE] if tier in ("B", "C") else [])
    lines = []
    for name, args, desc in tools:
        if tier == "C":
            if name in TIER_C_HIDDEN:
                continue
            desc, args = (TIER_C_DOCS[name][0], TIER_C_DOCS[name][1] or args) if name in TIER_C_DOCS else (desc, args)
        a = ", ".join(f"{k}: {v}" for k, v in args.items())
        lines.append(f"- {name}({a})\n    {desc}")
    return "\n".join(lines)


def system_prompt(tier: str) -> str:
    base = SYSTEM.format(parts_source=PARTS_CREATED_ONLY if tier == "C" else PARTS_CATALOG)
    return base + {"B": SYSTEM_TIER_B, "C": SYSTEM_TIER_C}.get(tier, "")


class ToolError(Exception):
    pass


def _vec3(v, name):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ToolError(f"{name} must be a list of 3 numbers, got {v!r}")
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be a list of 3 numbers, got {v!r}")
    if not all(np.isfinite(out)) or max(abs(x) for x in out) > (MAX_COORD if name != "rotation" else 3600):
        raise ToolError(f"{name} has a non-finite or out-of-range value: {v!r}")
    return out


class AssemblyEnv:
    def __init__(self, task: Task, tier: str, condition_id: str, rounds: int = 2, budgets=None,
                 state: dict | None = None, workdir=None):
        self.task = task
        self.tier = tier.upper()
        assert self.tier in ("A", "B", "C")
        if self.tier == "C":
            budgets = dict(TIER_C_BUDGETS, **(budgets or {}))
        self.workdir = Path(workdir or (RESULTS / task.id / "scratch"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.s = state or {
            "protocol": PROTOCOL_VERSION,
            "task_id": task.id, "tier": self.tier, "condition_id": condition_id, "rounds": int(rounds),
            "round": 1, "done": False, "instances": {}, "created": {}, "joints": {}, "sequence": None,
            "counters": {"tool_calls": 0, "errors": 0, "check": 0, "render": 0, "create": 0},
            "round_counters": [], "budgets": dict(BUDGETS, **(budgets or {})), "trace": [], "started": time.time(),
            "pool_mesh_version": getattr(task, "POOL_MESH_VERSION", 1),
        }
        self._pool = {p["pool_part_id"]: p for p in task.pool}
        self._created_mesh: dict = {}
        self._occ: dict = {}
        self.occ_dir = None        # set by the shell session: occupancies persist between calls (each call is a new process)
        self._occ_saved: dict = {}

    # ------------------------------------------------------------ geometry
    def _local(self, source):
        if source["kind"] == "pool":
            return self.task.pool_mesh(source["id"])
        name = source["id"]
        if name not in self._created_mesh:
            m, _ = csg.compile_program(self.s["created"][name]["program"])
            self._created_mesh[name] = (np.asarray(m.vertices), np.asarray(m.faces))
        return self._created_mesh[name]

    def _world(self, iid):
        inst = self.s["instances"][iid]
        v, f = self._local(inst["source"])
        R = rotation_matrix(inst["rotation"])
        return v @ R.T + np.asarray(inst["position"]), f

    def _occupancy(self, iid):
        inst = self.s["instances"][iid]
        key = json.dumps([inst["source"], inst["position"], inst["rotation"]])
        hit = self._occ.get(iid)
        if (hit is None or hit[0] != key) and self.occ_dir is not None:
            import hashlib
            fp = Path(self.occ_dir) / (hashlib.sha1(key.encode()).hexdigest()[:32] + ".npz")
            if fp.exists():
                try:
                    o = voxel.occ_load(fp)
                    self._occ[iid] = (key, o)
                    self._occ_saved[key] = (o._core is not None, o._halo is not None)
                    hit = self._occ[iid]
                except Exception:
                    hit = None
        if hit is None or hit[0] != key:
            v, f = self._world(iid)
            try:
                self._occ[iid] = (key, voxel.voxelize(v, f))
            except voxel.VoxelTooLarge as e:
                raise ToolError(f"{iid}: {e}; parts must stay within a few metres of the origin")
        return self._occ[iid][1]

    def parts(self) -> list[Part]:
        out = []
        for iid, inst in self.s["instances"].items():
            v, f = self._world(iid)
            out.append(Part(iid, v, f, inst.get("role"), inst.get("material"),
                            {**inst["source"], "position": inst["position"], "rotation": inst["rotation"]}))
        return out

    def joints(self) -> list[Joint]:
        return [Joint(**j) for j in self.s["joints"].values()]

    def _bbox(self, iid):
        v, _ = self._world(iid)
        return v.min(0), v.max(0)

    def _feedback(self, iid):
        occ = self._occupancy(iid)
        lo, hi = self._bbox(iid)
        touches, collides = [], []
        jp = {frozenset((j["parent"], j["child"])) for j in self.s["joints"].values()}
        for other in self.s["instances"]:
            if other == iid:
                continue
            olo, ohi = self._bbox(other)
            if np.any(lo > ohi + 0.01) or np.any(olo > hi + 0.01):
                continue
            o = self._occupancy(other)
            if not voxel.touching(occ, o):
                continue
            deep = voxel.overlap_cells(occ, o) * occ.res ** 3
            frac = deep / max(min(occ.volume, o.volume), 1e-12)
            if deep > analysis.COLLIDE_ABS_M3 and frac > analysis.COLLIDE_FRAC and frozenset((iid, other)) not in jp:
                collides.append({"with": other, "overlap_cm3": round(deep * 1e6, 2), "frac_of_smaller_part": round(frac, 3)})
            else:
                touches.append(other)
        return {"instance_id": iid, "world_bbox_min": np.round(lo, 4).tolist(), "world_bbox_max": np.round(hi, 4).tolist(),
                "touches": touches, "collides": collides,
                "below_ground": bool(lo[2] < -0.005)}

    # ------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict | None) -> dict:
        args = args or {}
        c = self.s["counters"]
        t0 = time.time()
        if self.s["done"]:
            return {"ok": False, "error": "the episode has ended"}
        if c["tool_calls"] >= self.s["budgets"]["max_tool_calls"] * self.s["round"] and name != "submit":
            return {"ok": False, "error": "tool-call budget for this round is exhausted; call submit"}
        c["tool_calls"] += 1
        fn = getattr(self, "t_" + str(name), None)
        if fn is None or (name == "create_part" and self.tier == "A") or \
                (name in TIER_C_HIDDEN and self.tier == "C"):
            res = {"ok": False, "error": f"unknown tool {name!r}"}
        elif not isinstance(args, dict):
            res = {"ok": False, "error": "arguments must be a JSON object"}
        else:
            try:
                res = {"ok": True, **fn(**args)}
            except TypeError as e:
                res = {"ok": False, "error": f"bad arguments for {name}: {e}"}
            except (ToolError, csg.CSGError, KeyError) as e:
                res = {"ok": False, "error": str(e).strip("'\"")}
        if not res.get("ok"):
            c["errors"] += 1
        self.s["trace"].append({"t": round(time.time() - self.s["started"], 2), "round": self.s["round"], "tool": name,
                                "args": args, "ok": res.get("ok"), "error": res.get("error"),
                                "wall_s": round(time.time() - t0, 3)})
        return res

    # ------------------------------------------------------------ tools
    def t_list_parts(self, query: str | None = None):
        q = (query or "").lower()
        rows = [] if self.tier == "C" else [{"part_id": p["pool_part_id"], "name": p["name"], "size_m": p["size_m"]}
                                           for p in self.task.pool if q in p["name"].lower()]
        rows += [{"part_id": n, "name": n, "size_m": c["info"]["size_m"], "created": True}
                 for n, c in self.s["created"].items() if q in n.lower()]
        note = "size_m = [x, y, z] extents in the part's default orientation; Z is up"
        if self.tier == "C":
            note = ("there is no catalog in this episode: these are the parts you created. " + note
                    if rows else "there is no catalog in this episode and you have created no parts yet; "
                                 "design one with create_part, then place copies of it")
        return {"parts": rows, "note": note}

    def t_inspect_part(self, part_id: str):
        if part_id in self._pool and self.tier != "C":
            v, f = self.task.pool_mesh(part_id)
            import trimesh
            m = trimesh.Trimesh(v, f, process=False)
            thumb = task_dir(self.task.id) / "pool_sheet" / "thumbs" / f"{part_id}.png"
            return {"part_id": part_id, "name": self._pool[part_id]["name"], "size_m": np.round(v.max(0) - v.min(0), 4).tolist(),
                    "watertight": bool(m.is_watertight), "surface_area_m2": round(float(m.area), 4),
                    "thumbnail": str(thumb) if thumb.exists() else None}
        if part_id in self.s["created"]:
            return {"part_id": part_id, "created": True, **self.s["created"][part_id]["info"]}
        raise ToolError(self._no_part(part_id))

    def t_list_materials(self):
        return {"materials": materials.listing()}

    def t_view_catalog_sheet(self):
        from ppbench.v2 import render
        p = render.sheet_all(self.task.id)
        if not p:
            raise ToolError("the catalog sheet has not been rendered")
        return {"images": [p], "note": "labels: id, name, size x*y*z in metres"}

    def _material(self, m):
        if m is None:
            return None
        r = materials.resolve(m)
        if r is None:
            raise ToolError(f"unknown material {m!r}; call list_materials and use a 'material' value from it")
        return r[0]

    def t_place_part(self, part_id: str, instance_id: str, position, rotation=None, role=None, material=None):
        if not ID_RE.match(str(instance_id)):
            raise ToolError("instance_id must be 1-48 letters, digits, _ or -")
        if instance_id in self.s["instances"]:
            raise ToolError(f"instance {instance_id!r} already exists; use move_part or a new id")
        if len(self.s["instances"]) >= self.s["budgets"]["max_parts"]:
            raise ToolError("part limit reached")
        if part_id in self._pool and self.tier != "C":
            src = {"kind": "pool", "id": part_id}
        elif part_id in self.s["created"]:
            src = {"kind": "created", "id": part_id}
        else:
            raise ToolError(self._no_part(part_id))
        self.s["instances"][instance_id] = {"source": src, "position": _vec3(position, "position"),
                                            "rotation": _vec3(rotation or [0, 0, 0], "rotation"),
                                            "role": str(role) if role else None, "material": self._material(material)}
        return self._feedback(instance_id)

    def _no_part(self, part_id):
        if self.tier == "C":
            return (f"no part {part_id!r}: this episode has no catalog, so a part exists only after you "
                    f"create it with create_part; call list_parts to see the ones you have")
        return f"no part {part_id!r}; call list_parts"

    def _inst(self, iid):
        if iid not in self.s["instances"]:
            raise ToolError(f"no placed part {iid!r}; call get_scene")
        return self.s["instances"][iid]

    def t_move_part(self, instance_id: str, position=None, rotation=None, translate_by=None):
        inst = self._inst(instance_id)
        if position is not None:
            inst["position"] = _vec3(position, "position")
        if translate_by is not None:
            inst["position"] = _vec3((np.asarray(inst["position"]) + _vec3(translate_by, "translate_by")).tolist(), "position")
        if rotation is not None:
            inst["rotation"] = _vec3(rotation, "rotation")
        return self._feedback(instance_id)

    def t_attach(self, instance_id: str, anchor: str, target: str, target_anchor: str = "top", offset=None):
        inst = self._inst(instance_id)
        if anchor not in ANCHORS or target_anchor not in ANCHORS:
            raise ToolError(f"anchors are {sorted(ANCHORS)}")
        lo, hi = self._bbox(instance_id)
        c, h = (lo + hi) / 2, (hi - lo) / 2
        a = c + np.array(ANCHORS[anchor]) * h
        if target == "ground":
            t = np.array([a[0], a[1], 0.0])
        else:
            if target == instance_id:
                raise ToolError("a part cannot attach to itself")
            self._inst(target)
            tlo, thi = self._bbox(target)
            t = (tlo + thi) / 2 + np.array(ANCHORS[target_anchor]) * (thi - tlo) / 2
        off = np.asarray(_vec3(offset or [0, 0, 0], "offset"))
        inst["position"] = _vec3((np.asarray(inst["position"]) + (t + off - a)).tolist(), "position")
        return self._feedback(instance_id)

    def t_measure(self, a: str, b: str):
        self._inst(a)
        oa = self._occupancy(a)
        if b == "ground":
            v, _ = self._world(a)
            k = int(np.argmin(v[:, 2]))
            return {"a": a, "b": "ground", "gap_m": round(float(v[k, 2]), 4), "closest_point_a": np.round(v[k], 4).tolist(),
                    "note": "negative gap = below the ground"}
        if a == b:
            raise ToolError("a and b must differ")
        self._inst(b)
        ob = self._occupancy(b)
        from scipy.spatial import cKDTree
        va, _ = self._world(a)
        vb, _ = self._world(b)
        pa = np.vstack([oa.points, va[:: max(1, len(va) // 20000)]])
        pb = np.vstack([ob.points, vb[:: max(1, len(vb) // 20000)]])
        d, j = cKDTree(pb).query(pa)
        i = int(np.argmin(d))
        deep = voxel.overlap_cells(oa, ob) * oa.res ** 3
        touch = voxel.touching(oa, ob)
        return {"a": a, "b": b, "gap_m": round(float(d[i]), 4) if not touch or deep == 0 else 0.0,
                "closest_point_a": np.round(pa[i], 4).tolist(), "closest_point_b": np.round(pb[j[i]], 4).tolist(),
                "midpoint": np.round((pa[i] + pb[j[i]]) / 2, 4).tolist(), "touching": bool(touch),
                "overlap_cm3": round(deep * 1e6, 2),
                "note": "touching means the surfaces are within about 4 mm; a small overlap is allowed, a large one is a collision"}

    def t_remove_part(self, instance_id: str):
        self._inst(instance_id)
        del self.s["instances"][instance_id]
        self._occ.pop(instance_id, None)
        gone = [k for k, j in self.s["joints"].items() if instance_id in (j["parent"], j["child"])]
        for k in gone:
            del self.s["joints"][k]
        if self.s["sequence"]:
            self.s["sequence"] = [st for st in self.s["sequence"] if st["part"] != instance_id]
        return {"removed": instance_id, "removed_joints": gone}

    def t_set_part_info(self, instance_id: str, role=None, material=None):
        inst = self._inst(instance_id)
        if role is not None:
            inst["role"] = str(role)
        if material is not None:
            inst["material"] = self._material(material)
        return {"instance_id": instance_id, "role": inst["role"], "material": inst["material"]}

    def t_add_joint(self, joint_id: str, type: str, parent: str, child: str, axis=None, origin=None, limits=None):
        if not ID_RE.match(str(joint_id)):
            raise ToolError("joint_id must be 1-48 letters, digits, _ or -")
        if joint_id in self.s["joints"]:
            raise ToolError(f"joint {joint_id!r} exists; remove_joint first")
        if type not in JOINT_TYPES:
            raise ToolError(f"type must be one of {list(JOINT_TYPES)}")
        self._inst(parent)
        self._inst(child)
        if parent == child:
            raise ToolError("parent and child must differ")
        notes = []
        ax = None
        if type != "fixed":
            if axis is None and type != "ball":
                raise ToolError("axis is required for a moving joint")
            if axis is not None:
                ax = np.asarray(_vec3(axis, "axis"))
                if np.linalg.norm(ax) < 1e-9:
                    raise ToolError("axis must be non-zero")
                ax = (ax / np.linalg.norm(ax)).round(6).tolist()
            if origin is None:
                raise ToolError("origin is required for a moving joint")
        org = _vec3(origin, "origin") if origin is not None else None
        lim = None
        if limits is not None:
            if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                raise ToolError("limits must be [low, high]")
            lim = [float(limits[0]), float(limits[1])]
            if not lim[0] <= lim[1]:
                raise ToolError("limits must satisfy low <= high")
        elif type in ("revolute", "prismatic", "cylindrical"):
            notes.append(f"no limits given for a {type} joint; its range counts as undeclared")
        if any(j["child"] == child for j in self.s["joints"].values()):
            notes.append(f"{child} is already the child of another joint; a part should have one parent joint")
        self.s["joints"][joint_id] = {"id": joint_id, "type": type, "parent": parent, "child": child,
                                      "axis": ax, "origin": org, "limits": lim}
        if org is not None:
            dp = analysis.surface_distance(org, self._occupancy(parent))
            dc = analysis.surface_distance(org, self._occupancy(child))
            if max(dp, dc) > 0.02:
                notes.append(f"origin is {dp:.3f} m from {parent} and {dc:.3f} m from {child}")
        return {"joint_id": joint_id, "notes": notes}

    def t_remove_joint(self, joint_id: str):
        if joint_id not in self.s["joints"]:
            raise ToolError(f"no joint {joint_id!r}")
        del self.s["joints"][joint_id]
        return {"removed": joint_id}

    def t_set_assembly_sequence(self, steps):
        if not isinstance(steps, list) or not steps:
            raise ToolError("steps must be a non-empty list of {part, direction?}")
        seq, seen = [], set()
        for k, st in enumerate(steps):
            if isinstance(st, str):
                st = {"part": st}
            if not isinstance(st, dict) or "part" not in st:
                raise ToolError(f"step {k} must be an object with 'part'")
            self._inst(st["part"])
            if st["part"] in seen:
                raise ToolError(f"{st['part']} appears twice")
            seen.add(st["part"])
            d = st.get("direction")
            if d is not None:
                d = np.asarray(_vec3(d, "direction"))
                if np.linalg.norm(d) < 1e-9:
                    raise ToolError(f"step {k}: direction must be non-zero")
                d = (d / np.linalg.norm(d)).round(6).tolist()
            seq.append({"part": st["part"], "direction": d})
        self.s["sequence"] = seq
        missing = [i for i in self.s["instances"] if i not in seen]
        return {"n_steps": len(seq), "parts_missing_from_sequence": missing}

    def t_create_part(self, name: str, program):
        if self.tier == "A":
            raise ToolError("unknown tool 'create_part'")
        c = self.s["counters"]
        if c["create"] >= self.s["budgets"]["max_create_calls"]:
            raise ToolError("create_part budget exhausted")
        c["create"] += 1
        if not ID_RE.match(str(name)):
            raise ToolError("name must be 1-48 letters, digits, _ or -")
        if name in self.s["created"] or (name in self._pool and self.tier != "C"):
            raise ToolError(f"a part named {name!r} already exists; pick a new name")
        if isinstance(program, str):
            try:
                program = json.loads(program)
            except json.JSONDecodeError as e:
                raise ToolError(f"program is not valid JSON: {e}")
        m, info = csg.compile_program(program)
        self.s["created"][name] = {"program": program, "info": info}
        self._created_mesh[name] = (np.asarray(m.vertices), np.asarray(m.faces))
        warn = [] if info["n_pieces"] == 1 else [f"the part has {info['n_pieces']} separate pieces; a part should be one solid"]
        return {"part_id": name, **info, "warnings": warn}

    def t_get_scene(self):
        inst = []
        for iid, s in self.s["instances"].items():
            lo, hi = self._bbox(iid)
            inst.append({"instance_id": iid, "part_id": s["source"]["id"], "position": s["position"], "rotation": s["rotation"],
                         "world_bbox_min": np.round(lo, 4).tolist(), "world_bbox_max": np.round(hi, 4).tolist(),
                         "role": s["role"], "material": s["material"]})
        b, c = self.s["budgets"], self.s["counters"]
        return {"round": self.s["round"], "rounds": self.s["rounds"], "instances": inst,
                "joints": list(self.s["joints"].values()), "sequence": self.s["sequence"],
                "created_parts": sorted(self.s["created"]),
                "budget_left": {"tool_calls": b["max_tool_calls"] * self.s["round"] - c["tool_calls"],
                                "check": b["max_check_calls"] - c["check"], "render": b["max_render_calls"] - c["render"],
                                **({"create_part": b["max_create_calls"] - c["create"]} if self.tier != "A" else {})}}

    def _check(self):
        parts = self.parts()
        if not parts:
            return {"n_parts": 0}
        occs = [self._occupancy(p.id) for p in parts]
        rep, _, _ = analysis.summarize(parts, self.joints(), occs=occs)
        rep["parts_without_role"] = [p.id for p in parts if not p.role]
        rep["parts_without_material"] = [p.id for p in parts if not p.material]
        seq = self.s["sequence"] or []
        rep["parts_missing_from_sequence"] = [p.id for p in parts if p.id not in {s["part"] for s in seq}]
        rep["n_joints"] = len(self.s["joints"])
        return rep

    def t_check(self):
        c = self.s["counters"]
        if c["check"] >= self.s["budgets"]["max_check_calls"]:
            raise ToolError("check budget exhausted")
        c["check"] += 1
        return self._check()

    def _render(self, views, tag):
        parts = self.parts()
        if not parts:
            raise ToolError("nothing to render yet")
        from ppbench.v2 import render
        views = [v for v in (views or ["cond", "iso", "front", "side"]) if v in render.VIEWS] or ["iso"]
        out = self.workdir / "renders" / f"r{self.s['round']}_{tag}"
        paths = render.render_parts(parts, out, views=views, size=448, samples=16)
        g = render.grid([paths[v] for v in views], f"{out}_grid.png", [f"view: {v}" for v in views],
                        cols=2 if len(views) > 1 else 1, cell=448)
        return {"images": [g], "view_files": [paths[v] for v in views], "views": views,
                "note": "one picture with the views tiled; each placed part has its own colour; the grey plane is the ground z = 0. "
                        "front: camera at -Y looking toward +Y, +X to the right. side: camera at +X, the front (-Y) on the left. "
                        "top: seen from above with the front (-Y) at the bottom of the picture. iso and cond: from the front-right, above. "
                        "back_iso: from the back-left, above."}

    def t_render(self, views=None):
        c = self.s["counters"]
        if c["render"] >= self.s["budgets"]["max_render_calls"]:
            raise ToolError("render budget exhausted")
        c["render"] += 1
        return self._render(views, f"call{c['render']}")

    def t_submit(self):
        self.s["round_counters"].append(dict(self.s["counters"]))
        if self.s["round"] >= self.s["rounds"]:
            self.s["done"] = True
            return {"episode_done": True}
        review = {"check": self._check()}
        try:
            review.update(self._render(["cond", "iso", "front", "side"], "review"))
        except Exception as e:  # a failed render is reported, the round still advances
            review["render_error"] = str(e)[:300]
        self.s["round"] += 1
        return {"episode_done": False, "next_round": self.s["round"], "review": review,
                "message": "Review the check and pictures, fix what is wrong, then submit again."}

    # ------------------------------------------------------------ export
    def design(self, system: str, extra_meta=None) -> Design:
        c = self.s["counters"]
        meta = {"protocol": self.s.get("protocol", "v2.0"),
                "task_id": self.task.id, "tier": self.tier, "condition_id": self.s["condition_id"],
                "rounds": self.s["rounds"], "rounds_completed": len(self.s["round_counters"]),
                "submitted_final": self.s["done"], "counters": c, "round_counters": self.s["round_counters"],
                "created_parts": {n: v["info"] for n, v in self.s["created"].items()},
                "scale_mode": "metric", "declares": {"roles": True, "materials": True, "joints": True, "sequence": True},
                "task_source": self.task.source, "wall_s": round(time.time() - self.s["started"], 1),
                "pool_mesh_version": self.s.get("pool_mesh_version", 1), **(extra_meta or {})}
        return Design(system, self.tier, self.parts(), self.joints(), self.s["sequence"], meta)

    def save_occ_cache(self):
        """Write occupancies computed (or extended with core/halo) during this call to occ_dir."""
        if self.occ_dir is None:
            return
        import hashlib
        Path(self.occ_dir).mkdir(parents=True, exist_ok=True)
        for iid, (key, o) in list(self._occ.items()):
            flags = (o._core is not None, o._halo is not None)
            if self._occ_saved.get(key) == flags:
                continue
            voxel.occ_save(o, Path(self.occ_dir) / (hashlib.sha1(key.encode()).hexdigest()[:32] + ".npz"))
            self._occ_saved[key] = flags

    def to_state(self):
        return self.s

    @classmethod
    def from_state(cls, st, task: Task, workdir=None):
        return cls(task, st["tier"], st["condition_id"], st["rounds"], state=st, workdir=workdir)
