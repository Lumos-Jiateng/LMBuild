"""The part catalogue, memory-mapped and cached.

Three sources, all already on disk:

  inset/*.ply     watertight collision meshes, 21,084 parts, inset 0.25 LDU
  lib_2048/       2,048 surface points, normals and 324,998 port frames
  part_aliases    LDraw stem canonicalisation, so obsolete refs still resolve

Per part we derive and cache: exact enclosed volume, exact centroid, the axis
aligned bounding box, and an occupancy grid at the requested resolution.
Voxelising a complex part costs a couple of seconds, so the cache is what makes
whole-design evaluation practical -- a design reuses a handful of distinct parts.
"""
from __future__ import annotations

import json
import lzma
import re
from dataclasses import dataclass

import numpy as np

from ppbench import config as C
from ppbench.geom.voxel import mesh_volume_centroid, voxelize_watertight, watertight
from ppbench.core.plyio import read_ply


@dataclass
class Part:
    stem: str
    volume_ldu3: float
    centroid: np.ndarray          # (3,) LDraw part frame, LDU
    bbox: np.ndarray              # (2, 3) lo / hi
    occ: np.ndarray               # bool (nx, ny, nz)
    origin: np.ndarray            # (3,) grid origin
    res: float
    watertight: bool = True       # False => occ and volume are not trustworthy
    n_boundary_edges: int = 0
    volume_source: str = "mesh"   # "mesh" (exact) or "voxel" (fallback)

    @property
    def mass_kg(self) -> float:
        return self.volume_ldu3 * C.LDU3_TO_M3 * C.ABS_DENSITY_KG_M3


# LDraw prints a decoration onto a base part by suffixing the stem: `p`, `pb`,
# `pr`, `px` followed by a pattern number, optionally with a `cNN` colour
# variant. `3069bpb030` is tile 1x2 with pattern 030, and its geometry is the
# geometry of `3069b`. Printed variants mostly have no collision mesh of their
# own, so substituting the base part is not an approximation for our purposes --
# it is the same solid. The substitution is logged, because it does discard
# which decoration was asked for, and Level 3 will care about that.
_PATTERN_SUFFIX = re.compile(r"^(.*?)(p[a-z]*\d[0-9a-z]*)(c\d+)?$")


class PartLib:
    def __init__(self, res: float = C.DEFAULT_VOXEL_RES):
        self.res = float(res)
        self._parts: dict[str, Part] = {}
        self._cache_dir = C.CACHE / "parts" / f"r{self.res:g}"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._aliases = self._load_aliases()
        self._names = None
        self._lib = None

    # ------------------------------------------------------------- resolution

    @staticmethod
    def _load_aliases() -> dict[str, tuple[str, np.ndarray]]:
        """`src -> (dst, 4x4)`.

        The file is `{"rows": [...]}`, not a bare list -- reading it as a list
        silently yields an empty table, which then drops every aliased part from
        the collision test without any error. 401 of the 2,522 rows carry a
        non-identity transform, so the matrix has to come along: LDraw stores it
        as `[x y z a b c d e f g h i]`, translation first, then row-major 3x3.
        """
        p = C.PART_ALIASES
        if not p.exists():
            return {}
        with lzma.open(p, "rt") as fh:
            blob = json.load(fh)
        rows = blob.get("rows", blob) if isinstance(blob, dict) else blob
        out = {}
        for r in rows:
            m = r.get("final_matrix_3x4")
            T = np.eye(4)
            if m is not None and len(m) == 12:
                T[:3, 3] = m[0:3]
                T[:3, :3] = np.array(m[3:12], float).reshape(3, 3)
            out[str(r["src"]).lower().removesuffix(".dat")] = (
                str(r["dst"]).lower().removesuffix(".dat"), T)
        return out

    def canonical(self, stem: str) -> str:
        return self.resolve(stem)[0]

    def resolve(self, stem: str) -> tuple[str, np.ndarray]:
        """Canonical stem and the transform to compose into the placement.

        Alias chains are followed to their final target; the composed transform
        is the product along the chain, applied as `T_new = T_placement @ T_alias`.
        """
        return self.resolve_full(stem)[:2]

    def resolve_full(self, stem: str) -> tuple[str, np.ndarray, str]:
        """As `resolve`, plus how it was resolved: "exact", "alias",
        "pattern_base" or "unresolved"."""
        s0 = stem.lower().removesuffix(".dat")
        s = s0
        T = np.eye(4)
        seen = set()
        while s in self._aliases and s not in seen:
            seen.add(s)
            s, step = self._aliases[s]
            T = T @ step
        how = "alias" if s != s0 else "exact"

        if not (C.INSET / f"{s}.ply").exists():
            m = _PATTERN_SUFFIX.match(s)
            if m and m.group(1) and (C.INSET / f"{m.group(1)}.ply").exists():
                return m.group(1), T, "pattern_base"
            return s, T, "unresolved"
        return s, T, how

    def exists(self, stem: str) -> bool:
        return (C.INSET / f"{self.canonical(stem)}.ply").exists()

    def name(self, stem: str) -> str:
        if self._names is None:
            p = C.PART_NAMES
            self._names = json.load(open(p)) if p.exists() else {}
        return self._names.get(self.canonical(stem), "")

    # ------------------------------------------------------------------- load

    def get(self, stem: str) -> Part:
        s = self.canonical(stem)
        if s in self._parts:
            return self._parts[s]

        cache = self._cache_dir / f"{s}.npz"
        if cache.exists():
            try:
                d = np.load(cache)
                part = Part(s, float(d["volume"]), d["centroid"], d["bbox"],
                            np.unpackbits(d["occ"], count=int(np.prod(d["shape"])))
                              .astype(bool).reshape(tuple(d["shape"])),
                            d["origin"], self.res,
                            bool(d["watertight"]), int(d["n_boundary_edges"]),
                            str(d["volume_source"]))
                self._parts[s] = part
                return part
            except Exception:
                cache.unlink(missing_ok=True)

        ply = C.INSET / f"{s}.ply"
        if not ply.exists():
            raise KeyError(f"no collision mesh for part {stem!r} (canonical {s!r})")
        verts, faces = read_ply(ply)
        wt, nb, nnm = watertight(verts, faces)
        vol, cen = mesh_volume_centroid(verts, faces)
        occ, origin = voxelize_watertight(verts, faces, self.res)
        bbox = np.stack([verts.min(0), verts.max(0)])

        # The divergence theorem needs a closed, outward-wound surface, and 27%
        # of this library is neither. On an inverted or badly open mesh it
        # returns a NEGATIVE volume -- `32532a`, technic brick 6x8 with open
        # centre, comes out at -75,773 LDU^3 -- and a negative volume becomes a
        # negative mass, which drags the centre of mass the wrong way and
        # corrupts the equilibrium LP for every design containing that part.
        # So a non-positive exact volume falls back to counting occupied cells,
        # which cannot go negative, and the fallback is recorded.
        source = "mesh"
        if not np.isfinite(vol) or vol <= 0.0:
            vol = float(occ.sum()) * self.res ** 3
            cen = (origin + (np.argwhere(occ) + 0.5) * self.res).mean(0) \
                if occ.any() else verts.mean(0)
            source = "voxel"

        np.savez_compressed(cache, volume=vol, centroid=cen, bbox=bbox,
                            occ=np.packbits(occ.ravel()), shape=np.array(occ.shape),
                            origin=origin, watertight=wt, n_boundary_edges=nb + nnm,
                            volume_source=source)
        part = Part(s, vol, cen, bbox, occ, origin, self.res, wt, nb + nnm, source)
        self._parts[s] = part
        return part

    # ---------------------------------------------------------------- ports

    def ports(self, stem: str):
        """Port frames and kinds from lib_2048. Supervision only: nothing in
        Level 1 may read these, per the BrickForge geometry/supervision split.
        Returns (xf (K,4,4), kind (K,)) or (None, None) if the part is absent."""
        if self._lib is None:
            meta = np.load(C.LIB2048 / "part_meta.npz", allow_pickle=True)
            self._lib = {
                "stem": {str(v): i for i, v in enumerate(meta["stem"])},
                "port_ptr": np.load(C.LIB2048 / "port_ptr.npy"),
                "port_xf": np.load(C.LIB2048 / "port_xf.npy", mmap_mode="r"),
                "port_kind": np.load(C.LIB2048 / "port_kind.npy", mmap_mode="r"),
            }
        i = self._lib["stem"].get(self.canonical(stem))
        if i is None:
            return None, None
        a, b = self._lib["port_ptr"][i], self._lib["port_ptr"][i + 1]
        return np.asarray(self._lib["port_xf"][a:b]), np.asarray(self._lib["port_kind"][a:b])
