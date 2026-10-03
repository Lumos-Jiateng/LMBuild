"""The one primitive: filled voxel occupancy per part, on a shared world grid.

Cell (i, j, k) covers [i, i+1) * res in every axis, so any two parts can be
intersected by index arithmetic without resampling. A part is voxelised by
sampling its surface densely (about 16 samples per cell face), marking the
cells hit, and filling the enclosed interior with a 6-connected flood from
outside. The surface samples are kept, because the support polygon and
symmetry use points rather than cells.

Resolution limits every volume here, exactly as in v1 (LIMITATIONS §3): a
cell counts as full when the surface or the interior touches it, so thin
parts are over-counted. `res` travels with every number that depends on it.
Open meshes whose holes are larger than a cell cannot be filled and come back
as shells; `filled_frac` exposes that, and trimesh's watertight flag is
recorded next to it.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

RES = 0.004          # metres
SAMPLE_SPACING = 0.25  # of a cell
MAX_SAMPLES = 4_000_000


class Occ:
    """Occupancy of one part: `grid` filled, `core` eroded by one cell, `halo` dilated by one."""

    def __init__(self, lo, grid, points, res, shell_count):
        self.lo = np.asarray(lo, np.int64)
        self.grid = grid
        self.res = res
        self.points = points
        self.shell_count = int(shell_count)
        self._core = None
        self._halo = None

    @property
    def hi(self):
        return self.lo + np.array(self.grid.shape)

    @property
    def core(self):
        if self._core is None:
            self._core = ndimage.binary_erosion(self.grid)
        return self._core

    @property
    def halo(self):
        if self._halo is None:
            self._halo = ndimage.binary_dilation(self.grid, structure=np.ones((3, 3, 3), bool))
        return self._halo

    @property
    def count(self):
        return int(self.grid.sum())

    @property
    def volume(self):
        return self.count * self.res ** 3

    @property
    def filled_frac(self):
        return 1.0 - self.shell_count / max(self.count, 1)

    def centroid(self):
        idx = np.argwhere(self.grid)
        if not len(idx):
            return self.points.mean(0)
        return (idx.mean(0) + self.lo + 0.5) * self.res


def occ_save(o: Occ, path):
    """Store an occupancy exactly (bit-packed grids, lazily computed core/halo included when present)."""
    shape = np.array(o.grid.shape)
    arrays = {"lo": o.lo, "shape": shape, "grid": np.packbits(o.grid), "points": o.points, "shell_count": np.array(o.shell_count),
              "res": np.array(o.res)}
    if o._core is not None:
        arrays["core"] = np.packbits(o._core)
    if o._halo is not None:
        arrays["halo"] = np.packbits(o._halo)
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def occ_load(path) -> Occ:
    z = np.load(path)
    shape = tuple(int(x) for x in z["shape"])
    n = int(np.prod(shape))
    unpack = lambda k: np.unpackbits(z[k], count=n).reshape(shape).astype(bool)
    o = Occ(z["lo"], unpack("grid"), z["points"], float(z["res"]), int(z["shell_count"]))
    if "core" in z.files:
        o._core = unpack("core")
    if "halo" in z.files:
        o._halo = unpack("halo")
    return o


def surface_points(vertices, faces, res, seed=0):
    import trimesh
    m = trimesh.Trimesh(vertices, faces, process=False)
    area = float(m.area)
    n = int(np.clip(area / (res * SAMPLE_SPACING) ** 2, 2000, MAX_SAMPLES))
    if len(faces):
        pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
        pts = np.vstack([pts, vertices])
    else:
        pts = np.asarray(vertices, float)
    return pts


MAX_CELLS = 400_000_000   # 1.6 m x 1.6 m x 1.6 m of 4 mm cells is ~64M; beyond this a part is tens of metres across


class VoxelTooLarge(ValueError):
    """A part whose voxel grid would not fit in memory (typically placed or created tens of metres across)."""


def voxelize(vertices, faces, res=RES) -> Occ:
    ext = (np.asarray(vertices).max(0) - np.asarray(vertices).min(0)) / res + 5
    if float(np.prod(ext)) > MAX_CELLS:
        raise VoxelTooLarge(f"part spans {np.round(ext[:3] * res, 2).tolist()} m, too large to voxelise at {res * 1000:.0f} mm")
    pts = surface_points(vertices, faces, res)
    idx = np.floor(pts / res).astype(np.int64)
    lo = idx.min(0) - 2
    shape = idx.max(0) - lo + 3
    shell = np.zeros(shape, bool)
    shell[tuple((idx - lo).T)] = True
    filled = ndimage.binary_fill_holes(shell)
    keep = pts if len(pts) <= 60000 else pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]
    return Occ(lo, filled, keep, res, shell.sum())


def _slices(a: Occ, b: Occ):
    lo = np.maximum(a.lo, b.lo)
    hi = np.minimum(a.hi, b.hi)
    if np.any(hi <= lo):
        return None
    sa = tuple(slice(int(l - a.lo[i]), int(h - a.lo[i])) for i, (l, h) in enumerate(zip(lo, hi)))
    sb = tuple(slice(int(l - b.lo[i]), int(h - b.lo[i])) for i, (l, h) in enumerate(zip(lo, hi)))
    return sa, sb


def overlap_cells(a: Occ, b: Occ, deep=True) -> int:
    """Cells in both. deep=True intersects the eroded cores, so two parts that
    merely touch (sharing a boundary layer) read zero; anything reported is at
    least about two cells of interpenetration."""
    s = _slices(a, b)
    if s is None:
        return 0
    A = a.core if deep else a.grid
    B = b.core if deep else b.grid
    return int(np.count_nonzero(A[s[0]] & B[s[1]]))


def contact_cells(a: Occ, b: Occ) -> int:
    """Cells of b that lie in a's one-cell halo, and of a in b's: the smaller of the two counts.

    Times res^2 this is an estimate of the shared interface area (a flat 4 cm x 4 cm joint reads
    100 cells = 16 cm^2). It is a proxy: a rough interface reads larger than the area a glue line
    or a screw flange could actually use."""
    s = _slices(a, b)
    if s is None:
        return 0
    ab = int(np.count_nonzero(a.halo[s[0]] & b.grid[s[1]]))
    ba = int(np.count_nonzero(a.grid[s[0]] & b.halo[s[1]]))
    return min(ab, ba)


def contact_z_layers(a: Occ, b: Occ):
    """(lo, hi) grid-z indices of the contact region in world cells, or None when they do not touch."""
    s = _slices(a, b)
    if s is None:
        return None
    inter = a.halo[s[0]] & b.grid[s[1]]
    zs = np.nonzero(inter.any(axis=(0, 1)))[0]
    if not len(zs):
        return None
    z0 = max(a.lo[2], b.lo[2])
    return int(zs.min() + z0), int(zs.max() + z0)


def touching(a: Occ, b: Occ) -> bool:
    """Within one cell of each other (surface gap <= res)."""
    lo = np.maximum(a.lo - 1, b.lo)
    hi = np.minimum(a.hi + 1, b.hi)
    if np.any(hi <= lo):
        return False
    s = _slices(a, b)
    if s is None:
        return False
    return bool(np.any(a.halo[s[0]] & b.grid[s[1]]))


def union(occs, res=RES):
    occs = [o for o in occs if o is not None]
    if not occs:
        return None
    lo = np.min([o.lo for o in occs], 0)
    hi = np.max([o.hi for o in occs], 0)
    g = np.zeros(hi - lo, bool)
    for o in occs:
        sl = tuple(slice(int(o.lo[i] - lo[i]), int(o.hi[i] - lo[i])) for i in range(3))
        g[sl] |= o.grid
    pts = np.vstack([o.points for o in occs])
    return Occ(lo, g, pts, res, 0)


# ---------------------------------------------------------------- stability

def support_polygon(points, band):
    """Ground contact points (z within `band` of the lowest point), projected to XY."""
    z0 = points[:, 2].min()
    return points[points[:, 2] <= z0 + band][:, :2], float(z0)


def _hull(xy):
    from scipy.spatial import ConvexHull, QhullError
    xy = np.unique(np.round(xy, 6), axis=0)
    if len(xy) < 3:
        return xy
    try:
        h = ConvexHull(xy)
        return xy[h.vertices]
    except QhullError:
        return xy


def tilt_and_margin(points, com, band=0.005, n_dirs=720):
    """Critical tilt angle (deg, min over azimuth) and normalised support margin.

    Tilting the object by angle t about the hull's supporting line in direction
    u lifts the CoM over that line when atan(d(u) / h) < t, with d(u) = h_H(u) -
    u . c the distance from the CoM projection to the supporting line and h the
    CoM height above the ground. Negative values mean the CoM projection is
    already outside the support polygon.
    """
    xy, z0 = support_polygon(points, band)
    hv = _hull(xy)
    c = np.asarray(com[:2])
    h = max(float(com[2] - z0), 1e-6)
    th = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    U = np.stack([np.cos(th), np.sin(th)], 1)
    d = (hv @ U.T).max(0) - U @ c
    ang = np.degrees(np.arctan2(d, h))
    k = int(np.argmin(ang))
    # margin: signed distance to the hull boundary, normalised by the radius of a disc of equal area
    if len(hv) >= 3:
        x, y = hv[:, 0], hv[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    else:
        area = 0.0
    r_eq = np.sqrt(area / np.pi) if area > 0 else 0.0
    margin = float(d.min())
    return {"critical_tilt_deg": float(ang[k]), "tilt_direction_deg": float(np.degrees(th[k])),
            "support_margin_m": margin, "support_margin_norm": margin / r_eq if r_eq > 0 else float("-inf") if margin < 0 else 0.0,
            "support_area_m2": float(area), "n_support_vertices": int(len(hv)), "com_height_m": h, "ground_z": z0}
