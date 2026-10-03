"""The one geometric primitive.

Every check in Level 1 -- interpenetration volume, contact, connectivity, centre
of mass, the support polygon -- is an operation on a per-part occupancy grid in
the part's own LDraw frame, computed once and cached. Whole-scene queries are
built by mapping world points back into each part's local grid, never by
rasterising parts into a shared grid, because a shared grid would add its own
aliasing on top of the parts'.

Occupancy is computed by winding accumulation along +Z: shoot a ray through the
centre of every (x, y) column, collect every triangle crossing with the sign of
its facing, and fill the spans where the accumulated winding number is positive.
Winding rather than even/odd parity, because a ray that grazes a shared seam
gives a column an odd hit count, and parity then mis-fills that column and, if
the pairing is done in bulk, every column after it.

Mass does not come from this grid. `mesh_volume` computes the enclosed volume
exactly by the divergence theorem, which is both cheaper and not subject to the
grid's systematic overestimate of thin features like studs and walls.
"""
from __future__ import annotations

import numpy as np

# Nudge column centres off the grid by an irrational fraction of a cell so a ray
# never lands exactly on a shared triangle edge, where parity is ill-defined.
_JITTER = 0.0193731

# Chunk size for the column x triangle broadcast. Keeps peak memory bounded
# regardless of part complexity.
_CHUNK = 8192


def voxelize_watertight(verts, faces, res, pad=1):
    """Occupancy grid of a closed triangle mesh.

    Returns (occ, origin) where occ is a boolean array indexed [ix, iy, iz] and
    the centre of cell (ix, iy, iz) is origin + (idx + 0.5) * res.
    """
    lo = verts.min(0) - pad * res
    hi = verts.max(0) + pad * res
    dims = np.maximum(np.ceil((hi - lo) / res).astype(int), 1)
    nx, ny, nz_dim = (int(d) for d in dims)

    gx = lo[0] + (np.arange(nx) + 0.5 + _JITTER) * res
    gy = lo[1] + (np.arange(ny) + 0.5 - _JITTER) * res
    cols = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # 2D barycentric setup in the xy plane, one per triangle.
    a = v0[:, :2]
    e1 = v1[:, :2] - a
    e2 = v2[:, :2] - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    live = np.abs(det) > 1e-12          # drop triangles that project to a line
    a, e1, e2, det = a[live], e1[live], e2[live], det[live]
    z0, z1, z2 = v0[live, 2], v1[live, 2], v2[live, 2]
    inv = 1.0 / det

    # Facing of each triangle along the ray. With outward winding, a triangle
    # whose normal opposes +Z is where the ray enters the solid.
    nz = np.cross(v1[live] - v0[live], v2[live] - v0[live])[:, 2]
    step = np.where(nz < 0.0, 1, -1).astype(np.int32)

    occ = np.zeros((nx, ny, nz_ := nz_dim), dtype=bool)
    flat = occ.reshape(-1, nz_)

    for s in range(0, len(cols), _CHUNK):
        blk = cols[s:s + _CHUNK]
        d = blk[:, None, :] - a[None, :, :]                       # (C, T, 2)
        u = (d[..., 0] * e2[None, :, 1] - d[..., 1] * e2[None, :, 0]) * inv
        v = (d[..., 1] * e1[None, :, 0] - d[..., 0] * e1[None, :, 1]) * inv
        hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        if not hit.any():
            continue
        ci, ti = np.nonzero(hit)
        zh = z0[ti] + u[ci, ti] * (z1[ti] - z0[ti]) + v[ci, ti] * (z2[ti] - z0[ti])
        sg = step[ti]

        # Sort hits by (column, z), then accumulate winding within each column.
        order = np.lexsort((zh, ci))
        ci, zh, sg = ci[order], zh[order], sg[order]
        cum = np.cumsum(sg)
        first = np.searchsorted(ci, ci, side="left")              # first hit of each column
        base = np.where(first > 0, cum[first - 1], 0)
        wind = cum - base                                         # winding after this hit

        # A span runs from hit j to hit j+1 when both belong to the same column
        # and the winding between them is positive.
        same = ci[:-1] == ci[1:]
        span = same & (wind[:-1] > 0)
        if not span.any():
            continue
        za, zb, cc = zh[:-1][span], zh[1:][span], ci[:-1][span]

        ia = np.ceil((za - lo[2]) / res - 0.5).astype(np.int64)
        ib = np.floor((zb - lo[2]) / res - 0.5).astype(np.int64)
        np.clip(ia, 0, nz_ - 1, out=ia)
        np.clip(ib, -1, nz_ - 1, out=ib)
        keep = ib >= ia
        for c, i0, i1 in zip(cc[keep], ia[keep], ib[keep]):
            flat[s + c, i0:i1 + 1] = True

    return occ, lo


def mesh_volume(verts, faces):
    """Enclosed volume of a closed mesh, exactly, by the divergence theorem.

    Signed sum of tetrahedra from the origin. Returns LDU**3, positive for
    outward winding, so the sign is also a winding check.
    """
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def mesh_volume_centroid(verts, faces):
    """Enclosed volume and its centroid, exactly, assuming uniform density.

    Decomposes the solid into tetrahedra on the origin. Each tetrahedron's
    centroid is the mean of its four vertices, one of which is the origin.
    """
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    v = vol6.sum() / 6.0
    if abs(v) < 1e-12:
        return 0.0, verts.mean(0)
    cen = ((a + b + c) / 4.0 * vol6[:, None]).sum(0) / vol6.sum()
    return float(v), cen


def watertight(verts, faces):
    """Is the mesh closed and orientable?

    Every undirected edge of a closed, consistently oriented surface appears
    exactly twice, once in each direction. Returns (ok, n_boundary_edges,
    n_nonmanifold_edges).

    This matters because `voxelize_watertight` and `mesh_volume` both assume a
    closed surface, and BrickForge's own notes record that the inset PLYs are
    not all watertight -- meshlib mis-answers on the open ones. A part that
    fails here gets its occupancy flagged rather than trusted.
    """
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, counts = np.unique(key, axis=0, return_counts=True)
    n_boundary = int((counts == 1).sum())
    n_nonmanifold = int((counts > 2).sum())
    return (n_boundary == 0 and n_nonmanifold == 0), n_boundary, n_nonmanifold


def occupied_centres(occ, origin, res):
    """World/local coordinates of every occupied cell centre."""
    idx = np.argwhere(occ)
    return origin + (idx + 0.5) * res


def sample_occupancy(occ, origin, res, pts):
    """Nearest-cell occupancy lookup for arbitrary points. Out of grid is False."""
    idx = np.floor((pts - origin) / res).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.array(occ.shape)), axis=1)
    out = np.zeros(len(pts), dtype=bool)
    if ok.any():
        j = idx[ok]
        out[ok] = occ[j[:, 0], j[:, 1], j[:, 2]]
    return out


def dilate(occ, iters=1):
    """6-connected dilation. Used to turn 'overlapping' into 'touching', because
    the inset meshes leave a 0.5 LDU gap between parts that really do touch."""
    out = occ.copy()
    for _ in range(iters):
        nxt = out.copy()
        nxt[1:] |= out[:-1]
        nxt[:-1] |= out[1:]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        nxt[:, :, 1:] |= out[:, :, :-1]
        nxt[:, :, :-1] |= out[:, :, 1:]
        out = nxt
    return out
