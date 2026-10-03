"""A small software rasterizer: orthographic views of posed part meshes, with a per-pixel part id.

The Level-A part grounding shows a judge one component of a design at a time, highlighted inside the
whole object. That needs an id buffer (which part is visible where) and a solo mask per part (where the
part is, visible or hidden), for tens of thousands of parts. Blender is far too slow for that, and an
EGL context would land on somebody else's GPU, so this is numpy: every triangle is expanded to the
pixels of its bounding box, tested with edge functions, and the nearest fragment per pixel wins.
"""
from __future__ import annotations

import math

import numpy as np

VIEWS = {   # name -> (azimuth deg, elevation deg); the design's front faces -Y, so "front" looks along +Y
    "front_left": (-120.0, 25.0), "front_right": (-60.0, 25.0),
    "back_left": (120.0, 25.0), "back_right": (60.0, 25.0),
    "top": (-90.0, 75.0), "low_front": (-90.0, 5.0),
}


def camera(az_deg, el_deg):
    """Orthonormal (right, up, toward-viewer) for a camera at azimuth/elevation around +Z."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    back = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])   # points at the viewer
    right = np.cross([0.0, 0.0, 1.0], back)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return right, up, back


class Frame:
    """Projection of world points to pixels for one view, fitted to a bounding box."""

    def __init__(self, lo, hi, view, size=256, margin=0.06):
        self.right, self.up, self.back = camera(*VIEWS[view] if isinstance(view, str) else view)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        u, v = corners @ self.right, corners @ self.up
        self.cu, self.cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
        span = max(u.max() - u.min(), v.max() - v.min(), 1e-9)
        self.scale = size * (1 - 2 * margin) / span
        self.size = size

    def project(self, V):
        x = (V @ self.right - self.cu) * self.scale + self.size / 2
        y = self.size / 2 - (V @ self.up - self.cv) * self.scale
        z = V @ self.back            # larger = nearer the viewer
        return x, y, z


def rasterize(meshes, frame: Frame, ids=None, max_candidates=6_000_000):
    """meshes: [(V, F)]. Returns (id buffer int32 with -1 background, depth, shade in [0,1])."""
    S = frame.size
    ids = list(range(len(meshes))) if ids is None else ids
    Xs, Ys, Zs, Ns, Is = [], [], [], [], []
    for (V, F), pid in zip(meshes, ids):
        V = np.asarray(V, float)
        F = np.asarray(F, np.int64)
        if not len(F):
            continue
        x, y, z = frame.project(V)
        tri = F
        Xs.append(x[tri]); Ys.append(y[tri]); Zs.append(z[tri])
        n = np.cross(V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]])
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        Ns.append(n / np.maximum(nn, 1e-20))
        Is.append(np.full(len(tri), pid, np.int32))
    idbuf = np.full(S * S, -1, np.int32)
    depth = np.full(S * S, -np.inf)
    shade = np.zeros(S * S)
    if not Xs:
        return idbuf.reshape(S, S), depth.reshape(S, S), shade.reshape(S, S)
    X, Y, Z = np.vstack(Xs), np.vstack(Ys), np.vstack(Zs)
    N, I = np.vstack(Ns), np.concatenate(Is)
    light = 0.8 * frame.back + 0.45 * frame.up + 0.25 * frame.right
    light /= np.linalg.norm(light)
    lum = 0.30 + 0.70 * np.abs(N @ light)
    x0 = np.clip(np.floor(X.min(1)), 0, S - 1).astype(np.int64)
    x1 = np.clip(np.ceil(X.max(1)), 0, S - 1).astype(np.int64)
    y0 = np.clip(np.floor(Y.min(1)), 0, S - 1).astype(np.int64)
    y1 = np.clip(np.ceil(Y.max(1)), 0, S - 1).astype(np.int64)
    keep = (X.max(1) >= 0) & (X.min(1) < S) & (Y.max(1) >= 0) & (Y.min(1) < S)
    w = (x1 - x0 + 1) * (y1 - y0 + 1) * keep
    order = np.nonzero(w > 0)[0]
    # chunk so no pass materialises more than max_candidates fragments
    cw = np.cumsum(w[order])
    starts = [0]
    while starts[-1] < len(order):
        base = cw[starts[-1] - 1] if starts[-1] > 0 else 0
        nxt = int(np.searchsorted(cw, base + max_candidates, side="right"))
        starts.append(max(nxt, starts[-1] + 1))
    for a, b in zip(starts[:-1], starts[1:]):
        t = order[a:b]
        wt = w[t]
        rep = np.repeat(np.arange(len(t)), wt)
        off = np.arange(int(wt.sum())) - np.repeat(np.cumsum(wt) - wt, wt)
        bw = (x1[t] - x0[t] + 1)[rep]
        px = x0[t][rep] + off % bw
        py = y0[t][rep] + off // bw
        cx, cy = px + 0.5, py + 0.5
        ax, ay, bx, by, qx, qy = (X[t, 0][rep], Y[t, 0][rep], X[t, 1][rep], Y[t, 1][rep], X[t, 2][rep], Y[t, 2][rep])
        area = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
        ok = np.abs(area) > 1e-12
        w0 = ((bx - cx) * (qy - cy) - (by - cy) * (qx - cx))
        w1 = ((qx - cx) * (ay - cy) - (qy - cy) * (ax - cx))
        w2 = ((ax - cx) * (by - cy) - (ay - cy) * (bx - cx))
        sgn = np.sign(area)
        inside = ok & (w0 * sgn >= -1e-9) & (w1 * sgn >= -1e-9) & (w2 * sgn >= -1e-9)
        if not inside.any():
            continue
        l0, l1, l2 = w0[inside] / area[inside], w1[inside] / area[inside], w2[inside] / area[inside]
        tt = t[rep[inside]]
        zz = l0 * Z[tt, 0] + l1 * Z[tt, 1] + l2 * Z[tt, 2]
        pix = (py[inside] * S + px[inside]).astype(np.int64)
        # nearest fragment per pixel: sort by (pixel, -depth) and keep the first of each pixel
        o = np.lexsort((-zz, pix))
        pix, zz, tt = pix[o], zz[o], tt[o]
        first = np.ones(len(pix), bool)
        first[1:] = pix[1:] != pix[:-1]
        pix, zz, tt = pix[first], zz[first], tt[first]
        better = zz > depth[pix]
        pix, zz, tt = pix[better], zz[better], tt[better]
        depth[pix] = zz
        idbuf[pix] = I[tt]
        shade[pix] = lum[tt]
    return idbuf.reshape(S, S), depth.reshape(S, S), shade.reshape(S, S)
