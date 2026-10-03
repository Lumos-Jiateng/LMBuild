"""The truth format.

A design is a list of placed catalogue parts plus a manifest of claims. LDraw is
the wire format because all three generators already speak it: BrickGPT writes
`.ldr` from its voxel grid, BrickNet has `graph_to_ldr`, and LegoACE emits `.ldr`
natively. Everything downstream reads this class and nothing else.

Geometry and claims are kept apart. `parts` is what the scorer measures;
`manifest` is what the design asserts about itself, and Level 1 never reads it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
_TYPE1 = re.compile(r"^\s*1\s+(\S+)\s+" + r"\s+".join([_NUM] * 12) + r"\s+(\S+)\s*$")


@dataclass
class Design:
    """N placed parts. `poses` are 4x4 homogeneous, LDraw frame, LDU, +Y down."""

    stems: list[str] = field(default_factory=list)
    colors: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    poses: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4), np.float64))
    steps: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    manifest: dict = field(default_factory=dict)
    source: str = "unknown"
    design_id: str = ""
    ingest: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.stems)

    # --------------------------------------------------------------- LDraw io

    @classmethod
    def from_ldr(cls, text: str, source: str = "ldr", design_id: str = "") -> "Design":
        """Parse a flat LDraw model. Type-1 lines only; `0 STEP` advances the step
        counter. Unparseable and non-part lines are counted, not silently dropped:
        the counts land in `ingest` and are reported next to every score."""
        stems, colors, poses, steps = [], [], [], []
        step = 0
        n_skipped = n_subfile = 0
        for ln in text.splitlines():
            t = ln.strip()
            if not t:
                continue
            if t.startswith("0"):
                if re.match(r"^0\s+STEP\b", t, re.I):
                    step += 1
                continue
            if not t.startswith("1"):
                n_skipped += 1
                continue
            m = _TYPE1.match(t)
            if m is None:
                n_skipped += 1
                continue
            g = m.groups()
            ref = g[13]
            if not ref.lower().endswith(".dat"):
                n_subfile += 1          # .ldr/.mpd subfile reference: not flat
                continue
            x, y, z = (float(v) for v in g[1:4])
            a, b, c, d, e, f, gg, h, i = (float(v) for v in g[4:13])
            T = np.eye(4)
            T[:3, :3] = [[a, b, c], [d, e, f], [gg, h, i]]
            T[:3, 3] = (x, y, z)
            stems.append(ref.lower().removesuffix(".dat"))
            try:
                colors.append(int(g[0]))
            except ValueError:
                colors.append(16)
            poses.append(T)
            steps.append(step)

        return cls(
            stems=stems,
            colors=np.array(colors, np.int32),
            poses=np.array(poses, np.float64).reshape(-1, 4, 4),
            steps=np.array(steps, np.int32),
            source=source,
            design_id=design_id,
            ingest={"lines_skipped": n_skipped, "subfile_refs": n_subfile},
        )

    @classmethod
    def load(cls, path, source: str = "ldr") -> "Design":
        p = Path(path)
        return cls.from_ldr(p.read_text(errors="replace"), source=source, design_id=p.stem)

    def to_ldr(self) -> str:
        out = ["0 FILE " + (self.design_id or "design") + ".ldr"]
        last = -1
        for k in range(len(self)):
            if len(self.steps) and self.steps[k] != last and last >= 0:
                out.append("0 STEP")
            last = self.steps[k] if len(self.steps) else 0
            M = self.poses[k]
            r = " ".join(f"{v:g}" for v in M[:3, :3].ravel())
            t = " ".join(f"{v:g}" for v in M[:3, 3])
            out.append(f"1 {int(self.colors[k])} {t} {r} {self.stems[k]}.dat")
        return "\n".join(out) + "\n"

    def save(self, path) -> None:
        Path(path).write_text(self.to_ldr())

    # ----------------------------------------------------------------- misc

    def hash(self) -> str:
        """Content hash over geometry only. Order-invariant: parts are sorted by
        stem and then by their rounded pose bytes, so the same set of placements
        listed in any order hashes the same. Rounded to 1e-3 LDU so float noise
        in a round trip does not change identity."""
        rows = sorted((self.stems[k], np.round(self.poses[k], 3).tobytes()) for k in range(len(self)))
        h = hashlib.sha256()
        for stem, blob in rows:
            h.update(stem.encode())
            h.update(blob)
        return h.hexdigest()[:16]

    def subset(self, idx) -> "Design":
        idx = np.asarray(idx, dtype=np.int64)
        return Design([self.stems[i] for i in idx], self.colors[idx], self.poses[idx],
                      self.steps[idx] if len(self.steps) else self.steps,
                      dict(self.manifest), self.source, self.design_id, dict(self.ingest))

    def summary(self) -> dict:
        return {
            "design_id": self.design_id,
            "source": self.source,
            "n_parts": len(self),
            "n_distinct_parts": len(set(self.stems)),
            "hash": self.hash(),
            **self.ingest,
        }
