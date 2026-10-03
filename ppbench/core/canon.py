"""Ingest canonicalisation.

Two things happen to a design before it is measured, and both are counted rather
than done quietly. Obsolete LDraw stems are rewritten to their canonical target
with the alias transform composed into the placement, and parts with no
collision mesh are separated out. The counts land in `Design.ingest` and travel
with every score, because a design that needed heavy repair before it could be
scored has already told you something.
"""
from __future__ import annotations

import numpy as np

from ppbench.core.ir import Design
from ppbench.core.partlib import PartLib


def resolve(design: Design, lib: PartLib | None = None) -> Design:
    lib = lib or PartLib()
    stems, poses, missing = [], [], []
    how_counts = {"exact": 0, "alias": 0, "pattern_base": 0, "unresolved": 0, "created": 0}
    for k in range(len(design)):
        s, T, how = lib.resolve_full(design.stems[k])
        how_counts[how] += 1
        stems.append(s)
        poses.append(design.poses[k] @ T)
        if how == "unresolved":
            missing.append((k, design.stems[k]))
    n_alias = how_counts["alias"]

    out = Design(
        stems=stems,
        colors=design.colors.copy(),
        poses=np.array(poses, np.float64).reshape(-1, 4, 4),
        steps=design.steps.copy(),
        manifest=dict(design.manifest),
        source=design.source,
        design_id=design.design_id,
        ingest={**design.ingest,
                "n_aliased": n_alias,
                "n_pattern_substituted": how_counts["pattern_base"],
                "n_created": how_counts["created"],
                "n_missing_mesh": len(missing),
                "missing_stems": sorted({m[1] for m in missing}),
                "missing_idx": [m[0] for m in missing],
                "resolve_rate": 1.0 - len(missing) / max(len(design), 1)},
    )
    return out
