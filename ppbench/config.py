"""Where the data lives, and the constants that are not ours to choose."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# LDraw data. The release ships the subset the 50 core tasks need (their LDraw references and LEGO pool parts) under
# benchmark/assets/ldraw. To load arbitrary LDraw outputs (e.g. brick generators), point PPBENCH_LDRAW_DATA at a full
# BrickNet data directory holding inset/<stem>.ply, part_aliases.json.xz and part_names.json.
LDRAW_DATA = Path(os.environ.get("PPBENCH_LDRAW_DATA", ROOT / "benchmark" / "assets" / "ldraw"))
INSET = LDRAW_DATA / "inset"             # watertight collision meshes, inset 0.25 LDU
PART_ALIASES = LDRAW_DATA / "part_aliases.json.xz"
PART_NAMES = LDRAW_DATA / "part_names.json"
LIB2048 = Path(os.environ.get("PPBENCH_LIB2048", LDRAW_DATA / "lib_2048"))   # connector ports (v1 lattice only; unused by v2)
# Brick-generator baselines only (ppbench/baselines): a BrickForge checkout with BrickNet, BrickGPT and LegoACE.
BRICKFORGE = Path(os.environ.get("PPBENCH_BRICKFORGE", ROOT / "third_party" / "BrickForge"))
BRICKNET_SRC = BRICKFORGE / "third_party/BrickNet/src"
BRICKGPT_SRC = BRICKFORGE / "third_party/BrickGPT/src"
CACHE = Path(os.environ.get("PPBENCH_CACHE", ROOT / ".cache"))

# LDraw units. 1 LDU = 0.4 mm. +Y points DOWN, so "up" is -Y and the ground is at max Y.
LDU_MM = 0.4
STUD_PITCH = 20.0        # LDU between stud centres
BRICK_HEIGHT = 24.0      # LDU, one brick course
PLATE_HEIGHT = 8.0       # LDU
UP = -1.0                # sign of the up direction along Y

# ABS. Density is standard; the joint capacities in metrics/stability.py are NOT
# and are marked as placeholders there.
ABS_DENSITY_KG_M3 = 1040.0
LDU3_TO_M3 = (LDU_MM * 1e-3) ** 3
GRAVITY = 9.80665

DEFAULT_VOXEL_RES = 2.0  # LDU
