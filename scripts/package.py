"""Build the two release archives.

    python scripts/package.py code  [--out dist/]   # LMBuild-code.zip : everything git tracks (code, docs, task JSON,
                                                    #   knowledge, eval fixtures, scores, demos pages and GIFs)
    python scripts/package.py data  [--out dist/]   # LMBuild-data.tar : the large binaries .gitignore leaves out
                                                    #   (pool meshes, reference CAD, images, renders, LDraw/Artiverse assets)
    python scripts/package.py designs [--out dist/] # LMBuild-designs.tar : every scored main-setting design (designs/,
                                                    #   see designs/index.json; stage with scripts/stage_designs.py)
    python scripts/package.py sizes                 # what each archive would hold, without writing it

Unpacking LMBuild-data.tar at the repository root puts every file where benchmark/core_v2.json points.
Hard-linked duplicates are stored once in the tar (the zip stores them again).
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ["benchmark/assets/*", "benchmark/tasks/*/pool/*", "benchmark/tasks/*/reference/*", "benchmark/tasks/*/pool_sheet/*",
        "benchmark/tasks/*/images/*", "benchmark/tasks/*/reference_renders/*", "human_evaluation/renders/*",
        "demos/trajectories/*/raw_episode/*.glb", "demos/outcomes/final_designs/*.glb"]
DESIGNS = ["designs/*"]   # the scored designs: their own archive (tens of GB), never in code or data
NEVER = [".venv_eval/*", ".venv_render/*", ".cache/*", "dist/*", ".git/*", "*/__pycache__/*", "*.pyc",
         "results/v2/*/runs/*", "results/v2/*/external_core/*", "results/v2/*/judge_views/*", "results/v2/_*",
         "results/scores/my_scores.csv", "vllm_*.log"]
NEVER += [f"results/v2/*/{d}/*" for d in ("eval_v3", "eval_v32", "eval_v33", "eval_v345", "eval_v35", "eval_v37", "scratch")]


def files():
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if not d.startswith(".venv") and d not in (".git", "dist", "__pycache__", ".cache")]
        for f in fn:
            rel = (Path(dp) / f).relative_to(ROOT).as_posix()
            if not any(fnmatch.fnmatch(rel, p) for p in NEVER):
                yield rel


def split():
    code, data, designs = [], [], []
    for rel in sorted(files()):
        if any(fnmatch.fnmatch(rel, p) for p in DESIGNS):
            designs.append(rel)
        else:
            (data if any(fnmatch.fnmatch(rel, p) for p in DATA) else code).append(rel)
    return code, data, designs


def size(rels):
    return sum((ROOT / r).stat().st_size for r in rels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["code", "data", "designs", "sizes"])
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ns = ap.parse_args()
    code, data, designs = split()
    if ns.what == "sizes":
        print(f"code: {len(code)} files, {size(code) / 1e6:.1f} MB (uncompressed)")
        print(f"data: {len(data)} files, {size(data) / 1e9:.2f} GB")
        print(f"designs: {len(designs)} files, {size(designs) / 1e9:.2f} GB")
        return
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    if ns.what == "code":
        p = out / "LMBuild-code.zip"
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for r in code:
                z.write(ROOT / r, f"LMBuild/{r}")
    else:
        p = out / ("LMBuild-data.tar" if ns.what == "data" else "LMBuild-designs.tar")
        with tarfile.open(p, "w") as t:
            for r in (data if ns.what == "data" else designs):
                t.add(ROOT / r, r)
    print(f"{p}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
